#include "adam_atan2.h"

#include <ATen/core/Tensor.h>
#include <ATen/native/cuda/ForeachFunctors.cuh>
#include <ATen/native/cuda/MultiTensorApply.cuh>
#include <ATen/native/cuda/Pow.cuh>
#include <utility>


namespace adam_atan2 {

using at::native::kILP;

constexpr int kArgsDepth = 4;

constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kExpAvgIdx = 2;
constexpr uint8_t kExpAvgSqIdx = 3;

template <typename T>
__device__ __forceinline__ T lerp(const T v0, const T v1, const T t) {
    // NOTE(one): Identical to PyTorch when t < 0.5
    // https://github.com/pytorch/pytorch/blob/b7f25226929e70187a9f36c393665abad0b25190/aten/src/ATen/native/Lerp.h#L21
    return fma(t, v1, fma(-t, v0, v0));
}

template <typename scalar_type, typename opmath_t>
__device__ __forceinline__ void adam_math(
    scalar_type r_args[kArgsDepth][kILP],
    const opmath_t &step_size,
    const opmath_t &wd_alpha,
    const opmath_t &mbeta1,
    const opmath_t &mbeta2,
    const opmath_t &bias_correction2_sqrt)
{
#pragma unroll
    for (int ii = 0; ii < kILP; ii++)
    {
        // Load values.
        opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
        const opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);

        opmath_t exp_avg = static_cast<opmath_t>(r_args[kExpAvgIdx][ii]);
        opmath_t exp_avg_sq = static_cast<opmath_t>(r_args[kExpAvgSqIdx][ii]);

        param *= wd_alpha;

        exp_avg = lerp(exp_avg, grad, mbeta1);
        exp_avg_sq = lerp(exp_avg_sq, grad * grad, mbeta2);

        const opmath_t denom = std::sqrt(exp_avg_sq) / bias_correction2_sqrt;
        param -= step_size * std::atan2(exp_avg, denom);

        // Store results.
        r_args[kParamIdx][ii] = param;
        r_args[kExpAvgIdx][ii] = exp_avg;
        r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
    }
}

template <typename scalar_type>
struct FusedAdamMathFunctor {
  using opmath_t = at::opmath_type<scalar_type>;
  __device__ __forceinline__ void operator()(
      int chunk_size,
      at::native::FusedOptimizerTensorListMetadata<kArgsDepth>& tl,
      const double& lr,
      const double& beta1,
      const double& beta2,
      const double& weight_decay) {
    const auto tensor_loc = tl.block_to_tensor[blockIdx.x];
    const auto chunk_idx = tl.block_to_chunk[blockIdx.x];

    const auto [step_size, wd_alpha, bias_correction2_sqrt, mbeta1, mbeta2] = [&]() -> std::tuple<opmath_t, opmath_t, opmath_t, opmath_t, opmath_t> {
      auto* step_count = reinterpret_cast<const float*>(tl.state_steps_addresses[tensor_loc]);
      const auto bias_correction1 = 1 - at::native::pow_(beta1, *step_count);
      const auto bias_correction2 = 1 - at::native::pow_(beta2, *step_count);
      const auto bias_correction2_sqrt = std::sqrt(bias_correction2);

      return {
        static_cast<opmath_t>(lr / bias_correction1),
        static_cast<opmath_t>(1 - lr * weight_decay),
        static_cast<opmath_t>(bias_correction2_sqrt),
        static_cast<opmath_t>(1 - beta1),
        static_cast<opmath_t>(1 - beta2)
      };
    }();

    scalar_type* args[kArgsDepth];
    scalar_type r_args[kArgsDepth][kILP];
    const auto n = tl.numel_for_tensor[tensor_loc] - chunk_idx * chunk_size;

    const bool all_aligned{
        at::native::init_args<kArgsDepth>(args, tl, chunk_idx, chunk_size, tensor_loc)};
    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
#pragma unroll
        for (int i = 0; i < kArgsDepth; i++) {
          at::native::load_store(r_args[i], args[i], 0, i_start);
        }
        adam_math(
            r_args,
            step_size,
            wd_alpha,
            mbeta1,
            mbeta2,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < kArgsDepth; i++) {
          if (i != kGradIdx) {
            at::native::load_store(args[i], r_args[i], i_start, 0);
          }
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        at::native::load_args<kArgsDepth>(r_args, args, i_start, chunk_size, n);
        adam_math(
            r_args,
            step_size,
            wd_alpha,
            mbeta1,
            mbeta2,
            bias_correction2_sqrt);
#pragma unroll
        for (int i = 0; i < kArgsDepth; i++) {
          if (i != kGradIdx) {
            at::native::store_args(args[i], r_args[i], i_start, chunk_size, n);
          }
        }
      }
    }
  }
};

void adam_atan2_cuda_impl_(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay) {
  std::vector<std::vector<at::Tensor>> tensor_lists{params, grads, exp_avgs, exp_avg_sqs};

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      params[0].scalar_type(),
      "adam_atan2_kernel_cuda",
      [&]() {
        at::native::multi_tensor_apply_for_fused_optimizer<kArgsDepth>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t>(),
            lr,
            beta1,
            beta2,
            weight_decay);
      });
}

} // namespace adam_atan2
