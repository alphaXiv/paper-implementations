"""Custom optimizers vendored into the project."""

# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch
import os
from typing import List, Tuple, Union
from torch import Tensor
from torch.optim.optimizer import Optimizer, ParamsT

# Compile the CUDA extension on-the-fly using torch.utils.cpp_extension.load
# This ensures the backend is built properly with the correct CUDA architecture
_adam_atan2_backend = None

def _get_adam_backend():
    global _adam_atan2_backend
    if _adam_atan2_backend is None:
        from torch.utils.cpp_extension import load
        
        # Get the directory containing this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csrc_dir = os.path.join(current_dir, "adam_atan2_csrc")
        
        # Compile the CUDA extension
        _adam_atan2_backend = load(
            name="adam_atan2_backend",
            sources=[
                os.path.join(csrc_dir, "ops.cu"),
                os.path.join(csrc_dir, "adam_atan2.cu"),
            ],
            extra_include_paths=[csrc_dir],
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=[
                "-O2",
                "-std=c++17",
                "--expt-extended-lambda",
            ],
            verbose=False,
        )
    return _adam_atan2_backend


class AdamATan2(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 1e-2
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps
    ):
        for p in group["params"]:
            if p.grad is None:
                continue

            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = (
                    torch.zeros((), dtype=torch.float32, device=p.device)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])

    def step(self):
        """Perform a single optimization step.
        """
        self._cuda_graph_capture_health_check()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps
            )

            _adam_atan2(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"]
            )


def _adam_atan2(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float
) -> None:
    if not params:
        return

    # We only support scalar lr.
    assert not isinstance(lr, Tensor)

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps])
    for (device, _), ((device_params,
                       device_grads,
                       device_exp_avgs,
                       device_exp_avg_sqs,
                       device_state_steps, ), _) in grouped_tensors.items():
        torch._foreach_add_(device_state_steps, 1)
        backend = _get_adam_backend()
        backend.adam_atan2_cuda_impl_(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_state_steps,
            lr,
            beta1,
            beta2,
            weight_decay
        )
