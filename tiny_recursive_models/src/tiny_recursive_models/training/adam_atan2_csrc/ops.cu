#include "adam_atan2.h"

#include <torch/extension.h>


namespace adam_atan2 {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adam_atan2_cuda_impl_", &adam_atan2_cuda_impl_, "Adam-atan2 Fused Implementation");
}

}  // namespace adam_atan2
