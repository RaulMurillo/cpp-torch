#include <torch/torch.h>
#include "linear_cpu.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &linear_forward, "linear forward (CPU)");
    m.def("backward", &linear_backward, "linear backward (CPU)");
}
