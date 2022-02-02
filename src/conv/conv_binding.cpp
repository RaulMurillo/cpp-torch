#include <torch/torch.h>
#include "conv_cpu.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &conv_forward, "conv forward (CPU)");
    m.def("backward", &conv_backward, "conv backward (CPU)");
}
