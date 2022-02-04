#include <torch/torch.h>
#include "mse_cpu.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &mse_forward, "mse forward (CPU)");
    m.def("backward", &mse_backward, "mse backward (CPU)");
}
