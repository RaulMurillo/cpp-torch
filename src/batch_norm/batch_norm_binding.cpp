#include <torch/torch.h>
#include "batch_norm_cpu.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &batch_norm_forward, "batch_norm forward (CPU)");
    m.def("backward", &batch_norm_backward, "batch_norm backward (CPU)");
}
