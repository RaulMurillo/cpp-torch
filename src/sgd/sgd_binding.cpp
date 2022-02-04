#include <torch/torch.h>
#include "sgd_cpu.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("sgd", &sgd_function, "sgd computation (CPU)");
}
