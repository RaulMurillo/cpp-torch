#include <torch/extension.h>

#include <iostream>
#include <vector>
#include <omp.h>

// ONLY NECCESARY FOR CUDA EXTENSIONS
// // C++ interface
// // NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
// #define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
// #define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be contiguous")
// #define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void param_add(torch::Tensor& param,
               const torch::Tensor& d_p,
               double lr)
{
    param.add_(d_p, lr);
}

void sgd_function(std::vector<torch::Tensor> params,
                  std::vector<torch::Tensor> d_p_list,
                  std::vector<std::optional<torch::Tensor>> momentum_buffer_list,
                  double weight_decay,
                  double momentum,
                  double lr,
                  double dampening,
                  bool nesterov)
{
    const size_t num_params = params.size();

    for (size_t i = 0; i < num_params; ++i)
    {
        torch::Tensor param = params[i];
        torch::Tensor d_p = d_p_list[i];
        d_p.add_(param, weight_decay);

        if (momentum != 0)
        {
            torch::Tensor buf = torch::empty_like(d_p);

            if (!momentum_buffer_list[i].has_value())
            {
                buf = d_p.clone().detach();
                momentum_buffer_list[i] = buf;
            }
            else
            {
                buf = momentum_buffer_list[i].value();
                buf.mul_(momentum).add_(d_p, 1 - dampening);
            }

            if (nesterov)
            {
                d_p.add_(buf, momentum);
            }
            else
            {
                d_p = buf;
            }
        }

        params[i].add_(d_p, -lr);
    }
}