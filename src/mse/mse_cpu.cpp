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

torch::Scalar mse_mean_forward(const torch::Tensor input,
                                const torch::Tensor target)
{
    // torch.mean((y_pred - y)**2)
    float d = 0.0f;
    const size_t elems = input.numel();
    float* y = (float*)input.data_ptr();
    float* t = (float*)target.data_ptr();
    for (size_t i = 0; i < elems; ++i) d += (y[i] - t[i]) * (y[i] - t[i]);

    return torch::Scalar(d/static_cast<float>(elems));
}

torch::Scalar mse_sum_forward(const torch::Tensor input,
                                const torch::Tensor target)
{
    // torch.sum((y_pred - y)**2)
    float d = 0.0f;
    const size_t elems = input.numel();
    float* y = (float*)input.data_ptr();
    float* t = (float*)target.data_ptr();
    for (size_t i = 0; i < elems; ++i) d += (y[i] - t[i]) * (y[i] - t[i]);

    return torch::Scalar(d);
}

torch::Tensor mse_none_forward(const torch::Tensor input,
                                const torch::Tensor target)
{
    // (y_pred - y)**2
    torch::Tensor diff = torch::sub(input, target) /*.cuda()*/;
    torch::Tensor output = diff.square() /*.cuda()*/;

    return output;
}


std::vector<torch::Tensor> mse_forward(const torch::Tensor input,
                                       const torch::Tensor target,
                                       int reduction_type)
{
    // CHECK_INPUT(input);
    // CHECK_INPUT(weights);
    // CHECK_INPUT(bias);

    torch::Tensor output;
    if (reduction_type == 1)
    {
        output = torch::zeros(torch::IntArrayRef({1}));
        output[0] = mse_mean_forward(input, target);
    }
    else if (reduction_type == 2)
    {
        output = torch::zeros(torch::IntArrayRef({1}));
        output[0] = mse_sum_forward(input, target);
    }
    else if (reduction_type == 3)
    {
        output = mse_none_forward(input, target);
    }

    return {output};
}

torch::Tensor mse_mean_backward(const torch::Tensor input,
                                const torch::Tensor target)
{
    // 2 * (y_pred - y) / y_pred.numel()
    torch::Scalar factor = 2.0f / static_cast<float>(input.numel());
    torch::Tensor diff = torch::sub(input, target) /*.cuda()*/;
    torch::Tensor output = torch::mul(diff, factor) /*.cuda()*/;

    return output;
}

torch::Tensor mse_sum_backward(const torch::Tensor input,
                                const torch::Tensor target)
{
    // 2 * (y_pred - y)
    torch::Scalar factor = 2.0f;
    torch::Tensor diff = torch::sub(input, target) /*.cuda()*/;
    torch::Tensor output = torch::mul(diff, factor) /*.cuda()*/;

    return output;
}

torch::Tensor mse_none_backward(const torch::Tensor input,
                                const torch::Tensor target)
{
    // TODO: Throw error
    torch::Tensor output;
    return output;
}

std::vector<torch::Tensor> mse_backward(const torch::Tensor input,
                                        const torch::Tensor target,
                                        int reduction_type)
{
    // CHECK_INPUT(input);
    // CHECK_INPUT(weights);
    // CHECK_INPUT(bias);

    // TODO
    torch::Tensor output;
    if (reduction_type == 1)
    {
        output = mse_mean_backward(input, target);
    }
    else if (reduction_type == 2)
    {
        output = mse_sum_backward(input, target);
    }
    else if (reduction_type == 3)
    {
        output = mse_none_backward(input, target);
    }
    return {output};
}