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


std::vector<torch::Tensor> linear_forward(torch::Tensor input,
                                   torch::Tensor weights,
                                   torch::Tensor bias,
                                   bool is_bias) {

    // CHECK_INPUT(input);
    // CHECK_INPUT(weights);
    // CHECK_INPUT(bias);

    // std::cout<<output.dim()<<std::endl;
    int64_t batch_size = input.size(0);
    int64_t nInputFeatures = input.size(1);

    int64_t nOutputFeatures = weights.size(0);

    torch::Tensor output = torch::zeros(torch::IntArrayRef({batch_size, nOutputFeatures}));/*.cuda()*/;
    
    weights = weights.t();  // Y = X*w.T + b

    for(int elt = 0; elt < batch_size; elt++){
        torch::Tensor input_n = input[elt];
        input_n = input_n.reshape(torch::IntArrayRef({1, nInputFeatures}))/*.cuda()*/;

        if(is_bias){
            output[elt].add_(bias/*.cuda()*/, 1);
        }

        // weights.dim: out_features,in_features
        output[elt].add_(input_n.mm(weights).reshape(torch::IntArrayRef({nOutputFeatures})), 1);
    }
    return {output};
}

torch::Tensor backward_gradInput(torch::Tensor input,
                                    torch::Tensor gradOutput,
                                    torch::Tensor weights){

    int64_t batch_size = input.size(0);
    int64_t nInputFeatures = input.size(1);

    int64_t nOutputFeatures = gradOutput.size(1);

    torch::Tensor gradInput = torch::zeros(torch::IntArrayRef({batch_size, nInputFeatures}))/*.cuda()*/;

    for(int elt = 0; elt < batch_size; elt++){
        torch::Tensor gradOutput_n = gradOutput[elt];
        gradOutput_n = gradOutput_n.reshape(torch::IntArrayRef({1, nOutputFeatures}))/*.cuda()*/;

        gradInput[elt] = gradOutput_n.mm(weights).reshape(torch::IntArrayRef({nInputFeatures}))/*.cuda()*/;
    }

    return gradInput;
}

std::vector<torch::Tensor> backward_gradParameters(torch::Tensor input,
                                                   torch::Tensor gradOutput,
                                                   torch::Tensor weights,
                                                   bool is_bias){

    int64_t batch_size = input.size(0);
    int64_t nInputFeatures = input.size(1);

    int64_t nOutputFeatures = gradOutput.size(1);

    torch::Tensor gradWeights = torch::zeros(torch::IntArrayRef({weights.size(0), weights.size(1)}))/*.cuda()*/;
    torch::Tensor gradBias = torch::zeros(torch::IntArrayRef({nOutputFeatures}))/*.cuda()*/;

    for(int elt = 0; elt < batch_size; elt++){
        torch::Tensor input_n = input[elt];
        torch::Tensor gradOutput_n = gradOutput[elt];
        input_n = input_n.reshape(torch::IntArrayRef({1, nInputFeatures}));

        gradWeights.add_(gradOutput_n.reshape(torch::IntArrayRef({1, nOutputFeatures})).t().mm(input_n), 1);

        if(is_bias){
            gradBias.add_(gradOutput_n, 1);
        }

    }
    return {gradWeights, gradBias};
}

std::vector<torch::Tensor> linear_backward(torch::Tensor input,
                                    torch::Tensor gradOutput,
                                    torch::Tensor weights,
                                    bool is_bias) {

    // CHECK_INPUT(gradOutput);
    // CHECK_INPUT(weights);
    // CHECK_INPUT(input);

    torch::Tensor gradInput = backward_gradInput(input, gradOutput, weights);
    std::vector<torch::Tensor> gradParams = backward_gradParameters(input, gradOutput, weights, is_bias);

    torch::Tensor gradWeights = gradParams[0];
    torch::Tensor gradBias = gradParams[1];

    return {gradInput, gradWeights, gradBias};

}
