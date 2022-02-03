from torch import nn
import torch
import torch.nn.functional as F
import linear_cpu as linear


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights, bias, params):

        is_bias = int(params[0])

        outputs = linear.forward(input, weights, bias, is_bias)[0]

        variables = [input, weights, bias, params]
        ctx.save_for_backward(*variables)

        return outputs

    @staticmethod
    def backward(ctx, gradOutput):
        _ = torch.autograd.Variable(torch.zeros(5))

        input, weights, bias, params = ctx.saved_tensors

        is_bias = int(params[0])

        gradInput, gradWeight, gradBias = linear.backward(input, gradOutput, weights, is_bias)
        return gradInput, gradWeight, gradBias, _


class Linear(nn.Module):
    def __init__(self, in_features, out_features, is_bias=True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.is_bias = is_bias

        self.params = torch.autograd.Variable(torch.Tensor([is_bias]))

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        if self.is_bias:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return LinearFunction.apply(input, self.weight, self.bias, self.params)