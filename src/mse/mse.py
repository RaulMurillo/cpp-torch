from distutils.log import error
from torch import nn
import torch
import torch.nn.functional as F
import mse_cpu as mse

class MSELossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, target, reduction):
        variables = [input, target, reduction]
        ctx.save_for_backward(*variables)
        red = reduction.item()
        # y_pred , y = input, target
        # if red == 1: # 'mean':
        #     output = torch.mean((y_pred - y)**2)
        # elif red == 2: # 'sum':
        #     output = torch.sum((y_pred - y)**2)
        # elif red == 3: # 'none':
        #     output = (y_pred - y)**2
        # else:
        #     raise ValueError("{} is not a valid value for reduction".format(reduction))
        output = mse.forward(input, target, red)[0]
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        input, target, reduction = ctx.saved_tensors
        red = reduction.item()
        # y_pred , y = input, target
        if red == 3: # 'none':
            raise RuntimeError("grad can be implicitly created only for scalar outputs")
        # if red == 1: # 'mean':
        #     gradInput = 2 * (y_pred - y) / y_pred.numel()
        # elif red == 2: # 'sum':
        #     gradInput = 2 * (y_pred - y)
        # else:
        #     raise ValueError("{} is not a valid value for reduction".format(reduction))
        gradInput = mse.backward(input, target, red)[0]
        return gradInput, None, None


class MSELoss(nn.Module):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction= 'mean'):
        super(MSELoss, self).__init__()

        self.size_average = size_average
        self.reduce = reduce
        red = 0
        if reduction == 'mean':
            red = 1
        elif reduction == 'sum':
            red = 2
        elif reduction == 'none':
            red = 3
        else:
            raise ValueError("{} is not a valid value for reduction".format(reduction))

        self.reduction = torch.autograd.Variable(torch.tensor([red], dtype=int))

    def forward(self, input, target):
        # return F.mse_loss(input, target, reduction=self.reduction)
        return MSELossFunction.apply(input, target, self.reduction)
