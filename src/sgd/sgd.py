from distutils.log import error
from torch import nn
from torch import optim
import torch
import torch.optim._functional as F
from torch import Tensor
from typing import List, Optional
import sgd_cpu


def sgd_function(params: List[Tensor],
                 d_p_list: List[Tensor],
                 momentum_buffer_list: List[Optional[Tensor]],
                 *,
                 weight_decay: float,
                 momentum: float,
                 lr: float,
                 dampening: float,
                 nesterov: bool):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    # for i, param in enumerate(params):
    #     d_p = d_p_list[i]
    #     if weight_decay != 0:
    #         d_p = d_p.add(param, alpha=weight_decay)
    #     if momentum != 0:
    #         buf = momentum_buffer_list[i]
    #         if buf is None:
    #             buf = torch.clone(d_p).detach()
    #             momentum_buffer_list[i] = buf
    #         else:
    #             buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
    #         if nesterov:
    #             d_p = d_p.add(buf, alpha=momentum)
    #         else:
    #             d_p = buf
    #     param.add_(d_p, alpha=-lr)
    sgd_cpu.sgd(params,
                d_p_list,
                momentum_buffer_list,
                weight_decay,
                momentum,
                lr,
                dampening,
                nesterov)


class SGD(optim.Optimizer):
    """Implements stochastic gradient descent (optionally with momentum).

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")

        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            # F.sgd(params_with_grad,
            #       d_p_list,
            #       momentum_buffer_list,
            #       weight_decay=weight_decay,
            #       momentum=momentum,
            #       lr=lr,
            #       dampening=dampening,
            #       nesterov=nesterov)
            sgd_function(params_with_grad,
                         d_p_list,
                         momentum_buffer_list,
                         weight_decay=weight_decay,
                         momentum=momentum,
                         lr=lr,
                         dampening=dampening,
                         nesterov=nesterov)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

        return loss
