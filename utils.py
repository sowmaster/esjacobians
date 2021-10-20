
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init


class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fns

    def forward(self, x):
        return self.fn(x)

# save checkpoint
def save_checkpoint(state_dict, save_path):
    torch.save(state_dict, save_path)

# load checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print('Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt


def init_params(net):
    '''Initialize network parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant(m.bias, 0)


# Jacobian function by PyTorch author
def jacobian(y, x, create_graph=False):
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y, retain_graph=True, create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    return jacobian(jacobian(y, x, create_graph=True), x)


def f(x):
    return x * x * torch.arange(4, dtype=torch.float)


# x = torch.ones(4, requires_grad=True)
# print(jacobian(f(x), x))
# print(hessian(f(x), x))