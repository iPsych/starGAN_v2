import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function

from math import sqrt

import random


# class AdaptiveInstanceNorm2dWithAffine(nn.Module):
#     def __init__(self, num_features, style_dim, eps=1e-5, momentum=0.1):
#         super(AdaptiveInstanceNorm2dWithAffine, self).__init__()
#         self.num_features = num_features * 2
#         self.eps = eps
#         self.momentum = momentum
#
#         # weight and bias are dynamically assigned
#         self.weight = None
#         self.bias = None
#
#         self.style = EqualLinear(style_dim, num_features * 2)
#
#         # just dummy buffers, not used
#         self.register_buffer('running_mean', torch.zeros(num_features))
#         self.register_buffer('running_var', torch.ones(num_features))
#
#     def forward(self, x):
#         assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
#         b, c = x.size(0), x.size(1)
#         running_mean = self.running_mean.repeat(b)
#         running_var = self.running_var.repeat(b)
#
#         print("adain", x.shape, self.running_mean.shape, running_mean.shape)
#         print("adain", self.num_features, running_mean.shape, running_var.shape, self.weight.shape, self.bias.shape)
#
#         # Apply instance norm
#         x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
#
#         out = F.batch_norm(
#             x_reshaped, running_mean, running_var, self.weight, self.bias,
#             True, self.momentum, self.eps)
#         print('adain', x_reshaped.shape, out.shape, end='\n\n')
#         return out.view(b, c, *x.size()[2:])
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(' + str(self.num_features) + ')'


class AdaptiveInstanceNormWithAffineTransform(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input):
        out = self.norm(input)
        out = self.gamma * out + self.beta
        return out


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)
