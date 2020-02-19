from torch import nn
import torch
import torch.nn.functional as F
import math


# ------------------------------------------------------------
# --- Model Structures Template
# ------------------------------------------------------------
# src: https://www.pyimagesearch.com/wp-content/uploads/2017/03/imagenet_resnet_identity.png
class ResBlockPreActivation(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, norm='in', activation='relu', pad_type='zero'):
        super(ResBlockPreActivation, self).__init__()

        self.conv1 = Conv2dBlockPreActivation(dim, dim, kernel_size, stride, padding, norm, activation, pad_type)
        self.conv2 = Conv2dBlockPreActivation(dim, dim, kernel_size, 1, padding, norm, activation, pad_type)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        return out + residual


class ResBlockPreActivationWithAvgPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, norm='in', activation='relu',
                 pad_type='zero'):
        super(ResBlockPreActivationWithAvgPool, self).__init__()

        self.avgPool = nn.AvgPool2d(2)
        self.conv1 = Conv2dBlockPreActivation(input_dim, output_dim, kernel_size, stride, padding, norm, activation,
                                              pad_type)
        self.conv2 = Conv2dBlockPreActivation(output_dim, output_dim, kernel_size, 1, padding, norm, activation,
                                              pad_type)

        self.residual = Conv2dBlock(input_dim, output_dim, kernel_size, 1, 1, pad_type=pad_type)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.residual(x)
        return self.avgPool(out + residual)


class ResBlockPreActivationWithUpsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, norm='in', activation='relu',
                 pad_type='zero'):
        super(ResBlockPreActivationWithUpsample, self).__init__()

        self.upsample = Upsample(scale_factor=2, mode='nearest')

        self.conv1 = Conv2dBlockPreActivation(input_dim, output_dim, kernel_size, stride, padding, norm, activation,
                                              pad_type)
        self.conv2 = Conv2dBlockPreActivation(output_dim, output_dim, kernel_size, 1, padding, norm, activation,
                                              pad_type)

        self.residual = Conv2dBlock(input_dim, output_dim, kernel_size, 1, 1, pad_type=pad_type)

    def forward(self, x):
        x = self.upsample(x)
        out = self.conv1(x)
        out = self.conv2(out)
        residual = self.residual(x)
        return out + residual


# ------------------------------------------------------------
# --- Basic Computation Layers
# ------------------------------------------------------------
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()

        self.fc = nn.Linear(input_dim, output_dim, bias=norm == "none")
        self.fc.nonlinearity = activation
        self.norm = _normalization1d(norm, output_dim)
        self.non_linear = _activation(activation)

    def forward(self, x):
        out = self.fc(x)
        out = self.norm(out)
        out = self.non_linear(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='none', pad_type='zero'):
        super(Conv2dBlock, self).__init__()

        # downsampling : 7, 2, 3 / 6, 2, 2 / 5, 2, 2 / 4, 2, 1 / 3, 2, 1
        # maintain     : 7, 1, 3 / 6 X / 5, 1, 2 / 4 X / 3, 1, 1

        self.padding = Identity()
        if padding != 0:
            self.padding = _padding(pad_type, padding)
        self.net = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=norm == "none")
        self.net.nonlinearity = activation

        self.norm = _normalization2d(norm, output_dim)
        self.non_linear = _activation(activation)

    def forward(self, x):
        x = self.padding(x)
        x = self.net(x)
        x = self.norm(x)
        x = self.non_linear(x)
        return x


class Conv2dBlockPreActivation(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=1, norm='none', activation='none', pad_type='zero'):
        super(Conv2dBlockPreActivation, self).__init__()

        self.padding = Identity()
        if padding != 0:
            self.padding = _padding(pad_type, padding)

        self.norm = _normalization2d(norm, input_dim)
        self.non_linear = _activation(activation)
        self.net = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=norm == "none")
        self.net.nonlinearity = activation

    def forward(self, x):
        x = self.padding(x)
        x = self.norm(x)
        x = self.non_linear(x)
        x = self.net(x)
        return x


# --- Normalization layers
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None

        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'


# Adain with affine transformation
class AffineLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.bias.data[:out_features // 2] = 1  # init gamma close to 1
        self.bias.data[out_features // 2:] = 0  # init beta close to 0
        self.is_affine = True
        self.nonlinearity = 'none'


class AdaptiveInstanceNormWithAffineTransform(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = AffineLinear(style_dim, in_channel * 2)

    def forward(self, input):
        out = self.norm(input)
        out = self.gamma * out + self.beta
        return out


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class SpectralNorm(nn.Module):
    """
    Based on the paper "Spectral Normalization for Generative Adversarial Networks" by Takeru Miyato, Toshiki Kataoka, Masanori Koyama, Yuichi Yoshida
    and the Pytorch implementation https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# ------------------------------------------------------------
# --- Utils & Activation, Non-linearity etc..
# ------------------------------------------------------------
class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


class Identity(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()

    def forward(self, x):
        return x


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def _padding(pad_type, padding):
    if pad_type == 'reflect':
        return nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        return nn.ReplicationPad2d(padding)
    elif pad_type == 'zero':
        return nn.ZeroPad2d(padding)
    elif pad_type == 'none':
        return Identity()
    else:
        raise NotImplementedError("Unsupported padding type: {}".format(pad_type))


def _normalization1d(norm, norm_dim):
    if norm == 'bn':
        return nn.BatchNorm1d(norm_dim)
    elif norm == 'in':
        return nn.InstanceNorm1d(norm_dim)
    elif norm == 'ln':
        return LayerNorm(norm_dim)
    elif norm == 'none' or norm == 'sn':
        return Identity()
    else:
        raise NotImplementedError("Unsupported normalization: {}".format(norm))


def _normalization2d(norm, norm_dim):
    if norm == 'bn':
        return nn.BatchNorm2d(norm_dim)
    elif norm == 'in':
        return nn.InstanceNorm2d(norm_dim, affine=True, track_running_stats=True)
    elif norm == 'ln':
        return LayerNorm(norm_dim)
    elif norm == 'adain':
        return AdaptiveInstanceNorm2d(norm_dim)
    elif norm == 'adain_affine':
        return AdaptiveInstanceNormWithAffineTransform(norm_dim, style_dim=64)  # TODO:
    elif norm == 'none' or norm == 'sn':
        return Identity()
    else:
        raise NotImplementedError("Unsupported normalization: {}".format(norm))


def _activation(activation):
    if activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'lrelu':
        return nn.LeakyReLU(0.1, inplace=True)
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'selu':
        return nn.SELU(inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'none':
        return Identity()
    else:
        raise NotImplementedError("Unsupported activation: {}".format(activation))
