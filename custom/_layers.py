from torch import nn
import torch
import torch.nn.functional as F
from styleGAN import AdaptiveInstanceNormWithAffineTransform


# ------------------------------------------------------------
# --- Model Structures Template
# ------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()

        layers = []
        layers += [
            Conv2dBlock(dim, dim, kernel_size, stride, padding, norm=norm, activation=activation, pad_type=pad_type)]
        layers += [Conv2dBlock(dim, dim, kernel_size, 1, padding, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = x

        logit = self.model(x)
        logit += residual
        return logit


# src: https://www.pyimagesearch.com/wp-content/uploads/2017/03/imagenet_resnet_identity.png
class ResBlockPreActivation(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, norm='in', activation='relu', pad_type='zero'):
        super(ResBlockPreActivation, self).__init__()

        layers = []
        layers += [
            Conv2dBlockPreActivation(dim, dim, kernel_size, stride, padding, norm=norm, activation=activation,
                                     pad_type=pad_type)]
        layers += [Conv2dBlockPreActivation(dim, dim, kernel_size, 1, padding, norm=norm, activation=activation,
                                            pad_type=pad_type)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        logit = self.model(x)
        logit += residual
        return logit


class ResBlockPreActivationWithAvgPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, norm='in', activation='relu',
                 pad_type='zero'):
        super(ResBlockPreActivationWithAvgPool, self).__init__()

        layers = []
        layers += [Conv2dBlockPreActivation(input_dim, output_dim, kernel_size, stride, padding, norm=norm,
                                            activation=activation, pad_type=pad_type)]
        layers += [
            Conv2dBlockPreActivation(output_dim, output_dim, kernel_size, 1, padding, norm=norm, activation=activation,
                                     pad_type=pad_type)]

        self.model = nn.Sequential(*layers)
        self.residual = Conv2dBlock(input_dim, output_dim, kernel_size, 1, 1, pad_type=pad_type)
        self.avgPool = nn.AvgPool2d(2)

    def forward(self, x):
        logit = self.model(x)
        residual = self.residual(x)
        logit = self.avgPool(logit + residual)
        return logit


class ResBlockPreActivationWithUpsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, norm='in', activation='relu',
                 pad_type='zero'):
        super(ResBlockPreActivationWithUpsample, self).__init__()

        self.upsample = Upsample(scale_factor=2, mode='nearest')

        layers = []
        layers += [Conv2dBlockPreActivation(input_dim, output_dim, kernel_size, stride, padding, norm=norm,
                                            activation=activation, pad_type=pad_type)]
        layers += [
            Conv2dBlockPreActivation(output_dim, output_dim, kernel_size, 1, padding, norm=norm, activation=activation,
                                     pad_type=pad_type)]

        self.model = nn.Sequential(*layers)
        self.residual = Conv2dBlock(input_dim, output_dim, kernel_size, 1, 1, pad_type=pad_type)

    def forward(self, x):
        x = self.upsample(x)
        logit = self.model(x)
        residual = self.residual(x)
        logit += residual
        return logit


# ------------------------------------------------------------
# --- Basic Computation Layers
# ------------------------------------------------------------
class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu', bias=False):
        super(LinearBlock, self).__init__()
        # initialize fully connected layer
        if norm == 'sn':
            self.fc = SpectralNorm(nn.Linear(input_dim, output_dim, bias=bias))
        else:
            self.fc = nn.Linear(input_dim, output_dim, bias=bias)

        self.norm = _normalization1d(norm, output_dim)

        self.activation = _activation(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='none', pad_type='zero', bias=False):
        super(Conv2dBlock, self).__init__()

        # downsampling : 7, 2, 3 / 6, 2, 2 / 5, 2, 2 / 4, 2, 1 / 3, 2, 1
        # maintain     : 7, 1, 3 / 6 X / 5, 1, 2 / 4 X / 3, 1, 1

        layers = []
        if padding:
            layers += [_padding(pad_type, padding)]  # padding

        # conv method
        if norm == 'sn':
            layers += [SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=bias))]
        else:
            layers += [nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=bias)]
            layers += [_normalization2d(norm, output_dim)]  # normalizaion

        layers += [_activation(activation)]  # activation

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Conv2dBlockPreActivation(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='none', pad_type='zero', bias=False):
        super(Conv2dBlockPreActivation, self).__init__()

        # downsampling : 7, 2, 3 / 6, 2, 2 / 5, 2, 2 / 4, 2, 1 / 3, 2, 1
        # maintain     : 7, 1, 3 / 6 X / 5, 1, 2 / 4 X / 3, 1, 1

        layers = []
        if padding:
            layers += [_padding(pad_type, padding)]

        if norm == 'sn':
            layers += [SpectralNorm(nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=bias))]
            layers += [_activation(activation)]
        else:
            layers += [_normalization2d(norm, input_dim)]
            layers += [_activation(activation)]
            layers += [nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=bias)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Conv2dUpsampleBlock(nn.Module):
    def __init__(self, input_dim, output_dim, mode, kernel_size,
                 padding=0, norm='none', activation='relu', pad_type='zero', bias=False):
        super(Conv2dUpsampleBlock, self).__init__()

        if mode == 'transpose':
            layers = [
                # 6, 2 / 4, 1
                nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride=2, padding=padding, bias=bias),
                _normalization2d(norm, output_dim),
                _activation(activation),
            ]
        elif mode in ['nearest', 'linear', 'bilinear', 'trilinear']:
            layers = [
                # 7, 3 / 5, 2 / 3, 1
                Upsample(scale_factor=2, mode=mode),
                Conv2dBlock(input_dim, output_dim, kernel_size, stride=1, padding=padding, norm=norm,
                            activation=activation, pad_type=pad_type, bias=bias)
            ]
        else:
            raise NotImplementedError('Upsample layer [%s] not implemented' % mode)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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
