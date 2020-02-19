import torch
from torch import nn
from torch.autograd import Variable
from collections import OrderedDict

from common.layers import Conv2dBlock, LinearBlock, \
    ResBlockPreActivationWithAvgPool, ResBlockPreActivation, ResBlockPreActivationWithUpsample


def gather_domain(src, domain):  # only works at torch.gather(..., dim=1)
    domain = domain.repeat(1, 1, src.size(-1))
    return torch.gather(src, 1, domain.long())  # src: (batch, n_domain, style_dim), domain: (batch, 1, style_dim)


class Generator(nn.Module):
    def __init__(self, config_gen):
        super(Generator, self).__init__()

        dim = config_gen['dim']  # 32
        n_downs = config_gen['n_downs']  # 4
        n_intermediates = config_gen['n_intermediates']  # 4
        # activation = config_gen['activation']
        # pad_type = config_gen['pad_type']

        layers = []

        layers += [Conv2dBlock(3, dim, 1, 1, 0, 'none', 'none')]

        for n_down in range(n_downs):
            layers += [ResBlockPreActivationWithAvgPool(dim, dim * 2, 3, 1, 1, 'in', 'relu')]
            dim *= 2

        for n_intermediate in range(n_intermediates):
            if n_intermediate < n_intermediates // 2:
                layers += [ResBlockPreActivation(dim, 3, 1, 1, 'in', 'relu')]
            else:
                layers += [ResBlockPreActivation(dim, 3, 1, 1, 'adain_affine', 'relu')]
        #
        for n_up in range(n_downs):
            layers += [
                ResBlockPreActivationWithUpsample(dim, dim // 2, 3, 1, 1, 'adain_affine', 'relu')]
            dim //= 2

        # layers += [Conv2dBlock(dim, 3, 1, 1, 0, 'none', 'none')]  # TODO : TRICKY!! 1x1 raise cudnn bug
        layers += [Conv2dBlock(dim, 3, 3, 1, 1, 'none', 'none')]

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def assign_adain_params_features(self, adain_params):
        for module in self.model.modules():  # self.dec
            if module.__class__.__name__ == 'AdaptiveInstanceNormWithAffineTransform':
                affine_transformed = module.style(adain_params).view(adain_params.size(0), -1, 1, 1)
                module.gamma, module.beta = affine_transformed.chunk(2, 1)

    def forward(self, x, style):
        self.assign_adain_params_features(style)
        return self.model(x)


class MappingNetwork(nn.Module):
    def __init__(self, config_mapnet, num_domain, dim_style):
        super(MappingNetwork, self).__init__()

        input_dim = config_mapnet['dim_latent']
        dim = config_mapnet['dim']
        n_mlp = config_mapnet['n_mlp']
        norm = config_mapnet['norm']
        activation = config_mapnet['activation']
        self.num_domain = num_domain
        self.dim_style = dim_style

        layers = []
        layers += [LinearBlock(input_dim, dim, norm, activation)]
        for i in range(n_mlp):  # 6
            layers += [LinearBlock(dim, dim, norm, activation)]
        layers += [LinearBlock(dim, self.num_domain * self.dim_style, norm='none', activation='none')]

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, target_domain):
        batch_size = target_domain.size(0)
        x = self.model(x).repeat(batch_size, 1).view(batch_size, self.num_domain, self.dim_style)
        return gather_domain(x, target_domain)  # (batch, n_domain, 64) -> (batch, 1, 64)


class StyleEncoder(nn.Module):
    def __init__(self, config_enc, num_domain, img_size):
        super(StyleEncoder, self).__init__()

        ch = config_enc['ch']
        n_intermediates = config_enc['n_intermediates']
        activation = config_enc['activation']
        self.num_domain = num_domain
        self.dim_style = config_enc['output_dim']

        layers = []
        layers += [Conv2dBlock(3, ch, 1, 1, 0, 'none', 'none')]

        for n_intermediate in range(n_intermediates):
            if n_intermediate < n_intermediates - 1:
                layers += [ResBlockPreActivationWithAvgPool(ch, ch * 2, 3, 1, 1, 'none', activation)]
                ch *= 2
            else:
                # last intermediate resblock doesn't increase filter dimension
                layers += [
                    ResBlockPreActivation(ch, 3, 1, 1, 'none', activation),
                    nn.AvgPool2d(2)
                ]

        activation_height = img_size // 2 ** n_intermediates  # 4
        layers += [
            nn.LeakyReLU(0.1, inplace=True),
            Conv2dBlock(ch, ch, activation_height, 1, 0, 'none', 'lrelu'),
            Conv2dBlock(ch, self.num_domain * self.dim_style, 1, 1, 0, 'none', 'none')
        ]

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, target_domain):
        x = self.model(x).view(-1, self.num_domain, self.dim_style)
        return gather_domain(x, target_domain)  # (batch, n_domain, 64) -> (batch, 1, 64)


class Discriminator(nn.Module):
    def __init__(self, config_dis, num_domain, img_size):
        super(Discriminator, self).__init__()

        ch = config_dis['ch']
        n_intermediates = config_dis['n_intermediates']
        activation = config_dis['activation']
        self.num_domain = num_domain
        self.dim_style = config_dis['output_dim']

        layers = []
        layers += [Conv2dBlock(3, ch, 1, 1, 0, 'none', 'none')]

        for n_intermediate in range(n_intermediates):
            if n_intermediate < n_intermediates - 1:
                layers += [ResBlockPreActivationWithAvgPool(ch, ch * 2, 3, 1, 1, 'none', activation)]
                ch *= 2
            else:
                # last intermediate resblock doesn't increase filter dimension
                layers += [
                    ResBlockPreActivation(ch, 3, 1, 1, 'none', activation),
                    nn.AvgPool2d(2)
                ]

        activation_height = img_size // 2 ** n_intermediates
        layers += [
            nn.LeakyReLU(0.1, inplace=True),
            Conv2dBlock(ch, ch, activation_height, 1, 0, 'none', 'lrelu'),
            Conv2dBlock(ch, self.num_domain * self.dim_style, 1, 1, 0, 'none', 'none')
        ]

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, target_domain):
        x = self.model(x).view(-1, self.num_domain, self.dim_style)
        return gather_domain(x, target_domain)  # (batch, n_domain, 1) -> (batch, 1, 1)


if __name__ == '__main__':
    from common.utils_torch import count_params
    from utils import get_config

    config = get_config('./config/afhq.yaml')
    dim_style = config['dim_style']
    img_size = config['crop_size']
    num_domain = 3

    gen = Generator(config['gen'])  # 29072960
    mapping_network = MappingNetwork(config['mapping_network'], num_domain, dim_style)
    style_encoder = StyleEncoder(config['style_encoder'], num_domain, dim_style, img_size)
    discriminator = Discriminator(config['dis'], num_domain, 1, img_size)

    batch_size, height = config['batch_size'], config['re_size']
    random_noise = torch.randn(batch_size, 16)
    domain = torch.randint(num_domain, (batch_size, 1, 1))
    dummy_img = torch.zeros((batch_size, 3, height, height))
    random_domain = torch.randint(num_domain, (batch_size, 1, 1))

    style_mapped = mapping_network(random_noise, random_domain)
    fake = gen(dummy_img, style_mapped)
    style = style_encoder(dummy_img, domain)
    logit_real = discriminator(dummy_img, domain)
    logit_fake = discriminator(fake, random_domain)

    print(style_mapped.size(), fake.size(), style.shape, logit_real.shape, logit_fake.shape)
