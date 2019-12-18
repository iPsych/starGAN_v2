import torch
from torch import nn
from torch.autograd import Variable

from custom._layers import Conv2dBlock, LinearBlock, \
    ResBlockPreActivationWithAvgPool, ResBlockPreActivation, ResBlockPreActivationWithUpsample


def gather_domain(src, domain_index):  # only works at torch.gather(..., dim=1)
    domain_index = domain_index.repeat(1, 1, src.size(-1))
    return torch.gather(src, 1, domain_index.long())


class Generator(nn.Module):
    def __init__(self, config_gen):  # *a, **k : norm, activation, pad_type
        super(Generator, self).__init__()

        dim = config_gen['dim']  # 32
        n_downs = config_gen['n_downs']  # 4
        n_intermediates = config_gen['n_intermediates']  # 4
        # activation = config_gen['activation']
        # pad_type = config_gen['pad_type']

        layers = []

        layers += [Conv2dBlock(3, dim, 1, 1, 0, 'none', 'none', bias=False)]

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

        layers += [Conv2dBlock(dim, 3, 3, 1, 1, 'none', 'tanh', bias=False)]  # TODO : delete tanh

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def assign_adain_params_features(self, adain_params):
        for module in self.model.modules():  # self.dec
            if module.__class__.__name__ == 'AdaptiveInstanceNormWithAffineTransform':
                affine_transformed = module.style(adain_params).view(adain_params.size(0), -1, 1, 1)
                module.gamma, module.beta = affine_transformed.chunk(2, 1)

    def forward(self, x, style):
        self.assign_adain_params_features(style)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape)
        # return x
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
        layers += [LinearBlock(dim, num_domain * dim_style, norm='none', activation='none')]

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, target_domain):
        x = self.model(x).view(-1, self.num_domain, self.dim_style)
        return gather_domain(x, target_domain)  # .squeeze()


# TODO : discriminator 랑 구조,...
# TODO : pre-activation resblock 순서좀 봐야될듯...
class StyleEncoder(nn.Module):
    def __init__(self, config_what, num_domain, dim_style):
        super(StyleEncoder, self).__init__()

        ch = config_what['ch']
        n_intermediates = config_what['n_intermediates']
        activation = config_what['activation']
        self.num_domain = num_domain
        self.dim_style = dim_style

        layers = []
        layers += [Conv2dBlock(3, ch, 1, 1, 0, 'none', 'none', bias=False)]

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

        activation_height = 128 // 2 ** n_intermediates
        layers += [
            nn.LeakyReLU(0.1, inplace=True),
            Conv2dBlock(ch, ch, activation_height, 1, 0, 'none', 'lrelu'),
            Conv2dBlock(ch, num_domain * dim_style, 1, 1, 0, 'none', 'none')
        ]

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, target_domain):
        # print('style encoder', x.shape)
        x = self.model(x).view(-1, self.num_domain, self.dim_style)
        # print('style encoder', x.shape)
        return gather_domain(x, target_domain)


class Discriminator(nn.Module):
    def __init__(self, config_what, num_domain, dim_style):
        super(Discriminator, self).__init__()

        ch = config_what['ch']
        n_intermediates = config_what['n_intermediates']
        activation = config_what['activation']
        self.num_domain = num_domain
        self.dim_style = dim_style

        layers = []
        layers += [Conv2dBlock(3, ch, 1, 1, 0, 'none', 'none', bias=False)]

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

        activation_height = 128 // 2 ** n_intermediates
        layers += [
            nn.LeakyReLU(0.1, inplace=True),
            Conv2dBlock(ch, ch, activation_height, 1, 0, 'none', 'lrelu'),
            Conv2dBlock(ch, num_domain * dim_style, 1, 1, 0, 'none', 'none')
        ]

        self.layers = layers
        self.model = nn.Sequential(*layers)

    def forward(self, x, target_domain):
        x = self.model(x).view(-1, self.num_domain, self.dim_style)
        return gather_domain(x, target_domain)


if __name__ == '__main__':
    from custom._utils_torch import count_params
    from utils import get_config

    config = get_config('./config/celeba_HQ.yaml')
    num_domain = config['num_domain']
    dim_style = config['dim_style']

    gen = Generator(config['gen'])  # 29072960
    mapping_network = MappingNetwork(config['mapping_network'], num_domain, dim_style)
    style_encoder = StyleEncoder(config['style_encoder'], num_domain, dim_style)
    discriminator = Discriminator(config['dis'], num_domain, 1)

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

    print(style_mapped.shape, random_noise.shape, random_domain.shape)
    print(fake.shape, style.shape, style_mapped.shape, logit_real.shape, logit_fake.shape)

    # print(count_params(gen), count_params(mapping_network), count_params(style_encoder), count_params(discriminator))
    print()