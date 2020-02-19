import torch
from torch.nn import init
from torch.optim import lr_scheduler

import numpy as np
import yaml
import math


def get_config(path):
    with open(path, 'r') as stream:
        return yaml.load(stream, Loader=yaml.SafeLoader)


def get_scheduler(optimizer, config, iterations=-1):
    policy = config.get('lr_policy', None)

    if not policy or policy == 'constant':
        scheduler = None
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['step_size'], gamma=config['gamma'],
                                        last_epoch=iterations)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['niter'], eta_min=0)

    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


def weights_init(init_type):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print(classname, m)
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nonlinearity = "sigmoid" if m.nonlinearity == 'none' else "leaky_relu"
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity=nonlinearity)
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)

            if hasattr(m, 'bias') and m.bias is not None and not hasattr(m, 'is_affine'):  # TODO
                init.constant_(m.bias.data, 0.0)

    return init_fun


def seed_random(seed=123):
    import random
    # import tensorflow as tf
    import torch

    random.seed(seed)
    np.random.seed(seed)
    # tf.set_random_seed(seed)
    torch.manual_seed(seed)


def set_print_precision(precision=4):
    np.set_printoptions(precision=precision, linewidth=200, edgeitems=5, suppress=True)
    torch.set_printoptions(precision=precision, threshold=10)
