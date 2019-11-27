import functools
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import init

from . import config as cfg
from pretorched import models, optim, utils


optimizer_defaults = {
    'SGD': {
        'momentum': 0.9,
        'weight_decay': 1e-4,
    },
}

scheduler_defaults = {
    'CosineAnnealingLR': {
        'T_max': 100
    }
}


def get_optimizer(model, optimizer_name='SGD', lr=0.001, **kwargs):
    optim_func = getattr(optim, optimizer_name)
    func_kwargs, _ = utils.split_kwargs_by_func(optim_func, kwargs)
    optim_kwargs = {**optimizer_defaults.get(optimizer_name, {}), **func_kwargs}
    optimizer = optim_func(model.parameters(), lr=lr, **optim_kwargs)
    return optimizer


def get_scheduler(optimizer, scheduler_name='CosineAnnealingLR', **kwargs):
    sched_func = getattr(torch.optim.lr_scheduler, scheduler_name)
    func_kwargs, _ = utils.split_kwargs_by_func(sched_func, kwargs)
    sched_kwargs = {**scheduler_defaults.get(scheduler_name, {}), **func_kwargs}
    scheduler = sched_func(optimizer, **sched_kwargs)
    return scheduler


def _get_scheduler(optimizer, sched_name='ReduceLROnPlateau', **kwargs):
    sched_func = getattr(torch.optim.lr_scheduler, sched_name)
    if sched_name == 'ReduceLROnPlateau':
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 5)
        scheduler = sched_func(optimizer, factor=factor, patience=patience, verbose=True)
    elif sched_name == 'CosineAnnealingLR':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        scheduler = sched_func(optimizer, T_max, eta_min=eta_min)
    return scheduler


def init_weights_old(model, init_name='ortho'):
    for module in model.modules():
        if (isinstance(module, nn.Conv2d)
            or isinstance(module, nn.Linear)
                or isinstance(module, nn.Embedding)):
            if init_name == 'ortho':
                init.orthogonal_(module.weight)
            elif init_name == 'N02':
                init.normal_(module.weight, 0, 0.02)
            elif init_name in ['glorot', 'xavier']:
                init.xavier_normal_(module.weight)
    else:
        print('Init style not recognized...')
    return model


def init_weights(model, init_name='ortho'):

    def _init_weights(m, init_func):
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
            init_func(m.weight)
        for l in m.children():
            _init_weights(l, init_func)

    init_func = {
        'ortho': init.orthogonal_,
        'N02': functools.partial(init.normal_, mean=0, std=0.02),
        'glorot': init.xavier_normal_,
        'xavier': init.xavier_normal_,
        'kaiming': init.kaiming_normal_,
    }.get(init_name, 'kaiming')

    _init_weights(model, init_func)
    return model


def get_model(model_name, num_classes, pretrained='imagenet', init_name=None, **kwargs):
    model_func = getattr(models, model_name)
    if pretrained is not None:
        # TODO Update THIS!
        nc = {k.lower(): v for k, v in cfg.num_classes_dict.items()}.get(pretrained)
        model = model_func(num_classes=nc, pretrained=pretrained, **kwargs)
        if nc != num_classes:
            in_feat = model.last_linear.in_features
            last_linear = nn.Linear(in_feat, num_classes)
            if init_name is not None:
                print(f'Re-initializing last_linear of {model_name} with {init_name}.')
                last_linear = init_weights(last_linear, init_name)
            model.last_linear = last_linear
    else:
        model = model_func(num_classes=num_classes, pretrained=pretrained, **kwargs)
        if init_name is not None:
            print(f'Initializing {model_name} with {init_name}.')
            model = init_weights(model, init_name)
    return model
