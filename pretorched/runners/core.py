import functools
import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pretorched import models, optim, utils, data
from pretorched.data import transforms

from . import config as cfg

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


def get_transform(name='ImageNet', split='train', size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])
    }
    return data_transforms.get(split)


def get_dataset(name, root, split='train', size=224, resolution=256,
                dataset_type='ImageFolder', **kwargs):

    Dataset = getattr(data, dataset_type, 'ImageFolder')

    kwargs = {**kwargs,
              'root': root,
              'metafile': os.path.join(root, f'{split}.txt'),
              'transform': get_transform(name, split, size)}
    dataset_kwargs, _ = utils.split_kwargs_by_func(Dataset, kwargs)
    return Dataset(**dataset_kwargs)


def get_hybrid_dataset(root_dir=None, resolution=128, dataset_type='ImageHDF5', load_in_mem=False):
    imagenet_root = cfg.get_root_dirs('ImageNet', dataset_type=dataset_type,
                                      resolution=resolution, data_root=root_dir)
    places365_root = cfg.get_root_dirs('Places365', dataset_type=dataset_type,
                                       resolution=resolution, data_root=root_dir)
    imagenet_dataset = get_dataset('ImageNet', resolution=resolution,
                                   dataset_type=dataset_type, load_in_mem=load_in_mem,
                                   root_dir=imagenet_root)
    placess365_dataset = get_dataset('Places365', resolution=resolution,
                                     dataset_type=dataset_type, load_in_mem=load_in_mem,
                                     target_transform=functools.partial(add, 1000),
                                     root_dir=places365_root)
    return torch.utils.data.ConcatDataset((imagenet_dataset, placess365_dataset))


def get_dataloader(name, data_root=None, split='train', size=224, resolution=256,
                   dataset_type='ImageFolder', batch_size=64, num_workers=8, shuffle=True,
                   load_in_mem=False, pin_memory=True, drop_last=True, distributed=False,
                   **kwargs):
    root = cfg.get_root_dirs(name, dataset_type=dataset_type,
                             resolution=resolution, data_root=data_root)

    dataset = get_dataset(name=name, root=root,
                          resolution=resolution,
                          dataset_type=dataset_type,
                          load_in_mem=load_in_mem)

    sampler = DistributedSampler(dataset) if (distributed and split == 'train') else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None and shuffle), num_workers=num_workers,
                      pin_memory=pin_memory, drop_last=drop_last)


def get_dataloaders(name, root, dataset_type='ImageFolder', size=224, resolution=256,
                    batch_size=32, num_workers=12, shuffle=True, distributed=False,
                    load_in_mem=False, pin_memory=True, drop_last=True,
                    splits=['train', 'val'], **kwargs):
    dataloaders = {
        split: get_dataloader(name, data_root=root, split=split, size=size, resolution=resolution,
                              dataset_type=dataset_type, batch_size=batch_size, num_workers=num_workers,
                              shuffle=shuffle, load_in_mem=load_in_mem, pin_memory=pin_memory, drop_last=drop_last,
                              distributed=distributed, **kwargs)
        for split in splits}
    return dataloaders


def _miniplaces_get_dataloaders(data_dir, size, batch_size, shuffle=True, num_workers=12):
    # How to transform the image when you are loading them.
    # you'll likely want to mess with the transforms on the training set.

    # For now, we resize/crop the image to the correct input size for our network,
    # then convert it to a [C,H,W] tensor, then normalize it to values with a given mean/stdev.
    # These normalization constants are derived from aggregating lots of data and happen to
    # produce better results.
    data_transforms = {split: get_transform(split, size) for split in ['train', 'val']}

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in data_transforms.keys()}
    # image_datasets['train'] = torch.utils.data.ConcatDataset([image_datasets['train'], image_datasets['val']])
    # Create training and validation dataloaders
    # Never shuffle the test set
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                       batch_size=batch_size,
                                                       shuffle=False if x != 'train' else shuffle,
                                                       num_workers=num_workers,
                                                       pin_memory=True)
                        for x in data_transforms.keys()}
    return dataloaders_dict


def _biggan_get_dataset(name, root=None, resolution=128, dataset_type='ImageFolder',
                        split='train', transform=None, target_transform=None, load_in_mem=False,
                        download=False):

    if name == 'Hybrid':
        return get_hybrid_dataset(root_dir=root, resolution=resolution,
                                  dataset_type=dataset_type, load_in_mem=load_in_mem)

    if dataset_type == 'ImageFolder':
        # Get torchivision dataset class for desired dataset.
        dataset_func = getattr(torchvision.datasets, name)

        if name in ['CIFAR10', 'CIFAR100']:
            kwargs = {'train': True if split == 'train' else False}
        else:
            kwargs = {'split': split}
        if name == 'CelebA':
            def tf(x):
                return 0 if target_transform is None else target_transform
            kwargs = {**kwargs, 'target_transform': tf}

        if transform is None:
            transform = transforms.Compose([
                CenterCropLongEdge(),
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))])
        kwargs = {**kwargs,
                  'download': download,
                  'transform': transform}

        # Create dataset class based on config selection.
        dataset = dataset_func(root=root_dir, **kwargs)

    elif dataset_type == 'ImageHDF5':
        if download:
            raise NotImplementedError('Automatic Dataset Download not implemented yet...')

        hdf5_name = '{}-{}.hdf5'.format(name, resolution)
        hdf5_file = os.path.join(root_dir, hdf5_name)
        if not os.path.exists(hdf5_file):
            raise ValueError('Cannot find hdf5 file. You should download it, or create if yourself!')

        dataset = ImageHDF5(hdf5_file, load_in_mem=load_in_mem,
                            target_transform=target_transform)
    return dataset
