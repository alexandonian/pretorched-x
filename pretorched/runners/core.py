import functools
import os
from operator import add

import torch
import torch.distributed as dist
import torch.nn as nn
from pretorched import data, models, optim, utils
from pretorched.data import samplers, transforms
from torch.nn import init
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from . import config as cfg


def get_optimizer(model, optimizer_name='SGD', lr=0.001, **kwargs):
    optim_func = getattr(optim, optimizer_name)
    func_kwargs, _ = utils.split_kwargs_by_func(optim_func, kwargs)
    optim_kwargs = {**cfg.optimizer_defaults.get(optimizer_name, {}), **func_kwargs}
    return optim_func(model.parameters(), lr=lr, **optim_kwargs)


def get_scheduler(optimizer, scheduler_name='CosineAnnealingLR', **kwargs):
    sched_func = getattr(torch.optim.lr_scheduler, scheduler_name)
    func_kwargs, _ = utils.split_kwargs_by_func(sched_func, kwargs)
    sched_kwargs = {**cfg.scheduler_defaults.get(scheduler_name, {}), **func_kwargs}
    return sched_func(optimizer, **sched_kwargs)


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


def get_model(
    model_name, num_classes=None, pretrained='imagenet', init_name=None, **kwargs
):
    model_func = getattr(models, model_name)
    if pretrained is not None:
        # TODO Update THIS!
        nc = {k.lower(): v for k, v in cfg.NUM_CLASSES.items()}.get(pretrained)
        model = model_func(num_classes=nc, pretrained=pretrained, **kwargs)
        if num_classes is not None and nc != num_classes:
            in_feat = model.last_linear.in_features
            last_linear = nn.Linear(in_feat, num_classes)
            if init_name is not None:
                print(f'Re-initializing last_linear of {model_name} with {init_name}.')
                last_linear = init_weights(last_linear, init_name)
            model.last_linear = last_linear
    else:
        if num_classes is None:
            # num_classes = 1000
            # print(f'num_classes not specified! Defaulting to {num_classes}')
            model = model_func(pretrained=pretrained, **kwargs)
        else:
            model = model_func(num_classes=num_classes, pretrained=pretrained, **kwargs)
        if init_name is not None:
            print(f'Initializing {model_name} with {init_name}.')
            model = init_weights(model, init_name)
    return model


def get_transform(
    name='ImageNet',
    split='train',
    size=224,
    resolution=256,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    normalize = transforms.Normalize(mean=mean, std=std)
    data_transforms = {
        'train': transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        'val': transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        'test': transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }
    return data_transforms.get(split)


def get_video_transform(
    name='Moments',
    split='train',
    size=224,
    resolution=256,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    normalize=True,
):
    norm = transforms.NormalizeVideo(mean=mean, std=std)
    cropping = {
        'train': transforms.Compose(
            [
                transforms.RandomResizedCropVideo(size),
                transforms.RandomHorizontalFlipVideo(),
            ]
        ),
        'val': transforms.Compose(
            [transforms.ResizeVideo(resolution), transforms.CenterCropVideo(size)]
        ),
        'test': transforms.Compose(
            [transforms.ResizeVideo(resolution), transforms.CenterCropVideo(size)]
        ),
    }.get(split, 'val')
    return transforms.Compose(
        [
            cropping,
            transforms.CollectFrames(),
            transforms.PILVideoToTensor(),
            norm if normalize else transforms.IdentityTransform(),
        ]
    )


def get_dataset(
    name,
    root,
    split='train',
    size=224,
    resolution=256,
    dataset_type='ImageFolder',
    **kwargs,
):

    Dataset = getattr(data, dataset_type, 'ImageFolder')

    kwargs = {
        'root': root,
        'metafile': os.path.join(root, f'{split}.txt'),
        'transform': get_transform(name, split, size, resolution),
        **kwargs,
    }
    dataset_kwargs, _ = utils.split_kwargs_by_func(Dataset, kwargs)
    return Dataset(**dataset_kwargs)


def get_video_dataset(
    name,
    data_root=None,
    split='train',
    num_frames=16,
    size=224,
    resolution=256,
    dataset_type='VideoRecordDataset',
    sampler_type='TSNFrameSampler',
    record_set_type='RecordSet',
    load_in_mem=False,
    segment_count=None,
    **kwargs,
):

    segment_count = num_frames if segment_count is None else segment_count

    metadata = cfg.get_metadata(
        name,
        split=split,
        dataset_type=dataset_type,
        record_set_type=record_set_type,
        data_root=data_root,
    )
    kwargs = {**metadata, **kwargs, 'segment_count': segment_count}

    Dataset = getattr(data, dataset_type, 'VideoRecordDataset')
    RecSet = getattr(data, record_set_type, 'RecordSet')
    Sampler = getattr(samplers, sampler_type, 'TSNFrameSampler')

    r_kwargs, _ = utils.split_kwargs_by_func(RecSet, kwargs)
    s_kwargs, _ = utils.split_kwargs_by_func(Sampler, kwargs)
    record_set = RecSet(**r_kwargs)
    sampler = Sampler(**s_kwargs)
    full_kwargs = {
        'record_set': record_set,
        'sampler': sampler,
        'transform': get_video_transform(split=split, size=size),
        **kwargs,
    }
    dataset_kwargs, _ = utils.split_kwargs_by_func(Dataset, full_kwargs)
    return Dataset(**dataset_kwargs)


def get_video_dataloader(
    name,
    data_root=None,
    split='train',
    num_frames=16,
    size=224,
    resolution=256,
    dataset_type='VideoRecordDataset',
    sampler_type='TSNFrameSampler',
    record_set_type='RecordSet',
    batch_size=64,
    num_workers=8,
    shuffle=True,
    load_in_mem=False,
    pin_memory=True,
    drop_last=False,
    distributed=False,
    segment_count=None,
    **kwargs,
):

    segment_count = num_frames if segment_count is None else segment_count

    metadata = cfg.get_metadata(
        name,
        split=split,
        dataset_type=dataset_type,
        record_set_type=record_set_type,
        data_root=data_root,
    )
    kwargs = {**metadata, **kwargs, 'segment_count': segment_count}

    Dataset = getattr(data, dataset_type, 'VideoRecordDataset')
    RecSet = getattr(data, record_set_type, 'RecordSet')
    Sampler = getattr(samplers, sampler_type, 'TSNFrameSampler')

    r_kwargs, _ = utils.split_kwargs_by_func(RecSet, kwargs)
    s_kwargs, _ = utils.split_kwargs_by_func(Sampler, kwargs)
    record_set = RecSet(**r_kwargs)
    sampler = Sampler(**s_kwargs)
    full_kwargs = {
        'record_set': record_set,
        'sampler': sampler,
        'transform': get_video_transform(split=split, size=size),
        **kwargs,
    }
    dataset_kwargs, _ = utils.split_kwargs_by_func(Dataset, full_kwargs)
    dataset = Dataset(**dataset_kwargs)

    loader_sampler = (
        DistributedSampler(dataset) if (distributed and split == 'train') else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=loader_sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_hybrid_dataset(
    name,
    root,
    split='train',
    size=224,
    resolution=256,
    dataset_type='ImageFolder',
    load_in_mem=False,
):
    if name != 'Hybrid1365':
        raise ValueError(f'Hybrid Dataset: {name} not implemented')
    imagenet_root = cfg.get_root_dirs(
        'ImageNet', dataset_type=dataset_type, resolution=resolution, data_root=root
    )
    places365_root = cfg.get_root_dirs(
        'Places365', dataset_type=dataset_type, resolution=resolution, data_root=root
    )
    imagenet_dataset = get_dataset(
        'ImageNet',
        resolution=resolution,
        size=size,
        dataset_type=dataset_type,
        load_in_mem=load_in_mem,
        split=split,
        root=imagenet_root,
    )
    placess365_dataset = get_dataset(
        'Places365',
        resolution=resolution,
        size=size,
        dataset_type=dataset_type,
        load_in_mem=load_in_mem,
        target_transform=functools.partial(add, 1000),
        split=split,
        root=places365_root,
    )
    return torch.utils.data.ConcatDataset((imagenet_dataset, placess365_dataset))


def get_dataloader(
    name,
    data_root=None,
    split='train',
    size=224,
    resolution=256,
    dataset_type='ImageFolder',
    batch_size=64,
    num_workers=8,
    shuffle=True,
    load_in_mem=False,
    pin_memory=True,
    drop_last=True,
    distributed=False,
    **kwargs,
):
    if name in cfg.VIDEO_DATASETS:
        return get_video_dataloader(
            name,
            data_root=data_root,
            split=split,
            size=size,
            resolution=resolution,
            dataset_type=dataset_type,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            load_in_mem=load_in_mem,
            pin_memory=pin_memory,
            drop_last=drop_last,
            distributed=distributed,
            **kwargs,
        )
    root = cfg.get_root_dirs(
        name, dataset_type=dataset_type, resolution=resolution, data_root=data_root
    )
    get_dset_func = get_hybrid_dataset if name == 'Hybrid1365' else get_dataset
    dataset = get_dset_func(
        name=name,
        root=root,
        size=size,
        split=split,
        resolution=resolution,
        dataset_type=dataset_type,
        load_in_mem=load_in_mem,
        **kwargs,
    )

    sampler = (
        DistributedSampler(dataset) if (distributed and split == 'train') else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and shuffle),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def get_dataloaders(
    name,
    root,
    dataset_type='ImageFolder',
    size=224,
    resolution=256,
    batch_size=32,
    num_workers=12,
    shuffle=True,
    distributed=False,
    load_in_mem=False,
    pin_memory=True,
    drop_last=True,
    splits=['train', 'val'],
    **kwargs,
):
    return {
        split: get_dataloader(
            name,
            data_root=root,
            split=split,
            size=size,
            resolution=resolution,
            dataset_type=dataset_type,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            load_in_mem=load_in_mem,
            pin_memory=pin_memory,
            drop_last=drop_last,
            distributed=distributed,
            **kwargs,
        )
        for split in splits
    }


def get_rank(group=None):
    try:
        return dist.get_rank() if group is None else dist.get_rank(group)
    except AssertionError:
        return 0


def is_rank_zero(group=None):
    return get_rank(group) == 0


def init_ddp_env(
    gpu_idx,
    node_rank=-1,
    ngpus_per_node=None,
    dist_backend='nccl',
    dist_url='tcp://localhost:23456',
    world_size=-1,
    group_name='',
):
    if dist_url == "env://" and node_rank == -1:
        node_rank = int(os.environ["RANK"])
    if node_rank == -1:
        node_rank = int(os.environ.get('OMPI_COMM_WORLD_RANK'))
    rank = node_rank * ngpus_per_node + gpu_idx
    if ngpus_per_node is None:
        ngpus_per_node = torch.cuda.device_count()
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank,
    )


def distribute_model(model, device='cuda', gpu_idx=None):

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    if device == 'cpu':
        model.cpu()
        return model
    if gpu_idx is not None:
        torch.cuda.set_device(gpu_idx)
        model.to(f'{device}:{gpu_idx}')
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_idx])
    else:
        model.cuda()
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        model = torch.nn.parallel.DistributedDataParallel(model)
    return model


def resume_checkpoint(
    checkpoint_path,
    model=None,
    optimizer=None,
    scheduler=None,
    model_key='state_dict',
    optimizer_key='optimizer',
    scheduler_key='scheduler',
):
    # TODO: FINISH THIS
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"=> no checkpoint found at '{checkpoint_path}'")
    print(f"=> loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args.start_epoch = checkpoint['epoch']
    best_acc1 = checkpoint['best_acc1']
    if model is not None:
        model.load_state_dict(checkpoint[model_key])
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint[optimizer_key])
        except ValueError('Could not find optimizer state'):
            pass
    try:
        scheduler.load_state_dict(checkpoint[scheduler_key])
    except Exception:
        print(f'Could not load scheduler state_dict for {args.scheduler}')
        try:
            scheduler.step(checkpoint['epoch'])
            print(f'setting scheduler learning rate to: {scheduler.get_lr()}')
        except Exception:
            pass

    print(
        "=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_path, checkpoint['epoch']
        )
    )


def featurize(
    image_dir,
    model_name='resnet50',
    pretrained='imagenet',
    batch_size=64,
    device='cuda',
    num_workers=2,
    pin_memory=True,
):
    model = get_model(model_name, pretrained=pretrained)
    dataset = data.ImageDir(image_dir, transform=get_transform(split='val'))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    model.eval()
    model = model.to(device)
    features = []
    with torch.no_grad():
        for i, (inp, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            inp = inp.to(device)
            feats = model.features(inp)
            features.append(feats)
    features = torch.cat(features)
    return features
