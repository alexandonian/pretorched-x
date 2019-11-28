import argparse
import os
from collections import defaultdict

import pretorched.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

DATA_ROOT = '/data/vision/oliva/scratch/datasets'

num_classes_dict = {
    'ImageNet': 1000,
    'Places365': 365}

root_dirs = {
    'ImageNet': os.path.join(DATA_ROOT, 'ImageNet'),
    'Places365': os.path.join(DATA_ROOT, 'Places365'),
}


def get_root_dirs(name, dataset_type='ImageHDF5', resolution=128, data_root=DATA_ROOT):
    root_dirs = {
        'ImageNet': {
            'ImageHDF5': defaultdict(lambda: os.path.join(data_root, 'ImageNet'), {}),
            'ImageFolder': defaultdict(lambda: os.path.join(data_root, 'ImageNet'), {}),
        },
        'Places365': {
            'ImageHDF5': defaultdict(lambda: os.path.join(data_root, 'Places365'), {}),
            'ImageFolder': defaultdict(lambda: os.path.join(data_root, 'Places365'), {}),
        },
        'Hybrid': {
            'ImageHDF5': defaultdict(lambda: data_root, {}),
        }
    }
    return root_dirs[name][dataset_type][resolution]


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--data_root', metavar='DIR', default=None,
                        help='path to data_root containing all datasets')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('--pretrained', type=str, default=None, dest='pretrained',
                        help='use a pre-trained model')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR')
    parser.add_argument('--init', type=str, default='ortho')

    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--weights_dir', default='weights', type=str, metavar='PATH')
    parser.add_argument('--logs_dir', default='logs', type=str, metavar='PATH')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--ddp', '--multiprocessing-distributed', action='store_true',
                        dest='multiprocessing_distributed',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = DATA_ROOT
    args.num_classes = num_classes_dict[args.dataset]
    return args
