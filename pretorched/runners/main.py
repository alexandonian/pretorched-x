import json
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from pretorched.metrics import accuracy

from . import core
from . import config as cfg

from .utils import AverageMeter, ProgressMeter

best_acc1 = 0


def main():
    args = cfg.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    dir_path = os.path.dirname(os.path.realpath(__file__))
    args.weights_dir = os.path.join(dir_path, args.weights_dir)
    args.logs_dir = os.path.join(dir_path, args.logs_dir)
    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)

    save_name = '_'.join([args.arch,
                          args.dataset.lower(),
                          f'init-{"-".join([args.pretrained, args.init]) if args.pretrained else args.init}',
                          f'optim-{args.optimizer}',
                          f'lr-{args.lr}',
                          f'sched-{args.scheduler}',
                          f'bs-{args.batch_size}',
                          ])
    print(f'Starting: {save_name}')

    args.log_file = os.path.join(args.logs_dir, save_name + '.json')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = core.get_model(args.arch, args.num_classes,
                           pretrained=args.pretrained,
                           init_name=args.init)
    input_size = model.input_size[-1]

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = core.get_optimizer(model, args.optimizer,
                                   lr=args.lr, momentum=args.momentum,
                                   weight_decay=args.weight_decay)

    scheduler = core.get_scheduler(optimizer, args.scheduler)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    dataloaders = core.get_dataloaders(args.dataset, args.data_root,
                                       batch_size=args.batch_size,
                                       num_workers=args.num_workers,
                                       distributed=args.distributed,
                                       size=input_size)
    train_loader, val_loader = dataloaders['train'], dataloaders['val']
    train_sampler = train_loader.sampler

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    history = {
        'epoch': [],
        'loss': [],
        'val_loss': [],
        'acc': {
            'avg': [],
            'top1': [],
            'top5': [],
        },
        'val_acc': {
            'avg': [],
            'top1': [],
            'top5': [],
        },
    }

    val_acc_history = {
        'avg': [],
        'top1': [],
        'top5': [],
    }
    train_acc_history = {
        'avg': [],
        'top1': [],
        'top5': [],
    }

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        display = is_rank0(args, ngpus_per_node)

        # Train for one epoch.
        train_acc1, train_acc5, train_loss = train(train_loader, model,
                                                   criterion, optimizer,
                                                   epoch, args, display)

        train_acc_history['top1'].append(train_acc1.item())
        train_acc_history['top5'].append(train_acc5.item())
        train_acc_history['avg'].append(((train_acc1 + train_acc5) / 2).item())

        history['acc']['top1'].append(train_acc1.item())
        history['acc']['top5'].append(train_acc5.item())
        history['acc']['avg'].append(((train_acc1 + train_acc5) / 2).item())
        history['loss'].append(train_loss)

        # Evaluate on validation set.
        val_acc1, val_acc5, val_loss = validate(val_loader, model, criterion, args, display)

        val_acc_history['top1'].append(val_acc1)
        val_acc_history['top5'].append(val_acc5)
        val_acc_history['avg'].append((val_acc1 + val_acc5) / 2)

        history['val_acc']['top1'].append(val_acc1)
        history['val_acc']['top5'].append(val_acc5)
        history['val_acc']['avg'].append((val_acc1 + val_acc5) / 2)
        history['val_loss'].append(val_loss)
        history['epoch'].append(epoch + 1)

        # Update the learning rate.
        if type(scheduler).__name__ == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        if is_rank0(args, ngpus_per_node):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }, is_best, filename=os.path.join(args.weights_dir, save_name))

            with open(args.log_file, 'w') as f:
                json.dump(history, f, indent=4)


def is_rank0(args, ngpus_per_node):
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0)


def train(train_loader, model, criterion, optimizer, epoch, args, display=True):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1, images.size(0))
        top5.update(acc5, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 and display:
            progress.display(i)

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, args, display=True):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 and display:
                progress.display(i)

        if display:
            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
                  .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, filename='', suffix='checkpoint.pth.tar'):
    checkpoint = '_'.join([filename, suffix])
    torch.save(state, checkpoint)
    if is_best:
        best_name = '_'.join([filename, 'best.pth.tar'])
        shutil.copyfile(checkpoint, best_name)


if __name__ == '__main__':
    main()
