from collections import defaultdict
from typing import Dict

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from torchvision import models

from .torchvision_models import (
    load_pretrained,
    modify_resnets,
    resnet50 as base_resnet50,
)


__all__ = [
    'resnet50',
]

model_urls = {
    'imagenet': 'http://pretorched-x.csail.mit.edu/models/moco_v2_800ep_imagenet-9e6149f4.pth',
    'imagenet_v1_200ep': 'http://pretorched-x.csail.mit.edu/models/moco_v1_200ep_imagenet-4e921892.pth',
    'imagenet_v2_200ep': 'http://pretorched-x.csail.mit.edu/models/moco_v2_200ep_imagenet-8a51f46e.pth',
    'imagenet_v2_800ep': 'http://pretorched-x.csail.mit.edu/models/moco_v2_800ep_imagenet-9e6149f4.pth',
    # 'imagenet': 'http://pretorched-x.csail.mit.edu/models/moco_v2_800ep_imagenet-9783bd5b.pth',
    # 'imagenet_v1_200ep': 'http://pretorched-x.csail.mit.edu/models/moco_v1_200ep_imagenet-515bf1aa.pth',
    # 'imagenet_v2_200ep': 'http://pretorched-x.csail.mit.edu/models/moco_v2_200ep_imagenet-a4418128.pth',
    # 'imagenet_v2_800ep': 'http://pretorched-x.csail.mit.edu/models/moco_v2_800ep_imagenet-9783bd5b.pth',
}

has_mlp = {
    'imagenet': True,
    'imagenet_v1_200ep': False,
    'imagenet_v2_200ep': True,
    'imagenet_v2_800ep': True,
}
embed_dim = 128

pretrained_settings: Dict[str, Dict[str, Dict]] = defaultdict(dict)
input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 1, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in __all__:
    if model_name in ['ResNet3D']:
        continue
    for dataset, url in model_urls.items():
        pretrained_settings[model_name][dataset] = {
            'input_space': 'RGB',
            'input_range': [0, 1],
            'url': url,
            'std': stds[model_name],
            'mean': means[model_name],
            'num_classes': embed_dim,
            'input_size': input_sizes[model_name],
            'mlp': has_mlp[dataset],
        }


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, pretrained=None)
        self.encoder_k = base_encoder(num_classes=dim, pretrained=None)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer('memory', torch.randn(K, dim))
        self.memory = nn.functional.normalize(self.memory, dim=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, return_q=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        if return_q:
            return logits, labels, q
        else:
            return logits, labels


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def resnet50(num_classes=128, pretrained='imagenet', mlp=False):
    """Constructs a ResNet-50 model.
    """
    if pretrained is not None:
        settings = pretrained_settings['resnet50'][pretrained]
        model = MoCo(models.resnet50, mlp=settings['mlp'],)
        model = load_pretrained(model.encoder_q, num_classes, settings)
    else:
        model = MoCo(models.resnet50, mlp=mlp).encoder_q
        print(
            'WARNING: no pretrained weights loaded... Consider calling the MoCo constructor directly'
        )
    return model
