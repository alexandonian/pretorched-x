import pytest

import torch

from pretorched.metrics import accuracy


def pt_accuracy(output, target, topk=(1,)):
    """Compute the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@pytest.mark.parametrize('topk', [(1,), (1, 5)])
def test_accuracy(topk):
    bs = 128
    num_classes = 100
    target = torch.ones(bs)
    output = torch.randn(bs, num_classes)
    acc1 = accuracy(output, target, topk=topk)
    acc2 = pt_accuracy(output, target, topk=topk)
    assert acc1 == acc2


@pytest.mark.parametrize('topk', [(1, 5)])
def test_accuracy_multi_target(topk):
    bs = 128
    num_classes = 100
    for d in [1, 2]:
        target = torch.ones(bs, d)
        output = torch.randn(bs, num_classes, d)
        acc = accuracy(output, target, topk=topk)
        acc_list = [pt_accuracy(output[..., i], target[..., i], topk=(1, 5))
                    for i in range(d)]
        acc1 = [[a.tolist() for a in acc]]
        acc2 = [torch.tensor(list(zip(*acc_list))).mean(1).tolist()]
        assert acc1 == acc2
