import torch


def accuracy(output, target, topk=(1, 5)):
    """Compute the precision@k for the specified values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.reshape(-1).size(0)

        _, pred = output.topk(maxk, 1, True, True)
        correct = pred.eq(target.unsqueeze(1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:, :k, ...].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Accuracy(object):
    def __init__(self, topk=(1, 5)):
        self.topk = topk

    def __call__(self, output, target, topk=None):
        topk = topk or self.topk
        return accuracy(output, target, topk=topk)
