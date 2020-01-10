
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, verbose=True):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        msg = '\t'.join(entries)
        print(msg) if verbose else None
        return msg

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def reset(self):
        for meter in self.meters:
            meter.reset()


class MovingAverage(object):
    """Computes the moving average of a given float."""

    def __init__(self, momentum=0):
        self.momentum = momentum
        self.val = None
        self.previous = None

    def reset(self):
        self.val = None

    def update(self, val):
        self.previous = self.val
        if self.val is None:
            self.val = val
        else:
            self.val = self.momentum * self.val + (1 - self.momentum) * val
        return self.val

    @property
    def relative_change(self):
        if None not in [self.val, self.previous]:
            relative_change = (self.previous - self.val) / self.previous
            return relative_change
        else:
            return None
