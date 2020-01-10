from functools import wraps


def rank_zero_only(fn):
    """Decorate a logger method to run it only on the process with rank 0.
    :param fn: Function to decorate
    """

    @wraps(fn)
    def wrapped_fn(self, *args, **kwargs):
        if self.rank == 0:
            fn(self, *args, **kwargs)

    return wrapped_fn


class LoggerBase:
        # batch_time = AveraeMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    # losses = AverageMeter('Loss', ':.4e')
    # top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
        # len(train_loader),
        # [batch_time, data_time, losses, top1, top5],
        # prefix="Epoch: [{}]".format(epoch))

    def __init__(self, logs_root='logs', comment='', filename_suffix='', rank=0):
        self.logs_root = logs_root
        self.rank = rank

    @rank_zero_only
    def log(self, itr, meters, mode='train'):
        pass

    @rank_zero_only
    def display(self, batch):
        self.progress.display(batch)

    @property
    def rank(self):
        """Process rank. In general, metrics should only be logged by the process with rank 0."""
        return self._rank

    @rank.setter
    def rank(self, value):
        """Set the process rank."""
        self._rank = value

