from functools import wraps
from collections import defaultdict


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


class History:

    def __init__(self, filename=None):
        self.filename = filename
        self.steps = []
        self.scalars = defaultdict(list)

    def add_scalar(self, k, v, step):
        self.steps.append(step)
        self.scalars[k].append(v)
