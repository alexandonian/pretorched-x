import json
import os
from collections import defaultdict
from functools import wraps

import torch


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
    def __init__(
        self, logs_root='logs', name='default', filename_suffix='.json', rank=0
    ):
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

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.history.add_scalar(k, v, step)


class History:
    def __init__(self, filename=None):
        self.filename = filename
        self.scalars = defaultdict(list)

    def add_scalar(self, k, v, step):
        self.scalars[k].append((step, v))

    def add_scalars(self, scalars_dict, step, main_tag=None):
        for k, v in scalars_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            if main_tag is not None:
                k = os.path.join(main_tag, k)
            self.add_scalar(k, v, step)

    def update(self, vars_dict):
        self.__dict__.update(vars_dict)

    def to_json(self, filename=None):
        fname = filename if filename is not None else self.filename
        with open(fname, 'w') as f:
            json.dump(vars(self), f)

    @classmethod
    def from_json(cls, filename):
        h = cls(filename)
        if os.path.exists(filename):
            with open(filename) as f:
                try:
                    data = json.load(f)
                except json.decoder.JSONDecodeError:
                    print('Could not load json log file. Re-initializing file...')
                else:
                    h.update(data)
        return h


class HistoryV0:
    def __init__(self, filename=None):
        self.filename = filename
        self.steps = defaultdict(list)
        self.scalars = defaultdict(list)

    def add_scalar(self, k, v, step):
        self.steps[k].append(step)
        self.scalars[k].append(v)
