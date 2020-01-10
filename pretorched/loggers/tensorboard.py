"""Adapted from pytorch-lightning:
https://github.com/williamFalcon/pytorch-lightning/blob/master/pytorch_lightning/logging/tensorboard.py
"""
import os
from warnings import warn

import torch
from pkg_resources import parse_version
from torch.utils.tensorboard import SummaryWriter

from pretorched.utils import cache
from .base import LoggerBase, rank_zero_only


class TensorBoardLogger(LoggerBase):
    r"""Log to local file system in TensorBoard format

    Implemented using :class:`torch.utils.tensorboard.SummaryWriter`. Logs are saved to
    `os.path.join(save_dir, name, version)`

    :example:

    .. code-block:: python

        logger = TensorBoardLogger("tb_logs", name="my_model")
        trainer = Trainer(logger=logger)
        trainer.train(model)

    :param str save_dir: Save directory
    :param str name: Experiment name. Defaults to "default".
    :param int version: Experiment version. If version is not specified the logger inspects the save
        directory for existing versions, then automatically assigns the next available version.
    :param \**kwargs: Other arguments are passed directly to the :class:`SummaryWriter` constructor.


    """

    def __init__(self, logs_root='logs', name="default", version=None, rank=0, **kwargs):
        super().__init__()
        self.logs_root = logs_root
        self.name = name
        self.rank = rank
        self._version = version

        self.kwargs = kwargs

    # @rank_zero_only
    @cache
    def experiment(self):
        """The underlying :class:`torch.utils.tensorboard.SummaryWriter`.

        :rtype: torch.utils.tensorboard.SummaryWriter
        """
        root_dir = os.path.join(self.logs_root, self.name)
        os.makedirs(root_dir, exist_ok=True)
        log_dir = os.path.join(root_dir, str(self.version))
        return SummaryWriter(log_dir=log_dir, **self.kwargs)

    @rank_zero_only
    def log_hyperparams(self, hparams):
        if parse_version(torch.__version__) < parse_version("1.3.0"):
            warn(
                f"Hyperparameter logging is not available for Torch version {torch.__version__}."
                " Skipping log_hyperparams. Upgrade to Torch 1.3.0 or above to enable"
                " hyperparameter logging."
            )
            # TODO: some alternative should be added
            return
        try:
            # in case converting from namespace, todo: rather test if it is namespace
            hparams = vars(hparams)
        except TypeError:
            pass
        if hparams is not None:
            self.experiment.add_hparams(hparam_dict=dict(hparams), metric_dict={})

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.experiment.add_scalar(k, v, step)

    @rank_zero_only
    def save(self):
        try:
            self.experiment.flush()
        except AttributeError:
            # you are using PT version (<v1.2) which does not have implemented flush
            self.experiment._get_file_writer().flush()

    @rank_zero_only
    def finalize(self, status):
        self.save()

    @property
    def version(self):
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self):
        root_dir = os.path.join(self.logs_root, self.name)
        existing_versions = [
            int(d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.isdigit()
        ]
        if len(existing_versions) == 0:
            return 0
        else:
            return max(existing_versions) + 1


if __name__ == '__main__':

    # logger = Logger('test')
    hparams = {'arg': 'asdf', 'optimizer': 'Adam', 'bs': 128}
    metric_dict = {'hparam/best_train_acc': 14}
    writer = SummaryWriter()
    # writer.add_hparams(hparam_dict=hparams, metric_dict=metric_dict)
    import numpy as np
    itr = 0
    for epoch in range(10):
        for n_iter in range(100):
            itr += 1
            writer.add_scalars('Loss', {'train': np.random.random(), 'val': np.random.random()}, itr)
        # writer.add_scalar('Loss/train', np.random.random(), n_iter)
        # writer.add_scalar('Loss/test', np.random.random(), n_iter)

        writer.add_scalar('Accuracy/train', np.random.random(), itr)
        writer.add_scalar('Accuracy/test', np.random.random(), itr)

    # metric_dict = {'hparam/best_train_acc': 94}
    # hparams = {'arg': 'asdf', 'optimizer': 'Adam', 'bs': 256}
    for i in range(2):
        writer.add_hparams({'lr': 0.1 * i, 'bsize': i},
                           {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i})
    for i in range(2):
        writer.add_hparams({'lr': 0.1 * i, 'bsize': i},
                           {'hparam/accuracy': 10 * i, 'hparam/loss': 10 * i})
