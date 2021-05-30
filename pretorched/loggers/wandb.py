import wandb
from .base import LoggerBase, rank_zero_only


class WandbLogger(LoggerBase):
    def __init__(self, name, project='project', resume=True):
        self.name = name
        self.project = project
        self.run = wandb.init(name=name, project=project, resume=resume)

