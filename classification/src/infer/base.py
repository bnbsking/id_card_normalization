import os

import torch
import yaml

from src.models.probing import TimmProbingModel


class BaseInference:
    def __init__(self, pretrain_weights: str, threshold: float):
        self.pretrain_weights = None
        self.threshold = None
        self.model = None
        self.set_cfg(pretrain_weights, threshold)
        
    def set_cfg(self, pretrain_weights: str | None = None, threshold: float | None = None):
        if pretrain_weights is not None and pretrain_weights != self.pretrain_weights:
            self.pretrain_weights = pretrain_weights
            with open(os.environ["BASE_CFG_PATH"], "r") as f:
                cfg = yaml.safe_load(f)["train"]
            self.model = TimmProbingModel(cfg["model"]["name"], cfg["model"]["num_classes"])
            self.model.load_state_dict(torch.load(pretrain_weights))
            self.model.eval()
            self.img_size = cfg["dataset"]["img_size"]
        if threshold is not None:
            self.threshold = threshold

    def inference(self, img_path: str):
        raise NotImplementedError("The inference method must be implemented by the subclass.")
    