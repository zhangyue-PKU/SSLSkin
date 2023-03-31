# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import logging
from typing import Any, Callable, Dict, List, Tuple, Union
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.methods.base import BaseMethod
from model.utils.lars import LARS
from model.utils.knn import WeightedKNNClassifier
from model.utils.metrics import (
    accuracy_at_k, 
    weighted_mean,
    binary_metrics,
    multiclass_metrics
)
from model.utils.misc import (
    omegaconf_select,
    remove_bias_and_norm_from_weight_decay,
)
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from model.losses import SoftTargetFocalLoss, FocalLoss

import matplotlib.pyplot as plt
import wandb


class KNN(pl.LightningModule):
         
    _OPTIMIZERS = {
        "sgd": torch.optim.SGD,
        "lars": LARS,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }
    
    _SCHEDULERS = [
        "reduce",
        "warmup_cosine",
        "step",
        "exponential"
    ]
    
    _LOSSES = {
        "ce": torch.nn.CrossEntropyLoss,
        "focal": FocalLoss,
        "label_smoothing_ce": LabelSmoothingCrossEntropy,
        "soft_target_ce": SoftTargetCrossEntropy,
        "soft_target_focal": SoftTargetFocalLoss
    }

    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        """Implements linear and finetune evaluation.

        .. note:: Cfg defaults are set in init by calling `cfg = add_and_assert_specific_cfg(cfg)`

        backbone (nn.Module): backbone architecture for feature extraction.
        Cfg basic structure:
            data:
                num_classes (int): number of classes in the dataset.
            max_epochs (int): total number of epochs.

            optimizer:
                name (str): name of the optimizer.
                batch_size (int): number of samples in the batch.
                lr (float): learning rate.
                weight_decay (float): weight decay for optimizer.
                kwargs (Dict): extra named arguments for the optimizer.
            scheduler:
                name (str): name of the scheduler.
                min_lr (float): minimum learning rate for warmup scheduler. Defaults to 0.0.
                warmup_start_lr (float): initial learning rate for warmup scheduler.
                    Defaults to 0.00003.
                warmup_epochs (float): number of warmup epochs. Defaults to 10.
                lr_decay_steps (Sequence, optional): steps to decay the learning rate if scheduler is
                    step. Defaults to None.
                interval (str): interval to update the lr scheduler. Defaults to 'step'.

            finetune (bool): whether or not to finetune the backbone. Defaults to False.

            performance:
                disable_channel_last (bool). Disables channel last conversion operation which
                speeds up training considerably. Defaults to False.
                https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html#converting-existing-models

        loss_func (Callable): loss function to use (for mixup, label smoothing or default). Defaults to None
        mixup_func (Callable, optional). function to convert data and targets with mixup/cutmix.
            Defaults to None.
        """

        super().__init__()

        # add default values and assert that config has the basic needed settings
        cfg = self.add_and_assert_specific_cfg(cfg)
        
        # backbone
        self.backbone_name = cfg.backbone.name
        base_model: Callable = BaseMethod._BACKBONES[cfg.backbone.name]
        self.backbone: nn.Module = base_model(cfg.pretrain.method, **cfg.backbone.kwargs)
        # num_feature
        if self.backbone_name.startswith("resnet"):
            features_dim: int = self.backbone.inplanes
            # remove fc layer
            self.backbone.fc = nn.Identity()
            if cfg.data.dataset in ["cifar10", "cifar100"]:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False                    
                )
                self.backbone.maxpool = nn.Identity()
        else:
            features_dim: int = self.backbone.num_features
        self.features_dim: int = features_dim
        
        # load checkpoint from pretrain result
        # cfg.pretrain.ckpt_key choose from ["backbone", "momentum_backbone"]
        ckpt_path = cfg.pretrain.ckpt
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        replace_prefix = cfg.pretrain.ckpt_key + "."
        state_dict = {k.replace(replace_prefix, ""): v for k, v in state_dict.items()}
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded {ckpt_path} with {msg}")
        
        # disable grad track
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        # knn classifier
        self.num_classes = cfg.data.num_classes
        self.knn_k: int = cfg.knn_eval.k
        self.knn = WeightedKNNClassifier(k=self.self.knn_k, distance_fx=cfg.knn.distance_func)


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        from omegaconf import OmegaConf
        
        # default parameters for knn eval
        cfg.knn_eval = omegaconf_select(cfg, "knn_eval", {})
        cfg.knn_eval.enabled = omegaconf_select(cfg, "knn_eval.enabled", False)
        cfg.knn_eval.k = omegaconf_select(cfg, "knn_eval.k", 20)
        cfg.knn_eval.distance_func = omegaconf_select(cfg, "knn_eval.distance_func", "euclidean")
        
        # check pretrain configs
        assert not OmegaConf.is_missing(cfg, "pretrain.method")
        assert not OmegaConf.is_missing(cfg, "pretrain.ckpt")
        cfg.pretrain.ckpt_key = omegaconf_select(cfg, "pretrain.ckpt_key", "backbone")
        assert cfg.pretrain.ckpt_key in ["backbone", "momentum_backbone"]
        assert cfg.pretrain.ckpt.endswith(".ckpt") \
            or cfg.pretrain.ckpt.endswith(".pth") \
            or cfg.pretrain.ckpt.endswith(".pt")
        
        # check backbone name
        assert not OmegaConf.is_missing(cfg, "backbone")
        assert not OmegaConf.is_missing(cfg, "backbone.name")
        assert cfg.backbone.name in BaseMethod._BACKBONES.keys()
        cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})

        return cfg


    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of frozen backbone

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        with torch.no_grad():
            feats = self.backbone(X)
            
        return {"feats": feats}


    def training_step(self, batch: List[Any], batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """
        _, X, targets = batch
        
        out = self(X)
        
        self.knn(
            train_features=out["feats"].detach(),
            train_targets=targets.detach()
        )
        


    def validation_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Performs the validation step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Dict[str, Any]:
                dict with the batch_size (used for averaging),
                the classification loss and accuracies.
        """
        _, X, targets = batch
        
        out = self(X) # keys = [feats, logits]
        self.knn(
            test_features=out["feats"].detach(),
            test_targets=targets.detach()
        )



    def validation_epoch_end(self, outputs: List[Dict[str, Any]]):
        """knn compute

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        metrics = self.knn.compute()
        
        self.log_dict(metrics, sync_dist=True)
    
        