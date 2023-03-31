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


class SupervisedModel(pl.LightningModule):
         
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
        """Implements Supervised Learning.

        .. note:: Cfg defaults are set in init by calling `cfg = add_and_assert_specific_cfg(cfg)`

        backbone (nn.Module): backbone architecture for feature extraction.
        Cfg basic structure:
            name: "Your running name"
            backbone:
                name: "resnet18"
                kwargs: backbone kwargs, default {}
            loss_fn: 
                name: "ce" # choose from [ce, focal, label_smoothing_ce, soft_target_ce]
                kwargs: default {}
            data:
                dataset: isic2016
                data_path: "data/pretrain/images"
                train_label: "data/pretrain/ISIC_2016_train.csv"
                val_label: "data/pretrain/ISIC_2016_test.csv"
                num_workers: 16
                debug_transform: True
            optimizer:
                name: "sgd"
                batch_size: 512
                lr: 0.001
                weight_decay: 0
            scheduler:
                name: None
            checkpoint:
                enabled: True
                dir: "trained_models"
                frequency: 1
            auto_resume:
                enabled: False
        """

        super().__init__()

        # add default values and assert that config has the basic needed settings
        cfg = self.add_and_assert_specific_cfg(cfg)
        
        # backbone
        self.backbone_name = cfg.backbone.name
        base_model: Callable = BaseMethod._BACKBONES[cfg.backbone.name]
        self.backbone: nn.Module = base_model(None, **cfg.backbone.kwargs)
        # num_feature
        if self.backbone_name.startswith("resnet"):
            features_dim: int = self.backbone.inplanes
            # remove fc layer
            self.backbone.fc = nn.Identity()
            if cfg.data.dataset in ["cifar10", "cifar100"]:
                # change first conv layer stride and cancel maxpooling layer
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False                    
                )
                self.backbone.maxpool = nn.Identity()
        else:
            features_dim: int = self.backbone.num_features #output feature dimension by backbone module
        self.features_dim: int = features_dim

        # classifier
        self.num_classes = cfg.data.num_classes
        self.classifier = nn.Linear(features_dim, cfg.data.num_classes)  # type: ignore
        
        # loss and metrics
        self.loss_func: nn.Module = self._LOSSES[cfg.loss_fn.name](**cfg.loss_fn.kwargs)
        self.metric_func: Callable = accuracy_at_k
        # online evaluation metrics 
        # TODO: other metrics
        self.metric_args = {"top_k": (1, max(1, min(5, cfg.data.num_classes)))}

        # mixup/cutmix function
        mixup_func: Union[Mixup, None] = None
        if cfg.mixup > 0 or cfg.cutmix > 0:
            logging.info(f"Mixup activated")
            mixup_func = Mixup(
                mixup_alpha=cfg.mixup,
                cutmix_alpha=cfg.cutmix,
                prob=1.0,
                switch_prob=0.5,
                mode="batch",
                label_smoothing=cfg.label_smoothing,
                num_classes=cfg.data.num_classes
            )
            assert cfg.loss_fn.name.startswith("soft"), "You must use soft target loss function if Mixup activated"
        self.mixup_func: Union[Mixup, None] = mixup_func

        # training related
        self.max_epochs: int = cfg.max_epochs
        self.accumulate_grad_batches: int = cfg.accumulate_grad_batches

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr * self.accumulate_grad_batches
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimizer_kwargs: Dict[str, Any] = cfg.optimizer.kwargs
        self.exclude_bias_n_norm_wd: bool = cfg.optimizer.exclude_bias_n_norm_wd

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr * self.accumulate_grad_batches
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr * self.accumulate_grad_batches
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval
        if self.scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        from omegaconf import OmegaConf
        
        # check backbone name
        assert not OmegaConf.is_missing(cfg, "backbone")
        assert not OmegaConf.is_missing(cfg, "backbone.name")
        assert cfg.backbone.name in BaseMethod._BACKBONES.keys()
        cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})
        
        # check loss_fn
        assert not OmegaConf.is_missing(cfg, "loss_fn")
        assert not OmegaConf.is_missing(cfg, "loss_fn.name")
        cfg.loss_fn.kwargs = omegaconf_select(cfg, "loss_fn.kwargs", {})
        assert cfg.loss_fn.name in ["ce", "focal", "label_smoothing_ce", "soft_target_ce", "soft_target_focal" ], \
                        "Only ce and focal loss are supported currently"    
        
        # optimizer 
        cfg.optimizer = omegaconf_select(cfg, "optimizer", {})
        assert not OmegaConf.is_missing(cfg, "optimizer.name")
        assert not OmegaConf.is_missing(cfg, "optimizer.lr")
        assert not OmegaConf.is_missing(cfg, "optimizer.batch_size")
        assert not OmegaConf.is_missing(cfg, "optimizer.weight_decay")
        scale_factor = cfg.optimizer.batch_size * len(cfg.devices) * cfg.num_nodes / 256
        cfg.optimizer.lr = cfg.optimizer.lr * scale_factor
        # adjust lr according to batch size
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", 1)
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(cfg, "optimizer.exclude_bias_n_norm_wd", False)
        # extra optimizer kwargs
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        if cfg.optimizer.name == "sgd":
            cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        elif cfg.optimizer.name == "lars":
            cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
            cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
            cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
            cfg.optimizer.kwargs.exclude_bias_n_norm_wd = omegaconf_select(
                cfg,
                "optimizer.kwargs.exclude_bias_n_norm_wd",
                False,
            )
        elif cfg.optimizer.name == "adamw":
            cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])
        
        # extra augment parameters
        cfg.auto_augment = omegaconf_select(cfg, "auto_augment", False)
        cfg.label_smoothing = omegaconf_select(cfg, "label_smoothing", 0.0)
        cfg.mixup = omegaconf_select(cfg, "mixup", 0.0)
        cfg.cutmix = omegaconf_select(cfg, "cutmix", 0.0)

        # default parameters for the scheduler
        cfg.scheduler = omegaconf_select(cfg, "scheduler", {})
        cfg.scheduler.name = omegaconf_select(cfg, "scheduler.name", None)
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 0)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "epoch")
        assert cfg.scheduler.interval in ["step", "epoch"]
        
        return cfg


    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        learnable_params = [{
                "name": "classifier",
                "params": self.classifier.parameters(),
                "weight_decay": 0
            },
            {
                "name": "backbone",
                "params": self.backbone.parameters()
            }]

        # exclude bias and norm from weight decay
        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_kwargs,
        )

        # select scheduler
        if self.scheduler is None:
            return optimizer
        
        if self.scheduler == "warmup_cosine":
            max_warmup_steps = (
                self.warmup_epochs * (self.trainer.estimated_stepping_batches / self.max_epochs)
                if self.scheduler_interval == "step"
                else self.warmup_epochs
            )
            max_scheduler_steps = (
                self.trainer.estimated_stepping_batches
                if self.scheduler_interval == "step"
                else self.max_epochs
            )
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=max_warmup_steps,
                    max_epochs=max_scheduler_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        elif self.scheduler == "reduce":
            scheduler = ReduceLROnPlateau(optimizer)
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps, gamma=0.1)
        elif self.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, self.weight_decay)
        else:
            raise ValueError(
                f"{self.scheduler} not in (warmup_cosine, cosine, reduce, step, exponential)"
            )

        return [optimizer], [scheduler]


    def forward(self, X: torch.tensor) -> Dict[str, Any]:
        """Performs forward pass of the frozen backbone and the linear layer for evaluation.

        Args:
            X (torch.tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing features and logits.
        """
        feats = self.backbone(X)
        logits = self.classifier(feats)
    
        return {"logits": logits, "feats": feats}


    def training_step(self, batch: List[Any], batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """
        _, X, targets = batch
        
        if self.mixup_func is not None:
                # targets change to soft target
            X, targets = self.mixup_func(X, targets)
        
        out = self(X)
        loss = self.loss_func(out["logits"], targets)
        out.update({"loss": loss}) # keys=["logits", "feats", "loss"]
        if self.mixup_func is None:
            # get metrics on current batch
            metrics = self.metric_func(out["logits"], targets, **self.metric_args)
            # modify metrics keys
            log = {"train_" + key : metrics[key] for key in metrics.keys()}
            self.log_dict(log, on_epoch=True, sync_dist=True)
        
        return out["loss"]


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
        val_loss = self.loss_func(out["logits"], targets)
        
        # get metrics on batch
        metrics = {
            "loss": val_loss,
            "batch_size" : X.size(0)
        }
        metrics.update(self.metric_func(out["logits"], targets, **self.metric_args))
        # modify keys
        log = {"val_" + key : metrics[key] for key in metrics.keys() if key != "batch_size"}
        # log  
        self.log_dict(log, on_epoch=True, sync_dist=True)
        
        # update preds, targets, metrics to out
        out.update({"targets": targets, "metrics": metrics}) # keys = [feats, logits, targets, metrics]
        
        return out


    def validation_epoch_end(self, outputs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        # mean of validation metrics
        metrics = [out["metrics"] for out in outputs]
        log = {}
        for key in metrics[0].keys():
            if key == "batch_size":
                continue
            log.update({f"val_{key}": weighted_mean(metrics, key, "batch_size")})

        self.log_dict(log, on_epoch=True, sync_dist=True)
        
        # then calculate metrics of epoch
        all_logits = []
        all_targets = []
        for out in outputs:
            all_logits.append(out["logits"])
            all_targets.append(out["targets"])

        all_logits = torch.cat(all_logits)
        all_targets = torch.cat(all_targets).view(-1)
        
        if self.num_classes == 2:
            # binary classification
            all_metrics = binary_metrics(all_logits, all_targets)
            metrics_to_report = [
                "acc",
                "best_f1",
                "best_threshold",
                "p@bestf1",
                "r@bestf1",
                "ap",
                "auc",
            ]
            log = {key: all_metrics[key] for key in metrics_to_report if key in all_metrics}
            self.log_dict(log, on_epoch=True, sync_dist=True)
            if self.trainer.current_epoch == self.max_epochs - 1:
                # if this is the last epoch, draw PR curve and ROC curve
                # 绘制PR曲线
                fig_pr, ax_pr = plt.subplots(1, 1, figsize=(8, 8), dpi=200)
                ax_pr.set_aspect('equal')
                ax_pr.plot(*all_metrics["pr_curve"])
                ax_pr.set_title("P-R Curve")
                ax_pr.set_xlabel("R")
                ax_pr.set_ylabel("P")
                wandb.log({'PR Curve': wandb.Image(fig_pr)}, commit=False)

                # 绘制ROC曲线
                fig_roc, ax_roc = plt.subplots(1, 1, figsize=(8, 8), dpi=200)
                ax_roc.plot(*all_metrics["roc_curve"])
                ax_roc.set_title("ROC Curve")
                ax_roc.set_xlabel("FPR")
                ax_roc.set_ylabel("TPR")
                wandb.log({'ROC Curve': wandb.Image(fig_roc)}, commit=False)
        else:
            all_metrics = multiclass_metrics(all_logits, all_targets)
            metrics_to_report = [
                "acc",
                "f1",
                "macro_f1",
                "micro_f1",
                "p",
                "macro_p",
                "r",
                "macro_r"
            ]
        
            log = {key: all_metrics[key] for key in metrics_to_report if key in all_metrics}
            ap = all_metrics["ap"]
            macro_ap = ap["macro"]
            micro_ap = ap["micro"]
            auc = all_metrics["auc"]
            macro_auc = auc["macro"]
            micro_auc = auc["micro"]
            log.update({
                "macro_ap": macro_ap,
                "micro_ap": micro_ap,
                "macro_auc": macro_auc,
                "micro_auc": micro_auc
            })
            self.log_dict(log, on_epoch=True, sync_dist=True)
            if self.trainer.current_epoch == self.max_epochs -1:
                r, p = all_metrics["pr_curve"]
                fpr, tpr = all_metrics["roc_curve"]

                # plot PR curves
                fig, axs = plt.subplots(1, self.num_classes + 1, figsize=(8 * (self.num_classes + 1), 8), dpi=200, sharex='col', sharey='row')
                fig.suptitle("P-R Curves")
                for i in range(self.num_classes + 1):
                    axs[i].set_aspect('equal')
                    if i == self.num_classes:
                        axs[i].plot(r["micro"], p["micro"])
                        axs[i].set_title("micro")
                    else:
                        axs[i].plot(r[f"class{i}"], p[f"class{i}"])
                        axs[i].set_title(f"class{i}")
                    axs[i].set_xlabel("R")
                    axs[i].set_ylabel("P")
                wandb.log({"P-R": wandb.Image(fig)}, commit=False)

                # plot ROC curves
                fig, axs = plt.subplots(1, self.num_classes + 1, figsize=(8 * (self.num_classes + 1), 8), dpi=200, sharex='col', sharey='row')
                fig.suptitle("ROC Curves")
                for i in range(self.num_classes + 1):
                    axs[i].set_aspect('equal')
                    if i == self.num_classes:
                        axs[i].plot(fpr["micro"], tpr["micro"])
                        axs[i].set_title("micro")
                    else:
                        axs[i].plot(fpr[f"class{i}"], tpr[f"class{i}"])
                        axs[i].set_title(f"class{i}")
                    axs[i].set_xlabel("FPR")
                    axs[i].set_ylabel("TPR")
                wandb.log({"ROC": wandb.Image(fig)}, commit=False)          