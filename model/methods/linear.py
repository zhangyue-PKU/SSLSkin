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
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from model.methods.base import BaseMethod
from model.utils.lars import LARS
from model.utils.metrics import accuracy_at_k, weighted_mean
from model.utils.misc import (
    omegaconf_select,
    param_groups_layer_decay,
    remove_bias_and_norm_from_weight_decay,
)
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, ReduceLROnPlateau
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from model.losses import SoftTargetFocalLoss, FocalLoss


class LinearModel(pl.LightningModule):
         
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
        
        # ckpt loadding
        ckpt_path = cfg.pretrain.ckpt
        state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        replace_prefix = cfg.pretrain.ckpt_key + "."
        state_dict = {k.replace(replace_prefix, ""): v for k, v in state_dict.items()}
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        logging.info(f"Loaded {ckpt_path} with {msg}")
        
        # classifier
        self.classifier = nn.Linear(features_dim, cfg.data.num_classes)  # type: ignore
        self.loss_func: nn.Module = self._LOSSES[cfg.loss_fn.name](**cfg.loss_fn.kwargs)
        self.metric_func: Callable = accuracy_at_k
        # BUG metirc_args
        self.metric_args = {"top_k": (1, min(1, min(5, cfg.data.num_classes)))}
        self.metric_keys = [f"acc{k}" for k in set(self.metric_args["top_k"])]

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
        self.layer_decay: float = cfg.optimizer.layer_decay

        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr * self.accumulate_grad_batches
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr * self.accumulate_grad_batches
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        self.scheduler_interval: str = cfg.scheduler.interval
        assert self.scheduler_interval in ["step", "epoch"]
        if self.scheduler_interval == "step":
            logging.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

        # if finetuning the backbone
        self.finetune: bool = cfg.finetune
        
        if not self.finetune:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        from omegaconf import OmegaConf
        
        # check eval state
        assert not OmegaConf.is_missing(cfg, "finetune")
        
        # check pretrain configs
        assert not OmegaConf.is_missing(cfg, "pretrain.method")
        assert not OmegaConf.is_missing(cfg, "pretrain.ckpt")
        cfg.pretrain.ckpt_key = omegaconf_select(cfg, "pretrain.ckpt_key", "backbone")
        assert cfg.pretrain.ckpt.endswith(".ckpt") \
            or cfg.pretrain.ckpt.endswith(".pth") \
            or cfg.pretrain.ckpt.endswith(".pt")
        
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
        cfg.optimizer.layer_decay = omegaconf_select(cfg, "optimizer.layer_decay", 0.)
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

        return cfg


    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        learnable_params = [{
            "name": "classifier",
            "params": self.classifier.parameters()
        }]
        if self.finetune:
            if self.layer_decay > 0:
                msg = "Method should implement no_weight_decay() that returns a set of parameter names to ignore from weight decay"
                assert hasattr(self.backbone, "no_weight_decay"), msg

                extra_learnable_params = param_groups_layer_decay(
                    self.backbone,
                    self.weight_decay,
                    no_weight_decay_list=self.backbone.no_weight_decay(),
                    layer_decay=self.layer_decay,
                )
                learnable_params += extra_learnable_params 
            else:
                learnable_params += [{
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
        with torch.set_grad_enabled(self.finetune):
            feats = self.backbone(X)
        logits = self.classifier(feats)
    
        return {"logits": logits, "feats": feats}


    def _shared_step(
        self, batch: List[Any], batch_idx: int
    ) -> Dict:
        """Performs operations that are shared between the training nd validation steps.

        Args:
            batch (Tuple): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
                batch size, loss, accuracy @1 and accuracy @5.
        """
        _, X, targets = batch
        
        outs = {"batch_size": X.size(0)} 
        
        if self.training:
            if self.mixup_func is not None:
                # targets change to soft target
                X, targets = self.mixup_func(X, targets)
            outs.update(self(X))
            logits = outs["logits"]
            loss = self.loss_func(logits, targets)
            outs.update({"loss": loss})
            if self.mixup_func is None:
                metrics = self.metric_func(logits, targets, **self.metric_args)
                outs.update({**metrics})
        else:
            outs.update(self(X))
            logits = outs["logits"]
            loss = self.loss_func(logits, targets)
            outs.update({"loss": loss})
            metrics = self.metric_func(logits, targets, **self.metric_args)
            outs.update(metrics)

        return outs


    def training_step(self, batch: List[Any], batch_idx: int) -> torch.Tensor:
        """Performs the training step for the linear eval.

        Args:
            batch (torch.Tensor): a batch of images in the tensor format.
            batch_idx (int): the index of the batch.

        Returns:
            torch.Tensor: cross-entropy loss between the predictions and the ground truth.
        """
        
        out = self._shared_step(batch, batch_idx)
        
        log = {"train_loss": out["loss"]}
        if self.mixup_func is None:
            for key in self.metric_keys:
                log.update({"train_" + key: out.pop(key, None)})
            
        # debug
        with torch.no_grad():  
            logits = out["logits"]
            probs = F.softmax(logits, dim=1)
            log.update({"train_prob": torch.mean(probs[:, 1])})

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
        out = self._shared_step(batch, batch_idx)
        
        log = {
            "batch_size": out["batch_size"],
            "val_loss": out["loss"]
        }

        for key in self.metric_keys:
            log.update({"val_" + key: out.pop(key, None)})
            
        self.log_dict(log, on_epoch=True, sync_dist=True)
        
        return log


    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches.
        This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """
        val_loss = weighted_mean(outs, "val_loss", "batch_size")
        log = {"val_loss": val_loss}
        for key in self.metric_keys:
            log.update({f"val_{key}": weighted_mean(outs, f"val_{key}", "batch_size")})

        self.log_dict(log, sync_dist=True)


    def predict_step(self, batch: List[Any], batch_idx: int) -> Any:
        """Predict step for linear eval or finefune"""
        
        fname, X, _ = batch
        logits = self(X)["logits"]
        
        return (fname, F.softmax(logits, dim=1))
        
