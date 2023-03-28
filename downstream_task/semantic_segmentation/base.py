import logging
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.lars import LARS
from model.utils.misc import omegaconf_select
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from model.backbones import (
    convnext_base,
    convnext_large,
    convnext_small,
    convnext_tiny,
    poolformer_m36,
    poolformer_m48,
    poolformer_s12,
    poolformer_s24,
    poolformer_s36,
    resnet18,
    resnet50,
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
    wide_resnet28w2,
    wide_resnet28w8,
)
from downstream_task.losses import DiceLoss, FocalLoss
from torch.nn import CrossEntropyLoss


def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class SkipHook:
    """Hook register class for backbone
    results are saving in featuers
    get access by obj.features
    """
    features = []
    def __init__(self, model: nn.Module, layer_names: List[str]):
        """
        Args:
            model (nn.Module): backbone module
            layer_names (List[str]): inner module name of backbone
        """
        self.hooks = []
        for name, module in model.named_children():
            if name in layer_names:
                hook = module.register_forward_hook(self._hook_fn)
                self.hooks.append(hook)
        
    def _hook_fn(self, module, input, output):
        self.features.append(output)
        
    def remove_hook(self):
        for hook in self.hooks:
            hook.remove()
    


class BaseSegmentationModel(pl.LightningModule):
    
    _BACKBONES = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
        "poolformer_s12": poolformer_s12,
        "poolformer_s24": poolformer_s24,
        "poolformer_s36": poolformer_s36,
        "poolformer_m36": poolformer_m36,
        "poolformer_m48": poolformer_m48,
        "convnext_tiny": convnext_tiny,
        "convnext_small": convnext_small,
        "convnext_base": convnext_base,
        "convnext_large": convnext_large,
        "wide_resnet28w2": wide_resnet28w2,
        "wide_resnet28w8": wide_resnet28w8,
    }
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
        "exponential",
        "none",
    ]
    _LOSSES = {
    "dice": DiceLoss,
    "focal": FocalLoss,
    "ce": CrossEntropyLoss
}
    
    def __init__(self, cfg: omegaconf.DictConfig):
        
        super().__init__()
        # base config
        self.cfg = self.add_and_assert_specific_cfg(cfg) 
        # ssl pretrain method
        self.method_name = cfg.pretrain_method
        # training related
        self.max_epochs: int = cfg.max_epochs
        
        # encoder related
        self.base_model: Callable = self._BACKBONES[cfg.backbone.name]
        self.backbone_kwargs: Dict[str, Any] = cfg.backbone.kwargs
        self.skip_layers: List[str] = cfg.backbone.skip_layers
        self.backbone: nn.Module = self.base_model(cfg.pretrain_method, **self.backbone_kwargs)
        # get num_feature from backbone 
        if self.backbone_name.startswith("resnet"):
            self.features_dim: int = self.backbone.inplanes
            # remove fc layer
            self.backbone.fc = nn.Identity()
            cifar = cfg.data.dataset in ["cifar10", "cifar100"]
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.backbone.maxpool = nn.Identity()
        else:
            self.features_dim: int = self.backbone.num_features
        #load ckpt
        state_dict = torch.load(cfg.backbone.ckpt, map_location="cpu")["state_dict"]
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
        msg = self.bakcbone.load_state_dict(state_dict, strict=False)
        print(f"Loaded {cfg.backbone.ckpt} with msg: {msg}")
        #register hook for backbone's sublayers
        self.skip_hook = SkipHook(self.backbone, self.skip_layers) # get features from .features
        
        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr
        self.classifier_lr = cfg.optimizer.classifier_lr
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.extra_optimzier_kwargs = cfg.optimizer.kwrags
        
        # scheduler related
        self.scheduler: str = cfg.scheduler.name
        self.lr_decay_steps: Union[List[int], None] = cfg.scheduler.lr_decay_steps
        self.min_lr: float = cfg.scheduler.min_lr
        self.warmup_start_lr: float = cfg.scheduler.warmup_start_lr
        self.warmup_epochs: int = cfg.scheduler.warmup_epochs
        
        # auxiliary classification head
        self.classifier = None
        if cfg.classifier.enabled:
            self.classifier = nn.Linear(self.features_dim, cfg.data.num_classes)
            self.loss_fn_classifier = self._LOSSES[cfg.classifier.loss](**cfg.classifier.loss_kwargs)

        # segmentation loss function
        self.loss_fn = self._LOSSES[cfg.loss.name](**cfg.loss.kwargs)


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds and assert params passed by config file"""
        from omegaconf import OmegaConf
        
        # backbone related
        assert not OmegaConf.is_missing(cfg, "backbone")
        assert not OmegaConf.is_missing(cfg, "backbone.name")
        assert not OmegaConf.is_missing(cfg, "backbone.ckpt")
        assert cfg.backbone.name in BaseSegmentationModel._BACKBONES
        cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})
        # skip_layers: layer name of backbone inner modules,
        # which is used in Segmentation head
        cfg.backbone.skip_layers = omegaconf_select(cfg, "backbone.skip_layers", [])
        
        # classifier
        cfg.classifier = omegaconf_select(cfg, "classifier", {})
        cfg.classifier.enabled = omegaconf_select(cfg, "classifier.enabled", False)
        cfg.classifier.loss = omegaconf_select(cfg, "classifier.loss", "ce")
        cfg.classifier.loss_kwargs = omegaconf_select(cfg, "classifier.loss_kwargs", {}) 
        
        # optimizer related
        cfg.optimizer = omegaconf_select(cfg, "optimizer", {})
        assert not OmegaConf.is_missing(cfg, "optimizer.name")
        assert not OmegaConf.is_missing(cfg, "optimizer.lr")
        assert not OmegaConf.is_missing(cfg, "optimizer.batch_size")
        assert not OmegaConf.is_missing(cfg, "optimizer.weight_decay")
        scale_factor = cfg.optimizer.batch_size * len(cfg.devices) * cfg.num_nodes / 256
        cfg.optimizer.lr = cfg.optimizer.lr * scale_factor
        # extra optimizer kwargs
        cfg.optimizer.kwargs = omegaconf_select(cfg, "optimizer.kwargs", {})
        if cfg.optimizer.name == "sgd":
            cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
        elif cfg.optimizer.name == "lars":
            cfg.optimizer.kwargs.momentum = omegaconf_select(cfg, "optimizer.kwargs.momentum", 0.9)
            cfg.optimizer.kwargs.eta = omegaconf_select(cfg, "optimizer.kwargs.eta", 1e-3)
            cfg.optimizer.kwargs.clip_lr = omegaconf_select(cfg, "optimizer.kwargs.clip_lr", False)
            cfg.optimizer.kwargs.exclude_bias_n_norm = omegaconf_select(
                cfg,
                "optimizer.kwargs.exclude_bias_n_norm",
                False,
            )
        elif cfg.optimizer.name == "adamw":
            cfg.optimizer.kwargs.betas = omegaconf_select(cfg, "optimizer.kwargs.betas", [0.9, 0.999])

        # scheduler related
        cfg.scheduler = omegaconf_select(cfg, "scheduler", {})
        cfg.scheduler.name = omegaconf_select(cfg, "scheduler.name", "")
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        
        # decoder loss
        cfg.loss = omegaconf_select(cfg, "loss", {})
        assert not OmegaConf.is_missing(cfg, "loss.name")
        assert cfg.loss.name in BaseSegmentationModel._LOSSES
        cfg.loss.kwargs = omegaconf_select(cfg, "loss.kwargs", {})
        
        return cfg


    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines parameter group"""
        learnable_params = [{
            "name": "backbone",
            "params": self.backbone.parameters(),
            "lr": self.lr
        }]
        if self.classifier is not None:
            learnable_params += [{
                "name" : "auxiliary_classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0
            }]
        
        return learnable_params 
        

    def configure_optimizers(self) -> Any:
        """Collects learnable params and configure optimizer"""
        
        learnable_params = self.learnable_params
        
        # indexes of parameter group with static lr
        idxs_no_scheduler = [i for i, m in enumerate(learnable_params) if m.pop("static_lr", False)]
        
        assert self.optimizer in BaseSegmentationModel._OPTIMIZERS
        optimizer = self._OPTIMIZERS(self.optimizer)
        
        # create optimzier
        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay
            **self.extra_optimzier_kwargs
        )
        
        if self.scheduler.lower() == "":
            return optimizer
        else:
            # with scheduler
            if self.scheduler == "warmup_cosine":
                scheduler = LinearWarmupCosineAnnealingLR(
                        optimizer,
                        warmup_epochs=self.warmup_epochs,
                        max_epochs=self.max_epochs,
                        warmup_start_lr=self.warmup_start_lr,
                        eta_min=self.min_lr,
                    )
            elif self.scheduler == "cosine":
                scheduler = CosineAnnealingLR(optimizer, self.max_epochs, eta_min=self.min_lr)
            elif self.scheduler == "step":
                scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
            else:
                raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

            if idxs_no_scheduler:
                partial_fn = partial(
                    static_lr,
                    get_lr=scheduler.get_lr,
                    param_group_indexes=idxs_no_scheduler,
                    lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
                )
                scheduler.get_lr = partial_fn

            return [optimizer], [scheduler]            


    def forward(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Base forward function for all segmentation models,
            Inner skip features are get by forward hooks
        """
        feats = self.backbone(X)
        inner_feats = self.skip_hook.features
        out = {"inner_feats": inner_feats}
        
        if self.classifier is not None:
            logits = self.classifier(feats)
            out.update({"logits": logits})
        
        return out
    
    
    def _shared_step(self, batch: List[Any], batch_idx: int):
        """Shared step for batch training and validation step"""
        if self.classifier is not None:
            _, X, masks, targets = batch
        else:
            _, X, masks = batch
        batch_size = X.size(0)
        
        # loss of segmentation branch
        out = self(X)
        loss = self.loss_fn(out["mask"], masks)
        metrics = self.metric_fn(out["mask"], masks, **self.metric_args)
        out.update({"loss": loss, "batch_size": batch_size, **metrics})
    
        # loss of auxliliary classification branch
        if self.classifier is not None:
            auxiliary_loss = self.loss_fn_classifier(out["logits"], targets)
            out.update({"auxiliary_loss": auxiliary_loss})
            
        return out


    def training_step(self, batch: List[Any], batch_idx: int):
        """Base training step for segmentation models"""
        
        out = self._shared_step(batch, batch_idx)
        
        assert "loss" in out
        metrics = {"loss": out["loss"]}
        metrics.update({key: out[key] for key in self.metric_keys})    # metric_keys   
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        loss = out["loss"]
        auxiliary_loss = out["auxiliary_loss"] if self.classifier is not None else 0.
        
        return loss + auxiliary_loss
    
    
    def validation_step(self, batch, batch_idx: int) -> Dict[str, Any]:
        """Base validation step for Segmentation models"""
        
        out = self._shared_step(batch, batch_idx)
        
        metrics = {
            "val_batch_size": out["batch_size"],
            "val_loss": out["loss"]
        } 
        if self.classifier is not None:
            metrics.update({"val_auxiliary_loss": out["auxiliary_loss"]})
        metrics.update({("val"+ k) : out[k] for k in self.metric_keys})
        
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        return metrics
    
    
        
        
            
        
            
        
        
        