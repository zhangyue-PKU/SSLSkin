# Copyright 2023 model-learn development team.

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
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.knn import WeightedKNNClassifier
from model.utils.lars import LARS
from model.utils.metrics import accuracy_at_k, weighted_mean
from model.utils.misc import omegaconf_select, remove_bias_and_norm_from_weight_decay
from model.utils.momentum import MomentumUpdater, initialize_momentum_params
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
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


def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs



def parser_list_of_dict(d: List[Dict]) -> Dict[str, List]:
    """ Parser list of same key dict to Dict[key, List]
    """
    out = {}
    if len(d) == 0:
        return out
    keys = d[0].keys()
    for k in keys:
        out.update({k: [dd[k] for dd in d]})

    return out



class BaseMethod(pl.LightningModule):
    
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
        "cosine"
        "none",
    ]

    def __init__(self, cfg: omegaconf.DictConfig):
        """Base model that implements all basic operations for all self-supervised methods.
        It adds shared arguments, extract basic learnable parameters, creates optimizers
        and schedulers, implements basic training_step for any number of crops,
        trains the online classifier and implements validation_step.

        .. note:: Cfg defaults are set in init by calling `cfg = add_and_assert_specific_cfg(cfg)`

        Cfg basic structure:
            backbone:
                name (str): architecture of the base backbone.
                kwargs (dict): extra backbone kwargs.
            data:
                dataset (str): name of the dataset.
                num_classes (int): number of classes.
                num_workers: workers of dataloader
            max_epochs (int): number of training epochs.

            backbone_params (dict): dict containing extra backbone args, namely:
                #! only for resnet
                zero_init_residual (bool): change the initialization of the resnet backbone.
                #! only for vit
                patch_size (int): size of the patches for ViT.
            optimizer:
                name (str): name of the optimizer.
                batch_size (int): number of samples in the batch.
                lr (float): learning rate.
                weight_decay (float): weight decay for optimizer.
                classifier_lr (float): learning rate for the online linear classifier.
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
            knn_eval:
                enabled (bool): enables online knn evaluation while training.
                k (int): the number of neighbors to use for knn.
            accumulate_grad_batches (Union[int, None]): number of batches for gradient accumulation.
            num_large_crops (int): number of big crops.
            num_small_crops (int): number of small crops.
            max_epochs: epochs for pretrain
            method: name of pretrain method
            linear_eval: True or False, whehter use linear eval in pretraining

        .. note::
            When using distributed data parallel, the batch size and the number of workers are
            specified on a per process basis. Therefore, the total batch size (number of workers)
            is calculated as the product of the number of GPUs with the batch size (number of
            workers).

        .. note::
            The learning rate (base, min and warmup) is automatically scaled linearly
            if using gradient accumulation.

        .. note::
            For CIFAR10/100, the first convolutional and maxpooling layers of the ResNet backbone
            are slightly adjusted to handle lower resolution images (32x32 instead of 224x224).

        """

        super().__init__()

        # add default values and assert that config has the basic needed settings
        cfg: omegaconf.DictConfig = self.add_and_assert_specific_cfg(cfg)

        self.cfg = cfg

        # training related
        self.max_epochs: int = cfg.max_epochs
        self.accumulate_grad_batches: int = cfg.accumulate_grad_batches

        # optimizer related
        self.optimizer: str = cfg.optimizer.name
        self.batch_size: int = cfg.optimizer.batch_size
        self.lr: float = cfg.optimizer.lr * self.accumulate_grad_batches
        self.weight_decay: float = cfg.optimizer.weight_decay
        self.exclude_bias_n_norm_wd: bool = cfg.optimizer.exclude_bias_n_norm_wd
        self.extra_optimizer_kargs: Dict[str, Any] = cfg.optimizer.kwargs

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

        # data-related
        self.num_small_crops: int = cfg.data.num_small_crops
        self.num_large_crops: int = cfg.data.num_large_crops
        self.num_crops = self.num_small_crops + self.num_large_crops
        # turn on multicrop if there are small crops
        self.multicrop: bool = self.num_small_crops != 0

        # knn online evaluation
        self.knn_eval: bool = cfg.knn_eval.enabled
        self.knn_k: int = cfg.knn_eval.k
        if self.knn_eval:
            self.knn = WeightedKNNClassifier(k=self.knn_k, distance_fx=cfg.knn.distance_func)
            
        # online classification eval
        self.linear_eval = cfg.linear_eval
        if cfg.linear_eval:
            self.num_classes: int = cfg.data.num_classes
            self.classifier: nn.Module = nn.Linear(self.features_dim, self.num_classes)
            self.classifier_lr: float = cfg.optimizer.classifier_lr * self.accumulate_grad_batches
            # classifier metrics
            self.loss_fn = F.cross_entropy
            self.loss_args = {"ignore_index": -1}
            self.metric_fn = accuracy_at_k
            self.metric_args = {"top_k": (1, min(5, self.num_classes))}
            self.metric_keys = [f"acc{k}" for k in set(self.metric_args["top_k"])]
            
        # get backbone module from _BACKBONES
        self.base_model: Callable = self._BACKBONES[cfg.backbone.name]
        self.backbone_name: str = cfg.backbone.name
        self.backbone_kwargs: Dict[str, Any] = cfg.backbone.kwargs
        
        # initialize backbone
        kwargs = self.backbone_kwargs.copy()
        self.backbone: nn.Module = self.base_model(cfg.method, **kwargs)
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


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """
        from omegaconf import OmegaConf
        
        # backbone related
        assert not OmegaConf.is_missing(cfg, "backbone")
        assert not OmegaConf.is_missing(cfg, "backbone.name")
        assert cfg.backbone.name in BaseMethod._BACKBONES
        cfg.backbone.kwargs = omegaconf_select(cfg, "backbone.kwargs", {})
        
        # method related
        assert not OmegaConf.is_missing(cfg, "method")
        cfg.method_kwargs = omegaconf_select(cfg, "method_kwargs", {})
        
        # default parameters for knn eval
        cfg.knn_eval = omegaconf_select(cfg, "knn_eval", {})
        cfg.knn_eval.enabled = omegaconf_select(cfg, "knn_eval.enabled", False)
        cfg.knn_eval.k = omegaconf_select(cfg, "knn_eval.k", 20)
        cfg.knn_eval.distance_func = omegaconf_select(cfg, "knn_eval.distance_func", "euclidean")

        # online evaluation
        cfg.linear_eval = omegaconf_select(cfg, "linear_eval", False)
        
        # optimizer 
        cfg.optimizer = omegaconf_select(cfg, "optimizer", {})
        assert not OmegaConf.is_missing(cfg, "optimizer.name")
        assert not OmegaConf.is_missing(cfg, "optimizer.lr")
        assert not OmegaConf.is_missing(cfg, "optimizer.batch_size")
        assert not OmegaConf.is_missing(cfg, "optimizer.weight_decay")
        scale_factor = cfg.optimizer.batch_size * len(cfg.devices) * cfg.num_nodes / 256
        cfg.optimizer.lr = cfg.optimizer.lr * scale_factor
        if cfg.linear_eval:
            assert not OmegaConf.is_missing(cfg, "optimizer.classifier_lr")
            cfg.optimizer.classifier_lr = cfg.optimizer.classifier_lr * scale_factor
        cfg.optimizer.exclude_bias_n_norm_wd = omegaconf_select(cfg, "optimizer.exclude_bias_n_norm_wd", False)
        # accumulate_grad_batches
        cfg.accumulate_grad_batches = omegaconf_select(cfg, "accumulate_grad_batches", 1)
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

        # default parameters for the scheduler
        cfg.scheduler = omegaconf_select(cfg, "scheduler", {})
        cfg.scheduler.name = omegaconf_select(cfg, "scheduler.name", None)
        cfg.scheduler.lr_decay_steps = omegaconf_select(cfg, "scheduler.lr_decay_steps", None)
        cfg.scheduler.min_lr = omegaconf_select(cfg, "scheduler.min_lr", 0.0)
        cfg.scheduler.warmup_start_lr = omegaconf_select(cfg, "scheduler.warmup_start_lr", 3e-5)
        cfg.scheduler.warmup_epochs = omegaconf_select(cfg, "scheduler.warmup_epochs", 10)
        cfg.scheduler.interval = omegaconf_select(cfg, "scheduler.interval", "epoch")

        return cfg


    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """
        learnable_parameters = [{"name": "backbone", "params": self.backbone.parameters()}]
        if self.linear_eval:
            learnable_parameters += [{
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            }]
            
        return learnable_parameters 
    

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        learnable_params = self.learnable_params

        # exclude bias and norm from weight decay
        if self.exclude_bias_n_norm_wd:
            learnable_params = remove_bias_and_norm_from_weight_decay(learnable_params)

        # indexes of parameters without lr scheduler
        idxs_no_scheduler = [i for i, m in enumerate(learnable_params) if m.pop("static_lr", False)]

        assert self.optimizer in self._OPTIMIZERS
        optimizer = self._OPTIMIZERS[self.optimizer]

        # create optimizer
        optimizer = optimizer(
            learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_kargs,
        )

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
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler["scheduler"].get_lr
                if isinstance(scheduler, dict)
                else scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            if isinstance(scheduler, dict):
                scheduler["scheduler"].get_lr = partial_fn
            else:
                scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]

   
    def forward(self, X: torch.Tensor):
        """Base forward"""
        feats = self.backbone(X)
        out = {"feats": feats}
   
        return out

    
    def _shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        metrics 

        Args:
            X (torch.Tensor): batch of images in tensor format
            targets (torch.Tensor): batch of labels for X

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5
        """

        out = self(X)
        feats = out["feats"]
                
        if self.linear_eval: 
            logits = self.classifier(feats.detach())
            loss = self.loss_fn(logits, targets, **self.loss_args)
            metrics = self.metric_fn(logits, targets, **self.metric_args)
            out.update({"loss": loss, "logits": logits, **metrics})
        else:
            out.update({"loss": 0.})
            
        return out


    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """

        _, X, targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops
        outs = [self._shared_step(x, targets) for x in X[: self.num_large_crops]]
        # parser outs of crops to dict of list
        outs = parser_list_of_dict(outs)
        
        # multicrops forward
        if self.multicrop:
            multicrop_outs = [self(x) for x in X[self.num_large_crops:]]
            for k in multicrop_outs[0].keys():
                outs[k] = outs.get(k, []) + [out[k] for out in multicrop_outs]

        if self.linear_eval:
            # log metrics and remove them from outs
            loss = sum(outs["loss"]) / self.num_large_crops
            metrics = {"train" + "_class_loss": loss}
            for key in self.metric_keys:
                metrics.update({("train" + key): sum(outs.pop(key, [])) / self.num_large_crops})
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

        if self.knn_eval:
            self.knn(
                train_features=torch.cat(outs["feats"][: self.num_large_crops]).detach(),
                train_targets=targets.repeat(self.num_large_crops).detach(),
            )

        return outs
    

    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, Any]:
        """Validation step for pytorch lightning. It does all the shared operations, such as
        forwarding a batch of images, computing logits and computing metrics.

        Args:
            batch (List[torch.Tensor]):a batch of data in the format of [img_indexes, X, Y].
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the batch_size (used for averaging), the classification loss
                and accuracies.
        """

        _, X, targets = batch
        batch_size = targets.size(0)
        
        out = self._shared_step(X, targets)

        if self.knn_eval and not self.trainer.sanity_checking:
            self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())

        metrics = None
        if self.linear_eval:
            metrics = {
                "batch_size": batch_size,
                "val_loss": out["loss"],
            }
            for key in self.metric_keys:
                metrics.update({f"val_{key}": out[key]})
        
        return metrics


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

        #BUG: disable_knn_eval not set in base method
        if not self.disable_knn_eval and not self.trainer.running_sanity_check:
            val_knn_acc1, val_knn_acc5 = self.knn.compute()
            log.update({"val_knn_acc1": val_knn_acc1, "val_knn_acc5": val_knn_acc5})

        self.log_dict(log, sync_dist=True)



class BaseMomentumMethod(BaseMethod):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
    ):
        """Base momentum model that implements all basic operations for all self-supervised methods
        that use a momentum backbone. It adds shared momentum arguments, adds basic learnable
        parameters, implements basic training and validation steps for the momentum backbone and
        classifier. Also implements momentum update using exponential moving average and cosine
        annealing of the weighting decrease coefficient.

        Extra cfg settings:
            momentum:
                base_tau (float): base value of the weighting decrease coefficient in [0,1].
                final_tau (float): final value of the weighting decrease coefficient in [0,1].
                momentum_linear_eval (bool): whether or not to train a classifier on top of the momentum backbone.
        """

        super().__init__(cfg)

        # initialize momentum backbone
        kwargs = self.backbone_kwargs.copy()
        
        # cfg is 
        cfg = self.cfg

        method: str = cfg.method
        base_tau_momentum: float = cfg.momentum.base_tau
        final_tau_momentum: float = cfg.momentum.final_tau
        momentum_linear_eval: bool = cfg.momentum.linear_eval
        
        self.momentum_backbone: nn.Module = self.base_model(method, **kwargs)
        if self.backbone_name.startswith("resnet"):
            # remove fc layer
            self.momentum_backbone.fc = nn.Identity()
            cifar = cfg.data.dataset in ["cifar10", "cifar100"]
            if cifar:
                self.momentum_backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
                self.momentum_backbone.maxpool = nn.Identity()

        # copy backbone param to momentum_backbone and disable require_grad
        initialize_momentum_params(self.backbone, self.momentum_backbone)

        # momentum classifier
        self.momentum_linear_eval = momentum_linear_eval
        if momentum_linear_eval:
            self.momentum_classifier: Any = nn.Linear(self.features_dim, self.num_classes)

        # momentum updater
        self.momentum_updater = MomentumUpdater(base_tau_momentum, final_tau_momentum)


    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(BaseMomentumMethod, BaseMomentumMethod).add_and_assert_specific_cfg(cfg)

        cfg.momentum = omegaconf_select(cfg, "momentum", {})
        cfg.momentum.base_tau = omegaconf_select(cfg, "momentum.base_tau", 0.99)
        cfg.momentum.final_tau = omegaconf_select(cfg, "momentum.final_tau", 1.0)
        cfg.momentum.linear_eval = omegaconf_select(cfg, "momentum.linear_eval", False)

        return cfg
    
    
    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Adds momentum classifier parameters to the parameters of the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        extra_learnable_parameters = []
        if self.momentum_linear_eval:
            extra_learnable_parameters.append(
                {
                    "name": "momentum_classifier",
                    "params": self.momentum_classifier.parameters(),
                    "lr": self.classifier_lr,
                    "weight_decay": 0,
                }
            )
        return super().learnable_params + extra_learnable_parameters


    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Defines base momentum pairs that will be updated using exponential moving average.

        Returns:
            List[Tuple[Any, Any]]: list of momentum pairs (two element tuples).
        """
        return [(self.backbone, self.momentum_backbone)]


    def on_train_start(self):
        """Resets the step counter at the beginning of training."""
        self.last_step = 0


    @torch.no_grad()
    def forward_momentum(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward a batch of images X in the momentum network. This function can be 
        rewrite in children class. Sub class must return a dict with key 'momentum_feats' """
        feats = self.momentum_backbone(X)
        
        return {"feats": feats}


    def _shared_step_momentum(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X in the momentum backbone and optionally computes the
        classification loss, the logits, the features, acc@1 and acc@5 for of momentum classifier.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict[str, Any]:
                a dict containing the classification loss, logits, features, acc@1 and
                acc@5 of the momentum backbone / classifier.
        """
        out = self.forward_momentum(X)
        feats = out["feats"]

        if self.momentum_linear_eval:
            logits = self.momentum_classifier(feats.detach())
            loss = self.loss_fn(logits, targets, **self.loss_args)
            results = self.metric_fn(logits, targets, **self.metric_args)
            out.update({"logits": logits, "loss": loss, **results})
        else:
            out.update({"loss": 0.})

        return out


    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        """Training step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding the crops in the momentum backbone
        and classifier, and computing statistics.
        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: a dict with the features of the momentum backbone and the classification
                loss and logits of the momentum classifier.
        """

        online_outs = super().training_step(batch, batch_idx)

        _, X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        # remove small crops
        X = X[: self.num_large_crops]

        momentum_outs = [self._shared_step_momentum(x, targets) for x in X]
        momentum_outs = parser_list_of_dict(momentum_outs)

        if self.momentum_linear_eval:
            loss = sum(momentum_outs["loss"]) / self.num_large_crops
            metrics = {"momentum_train" + "_class_loss": loss}
            for key in self.metric_keys:
                metrics.update({("momentum_train" + key): sum(momentum_outs.pop(key, [])) / self.num_large_crops})
            self.log_dict(metrics, on_epoch=True, sync_dist=True)

        # adds the momentum classifier loss together with the general loss
        online_outs.update(
            {("momentum_" + k) : momentum_outs[k] for k in momentum_outs.keys()}
        )

        return online_outs


    def on_train_batch_end(self, outputs: Dict[str, Any], batch: Sequence[Any], batch_idx: int):
        """Performs the momentum update of momentum pairs using exponential moving average at the
        end of the current training step if an optimizer step was performed.

        Args:
            outputs (Dict[str, Any]): the outputs of the training step.
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.
        """

        if self.trainer.global_step > self.last_step:
            # update momentum backbone and projector
            momentum_pairs = self.momentum_pairs
            for mp in momentum_pairs:
                self.momentum_updater.update(*mp)
            # log tau momentum
            self.log("tau", self.momentum_updater.cur_tau)
            # update tau
            self.momentum_updater.update_tau(
                cur_step=self.trainer.global_step * self.trainer.accumulate_grad_batches,
                max_steps=len(self.trainer.train_dataloader) * self.trainer.max_epochs,
            )
        self.last_step = self.trainer.global_step


    def validation_step(
        self, batch: List[torch.Tensor], batch_idx: int, dataloader_idx: int = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validation step for pytorch lightning. It performs all the shared operations for the
        momentum backbone and classifier, such as forwarding a batch of images in the momentum
        backbone and classifier and computing statistics.
        Args:
            batch (List[torch.Tensor]): a batch of data in the format of [X, Y].
            batch_idx (int): index of the batch.
        Returns:
            Tuple(Dict[str, Any], Dict[str, Any]): tuple of dicts containing the batch_size (used
                for averaging), the classification loss and accuracies for both the online and the
                momentum classifiers.
        """

        parent_metrics = super().validation_step(batch, batch_idx)

        X, targets = batch
        batch_size = targets.size(0)

        out = self._shared_step_momentum(X, targets)

        metrics = None
        if self.momentum_linear_eval:
            metrics = {
                "batch_size": batch_size,
                "momentum_val_loss": out["loss"]
            }
            for key in self.metric_keys:
                metrics.update({f"momentum_val_{key}": out[key]})

        return parent_metrics, metrics


    def validation_epoch_end(self, outs: Tuple[List[Dict[str, Any]]]):
        """Averages the losses and accuracies of the momentum backbone / classifier for all the
        validation batches. This is needed because the last batch can be smaller than the others,
        slightly skewing the metrics.
        Args:
            outs (Tuple[List[Dict[str, Any]]]):): list of outputs of the validation step for self
                and the parent.
        """

        parent_outs = [out[0] for out in outs]
        super().validation_epoch_end(parent_outs)

        if self.momentum_linear_eval:
            momentum_outs = [out[1] for out in outs]
            val_loss = weighted_mean(momentum_outs, "momentum_val_loss", "batch_size")
            log = {"val_loss": val_loss}
            for key in self.metric_keys:
                stats = weighted_mean(momentum_outs, f"momentum_val_{key}", "batch_size")
                log.update({f"momentum_val_{key}": stats})
            self.log_dict(log, sync_dist=True)
