from typing import Any, Dict, List, Sequence, Tuple

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.losses.mocov2plus import mocov2plus_loss_func
from model.methods.base import BaseMomentumMethod
from model.utils.misc import gather
from model.utils.momentum import initialize_momentum_params


class MoBY(BaseMomentumMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """Implements MoCo V2+ (https://arxiv.org/abs/2011.10566).

        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of projected features.
                proj_hidden_dim (int): number of neurons of the hidden layers of the projector.
                temperature (float): temperature for the softmax in the contrastive loss.
                queue_size (int): number of samples to keep in the queue.
        """
        super().__init__(cfg)
        
        self.temparture = cfg.method_kwargs.temperature
        self.queue_size = cfg.method_kwargs.queue_size
        
        proj_output_dim = cfg.method_kwargs.proj_output_dim
        proj_hidden_dim = cfg.method_kwargs.proj_hidden_dim
        pred_hidden_dim = cfg.method_kwargs.pred_hidden_dim
        
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim)
        )
        
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )
        
        initialize_momentum_params(self.projector, self.momentum_projector)
        
        self.register_buffer("queue", torch.randn(2, proj_output_dim, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg = super(MoBY, MoBY).add_and_assert_specific_cfg(cfg)
        
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.temperature")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.queue_size")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")
        
        return cfg
    
    
    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Return a list of parameters 
        """
        extra_learnable_params = [
            {
                "name": "projector",
                "params": self.projector.parameters()
            },
            {
                "name": "predictor",
                "params": self.predictor.parameters()
            }
        ]
        
        return  super().learnable_params + extra_learnable_params
    
    
    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Add mometum pairs for base class
        """
        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        
        return super().momentum_pairs + extra_momentum_pairs
    
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        batch_size = keys.shape[1]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0
        
        keys = keys.permute(0, 2, 1)
        self.queue[:, :, ptr: ptr + batch_size] = keys
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
        
    
    def forward(self, X: torch.Tensor):
        out = super().forward(X)
        z = self.projector(out["feats"])
        p = nn.functional.normalize(self.predictor(z), dim=-1)        
        out.update({"z": z, "p": p})
        
        return out
        
        
    @torch.no_grad()
    def forward_momentum(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = super().forward_momentum(X)
        z = F.normalize(self.momentum_projector(out["feats"]), dim=-1)
        out.update({"z": z})
        
        return out
        
        
    def training_step(self, batch: List[Any], batch_idx: int) -> Dict[str, Any]:
        out = super().training_step(batch, batch_idx)
        p1, p2 = out["p"]
        k1, k2 = out["momentum_z"]
        
        with torch.no_grad():
            z_std = F.normalize(torch.stack([k1, k2]), dim=-1).std(dim=1).mean()
        
        queue = self.queue.clone().detach()
        nce_loss = (mocov2plus_loss_func(p1, k2, queue[1], self.temparture) + \
                    mocov2plus_loss_func(p2, k1, queue[0], self.temparture)
        ) / 2
        
        keys = torch.stack((gather(k1), gather(k2)))
        self._dequeue_and_enqueue(keys)
        
        metrics = {
            "train_nce_loss": nce_loss,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        class_loss = sum(out["loss"]) / self.num_large_crops
        momentum_class_loss = sum(out["momentum_loss"]) / self.num_large_crops
        
        return nce_loss + class_loss + momentum_class_loss
        
        