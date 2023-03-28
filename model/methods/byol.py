# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.
from typing import *

import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.methods.base import BaseMomentumMethod
from model.losses import byol_loss_func
from model.utils.momentum import initialize_momentum_params



class BYOL(BaseMomentumMethod):
    """BYOL implementation
    """
    def __init__(self,
                 cfg: omegaconf.DictConfig
                 ):
        super().__init__(cfg)

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        pred_hidden_dim: int = cfg.method_kwargs.pred_hidden_dim

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

        # momentum projector
        self.momentum_projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )
        initialize_momentum_params(self.projector, self.momentum_projector)

        # predictor
        self.predictor = nn.Sequential(
            nn.Linear(proj_output_dim, pred_hidden_dim),
            nn.BatchNorm1d(pred_hidden_dim),
            nn.ReLU(),
            nn.Linear(pred_hidden_dim, proj_output_dim),
        )
     
     
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        cfg = super(BYOL, BYOL).add_and_assert_specific_cfg(cfg)
        
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.pred_hidden_dim")
        
        return cfg
        
        
    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Add projector and predictor params to parent learnable parameters
        """
        extra_learnable_params = [
            {"name": "projector", "params": self.projector.parameters()},
            {"name": "predictor", "params": self.predictor.parameters()}
        ]
        
        return super().learnable_params + extra_learnable_params
    

    @property
    def momentum_pairs(self) -> List[Tuple[Any, Any]]:
        """Add (projector, momentum_projector) pair to parent momentum_pairs
        """
        extra_momentum_pairs = [(self.projector, self.momentum_projector)]
        
        return super().momentum_pairs + extra_momentum_pairs
      
      
    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs forward pass of the online encoder (encoder, projector and predictor).

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the logits of the head.
        """
        out = super().forward(X)
        z = self.projector(out["feats"])
        p = self.predictor(z)
        out.update({"z": z, "p": p})

        return out
    
    @torch.no_grad()
    def forward_momentum(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        momentum_outs = super().forward_momentum(X)
        feats = momentum_outs["feats"]
        momentum_z = self.momentum_projector(feats)
        momentum_outs.update({"z": momentum_z})
        
        return momentum_outs
    

    def training_step(self, batch: List[Any], batch_idx: int):
        
        out = super().training_step(batch, batch_idx)
        
        Z = out["z"]
        P = out["p"]
        Z_momentum = out["momentum_z"]
        
        neg_cos_sim = 0.
        for v1 in range(self.num_large_crops):
            for v2 in np.delete(range(self.num_crops), v1):
                neg_cos_sim += byol_loss_func(P[v2], Z_momentum[v1])
        
        with torch.no_grad():
            z_std = F.normalize(torch.stack(Z[: self.num_large_crops]), dim=-1).std(dim=1).mean()
        
        metrics = {
            "train_neg_cos_sim": neg_cos_sim,
            "train_z_std": z_std,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)
        
        # loss of linear classifier 
        class_loss = sum(out["loss"]) / self.num_large_crops
        momentum_class_loss = sum(out["momentum_loss"]) / self.num_large_crops
        
        return neg_cos_sim + class_loss + momentum_class_loss
