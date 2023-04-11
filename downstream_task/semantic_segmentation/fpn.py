from typing import *

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.misc import omegaconf_select
from .base import BaseSegmentationModel, SegmentationHead
from .module import *


class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), stride=1, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x
    

class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip_conv(skip)
        x = x + skip
        return x
    

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)
    
    
class MergeBlock(nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in ["add", "cat"]:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(policy))
        self.policy = policy

    def forward(self, x):
        if self.policy == "add":
            return sum(x)
        elif self.policy == "cat":
            return torch.cat(x, dim=1)
        else:
            raise ValueError("`merge_policy` must be one of: ['add', 'cat'], got {}".format(self.policy))
        
        
class FPN(BaseSegmentationModel):
    """Implementation of FPN segmentation arch

    """
    def __init__(self, cfg: omegaconf.DictConfig):
        
        super().__init__(cfg)
        
        encoder_channels: List[int] = cfg.decoder.encoder_channels
        encoder_depth: int = cfg.decoder.encoder_depth
        pyramid_channels: int =  cfg.decoder.pyramid_channels
        segmentation_channels: int = cfg.decoder.segmentation_channels
        dropout: float = cfg.decoder.dropout
        merge_policy: str = cfg.decoder.merge_policy
        
        out_channels = segmentation_channels if merge_policy == "add" else segmentation_channels * 4
        if encoder_depth < 3:
            raise ValueError("Encoder depth for FPN decoder cannot be less than 3, got {}.".format(encoder_depth))

        encoder_channels = encoder_channels[::-1]
        encoder_channels = encoder_channels[: encoder_depth + 1]

        self.p5 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=1)
        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.seg_blocks = nn.ModuleList(
            [
                SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=n_upsamples)
                for n_upsamples in [3, 2, 1, 0]
            ]
        )

        self.merge = MergeBlock(merge_policy)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)
        
        self.seg_head = SegmentationHead(
            in_channels=out_channels,
            out_channels=cfg.data.num_classes,
            kernel_size=1,
            upsampling=4,
        )
        

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Add and assert params"""
        from omegaconf import OmegaConf
        
        cfg = super(FPN, FPN).add_and_assert_specific_cfg(cfg)
        
        # assert deocoder not missing
        assert not OmegaConf.is_missing(cfg, "decoder")
        # decoder kwargs
        cfg.decoder.name = omegaconf_select(cfg, "decoder.name", "fpn")
        cfg.decoder.encoder_channels = omegaconf_select(cfg, "decoder.encoder_channels", [3, 64, 64, 128, 256, 512])
        cfg.decoder.encoder_depth = omegaconf_select(cfg, "decoder.encoder_depth", 5)
        cfg.decoder.pyramid_channels = omegaconf_select(cfg, "decoder.pyramid_channels", 256)
        cfg.decoder.segmentation_channels = omegaconf_select(cfg, "decoder.segmentation_channels", 128)
        cfg.decoder.dropout = omegaconf_select(cfg, "decoder.dropout", 0.2)
        cfg.decoder.merge_policy = omegaconf_select(cfg, "decoder.merge_policy", "add")
        assert cfg.decoder.merge_policy in ["add", "cat"]
        

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        decoder_learnable_params = [
                {"name": "p5", "params": self.p5.parameters()},
                {"name": "p4", "params": self.p4.parameters()},
                {"name": "p3", "params": self.p3.parameters()},
                {"name": "p2", "params": self.p2.parameters()},
                {"name": "seg_blocks", "params": self.seg_blocks.parameters()},
                {"name": "merge", "params": self.merge.parameters()},
                {"name": "seg_head", "params": self.seg_head.parameters()}
            ]
        
        return super().learnable_params + decoder_learnable_params
    
    
    def forward(self, X: torch.Tensor):
        """forward function for FPN"""
        out = super().forward(X)
        inner_feats = out["inner_feats"]
        
        c2, c3, c4, c5 = inner_feats[-4:]
        p5 = self.p5(c5)
        p4 = self.p4(p5, c4)
        p3 = self.p3(p4, c3)
        p2 = self.p2(p3, c2)

        feature_pyramid = [seg_block(p) for seg_block, p in zip(self.seg_blocks, [p5, p4, p3, p2])]
        x = self.merge(feature_pyramid)
        x = self.dropout(x)
        
        mask = self.seg_head(x)
        
        out.update({"mask": mask})
        
        return out
        
        
        