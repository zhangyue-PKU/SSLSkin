from typing import *

import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.misc import omegaconf_select
from .base import BaseSegmentationModel, SegmentationHead
from .module import *



class DecoderBlock(nn.Module):
    """Decoder block for UNet arch"""
    def __init__(self, 
                 in_channels, 
                 skip_channels, 
                 out_channels,
                 use_bn=True,
                 attention_type=None
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_bn
        )
        self.attn1 = Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_bn
        )
        self.attn2 = Attention(attention_type, in_channels=out_channels)
        
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attn1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attn2(x)
        
        return x
    


class CenterBlock(nn.Sequential):
    """CenterBlock for UNet"""
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        
        conv1 = Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
    )
        conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
    )
        super().__init__(conv1, conv2)
        


class UNet(BaseSegmentationModel):
    """Implemenation of UNet segmentation arch
       paper link: https://arxiv.org/pdf/1505.04597.pdf """
    def __init__(self, cfg: omegaconf.DictConfig):
        
        super().__init__(cfg)
        
        # additional args for UNet archtecture
        encoder_channels: List[int] = cfg.decoder.encoder_channels # [3, 64, 64, 128, 256, 512]
        decoder_channels: List[int] = cfg.decoder.decoder_channels # [256, 128, 64, 32, 16]
        use_batchnorm: bool = cfg.decoder.use_batchnorm
        attention_type: Union[None, str] = cfg.decoder.attention_type
        center: bool = cfg.decoder.center
        
        # computing blocks input and output channels
        encoder_channels = encoder_channels[1:]
        encoder_channels = encoder_channels[::-1]
        
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1]) # 512, 256, 128, 64, 32
        skip_channels = list(encoder_channels[1:]) + [0] # [256, 128, 64, 64, 0]
        out_channels = decoder_channels #[256, 128, 64, 32, 16]

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_bn=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        
        self.seg_head = SegmentationHead(
            in_channels=out_channels[-1],
            out_channels=cfg.data.num_classes
        )
        
  
    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Add and assert params"""
        from omegaconf import OmegaConf
        cfg = super(UNet, UNet).add_and_assert_specific_cfg(cfg)
        
        # assert deocoder not missing
        assert not OmegaConf.is_missing(cfg, "decoder")
        # decoder kwargs
        cfg.decoder.name = omegaconf_select(cfg, "decoder.name", "unet")
        cfg.decoder.encoder_channels = omegaconf_select(cfg, "decoder.encoder_channels", [3, 64, 64, 128, 256, 512])
        cfg.decoder.decoder_channels = omegaconf_select(cfg, "decoder.decoder_channels", [256, 128, 64, 32, 16])
        cfg.decoder.use_batchnorm = omegaconf_select(cfg, "decoder.use_batchnorm", True)
        cfg.decoder.attention_type = omegaconf_select(cfg, "decoder.attention_type", None)
        cfg.decoder.center = omegaconf_select(cfg, "decoder.center", True)


    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """extra learnable params: decoder and center block"""
        extra_learnable_params = [
            {"name": "decoder", "params": self.blocks.parameters()},
            {"name": "seg_head", "params": self.seg_head.parameters()}
        ]
        if self.center:
            extra_learnable_params += [{
                "name": "center_block", "params": self.center.parameters(),
            }]
    
        return super().learnable_params + extra_learnable_params
    
    
    def forward(self, X: torch.Tensor):
        """forward function for UNet segmentation"""
        out = super().forward(X)
        inner_feats = out["inner_feats"]
        
        inner_feats = inner_feats[::-1] # reverse skip features
        
        head = inner_feats[0]
        skips = inner_feats[1:]
        
        x = self.center(head)

        for i, block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = block(x, skip)
        
        mask = self.seg_head(x)
        # update segmentation mask feature into parent_out
        out.update({"mask": mask})
            
        return out

            
        
        
        
    

    
        
        

        

