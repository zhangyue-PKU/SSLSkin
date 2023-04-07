from .base import BaseSegmentationModel
from .unet import UNet

METHODS = {
    "unet": UNet
}

all = [
    "BaseSegmentationModel",
    "UNet"
]