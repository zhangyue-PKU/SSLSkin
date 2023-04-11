from .base import BaseSegmentationModel
from .unet import UNet
from .fpn import FPN

METHODS = {
    "unet": UNet,
    "fpn": FPN,
}

all = [
    "BaseSegmentationModel",
    "UNet",
    "FPN",
]