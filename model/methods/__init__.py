

from .base import BaseMethod
from .base import BaseMomentumMethod
from .byol import BYOL
from .dino import DINO
from .mocov2plus import MoCoV2Plus
from .mocov3 import MoCoV3
from .simsiam import SimSiam
from .moby import MoBY
from .simclr import SimCLR


METHODS = {
    # base classes
    "base": BaseMethod,
    # methods
    "byol": BYOL,
    "dino": DINO,
    "mocov2plus": MoCoV2Plus,
    "mocov3": MoCoV3,
    "simsam": SimSiam,
    "moby": MoBY,
    "simclr":SimCLR
}


__all__ = [
    "BaseMethod",
    "BaseMomentumMethod",
    "BYOL",
    "DINO",
    "MoCoV2Plus",
    "MoCoV3",
    "SimSiam",
    "MoBY",
    "SimCLR",
]