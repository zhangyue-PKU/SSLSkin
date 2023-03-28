from .dice import DiceLoss
from .focal import FocalLoss
from torch.nn import CrossEntropyLoss


LOSSES = {
    "dice": DiceLoss,
    "focal": FocalLoss,
    "ce": CrossEntropyLoss
}

__all__ = [
    "DiceLoss",
    "FocalLoss",
    "CrossEntropyLoss",
]