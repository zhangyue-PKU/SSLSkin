from .dice import DiceLoss
from torch.nn import CrossEntropyLoss


LOSSES = {
    "dice": DiceLoss,
    "ce": CrossEntropyLoss
}

__all__ = [
    "DiceLoss",
    "CrossEntropyLoss",
]