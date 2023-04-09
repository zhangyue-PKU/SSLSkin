from .ph2dataset import PH2Dataset
from .isicdataset import ISICDataset

DATASET = {
    "ph2": PH2Dataset,
    "isic2016": ISICDataset
}
__all__ = [
    "PH2Dataset"
    "ISICDataset"
]