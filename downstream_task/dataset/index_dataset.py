import os
from PIL import Image
from torch.utils.data import Dataset
import downstream_task.dataset.transform as t

from downstream_task.dataset.ph2dataset import PH2Dataset
from downstream_task.dataset.isicdataset import ISICDataset



# get filename by index
def _get_filename_by_index(dataset, index):
    """Default function which maps the index of an image to a filename.
    """
    if isinstance(dataset, PH2Dataset):
        file_path = os.path.join(dataset.root, 'images', dataset.file_list[index] + '.bmp')
        return file_path
    elif isinstance(dataset, ISICDataset):
        file_path = os.path.join(dataset.root, 'images', dataset.file_list[index] + '.jpg')
        return file_path
    else:
        return str(index)



class IndexDataset(Dataset):
    """
        Dataset with path returned
    """
    def __init__(self, dataset):     
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        item = self.dataset.__getitem__(index)
        path = _get_filename_by_index(self.dataset, index)
        
        return item, path