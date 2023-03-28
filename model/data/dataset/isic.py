from typing import *

import os
import torchvision.datasets as datasets


"""torchvision image loaders
(see https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html)

"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from PIL import Image


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)



def _make_dataset(directory, label_file, is_valid_file=None) -> List[Tuple[str, int]]:
    """This function get (sample path, label) pair for dataset

    Args:
        directory (str): dir of images 
        label_file (str): label_file file name
        columns (List[str], optional): columns to retrive. Defaults to None.
        map_func (List[], optional): map_function of columns. Defaults to None.

    Returns:
        List[Tuple[str, int]]: _description_
    """
    assert os.path.exists(label_file)
    
    instances = []
    
    with open(label_file, 'r') as f:
        for line in f.readlines():
            item = line.strip().split(',')
            full_path = os.path.join(directory, item[0] + ".JPG")
            label = int(float(item[1]))
            if is_valid_file is not None and not is_valid_file(full_path):
                continue
            instances.append((full_path, label))
        
    return sorted(instances, key=lambda x: x[0])


class ISICDataset(datasets.VisionDataset):
    """Implements a dataset represented by a metadata file.

    Attributes:
        root:
            Root directory path
        metafile:
            file of metadata
        columns:
            columns to retrive in metafile
        loader:
            Fuction that read an image from disk
        transfrom:
            Function that takes a PIL image and returns transformed version
        target_transform:
            As transform but for targets
        map:
            A map that transfer a column of data to require type
    Raises:
        RuntimeError: If no supported files are found in root.

    """
    def __init__(self,
                 root,
                 label_file,
                 loader=default_loader,
                 transform: Callable=None,
                 target_transform: Callable=None,
                 is_valid_file: Callable= None,
                 ):
        super(ISICDataset, self).__init__(root,
                         transform=transform,
                         target_transform=target_transform)
        
        samples = _make_dataset(root, label_file, is_valid_file=is_valid_file)
        
        if len(samples) == 0:
            msg = 'Found 0 files in metafile: {}\n'.format(label_file)
            raise RuntimeError(msg)

        self.loader = loader
        self.samples = samples
        self.targets = [s[1] for s in samples]
      
        
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
    
    def __len__(self):
        """Returns the number of samples in the dataset.

        """
        return len(self.samples)
    
    
if __name__ == "__main__":
    dataset = ISICDataset(
        root="/home/zhangyue/SSLSkinLesion/data/pretrain/images",
        label_file="/home/zhangyue/SSLSkinLesion/data/pretrain/ISIC_2016_train.csv",
    )
    cnt_1 = 0
    for i in range(len(dataset)):
        item = dataset[i]
        if item[1] == 1:
            cnt_1 += 1
    print(cnt_1)