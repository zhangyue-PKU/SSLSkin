"""Implemenation of PH2 dataset
    root/
        images/
            xxx.bmp
        masks/
            xxx_lesion.bmp
        train.txt
        val.txt
    
    translation refer to data/ph2/modify_ph2.py
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import downstream_task.dataset.transform as t

class PH2Dataset(Dataset):
    def __init__(self, root,  mode='train'):
        self.root = root
        self.mode = mode

        # 读取文件名和标签
        if mode == 'train':
            txt_file = os.path.join(root, 'train.txt')
        elif mode == 'val':
            txt_file = os.path.join(root, 'val.txt')
        else:
            raise ValueError('Invalid mode')
        with open(txt_file, 'r') as f:
            self.file_list = f.readlines()
        self.file_list = [x.strip() for x in self.file_list]

        # 数据增强
        if mode == 'train':
            self.transform = t.Compose([
                t.Resize(size=(224, 224)),
                t.RandomHorizontalFlip(),
                t.RandomVerticalFlip(),
                t.ToTensor(),
                t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = t.Compose([
                t.Resize(size=(224, 224)),
                t.ToTensor(),
                t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])


    def __getitem__(self, index):
        
        # 读取图像
        image_file = os.path.join(self.root, 'images', self.file_list[index] + '.bmp')
        image = Image.open(image_file).convert('RGB')

        # 读取标签
        mask_file = os.path.join(self.root,  'masks', self.file_list[index] + '_lesion.bmp')
        mask = Image.open(mask_file)
        
        image, mask = self.transform(image, mask)

        return image, mask.long()

    def __len__(self):
        return len(self.file_list)
    
    
if __name__ == "__main__":
    ph2_dataset = PH2Dataset(
        "/home/zhangyue/SSLSkinLesion/data/ph2",
        mode="train"
    )
    img, mask = ph2_dataset[0]
    import pdb; pdb.set_trace()
    

