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
import sys
sys.path.append("/home/zhangyue/SSLSkinLesion")
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
                t.RandomCenterCrop(scale=0.6),
                t.Resize(size=(224, 224)),
                t.RandomHorizontalFlip(),
                t.RandomVerticalFlip(),
                t.ToTensor(),
                t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = t.Compose([
                t.Resize(size=(256, 256)),
                t.CenterCrop(size=(224, 224)),
                t.ToTensor(),
                t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])


    def __getitem__(self, index):
        # 读取图像
        image_file = os.path.join(self.root, 'images', self.file_list[index] + '.bmp')
        image = Image.open(image_file).convert('RGB')

        # 读取标签
        mask_file = os.path.join(self.root,  'masks', self.file_list[index] + '_lesion.bmp')
        mask = Image.open(mask_file).convert("L")
        
        image, mask = self.transform(image, mask)

        return image, mask.long()

    def __len__(self):
        return len(self.file_list)
    
    
if __name__ == "__main__":
    from torchvision import transforms
    import numpy as np
    ph2_dataset = PH2Dataset(
        "/home/zhangyue/SSLSkinLesion/data/ph2",
        mode="train"
    )
    img, mask = ph2_dataset[0]
    import pdb; pdb.set_trace()
    # 创建一个反转换的转换对象
    mean = [0.485, 0.456, 0.406]   # ImageNet均值
    std = [0.229, 0.224, 0.225]    # ImageNet标准差
    
    denormalize_transform = transforms.Normalize(
        mean=[-m/s for m, s in zip(mean, std)],
        std=[1/s for s in std]
    )

    # 假设有一个名为normalized_tensor的归一化张量
    denormalized_tensor = denormalize_transform(img)

    # 将张量转换为NumPy数组
    array = denormalized_tensor.cpu().numpy().transpose(1, 2, 0)
    mask = mask.cpu().numpy()

    # 将数组的范围从[-1,1]转换为[0,255]
    # array = ((array + 1) / 2.0) * 255.0
    array = array * 255.0
    mask = mask * 255.0

    # 将数组转换为8位无符号整数类型
    array = array.astype(np.uint8)
    mask = mask.astype(np.uint8)

    # 将NumPy数组转换为PIL图像对象
    image = Image.fromarray(array)
    mask = Image.fromarray(mask)

    # 保存图像对象到磁盘
    image.save("image.png")
    mask.save("mask.png")
    