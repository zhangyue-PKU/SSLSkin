import torch
import random
import numpy as np
import torchvision.transforms.functional as F

from PIL import Image


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, img, mask):
        for transform in self.transforms:
            img, mask = transform(img, mask)

        # print("Shape after composed transform:", img.shape)
        # print(mask.shape)
        return img, mask



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, img, mask):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = np.array(img).astype(np.float32).transpose(2, 0, 1)
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float() / 255.0
        mask = torch.from_numpy(mask).float() / 255.0

        return img, mask



class RandomHorizontalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # print("Shape after HF", np.array(img).shape)
        # print(np.array(mask).shape)
        return img, mask



class RandomVerticalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        
        # print("Shape after VF", np.array(img).shape)
        # print(np.array(mask).shape)
        return img, mask



class RandomCenterCrop(object):
    def __init__(self, scale=0.9):
        self.scale = scale
    
    def __call__(self, img, mask):
        scale_rate = random.uniform(self.scale, 1.0)
        scale_size = [int(l * scale_rate) for l in img.size]
        #print(scale_size)
        img = F.center_crop(img, output_size=scale_size)
        mask = F.center_crop(mask, output_size=scale_size)

        return img, mask



class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, img, mask):
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, mask



class RandomAffine(object):
    def __init__(self, angle=10, shear=5, translate=20, scale=(0.9, 1.1)):
        self.angle = angle
        self.shear = shear
        self.translate = translate
        self.scale = scale

    def __call__(self, img, mask):

        scale = random.uniform(self.scale[0], self.scale[1])
        angle = random.uniform(-1 * self.angle, self.angle)

        shear_x = random.uniform(-1 * self.shear, self.shear)
        shear_y = random.uniform(-1 * self.shear, self.shear)
        shear = (shear_x, shear_y)

        translate_x = random.uniform(-1 * self.translate, self.translate)
        translate_y = random.uniform(-1 * self.translate, self.translate)
        translate = (translate_x, translate_y)

        img = F.affine(img, angle=angle, shear=shear, scale=scale, translate=translate)
        mask = F.affine(mask, angle=angle, shear=shear, scale=scale, translate=translate)

        return img, mask


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):

        img = F.resize(img, size=self.size)
        mask = F.resize(mask, size=self.size)   
     
        return img, mask
    

class CenterCrop():
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img, mask):
        img = F.center_crop(img, output_size=self.size)
        mask = F.center_crop(mask, output_size=self.size)
        
        return img, mask