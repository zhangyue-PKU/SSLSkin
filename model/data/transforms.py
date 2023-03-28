"""Transform fuction for SSL and Linear probbing"""
# Copyright 2023 solo-learn development team.

from torchvision import transforms

import torch
import random
import warnings
import omegaconf
from typing import Callable, List, Sequence
from PIL import Image, ImageFilter, ImageOps


# Mean and std for dataset
NormalizeMeanStd = {
            "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            "stl10": ((0.4914, 0.4823, 0.4466), (0.247, 0.243, 0.261)),
            "imagenet100": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        }


class GaussianBlur:
    def __init__(self, sigma: Sequence[float] = None):
        """Gaussian blur as a callable object.

        Args:
            sigma (Sequence[float]): range to sample the radius of the gaussian blur filter.
                Defaults to [0.1, 2.0].
        """

        if sigma is None:
            sigma = [0.1, 2.0]

        self.sigma = sigma

    def __call__(self, img: Image) -> Image:
        """Applies gaussian blur to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: blurred image.
        """

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img


class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: solarized image.
        """

        return ImageOps.solarize(img)


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)


class FTTransformPipeline:
    
    def __init__(self, cfg: omegaconf.DictConfig):
        """Create obj of Linear probbing Transform pipline

        Args:
            dataset (str): name of dataset, choose from [cifar10, cifar100, stl10, imagenet100, imagenet]
            stage (str): either train or val
        """
        
        pipeline = []
        
        if cfg.rrc.enabled:
            # Crop
            pipeline.append(
                transforms.RandomResizedCrop(
                    cfg.crop_size,
                    scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            )
        else:
            pipeline.extend([
                transforms.Resize(
                    cfg.resize_size, 
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(
                     cfg.crop_size
                )
            ])
            
        if cfg.horizontal_flip.prob:
            # Flip
            pipeline.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))
            
        if cfg.vertical_flip.prob:
            pipeline.append(transforms.RandomVerticalFlip(p=cfg.vertical_flip.prob))
            
        if cfg.rotate.prob:
            pipeline.append(transforms.RandomApply([transforms.RandomRotation(degrees=90)]))
            
        
        # ToTensor
        pipeline.append(transforms.ToTensor())
        pipeline.append(transforms.Normalize(mean=cfg.mean, std=cfg.std))
        
        # Compose
        self.pipeline = transforms.Compose(pipeline)


    def __call__(self, image: Image) -> torch.Tensor:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        return self.pipeline(image)
    
    
    def __repr__(self) -> str:
        return str(self.pipeline)



class TransformPipeline:
    """Creates a pipeline of transformations given a dataset and an augmentation Cfg node.
        The node needs to be in the following format:
            crop_size: int
            [OPTIONAL] mean: float
            [OPTIONAL] std: float
            rrc:
                enabled: bool
                crop_min_scale: float
                crop_max_scale: float
            color_jitter:
                prob: float
                brightness: float
                contrast: float
                saturation: float
                hue: float
            grayscale:
                prob: float
            gaussian_blur:
                prob: float
            solarization:
                prob: float
            equalization:
                prob: float
            horizontal_flip:
                prob: float
    """
    
    def __init__(self, 
                 dataset: str,
                 cfg: omegaconf.DictConfig, 
                 num_crops: int = 1):

        augmentations = []
        
        if cfg.rrc.enabled:
            # Crop
            augmentations.append(
                transforms.RandomResizedCrop(
                    cfg.crop_size,
                    scale=(cfg.rrc.crop_min_scale, cfg.rrc.crop_max_scale),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            )
        else:
            augmentations.append(
                transforms.Resize(
                    cfg.crop_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
            )

        if cfg.color_jitter.prob:
            # Color_jitter
            augmentations.append(
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            cfg.color_jitter.brightness,
                            cfg.color_jitter.contrast,
                            cfg.color_jitter.saturation,
                            cfg.color_jitter.hue,
                        )
                    ],
                    p=cfg.color_jitter.prob,
                ),
            )

        if cfg.grayscale.prob:
            # Grayscale
            augmentations.append(transforms.RandomGrayscale(p=cfg.grayscale.prob))

        if cfg.gaussian_blur.prob:
            # Gaussian_Blur
            augmentations.append(transforms.RandomApply([GaussianBlur()], p=cfg.gaussian_blur.prob))

        if cfg.solarization.prob:
            # Solarization
            augmentations.append(transforms.RandomApply([Solarization()], p=cfg.solarization.prob))

        if cfg.equalization.prob:
            # Equalization
            augmentations.append(transforms.RandomApply([Equalization()], p=cfg.equalization.prob))

        if cfg.horizontal_flip.prob:
            # Flip
            augmentations.append(transforms.RandomHorizontalFlip(p=cfg.horizontal_flip.prob))
        
        if cfg.vertical_flip.prob:
            # Vertical flip
            augmentations.append(transforms.RandomVerticalFlip(p=cfg.vertical_flip.prob))

        if cfg.rotate.prob:
            augmentations.append(transforms.RandomApply([transforms.RandomRotation(degrees=90)]))

        # ToTensor
        augmentations.append(transforms.ToTensor())
        # Normalize
        mean, std = NormalizeMeanStd.get(dataset, ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        augmentations.append(transforms.Normalize(mean=mean, std=std))
        
        # Compose
        self.augmentations = transforms.Compose(augmentations)
        
        # N Crops by this Transfrom
        self.num_crops = num_crops
        

    def __call__(self, image: Image) -> List[torch.Tensor]:
        """Applies transforms n times to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """
        return [self.augmentations(image) for _ in range(self.num_crops)]
    

    def __repr__(self) -> str:
        return f"Generate {self.num_crops} by {self.augmentations}"



class FullTransformPipeline:
    """ Pipline of transforms
    """
    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms
    
    
    def __call__(self, x: Image) -> List[torch.Tensor]:
        """Applies n transforms to a image x to generate n crops.

        Args:
            x (Image): an image in the PIL.Image format.

        Returns:
            List[torch.Tensor]: an image in the tensor format.
        """

        out = []
        for transform in self.transforms:
            out.extend(transform(x))
            
        return out


    def __repr__(self) -> str:
        return "\n".join([str(transform) for transform in self.transforms])