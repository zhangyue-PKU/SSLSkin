from typing import *
from PIL import Image, ImageFilter, ImageOps
import torchvision.transforms as transforms
import random


class Equalization:
    def __call__(self, img: Image) -> Image:
        return ImageOps.equalize(img)

# 读入数据
img = Image.open("ISIC_0000013.JPG")


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

# 定义增强函数



t = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1)),    
    transforms.RandomApply([transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.2,
        hue=0.1
    )], p=0.8),
    transforms.RandomGrayscale(p=0.5)]
t.append(
    transforms.RandomApply([
        GaussianBlur()], p=0.5)
)
t.append(
    transforms.RandomApply([Solarization()],p=0.2)
)
# t.append(transforms.RandomApply([Equalization()], p=0.5))
t.append(
    transforms.RandomHorizontalFlip()
)
t.append(
    transforms.RandomVerticalFlip()
)
t.append(transforms.RandomApply([transforms.RandomRotation(45)], p=0.5))

t = transforms.Compose(t)


# 循环增强并保存结果
for i in range(30):
    transformed_img = t(img)
    transformed_img.save(f"example_transformed_{i}.jpg")