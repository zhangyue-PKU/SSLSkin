import os
from PIL import Image

img_path = "pretrain/images/ISIC_0000022.JPG"


img_name = os.path.basename(img_path)

img = Image.open(img_path)

img.save(img_name)