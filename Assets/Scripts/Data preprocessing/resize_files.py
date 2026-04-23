import argparse
import os

from PIL import Image
from glob import glob
from torchvision import transforms
import torchvision.transforms as T

args = argparse.ArgumentParser()
args.add_argument('--input', type=str, default=r'D:\Users\manue\AppData\Local\SICE\Dark_cleaned_augmentated')
args, unknown = args.parse_known_args()

paths = glob(os.path.join(args.input, '*.*'))

for p in paths:
    filename = os.path.basename(p)
    with (Image.open(p) as my_image):
        image_transforms = transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(512),
                transforms.ToTensor(),
            ]
        )
        images = [image_transforms(my_image)]
        transform = T.ToPILImage()

    for i in images:
        img = transform(i)
        img.save(filename)