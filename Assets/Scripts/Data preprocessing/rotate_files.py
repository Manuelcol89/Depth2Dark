import argparse
import os
import cv2
from glob import glob
from PIL import Image

from torchvision import transforms
import torchvision.transforms as T
import torchvision.transforms.functional as TF

args = argparse.ArgumentParser()
args.add_argument('--input', type=str, default=r'D:\Users\manue\AppData\Local\SICE\Dark_cleaned_augmentated')
args, unknown = args.parse_known_args()

paths = glob(os.path.join(args.input, '*.*'))

for p in paths:
    filename = int(os.path.basename(p)[:-4]) + 580

    with Image.open(p) as my_image:
        img_rotated = transforms.functional.rotate(my_image,5.0, interpolation=transforms.InterpolationMode.BICUBIC)
        image_transforms = transforms.Compose(
            [
                transforms.CenterCrop(2048),
                transforms.ToTensor(),
            ]
        )

        images = [image_transforms(img_rotated)]
        transform = T.ToPILImage()

        for i in images:
            img = transform(i)
            img.save(f'{filename}.png')