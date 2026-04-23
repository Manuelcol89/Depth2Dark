import argparse
import os

from glob import glob
from PIL import Image


args = argparse.ArgumentParser()
args.add_argument('--input', type=str, default=r'D:\Users\manue\AppData\Local\SICE\Dark_cleaned')
args.add_argument('--output', type=str, default=r'D:\Users\manue\AppData\Local\SICE\Dark_cleaned_renamed')
args = args.parse_args()

paths = glob(os.path.join(args.input, '*.*'))
index = 1

for p in paths:
    with Image.open(p) as my_image:
        my_image.save(os.path.join(args.output, f"{index}.png"))
        index += 1