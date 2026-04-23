import argparse
import os
import cv2
from glob import glob

args = argparse.ArgumentParser()
args.add_argument('--input', type=str, default=r'D:\Users\manue\AppData\Local\SICE\Bright_cleaned_renamed')
args, unknown = args.parse_known_args()

paths = glob(os.path.join(args.input, '*.*'))
for p in paths:
    filename = int(os.path.basename(p)[:-4]) + 290
    img = cv2.imread(p)
    flipped = cv2.flip(img, 1)
    cv2.imwrite(str(filename) + '.png', flipped)