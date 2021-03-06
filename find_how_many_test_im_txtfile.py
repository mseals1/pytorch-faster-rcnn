from PIL import ImageEnhance
from PIL import Image
from PIL import ImageFilter
import numpy as np
import cv2
import skimage.util.noise as skinoise
import argparse
import random
import os


random.seed(1)

parser = argparse.ArgumentParser(description='Applies the different degradation methods to the directory of images')
parser.add_argument('inp_dir', help='Image directory')

args = parser.parse_args()

inp_d = args.inp_dir

names = [f[:-4] for f in os.listdir(inp_d) if os.path.isfile(os.path.join(inp_d, f))]
names = np.array(names)

with open(os.path.join(inp_d, 'deg_test.txt'), 'w') as f:
    for n in names:
        n += '\n'
        f.write(n)
