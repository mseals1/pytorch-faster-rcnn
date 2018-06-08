from PIL import ImageEnhance
from PIL import Image
from PIL import ImageFilter
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import cv2
import skimage.util.noise as skinoise
import argparse
import random
import os

# 1, 6, 98


# Adjusts the Guassian Blur applied to the image
# gblur_rad is the radius of the guassian blur being applied
# [0, 9+]
def gblur(im1, gblur_rad=0.0):
    return im1.filter(ImageFilter.GaussianBlur(radius=gblur_rad))


random.seed(1)

parser = argparse.ArgumentParser(description='Applies the different degradation methods to the directory of images')
parser.add_argument('inp_fn', help='Image directory')

args = parser.parse_args()

inp_fn = args.inp_fn

im = Image.open(inp_fn)
cv_im = cv2.imread(inp_fn)

ims = []
tiled_ims = []

# GAUSSIAN BLUR CODE
for i in range(0, 30):
    f = os.path.join(r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_gblur_r{}.jpg"
                     .format(os.path.split(inp_fn)[-1][:-4], i))
    # gblur(im, i).save(f)
    img = gblur(im, i)

    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 48)
    draw.text((0, 0), 'r'+str(i), (255, 255, 255), font=font)

    ims.append(np.array(img))

ims = np.array(ims)

vert = []

for i in range(3):
    horiz = []
    for j in range(10):
        if j == 0:
            horiz = ims[j+i*10]
        else:
            horiz = np.hstack((horiz, ims[j+i*10]))
    if i == 0:
        vert = horiz
    else:
        vert = np.vstack((vert, horiz))

Image.fromarray(vert).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_gblur_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))

