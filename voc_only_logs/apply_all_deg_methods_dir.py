"""
Matthew Seals, June 2018

Applies all 10 degradation functions to a directory of images
Most functions use the log space

- INPUT (on command line): directory of images
- OUTPUT: a folder containing ALL of the degraded images
    - N * 10 * 12 images in a single folder

The output folder can be sym linked to the network's input folder and
the test.txt can be modified to contain the correct image names to
feed through the network (iteratively go through all 12 different
function names)
"""


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
import glob
import os


# Adjust the colors of the image
# grayscale 0.0 -> original colors 1.0
def color_adj(im1, enhance_num):
    color = ImageEnhance.Color(im1)
    return color.enhance(enhance_num)


# Adjust the contrast of the image
# Solid gray image 0.0 -> original 1.0 -> over-saturated 2.0
def contrast_adj(im1, contrast_num):
    contrast = ImageEnhance.Contrast(im1)
    return contrast.enhance(contrast_num)


# Adjusts the brightness of the image
# black 0.0 -> original 1.0 -> white 2.0
def bright_adj(im1, bright_num):
    bright = ImageEnhance.Brightness(im1)
    return bright.enhance(bright_num)


# Adjusts the sharpness of the image
# perc1: percentage to be sharpened
# [0, 400+]
def sharp_adj(im1, perc1):
    return im1.filter(ImageFilter.UnsharpMask(percent=perc1))


# Adjusts the Gaussian Blur applied to the image
# gblur_rad is the radius of the gaussian blur being applied
def gblur(im1, gblur_rad):
    return im1.filter(ImageFilter.GaussianBlur(radius=gblur_rad))


# Resize from original -> ratio of the original -> the original
# reduces the quality of the image
def imresize(im1, ratio):
    filter1 = Image.NEAREST
    new_size = (int(im1.size[0] * ratio), int(im1.size[1] * ratio))
    return im1.resize(new_size, resample=filter1).resize(im1.size, resample=filter1)


# Gaussian Noise
# var1: original image 0 -> noisy image 1
def gnoise(im1, var1):
    out1 = skinoise.random_noise(im1, mode='gaussian', var=var1)
    out1 *= 255.0
    return Image.fromarray(out1.astype('uint8'))


# Salt and Pepper Noise
# amt: amount of s&p noise in the image
def snpnoise(im1, amt=0.05):
    out = skinoise.random_noise(im1, mode='s&p', amount=amt)
    out *= 255.0
    return Image.fromarray(out.astype('uint8'))


# Speckle Noise
# var1: original image 0 -> noisy image 1
def snoise(im1, var1):
    out = skinoise.random_noise(im1, mode='speckle', var=var1)
    out *= 255.0
    return Image.fromarray(out.astype('uint8'))


random.seed(1)

parser = argparse.ArgumentParser(description='Applies the different degradation methods to the directory of images')
parser.add_argument('inp_dir', help='Image directory')

args = parser.parse_args()

inp_d = args.inp_dir
ims = []

for dirs, _, files in os.walk(inp_d):
    ims += glob.glob(os.path.join(dirs, "*.jpg"))

# Color range
r_col = np.linspace(0, 1, num=4, endpoint=False)
r_col2 = np.geomspace(1, 10, num=8, endpoint=False)
r_col = np.concatenate((r_col, r_col2))

# Contrast range
r_con = np.linspace(0.05, 1, num=6, endpoint=False)
r_con2 = np.geomspace(1, 5, num=5)
r_con = np.concatenate(([0], r_con, r_con2))

# Brightness range
r_br = np.geomspace(0.1, 1, num=6, endpoint=False)
r_br2 = np.geomspace(1, 30, num=5)
r_br = np.concatenate(([0], r_br, r_br2))

# Brightness sqrt range
r_br_sqrt = [np.sqrt(2) ** (i - 6) for i in range(0, 11)]
r_br_sqrt = np.concatenate(([0], r_br_sqrt))

# Sharpness range
r_sh = [i*75 for i in range(0, 12)]

# Gaussian Blur range
r_gblur = [i for i in range(0, 12)]

# Imresize range
r_imr = np.geomspace(0.01, 1, num=12)
r_imr = r_imr[::-1]

# Gaussian Noise range
r_gau = np.geomspace(0.01, 5, num=11)
r_gau = np.concatenate(([0], r_gau))

# Salt and Pepper range
r_snp = np.geomspace(0.03, 1, num=11)
r_snp = np.concatenate(([0], r_snp))

# Speckle Noise range
r_spe = np.geomspace(0.03, 100, num=11)
r_spe = np.concatenate(([0], r_spe))

# with open(os.path.join(inp_d, 'LookupTable.txt'), 'w') as f:
#     f.write('Color (Saturation) Parameters\n' + str(r_col) + '\n\n')
#     f.write('Contrast Parameters\n' + str(r_con) + '\n\n')
#     f.write('Brightness Parameters\n' + str(r_br) + '\n\n')
#     f.write('Brightness (using sqrt space) Parameters\n' + str(r_br_sqrt) + '\n\n')
#     f.write('Sharpness Parameters\n' + str(r_sh) + '\n\n')
#     f.write('Gaussian Blur Parameters\n' + str(r_gblur) + '\n\n')
#     f.write('Image Resize Parameters\n' + str(r_imr) + '\n\n')
#     f.write('Gaussian Noise Parameters\n' + str(r_gau) + '\n\n')
#     f.write('Salt and Pepper Noise Parameters\n' + str(r_snp) + '\n\n')
#     f.write('Speckle Noise Parameters\n' + str(r_spe))

for fn in ims:
    im = Image.open(fn)
    numpy_im = np.array(im.convert('RGB'))

    ofn = r"voc_only_logs\degraded\JPEGImages"
    font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 48)
    text_color = (0, 255, 255)

    # COLOR CODE
    for e, i in enumerate(r_col):
        f = os.path.join(ofn, "{}_color_{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        color_adj(im, i).save(f)

    # CONTRAST CODE
    for e, i in enumerate(r_con):
        f = os.path.join(ofn, "{}_contrast_{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        contrast_adj(im, i).save(f)

    # BRIGHTNESS CODE
    for e, i in enumerate(r_br):
        f = os.path.join(ofn, "{}_brightness_{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        bright_adj(im, i).save(f)

    # BRIGHTNESS CODE using "Square Root space"
    for e, i in enumerate(r_br_sqrt):
        f = os.path.join(ofn, "{}_brightness_sqrt_{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        bright_adj(im, i).save(f)

    # SHARPNESS CODE
    for e, i in enumerate(r_sh):
        f = os.path.join(ofn, "{}_sharpness_{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        sharp_adj(im, i).save(f)

    # GAUSSIAN BLUR CODE
    for e, i in enumerate(r_gblur):
        f = os.path.join(ofn, "{}_gblur_r{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        gblur(im, i).save(f)

    # IMRESIZE CODE
    for e, i in enumerate(r_imr):
        f = os.path.join(ofn, "{}_resize_{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        imresize(im, i).save(f)

    # GAUSSIAN NOISE CODE
    for e, i in enumerate(r_gau):
        f = os.path.join(ofn, "{}_gnoise_{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        gnoise(numpy_im, i).save(f)

    # SALT AND PEPPER NOISE CODE
    for e, i in enumerate(r_snp):
        f = os.path.join(ofn, "{}_snpnoise_{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        snpnoise(numpy_im, i).save(f)

    # SPECKLE NOISE CODE
    for e, i in enumerate(r_spe):
        f = os.path.join(ofn, "{}_snoise_{}.jpg".format(os.path.split(fn)[-1][:-4], e))
        snoise(numpy_im, i).save(f)

