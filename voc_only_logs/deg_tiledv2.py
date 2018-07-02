# Applies all 9 degradation functions to a single image
# Most functions use the log space

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

# 4545.jpg


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
parser.add_argument('inp_fn', help='Image directory')

args = parser.parse_args()

inp_fn = args.inp_fn

im = Image.open(inp_fn)
numpy_im = np.array(im.convert('RGB'))

ims_color = []
ims_contr = []
ims_brigh = []
ims_sharp = []
ims_gblur = []
ims_imres = []
ims_gnois = []
ims_snpno = []
ims_snois = []

font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 48)
text_color = (0, 255, 255)

# Color range
r_col = np.linspace(0, 1, num=4, endpoint=False)
r_col2 = np.geomspace(1, 10, num=8, endpoint=False)
r_col = np.concatenate((r_col, r_col2))

# Contrast range
r_con = np.linspace(0.05, 1, num=6, endpoint=False)
r_con2 = np.geomspace(1, 20, num=5)
r_con = np.concatenate(([0], r_con, r_con2))

# Brightness range
r_br = np.geomspace(0.1, 1, num=6, endpoint=False)
r_br2 = np.geomspace(1, 30, num=5)
r_br = np.concatenate(([0], r_br, r_br2))

# Sharpness range
r_sh = [i * 34 for i in range(0, 12)]

# Gaussian Blur range
# r_gblur = [i for i in range(0, 12)]
r_gblur = np.linspace(0.5, 2.5, num=4, endpoint=False)
r_gblur2 = np.linspace(3, 9, num=7)
r_gblur = np.concatenate(([0], r_gblur, r_gblur2))

# Imresize range
r_imr = np.geomspace(0.04, 1, num=12)
r_imr = r_imr[::-1]

# Gaussian Noise range
r_gau = np.geomspace(0.01, 10.24, num=11)
r_gau = np.concatenate(([0], r_gau))

# Salt and Pepper range
r_snp = np.linspace(0.03, 0.5, num=6, endpoint=False)
r_snp2 = np.linspace(0.5, 1, num=5)
r_snp = np.concatenate(([0], r_snp, r_snp2))

# Speckle Noise range
r_spe = np.geomspace(0.03, 30, num=11)
r_spe = np.concatenate(([0], r_spe))

# COLOR CODE
for i in r_col:
    img = color_adj(im, i)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "{:.2f}".format(i), text_color, font=font)
    ims_color.append(np.array(img))


# CONTRAST CODE
for i in r_con:
    img = contrast_adj(im, i)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "{:.2f}".format(i), text_color, font=font)
    ims_contr.append(np.array(img))


# BRIGHTNESS CODE
for i in r_br:
    img = bright_adj(im, i)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "{:.2f}".format(i), text_color, font=font)
    ims_brigh.append(np.array(img))


# SHARPNESS CODE
for i in r_sh:
    img = sharp_adj(im, i)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "{:.2f}".format(i / 100. + 1), text_color, font=font)
    ims_sharp.append(np.array(img))


# GAUSSIAN BLUR CODE
for i in r_gblur:
    img = gblur(im, i)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "{:.2f}".format(i), text_color, font=font)
    ims_gblur.append(np.array(img))


# IMRESIZE CODE
for i in r_imr:
    img = imresize(im, i)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "{:.3f}".format(i), text_color, font=font)
    ims_imres.append(np.array(img))


# GAUSSIAN NOISE CODE
for i in r_gau:
    img = gnoise(numpy_im, i)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "{:.2f}".format(i), text_color, font=font)
    ims_gnois.append(np.array(img))


# SALT AND PEPPER NOISE CODE
for i in r_snp:
    img = snpnoise(numpy_im, i)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "{:.2f}".format(i), text_color, font=font)
    ims_snpno.append(np.array(img))


# SPECKLE NOISE CODE
for i in r_spe:
    img = snoise(numpy_im, i)
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), "{:.2f}".format(i), text_color, font=font)
    ims_snois.append(np.array(img))


ims_color = np.array(ims_color)
ims_contr = np.array(ims_contr)
ims_brigh = np.array(ims_brigh)
ims_sharp = np.array(ims_sharp)
ims_gblur = np.array(ims_gblur)
ims_imres = np.array(ims_imres)
ims_gnois = np.array(ims_gnois)
ims_snpno = np.array(ims_snpno)
ims_snois = np.array(ims_snois)

vert_color = []
vert_contr = []
vert_brigh = []
vert_sharp = []
vert_gblur = []
vert_imres = []
vert_gnois = []
vert_snpno = []
vert_snois = []

for i in range(3):
    horiz_color = []
    horiz_contr = []
    horiz_brigh = []
    horiz_sharp = []
    horiz_gblur = []
    horiz_imres = []
    horiz_gnois = []
    horiz_snpno = []
    horiz_snois = []
    ii = 4
    for j in range(ii):
        if j == 0:
            horiz_color = ims_color[j+i*ii]
            horiz_contr = ims_contr[j+i*ii]
            horiz_brigh = ims_brigh[j+i*ii]
            horiz_sharp = ims_sharp[j+i*ii]
            horiz_gblur = ims_gblur[j+i*ii]
            horiz_imres = ims_imres[j+i*ii]
            horiz_gnois = ims_gnois[j+i*ii]
            horiz_snpno = ims_snpno[j+i*ii]
            horiz_snois = ims_snois[j+i*ii]
        else:
            horiz_color = np.hstack((horiz_color, ims_color[j+i*ii]))
            horiz_contr = np.hstack((horiz_contr, ims_contr[j+i*ii]))
            horiz_brigh = np.hstack((horiz_brigh, ims_brigh[j+i*ii]))
            horiz_sharp = np.hstack((horiz_sharp, ims_sharp[j+i*ii]))
            horiz_gblur = np.hstack((horiz_gblur, ims_gblur[j+i*ii]))
            horiz_imres = np.hstack((horiz_imres, ims_imres[j+i*ii]))
            horiz_gnois = np.hstack((horiz_gnois, ims_gnois[j+i*ii]))
            horiz_snpno = np.hstack((horiz_snpno, ims_snpno[j+i*ii]))
            horiz_snois = np.hstack((horiz_snois, ims_snois[j+i*ii]))
    if i == 0:
        vert_color = horiz_color
        vert_contr = horiz_contr
        vert_brigh = horiz_brigh
        vert_sharp = horiz_sharp
        vert_gblur = horiz_gblur
        vert_imres = horiz_imres
        vert_gnois = horiz_gnois
        vert_snpno = horiz_snpno
        vert_snois = horiz_snois
    else:
        vert_color = np.vstack((vert_color, horiz_color))
        vert_contr = np.vstack((vert_contr, horiz_contr))
        vert_brigh = np.vstack((vert_brigh, horiz_brigh))
        vert_sharp = np.vstack((vert_sharp, horiz_sharp))
        vert_gblur = np.vstack((vert_gblur, horiz_gblur))
        vert_imres = np.vstack((vert_imres, horiz_imres))
        vert_gnois = np.vstack((vert_gnois, horiz_gnois))
        vert_snpno = np.vstack((vert_snpno, horiz_snpno))
        vert_snois = np.vstack((vert_snois, horiz_snois))

Image.fromarray(vert_color).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_color_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))
Image.fromarray(vert_contr).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_contrast_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))
Image.fromarray(vert_brigh).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_brightness_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))
Image.fromarray(vert_sharp).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_sharpness_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))
Image.fromarray(vert_gblur).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_gblur_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))
Image.fromarray(vert_imres).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_resize_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))
Image.fromarray(vert_gnois).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_gnoise_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))
Image.fromarray(vert_snpno).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_snpnoise_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))
Image.fromarray(vert_snois).save(
    r"C:\Users\Matthew\Desktop\masters\pytorch-faster-rcnn\voc_only_logs\degraded\{}_snoise_tiled.jpg"
    .format(os.path.split(inp_fn)[-1][:-4]))
