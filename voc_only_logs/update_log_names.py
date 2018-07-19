import numpy as np
import os
import argparse
import glob
import natsort
import shutil

parser = argparse.ArgumentParser(description='Applies the different degradation methods to the directory of images')
parser.add_argument('inp_fn', help='Image directory')

args = parser.parse_args()

inp_fn = args.inp_fn

files = glob.glob(os.path.join(inp_fn, "*.txt.*"))
files = natsort.natsorted(files)

cntr = 0
n = 0

for f in files:
    if cntr % 12 == 0:
        n += 1
        cntr = 0

    if n == 1:
        new_fn = os.path.join(*f.split('\\')[:-1], '512_voc_color_' + str(cntr) + '_test_log.txt')
        shutil.move(f, new_fn)
    elif n == 2:
        new_fn = os.path.join(*f.split('\\')[:-1], '512_voc_contrast_' + str(cntr) + '_test_log.txt')
        shutil.move(f, new_fn)
    elif n == 3:
        new_fn = os.path.join(*f.split('\\')[:-1], '512_voc_brightness_' + str(cntr) + '_test_log.txt')
        shutil.move(f, new_fn)
    elif n == 4:
        new_fn = os.path.join(*f.split('\\')[:-1], '512_voc_sharpness_' + str(cntr) + '_test_log.txt')
        shutil.move(f, new_fn)
    elif n == 5:
        new_fn = os.path.join(*f.split('\\')[:-1], '512_voc_gblur_' + str(cntr) + '_test_log.txt')
        shutil.move(f, new_fn)
    elif n == 6:
        new_fn = os.path.join(*f.split('\\')[:-1], '512_voc_resize_' + str(cntr) + '_test_log.txt')
        shutil.move(f, new_fn)
    elif n == 7:
        new_fn = os.path.join(*f.split('\\')[:-1], '512_voc_gnoise_' + str(cntr) + '_test_log.txt')
        shutil.move(f, new_fn)
    elif n == 8:
        new_fn = os.path.join(*f.split('\\')[:-1], '512_voc_snpnoise_' + str(cntr) + '_test_log.txt')
        shutil.move(f, new_fn)
    elif n == 9:
        new_fn = os.path.join(*f.split('\\')[:-1], '512_voc_snoise_' + str(cntr) + '_test_log.txt')
        shutil.move(f, new_fn)

    cntr += 1
