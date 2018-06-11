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
    if cntr % 15 == 0:
        n += 1
        cntr = 0
    # 512_1_voc_gblur_rX_test_log.txt
    new_fn = os.path.join(*f.split('/')[:-1], '512_' + str(n) + '_voc_gblur_r' + str(cntr) + '_test_log.txt')
    shutil.move(f, new_fn)
    cntr += 1
