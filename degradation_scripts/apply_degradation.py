"""
Shuffles the test directory of images and puts 1024
random test images into the test_degradation file

"""

import os
import shutil
import random
import argparse
import pandas as pd

random.seed(1)

parser = argparse.ArgumentParser(description='Shuffles dataset and creates train/val/test')
parser.add_argument('inp_file', help='Entire image dataset')
args = parser.parse_args()

inp_d = args.inp_file

d = []

for dirs, subdirs, files in os.walk(inp_d):
    for f in files:
        d.append(os.path.join(dirs, f))

random.shuffle(d)

dlen = len(d)

testset = d[:1024]
te_dst = r'/home/mseals1/Documents/aphylla/annotated_images_only/test_degradation'

for f in testset:
    shutil.copy(f, te_dst)
