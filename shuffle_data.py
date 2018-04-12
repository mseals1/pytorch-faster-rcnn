"""
Shuffles the given directory of images and divides
them into train/val/test sets with ratios 50/25/25
(the ratios can easily be changed)

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
tr_r = 0.5
va_r = (1. - tr_r) / 2.
te_r = 1. - va_r - tr_r

trainset = d[:int(dlen * tr_r)]
tr_dst = os.path.join(inp_d, 'train')

valset = d[int(dlen * tr_r):int(dlen * (tr_r + va_r))]
va_dst = os.path.join(inp_d, 'val')

testset = d[int(dlen*(tr_r + va_r)):]
te_dst = os.path.join(inp_d, 'test')

for f in trainset:
    shutil.move(f, tr_dst)

for f in valset:
    shutil.move(f, va_dst)

for f in testset:
    shutil.move(f, te_dst)
