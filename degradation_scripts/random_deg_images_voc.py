"""
Shuffles the test directory of images and puts 1024
random test images into the test_degradation_voc file

"""

import os
import shutil
import random
import argparse
import pandas as pd

random.seed(1)

inp_d = r'/home/mseals1/Documents/pytorch-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt'

d = []

with open(inp_d, 'r') as f:
    for i in f.readlines():
        i = i[:-1] + '.jpg'
        i = os.path.join(r'/home/mseals1/Documents/pytorch-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/', i)
        d.append(i)

random.shuffle(d)

dlen = len(d)

testset = d[:1024]
te_dst = r'/home/mseals1/Documents/pytorch-faster-rcnn/data/test_deg_voc'

for f in testset:
    shutil.copy(f, te_dst)
