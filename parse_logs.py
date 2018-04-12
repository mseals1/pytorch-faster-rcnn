import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Translates CSV into XML')
parser.add_argument('inp_file', help='CSV input file')
args = parser.parse_args()

inp_fn = args.inp_file

iters = []
losses = []
rpn_cls = []
rpn_box = []
loss_cls = []
loss_box = []

with open(inp_fn, 'r') as f:
    lines = f.readlines()
    for l in lines:
        if 'iter: ' in l:
            losses.append(float(l[-9:]))
            i = l.split('/')[0]
            iters.append(int(i[6:]))
        if ' >>> rpn_loss_cls:' in l:
            rpn_cls.append(float(l[-9:]))
        if ' >>> rpn_loss_box:' in l:
            rpn_box.append(float(l[-9:]))
        if ' >>> loss_cls:' in l:
            loss_cls.append(float(l[-9:]))
        if ' >>> loss_box:' in l:
            loss_box.append(float(l[-9:]))

# plt.figure(figsize=(2, 3), dpi=500)

# print(len(iters))
plt.subplot(2, 3, 1)
plt.plot(iters, losses)
plt.title('losses')
plt.subplot(2, 3, 2)
plt.plot(iters, rpn_cls)
plt.title('rpn_loss_cls')
plt.subplot(2, 3, 3)
plt.plot(iters, rpn_box)
plt.title('rpn_loss_box')
plt.subplot(2, 3, 4)
plt.plot(iters, loss_cls)
plt.title('loss_cls')
plt.subplot(2, 3, 5)
plt.plot(iters, loss_box)
plt.title('loss_box')
plt.show()
