# read csv for each parameter and create the graph of the
# mean of all of the mean maxIoU lines

# 1 graph, 1 line for each of the different set sizes

# ERROR BARS for the 5 different set sizes

import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import glob
import natsort
import seaborn as sns
import numpy as np

parser = argparse.ArgumentParser(description='Translates CSV into XML')
parser.add_argument('inp_file', help='CSV input file')
args = parser.parse_args()

inp_d = args.inp_file

all_xs = []
all_yvals = []
all_labels = ['1024', '128', '256', '512', '64']
yerrs = []

for dirs, subdirs, files in os.walk(inp_d):

    if not dirs.split('\\')[-1] in ['512_1', '512_2', '512_3']:
        fs = glob.glob(os.path.join(dirs, "*.csv"))
        if not fs:
            continue
        fs = natsort.natsorted(fs)

        df_from_each_file = (pd.read_csv(f) for f in fs)
        df = pd.concat(df_from_each_file, ignore_index=True)

        xs = []
        yvals = []
        ylabels = []
        yerr = []

        g1 = df.groupby(['parameter'], sort=False)

        for _, g in g1:
            param = g['parameter'].iloc[0]
            g2 = g.groupby(['class']).mean()
            g2 = g2.drop(labels=['AP', 'mAP'], axis=1)

            xs.append(param[1:])
            yvals.append(g2.values)
            yerr.append(g2.values.std())

        # print(xs)
        yvals = np.squeeze(np.array(yvals))
        yvals = yvals.mean(axis=1)
        # print(yvals.shape)
        # print(ylabels.shape)

        # plt.errorbar(xs, yvals, yerr=yerr, fmt='-o', capsize=4, color='k', ecolor='r', markerfacecolor='k')
        # plt.xlabel('Radius (pixels)', fontsize=18)
        # plt.ylabel('Mean of the Mean MaxIoU', fontsize=18)
        # plt.title('Error Bars for ' + str(dirs.split("\\")[-1]), fontsize=28)
        # plt.axis(([-1, 16, 0, 1]))
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        # plt.grid(True)
        # plt.show()
        # exit()

        all_xs.append(xs)
        all_yvals.append(yvals)
        yerrs.append(yerr)
    else:
        continue

all_xs = np.array(all_xs)
all_xs = np.asarray(all_xs[0], dtype=int)

all_labels, all_yvals, yerrs = zip(*natsort.natsorted(zip(all_labels, all_yvals, yerrs), reverse=True))

all_yvals = np.array(all_yvals).transpose()
all_labels = np.array(all_labels)
yerrs = np.array(yerrs).transpose()

ax = plt.subplot(111)

ax.bar(all_xs-0.30, all_yvals[:, 0], width=0.15, color='b', align='center', yerr=yerrs[:, 0], capsize=2)
ax.bar(all_xs-0.15, all_yvals[:, 1], width=0.15, color='g', align='center', yerr=yerrs[:, 1], capsize=2)
ax.bar(all_xs-0.00, all_yvals[:, 2], width=0.15, color='r', align='center', yerr=yerrs[:, 2], capsize=2)
ax.bar(all_xs+0.15, all_yvals[:, 3], width=0.15, color='c', align='center', yerr=yerrs[:, 3], capsize=2)
ax.bar(all_xs+0.30, all_yvals[:, 4], width=0.15, color='m', align='center', yerr=yerrs[:, 4], capsize=2)

plt.xlabel('Radius (pixels)', fontsize=18)
plt.ylabel('Mean of the Mean MaxIoU', fontsize=18)
plt.title('Error Bars for all five set sizes', fontsize=28)
plt.axis(([-1, 15, 0, 1]))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(all_labels, fontsize=14)

plt.show()
