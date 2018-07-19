# read csv for each parameter and create the graph of the
# ap scores for each parameter for each class

# 9 graphs (1 for each functions), 20 lines, 1 for each class, over all 12 parameters
# fixing the graphs, making them prettier

# FIXED RANGES


import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import glob
import natsort
import seaborn as sns
import numpy as np
import itertools

NUMBER_OF_CLASSES = 20

parser = argparse.ArgumentParser(description='Translates CSV into XML')
parser.add_argument('inp_file', help='CSV input file')
args = parser.parse_args()

inp_d = args.inp_file

# Color range
r_col = np.linspace(0, 1, num=4, endpoint=False)
r_col2 = np.geomspace(1, 10, num=8, endpoint=False)
r_col = (np.concatenate((r_col, r_col2))*100).round(decimals=0).astype(int)

# Contrast range
r_con = np.linspace(0.05, 1, num=6, endpoint=False)
r_con2 = np.geomspace(1, 20, num=5)
r_con = (np.concatenate(([0], r_con, r_con2))*100).round(decimals=0).astype(int)

# Brightness range
r_br = np.geomspace(0.1, 1, num=6, endpoint=False)
r_br2 = np.geomspace(1, 30, num=5)
r_br = (np.concatenate(([0], r_br, r_br2))*100).round(decimals=0).astype(int)

# Sharpness range
r_sh = [i * 34 for i in range(0, 12)]
r_sh = np.array(r_sh).round(decimals=0).astype(int)

# Gaussian Blur range
r_gblur = np.linspace(0.5, 2.5, num=4, endpoint=False)
r_gblur2 = np.linspace(3, 9, num=7)
r_gblur = np.concatenate(([0], r_gblur, r_gblur2)).round(decimals=2)

# Imresize range
r_imr = np.geomspace(0.04, 1, num=12)
r_imr = r_imr[::-1]
r_imr = (np.array(r_imr)*100).round(decimals=0).astype(int)

# Gaussian Noise range
r_gau = np.geomspace(0.01, 2.56, num=11)
r_gau = np.concatenate(([0], r_gau)).round(decimals=2)

# Salt and Pepper range
r_snp = np.linspace(0, 0.5, num=12)
r_snp = (r_snp*100).round(decimals=0).astype(int)

# Speckle Noise range
r_spe = np.geomspace(0.03, 15, num=11)
r_spe = np.concatenate(([0], r_spe)).round(decimals=2)

all_xs = []
all_yvals = []
all_ylabels = []
methods = {'brightness': r_br, 'color': r_col, 'contrast': r_con,
           'gblur': r_gblur, 'gnoise': r_gau, 'resize': r_imr,
           'sharpness': r_sh, 'snoise': r_spe, 'snpnoise': r_snp}
xlbls = {'brightness': 'Ratio (percentage)', 'color': 'Ratio (percentage)', 'contrast': 'Ratio (percentage)',
         'gblur': 'Blur Radius (pixels)', 'gnoise': 'Variance', 'resize': 'Ratio (percentage)',
         'sharpness': 'Ratio (percentage)', 'snoise': 'Variance', 'snpnoise': 'Ratio (percentage)'}

for dirs, subdirs, files in os.walk(inp_d):
    m = str(dirs.split('\\')[-1])
    if m in methods.keys():
        fs = glob.glob(os.path.join(dirs, "*.csv"))
        if not fs:
            continue
        fs = natsort.natsorted(fs)

        df_from_each_file = (pd.read_csv(f) for f in fs)
        df = pd.concat(df_from_each_file, ignore_index=True)

        xs = []
        yvals = []
        ylabels = []

        g1 = df.groupby(['parameter'], sort=False)

        for _, g in g1:
            param = g['parameter'].iloc[0]
            g2 = g.groupby(['class']).mean()
            g2 = g2.drop(labels=['parameter', 'maxIoU', 'mAP'], axis=1)

            xs.append(param)
            yvals.append(g2.values)
            ylabels.append(g2.index)

        xs = np.array(xs)
        yvals = np.squeeze(np.array(yvals))
        ylabels = np.array(ylabels[0])

        all_xs.append(xs)
        all_yvals.append(yvals)
        all_ylabels.append(ylabels)
    else:
        continue

all_xs = np.array(all_xs)
all_xs = np.asarray(all_xs[0], dtype=int)
all_yvals = np.array(all_yvals)
all_ylabels = np.array(all_ylabels)[0]

colors = ['b', 'g', 'y', 'c', 'm']
linestyles = [':', '-.', '--', '-']
color_and_linestyle = []

# print(all_xs.shape)
# print(all_yvals.shape)
# print(all_ylabels)

for i in itertools.product(linestyles, colors):
    color_and_linestyle.append(i[::-1])

color_and_linestyle = np.array(color_and_linestyle)

for e, m in enumerate(methods.keys()):
    for i in range(NUMBER_OF_CLASSES):
        plt.plot(all_xs, all_yvals[e, :, i], color=color_and_linestyle[i][0], linestyle=color_and_linestyle[i][1])

        plt.xlabel(xlbls[m], fontsize=18, weight='bold')
        plt.ylabel('AP score', fontsize=18, weight='bold')
        if m == 'color':
            plt.title('Effect of the ' + 'Saturation' + ' on Model Performance', fontsize=28)
        elif m == 'gblur':
            plt.title('Effect of the ' + 'Gaussian Blur' + ' on Model Performance', fontsize=28)
        elif m == 'gnoise':
            plt.title('Effect of the ' + 'Gaussian Noise' + ' on Model Performance', fontsize=28)
            plt.xscale('log')
        elif m == 'snoise':
            plt.title('Effect of the ' + 'Speckle Noise' + ' on Model Performance', fontsize=28)
            plt.xscale('log')
        elif m == 'snpnoise':
            plt.title('Effect of the ' + 'Salt and Pepper Noise' + ' on Model Performance', fontsize=28)
        elif m == 'resize':
            plt.title('Effect of the ' + 'Image Size' + ' on Model Performance', fontsize=28)
        elif m == 'brightness':
            plt.title('Effect of the ' + 'Brightness' + ' on Model Performance', fontsize=28)
            plt.xscale('log')
        elif m == 'contrast':
            plt.title('Effect of the ' + 'Contrast' + ' on Model Performance', fontsize=28)
            plt.xscale('log')
        else:
            plt.title('Effect of the ' + m.capitalize() + ' on Model Performance', fontsize=28)
        # plt.axis(([0, 13, 0, 1]))
        plt.ylim([0, 1])
        # plt.xticks(all_xs, methods[m], fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(all_ylabels, fontsize=15, frameon=False)

    plt.show()









