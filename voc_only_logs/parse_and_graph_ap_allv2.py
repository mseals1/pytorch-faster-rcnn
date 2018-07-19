# read csv for each parameter and create the graph of the
# mean of all of the mean maxIoU

# 1 graph, 12 bars, one for each parameter

# ERROR BARS on a bar graph
# one graph for each method (9 methods)
# making the graphs have the original scale (was linear scale no matter the values)

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
# r_gblur = [i for i in range(0, 12)]
r_gblur = np.linspace(0.5, 2.5, num=4, endpoint=False)
r_gblur2 = np.linspace(3, 9, num=7)
r_gblur = np.concatenate(([0], r_gblur, r_gblur2)).round(decimals=2)

# Imresize range
r_imr = np.geomspace(0.04, 1, num=12)
r_imr = r_imr[::-1]
r_imr = (np.array(r_imr)*100).round(decimals=0).astype(int)

# Gaussian Noise range
r_gau = np.geomspace(0.01, 10.24, num=11)
r_gau = np.concatenate(([0], r_gau)).round(decimals=2)

# Salt and Pepper range
r_snp = np.linspace(0.03, 0.5, num=6, endpoint=False)
r_snp2 = np.linspace(0.5, 1, num=5)
r_snp = (np.concatenate(([0], r_snp, r_snp2))*100).round(decimals=0).astype(int)

# Speckle Noise range
r_spe = np.geomspace(0.03, 30, num=11)
r_spe = np.concatenate(([0], r_spe)).round(decimals=2)

all_xs = []
all_yvals = []
methods = {'brightness': r_br, 'color': r_col, 'contrast': r_con,
           'gblur': r_gblur, 'gnoise': r_gau, 'resize': r_imr,
           'sharpness': r_sh, 'snoise': r_spe, 'snpnoise': r_snp}
xlbls = {'brightness': 'Ratio (percentage)', 'color': 'Ratio (percentage)', 'contrast': 'Ratio (percentage)',
         'gblur': 'Blur Radius (pixels)', 'gnoise': 'Variance', 'resize': 'Ratio (percentage)',
         'sharpness': 'Ratio (percentage)', 'snoise': 'Variance', 'snpnoise': 'Ratio (percentage)'}
yerrs = []

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
        yerr = []

        g1 = df.groupby(['parameter'], sort=False)

        for _, g in g1:
            param = g['parameter'].iloc[0]
            g2 = g.groupby(['class']).mean()
            g2 = g2.drop(labels=['parameter', 'maxIoU', 'mAP'], axis=1)

            xs.append(param)
            yvals.append(g2.values)
            yerr.append(g2.values.std())

        xs = np.array(xs)
        yvals = np.squeeze(np.array(yvals))
        yvals = yvals.mean(axis=1)

        plt.errorbar(methods[m], yvals, yerr=yerr, fmt='-o', capsize=4, color='k',
                     ecolor='r', markerfacecolor='k')
        plt.xlabel(xlbls[m], fontsize=18)
        plt.ylabel('mAP score', fontsize=18)
        if m == 'color':
            plt.title('Effect of the ' + 'Saturation' + ' on Model Performance', fontsize=28)
        elif m == 'gblur':
            plt.title('Effect of the ' + 'Gaussian Blur' + ' on Model Performance', fontsize=28)
        elif m == 'gnoise':
            plt.title('Effect of the ' + 'Gaussian Noise' + ' on Model Performance', fontsize=28)
        elif m == 'snoise':
            plt.title('Effect of the ' + 'Speckle Noise' + ' on Model Performance', fontsize=28)
        elif m == 'snpnoise':
            plt.title('Effect of the ' + 'Salt and Pepper Noise' + ' on Model Performance', fontsize=28)
        elif m == 'resize':
            plt.title('Effect of ' + 'Image Resizing' + ' on Model Performance', fontsize=28)
        else:
            plt.title('Effect of the ' + m.capitalize() + ' on Model Performance', fontsize=28)
        # plt.axis(([-1, 12, 0, 1]))
        plt.ylim([0, 1])
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(True, axis='y')
        plt.show()
    else:
        continue
