# read csv for each parameter and create the graph of the
# mean maxIoUs for each parameter for each class

# 1 graph, 20 lines, one for each class, over all 15 parameters (0 - 14)

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

files = glob.glob(os.path.join(inp_d, "*.csv"))

files = natsort.natsorted(files)

df_from_each_file = (pd.read_csv(f) for f in files)
df = pd.concat(df_from_each_file, ignore_index=True)

xs = []
yvals = []
ylabels = []

g1 = df.groupby(['parameter'], sort=False)

for _, g in g1:
    param = g['parameter'].iloc[0]
    g2 = g.groupby(['class']).mean()
    g2 = g2.drop(labels=['parameter', 'AP', 'mAP'], axis=1)

    xs.append(param)
    yvals.append(g2.values)
    ylabels.append(g2.index)

xs = np.array(xs)
# print(xs.shape)
yvals = np.squeeze(np.array(yvals))
# print(yvals.shape)
ylabels = np.array(ylabels[0])
# print(ylabels.shape)
r_imr = np.geomspace(0.04, 1, num=12)
r_imr = r_imr[::-1]
r_imr = (np.array(r_imr)*100).round(decimals=2)
# exit()

cmap = plt.cm.tab20((np.arange(20)).astype(int))
plt.gca().set_color_cycle(cmap)

plt.plot(xs, yvals)

plt.xlabel('Ratio (percentage)', fontsize=18)
plt.ylabel('Mean MaxIoU', fontsize=18)
plt.title('Effect of Image Resizing on Model Performance by class', fontsize=28)
plt.axis(([0, 13, 0, 1]))
plt.xticks(xs, r_imr, fontsize=14)
plt.yticks(fontsize=14)
plt.grid(True)
plt.legend(ylabels, fontsize=14)

plt.show()













