# read csv for each parameter and create the graph of the
# mean maxIoUs for each parameter for each class

# 1 graph, 20 lines, one for each class, over all 15 parameters (0 - 15)

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
    g2 = g2.drop(labels=['AP', 'mAP'], axis=1)

    xs.append(param)
    yvals.append(g2.values)
    ylabels.append(g2.index)

# print(xs)
yvals = np.squeeze(np.array(yvals))
# print(yvals)
ylabels = np.array(ylabels[0])
# print(ylabels.shape)

plt.plot(xs, yvals)

plt.xlabel('Gaussian Blur')
plt.ylabel('mean maxIoU')
plt.title('Mean maxIoU vs. Gaussian Blur radius')
plt.axis(([0, 17, 0, 1]))
plt.grid(True)
plt.legend(ylabels)

plt.show()













