# read csv for each parameter and create the graph of the
# mean/max maxIoUs for each parameter for each class

import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import glob
import natsort


parser = argparse.ArgumentParser(description='Translates CSV into XML')
parser.add_argument('inp_file', help='CSV input file')
args = parser.parse_args()

inp_d = args.inp_file

files = glob.glob(os.path.join(inp_d, "*.csv"))

files = natsort.natsorted(files)

df_from_each_file = (pd.read_csv(f) for f in files)
df = pd.concat(df_from_each_file, ignore_index=True)

# print(df)

g1 = df.groupby(['parameter'])

for _, g in g1:
    print(g.drop_duplicates(subset=['parameter'])['parameter'].iloc[0])
    g2 = g.groupby(['class']).mean()
    print(g2)
