# parses the txt files and writes csv files from the given dir of text files

import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import glob
import natsort
import shutil


def parse(inputfn):
    clsns = []
    imns = []
    ious = []
    avgps = {}
    meanap = 0

    classes = [' aeroplane', ' bicycle', ' bird', ' boat',
               ' bottle', ' bus', ' car', ' cat', ' chair',
               ' cow', ' diningtable', ' dog', ' horse',
               ' motorbike', ' person', ' pottedplant',
               ' sheep', ' sofa', ' train', ' tvmonitor']

    with open(inputfn, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if 'Mean AP = ' in l:
                meanap = l.split(" ")[-1][:-1]

            if 'AP for ' in l:
                c = l.split(" ")[2]
                ap = l.split(" ")[-1][:-1]
                avgps[c] = ap

            if any(l[:-1] == cls for cls in classes):
                clsns.append(lines[i][1:-1])
                if 'im_name = ' in lines[i + 1]:
                    imns.append(lines[i + 1].split(" ")[2][:-1])
                if 'max IoU = ' in lines[i + 2]:
                    ious.append(float(lines[i + 2].split(" ")[3][:-1]))

    return clsns, imns, ious, avgps, meanap


def csvwriter(inputdir, clsns, imns, ious, ap, meanap):
    csvfn = inputdir[:-4] + '.csv'

    with open(csvfn, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'parameter', 'class', 'maxIoU', 'AP', 'mAP'])

        for i in range(len(clsns)):
            spl = imns[i].split("_")
            if "r" in spl[-1]:
                writer.writerow([imns[i], spl[-1][1:], clsns[i], ious[i], ap[clsns[i]], meanap])
            else:
                writer.writerow([imns[i], spl[-1], clsns[i], ious[i], ap[clsns[i]], meanap])

    return csvfn


parser = argparse.ArgumentParser(description='Translates CSV into XML')
parser.add_argument('inp_file', help='CSV input file')
args = parser.parse_args()

inp_fn = args.inp_file

files = glob.glob(os.path.join(inp_fn, "*.txt"))
files = natsort.natsorted(files)

for f in files:
    classnames, imnames, maxious, aps, meap = parse(f)

    fn = csvwriter(f, classnames, imnames, maxious, aps, meap)

files = glob.glob(os.path.join(inp_fn, "*.csv"))
files = natsort.natsorted(files)

if not os.path.exists(os.path.join(inp_fn, "brightness")):
    os.makedirs(os.path.join(inp_fn, "brightness"))
if not os.path.exists(os.path.join(inp_fn, "color")):
    os.makedirs(os.path.join(inp_fn, "color"))
if not os.path.exists(os.path.join(inp_fn, "contrast")):
    os.makedirs(os.path.join(inp_fn, "contrast"))
if not os.path.exists(os.path.join(inp_fn, "sharpness")):
    os.makedirs(os.path.join(inp_fn, "sharpness"))
if not os.path.exists(os.path.join(inp_fn, "gblur")):
    os.makedirs(os.path.join(inp_fn, "gblur"))
if not os.path.exists(os.path.join(inp_fn, "resize")):
    os.makedirs(os.path.join(inp_fn, "resize"))
if not os.path.exists(os.path.join(inp_fn, "gnoise")):
    os.makedirs(os.path.join(inp_fn, "gnoise"))
if not os.path.exists(os.path.join(inp_fn, "snpnoise")):
    os.makedirs(os.path.join(inp_fn, "snpnoise"))
if not os.path.exists(os.path.join(inp_fn, "snoise")):
    os.makedirs(os.path.join(inp_fn, "snoise"))

for f in files:
    if "brightness" in f:
        new_f = os.path.join(inp_fn, "brightness", f.split("\\")[-1])
        shutil.move(f, new_f)
    if "color" in f:
        new_f = os.path.join(inp_fn, "color", f.split("\\")[-1])
        shutil.move(f, new_f)
    if "contrast" in f:
        new_f = os.path.join(inp_fn, "contrast", f.split("\\")[-1])
        shutil.move(f, new_f)
    if "sharpness" in f:
        new_f = os.path.join(inp_fn, "sharpness", f.split("\\")[-1])
        shutil.move(f, new_f)
    if "gblur" in f:
        new_f = os.path.join(inp_fn, "gblur", f.split("\\")[-1])
        shutil.move(f, new_f)
    if "resize" in f:
        new_f = os.path.join(inp_fn, "resize", f.split("\\")[-1])
        shutil.move(f, new_f)
    if "gnoise" in f:
        new_f = os.path.join(inp_fn, "gnoise", f.split("\\")[-1])
        shutil.move(f, new_f)
    if "snpnoise" in f:
        new_f = os.path.join(inp_fn, "snpnoise", f.split("\\")[-1])
        shutil.move(f, new_f)
    if "snoise" in f:
        new_f = os.path.join(inp_fn, "snoise", f.split("\\")[-1])
        shutil.move(f, new_f)
