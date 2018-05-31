# writes csv files from the given dir of text files

import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
import glob
import natsort


def parse(inputfn):
    clsns = []
    imns = []
    ious = []
    # recs = []
    # precs = []
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
                # print(lines[i:i+7])

                clsns.append(lines[i][1:-1])
                if 'im_name = ' in lines[i + 1]:
                    imns.append(lines[i + 1].split(" ")[2][:-1])
                if 'max IoU = ' in lines[i + 2]:
                    ious.append(float(lines[i + 2].split(" ")[3][:-1]))
                # if 'confidence = ' in lines[i + 3]:
                #     confs.append(float(lines[i + 3].split(" ")[2][:-1]))
                # if 'recall = ' in lines[i + 4]:
                #     arr1 = lines[i + 4][11:-2]
                #     arr = []
                #     for a in arr1.split():
                #         arr.append(float(a))
                #     recs.append(arr)
                # else:
                #     recs.append([])
                # if 'precision = ' in lines[i + 5]:
                #     arr2 = lines[i + 5][13:-4]
                #     arr = []
                #     for a in arr2.split():
                #         arr.append(float(a))
                #     precs.append(arr)
                # else:
                #     precs.append([])

    return clsns, imns, ious, avgps, meanap


def csvwriter(inputdir, clsns, imns, ious, ap, meanap):
    csvfn = inputdir[:-4] + '.csv'

    with open(csvfn, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'parameter', 'class', 'maxIoU', 'AP', 'mAP'])

        for i in range(len(clsns)):
            writer.writerow([imns[i], imns[i].split("_")[-1], clsns[i], ious[i], ap[clsns[i]], meanap])

    return csvfn


parser = argparse.ArgumentParser(description='Translates CSV into XML')
parser.add_argument('inp_file', help='CSV input file')
args = parser.parse_args()

inp_fn = args.inp_file

files = glob.glob(os.path.join(inp_fn, "*.txt"))

files = natsort.natsorted(files)

for f in files:
    classnames, imnames, maxious, aps, meap = parse(f)

    # print(classnames, '\n', imnames, '\n', maxious, '\n', aps, '\n', meap)

    fn = csvwriter(f, classnames, imnames, maxious, aps, meap)

# exit()

# df = pd.read_csv(fn)
#
# df_ap = df.drop_duplicates(subset='AP')
# df_ap = df_ap.drop(labels=['image', 'maxIoU', 'mAP'], axis=1)
#
# df_iou = df.drop(labels=['AP', 'mAP'], axis=1)
# boxplot = df_iou.boxplot(by=['class'])
#
# plt.show()
