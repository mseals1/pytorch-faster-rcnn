import argparse
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv


def parse(inputfn):
    clsns = []
    imns = []
    ious = []
    confs = []
    recs = []
    precs = []
    aps = {}

    classes = [' aeroplane', ' bicycle', ' bird', ' boat',
               ' bottle', ' bus', ' car', ' cat', ' chair',
               ' cow', ' diningtable', ' dog', ' horse',
               ' motorbike', ' person', ' pottedplant',
               ' sheep', ' sofa', ' train', ' tvmonitor']

    with open(inputfn, 'r') as f:
        lines = f.readlines()
        for i, l in enumerate(lines):
            if any(l[:-1] == cls for cls in classes):
                # print(lines[i:i+7])

                clsns.append(lines[i][1:-1])
                if 'im_name = ' in lines[i + 1]:
                    imns.append(lines[i + 1].split(" ")[2][:-1])
                if 'max IoU = ' in lines[i + 2]:
                    ious.append(float(lines[i + 2].split(" ")[3][:-1]))
                if 'confidence = ' in lines[i + 3]:
                    confs.append(float(lines[i + 3].split(" ")[2][:-1]))
                if 'recall = ' in lines[i + 4]:
                    arr1 = lines[i + 4][11:-2]
                    arr = []
                    for a in arr1.split():
                        arr.append(float(a))
                    recs.append(arr)
                else:
                    recs.append([])
                if 'precision = ' in lines[i + 5]:
                    arr2 = lines[i + 5][13:-4]
                    arr = []
                    for a in arr2.split():
                        arr.append(float(a))
                    precs.append(arr)
                else:
                    precs.append([])

    return clsns, imns, ious, confs, recs, precs


def csvwriter(clsns, imns, ious, confs, recs, precs):
    with open(r'/home/mseals1/Documents/aphylla/logs/voc_only/log.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image', 'class', 'maxIoU', 'confidence', 'recall', 'precision'])

        for i in range(len(clsns)):
            writer.writerow([imns[i], clsns[i], ious[i], confs[i], recs[i], precs[i]])


parser = argparse.ArgumentParser(description='Translates CSV into XML')
parser.add_argument('inp_file', help='CSV input file')
args = parser.parse_args()

inp_fn = args.inp_file

classnames, imnames, maxious, confidences, recalls, precisions = parse(inp_fn)

# print(classnames, '\n', imnames, '\n', maxious, '\n', confidences, '\n', recalls, '\n', precisions)

csvwriter(classnames, imnames, maxious, confidences, recalls, precisions)

exit()

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
