#!/bin/bash

for ((i=2; i<=14; i++)); do
   sed "s/$/_gblur_r"$i"/" data/test_deg_voc_1024/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
   ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
done

