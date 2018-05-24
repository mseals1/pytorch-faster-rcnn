#!/bin/bash

for ((i=512; i>=64; i/=2)); do
    for ((j=1; j<=14; j++)); do
        sed "s/$/_gblur_r"$j"/" data/test_deg_voc_$i/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
        ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
    done
done

