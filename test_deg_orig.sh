#!/bin/bash

for ((i=512; i>=64; i/=2)); do
    cp data/test_deg_voc_$i/test.txt data/VOCdevkit2007/VOC2007/ImageSets/Main/
    ./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
done

