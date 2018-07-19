#!/bin/bash

for ((i=0; i<3; i++)); do
    for ((j=0; j<12; j++)); do
		if(($i == 0)); then
			sed "s/$/_gnoise_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 1)); then
			sed "s/$/_snpnoise_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 2)); then
			sed "s/$/_snoise_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
    done
done

