#!/bin/bash

for ((i=0; i<9; i++)); do
    for ((j=0; j<12; j++)); do
        if(($i == 0)); then
			sed "s/$/_color_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 1)); then
			sed "s/$/_contrast_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 2)); then
			sed "s/$/_brightness_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 3)); then
			sed "s/$/_sharpness_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 4)); then
			sed "s/$/_gblur_r"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 5)); then
			sed "s/$/_resize_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 6)); then
			sed "s/$/_gnoise_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 7)); then
			sed "s/$/_snpnoise_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
		if(($i == 8)); then
			sed "s/$/_snoise_"$j"/" data/test_deg_voc_512/test.txt > data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt
			./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc vgg16
    done
done

