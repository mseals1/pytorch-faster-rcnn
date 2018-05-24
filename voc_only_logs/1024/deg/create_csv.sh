#!/bin/bash

for ((i=1; i<=14; i++)); do
    python3 voc_only_logs/parse_and_graph.py voc_only_logs/1024/deg/voc_gblur_r"$i"_test_log.txt
done

