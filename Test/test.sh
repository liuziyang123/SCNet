#!/bin/bash

echo 'Data path is: '$1
echo 'Save path is: '$2
echo 'Ground truth path is: '$3

python Test/test.py --data_path $1 --save_path Saved/$2 #--directly_model pretrained_model/depth_completion_KITTI.tar


# Arguments for evaluate_depth file: 
# - ground truth directory
# - results directory

Test/devkit/cpp/evaluate_depth $3 Saved/$2/results_test
