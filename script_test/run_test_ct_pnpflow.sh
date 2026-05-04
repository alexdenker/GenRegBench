#!/bin/bash

sigma_n=0.01
part="test"
dataset_name=walnut
task=tomography_sparseview

trained_model=walnut

gamma=1000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 16 --sigma_n $sigma_n

gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 32 --sigma_n $sigma_n

gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 64 --sigma_n $sigma_n

gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n
