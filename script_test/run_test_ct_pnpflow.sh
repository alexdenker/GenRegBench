#!/bin/bash

sigma_n=0.01
part="test"
dataset_name=walnut
task=tomography_sparseview

trained_model=walnut

# for the varying angles figure
gamma=10000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 8 --sigma_n $sigma_n

# for the varying angles figure and table
gamma=1000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 16 --sigma_n $sigma_n

gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 32 --sigma_n $sigma_n

gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 64 --sigma_n $sigma_n

gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n

# varying noise level

sigma_n=0.005
gamma=1000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n

sigma_n=0.002
gamma=5000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n

sigma_n=0.001
gamma=20000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n

sigma_n=0
gamma=100000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n

# ood settings
sigma_n=0.01
trained_model=aapm

gamma=2000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 32 --sigma_n $sigma_n

gamma=1000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n


trained_model=diskellipses

gamma=2000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 32 --sigma_n $sigma_n

gamma=1000
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n


trained_model=celebahq

gamma=200
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 32 --sigma_n $sigma_n

gamma=200
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n

# misaligned angles

trainde_model=walnut
gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 32 --sigma_n $sigma_n --misaligned_angles

gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n --misaligned_angles

# misaligned noise

gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 32 --sigma_n $sigma_n --misaligned_noise

gamma=500
python main_pnpflow.py --trained_model $trained_model --dataset_name $dataset_name --part $part --task $task --alpha 1.0 --gamma $gamma --num_angles 128 --sigma_n $sigma_n --misaligned_noise