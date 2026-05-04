#!/bin/bash


### reddiff, celebahq, val set
sigma_n=0.01
dataset="walnut"
task="tomography_sparseview"

alpha=0.0005
python main_tv.py --dataset_name $dataset --part test --task $task --method tv --alpha $alpha --num_angles 16 --sigma_n $sigma_n

alpha=0.0005
python main_tv.py --dataset_name $dataset --part test --task $task --method tv --alpha $alpha --num_angles 32 --sigma_n $sigma_n

alpha=0.0005
python main_tv.py --dataset_name $dataset --part test --task $task --method tv --alpha $alpha --num_angles 64 --sigma_n $sigma_n

alpha=0.0005
python main_tv.py --dataset_name $dataset --part test --task $task --method tv --alpha $alpha --num_angles 128 --sigma_n $sigma_n


sigma_n=0.005
alpha=0.0002
python main_tv.py --dataset_name $dataset --part test --task $task --method tv --alpha $alpha --num_angles 128 --sigma_n $sigma_n

sigma_n=0.002
alpha=5e-5
python main_tv.py --dataset_name $dataset --part test --task $task --method tv --alpha $alpha --num_angles 128 --sigma_n $sigma_n

sigma_n=0.001
alpha=2e-5
python main_tv.py --dataset_name $dataset --part test --task $task --method tv --alpha $alpha --num_angles 128 --sigma_n $sigma_n

sigma_n=0
alpha=5e-6
python main_tv.py --dataset_name $dataset --part test --task $task --method tv --alpha $alpha --num_angles 128 --sigma_n $sigma_n

