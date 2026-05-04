#!/bin/bash


sigma_n=0.01
dataset="walnut"
task="tomography_sparseview"


num_angles=16
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n

num_angles=32
lam=0.01
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n

num_angles=64
lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n

num_angles=128
lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n
