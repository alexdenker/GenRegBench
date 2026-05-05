#!/bin/bash


sigma_n=0.01
dataset="walnut"
task="tomography_sparseview"
trained_model=BSD

# for the varying angles figure
num_angles=8
lam=1.0
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n

# for the varying angles figure and table
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

# varying noise level

num_angles=128
sigma_n=0.005
lam=0.002
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n

sigma_n=0.002
lam=0.00075
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n

sigma_n=0.001
lam=0.0002
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n

sigma_n=0
lam=2e-5
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n

# ood not relevant (not dataset specific)

# misaligned angles

num_angles=32
lam=0.01
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n --misaligned_angles

num_angles=128
lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n --misaligned_angles

# misaligned noise

num_angles=32
lam=0.01
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n --misaligned_noise

num_angles=128
lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type LSR --dataset_name $dataset --part test --task $task --lmbd $lam --num_angles $num_angles --sigma_n $sigma_n --misaligned_noise