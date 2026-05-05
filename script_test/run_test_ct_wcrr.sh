#!/bin/bash

sigma_n=0.01
part="test"
dataset_name=walnut
task=tomography_sparseview

trained_model=walnut

# for the varying angles figure
lam=0.05
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 8 

# for teh varying angles figure and table
lam=0.01
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 16 

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 32 

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 64 

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128

# for the varying noise level table
sigma_n=0.005
lam=0.002
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128

sigma_n=0.002
lam=0.00075
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128

sigma_n=0.001
lam=0.0002
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128

sigma_n=0
lam=0.00002
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128

# ood results
sigma_n=0.01
trained_model=aapm

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 32 

lam=0.01
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128

trained_model=diskellipses

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 32 

lam=0.01
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128

trained_model=celebahq

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 32 

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128

# misaligned angles
trained_model=walnut

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 32 --misaligned_angles

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128 --misaligned_angles

# misaligned noise

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 32 --misaligned_noise

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128 --misaligned_noise