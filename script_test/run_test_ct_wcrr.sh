#!/bin/bash

sigma_n=0.01
part="test"
dataset_name=walnut
task=tomography_sparseview

trained_model=walnut

lam=0.01
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 16 

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 32 

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 64 

lam=0.005
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --num_angles 128

