#!/bin/bash
trained_model=CBSD

# celebahq

sigma_n=0.05
part="test"
task=inpainting
dataset_name=celebahq
lam=0.075
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
task=deblurring
dataset_name=celebahq
lam=0.05
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
task=super_resolution
scale_factor=2
dataset_name=celebahq
lam=0.05
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor


sigma_n=0.05
part="test"
task=super_resolution
scale_factor=4
dataset_name=celebahq
lam=0.025
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor

# afhq
sigma_n=0.05
part="test"
task=inpainting
dataset_name=afhq
lam=0.075
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
task=deblurring
dataset_name=afhq
lam=0.05
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
task=super_resolution
scale_factor=2
dataset_name=afhq
lam=0.05
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor


sigma_n=0.05
part="test"
task=super_resolution
scale_factor=4
dataset_name=afhq
lam=0.025
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor


# ffhq

sigma_n=0.05
part="test"
task=inpainting
dataset_name=ffhq
lam=0.075
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
task=deblurring
dataset_name=ffhq
lam=0.05
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
task=super_resolution
scale_factor=2
dataset_name=ffhq
lam=0.05
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor


sigma_n=0.05
part="test"
task=super_resolution
scale_factor=4
dataset_name=ffhq
lam=0.025
python main_learned_reg.py --trained_model $trained_model --model_type=LSR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor
