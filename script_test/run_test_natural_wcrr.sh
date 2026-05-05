#!/bin/bash


### RUN WCRR, CelebaHQ to CelebaHQ
sigma_n=0.05
part="test"
trained_model=celebahq
task=inpainting
dataset_name=celebahq
lam=0.3
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
trained_model=celebahq
task=deblurring
dataset_name=celebahq
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
trained_model=celebahq
task=super_resolution
scale_factor=2
dataset_name=celebahq
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor


sigma_n=0.05
part="test"
trained_model=celebahq
task=super_resolution
scale_factor=4
dataset_name=celebahq
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor


### RUN WCRR, CelebaHQ to AFHQ
sigma_n=0.05
part="test"
trained_model=celebahq
task=inpainting
dataset_name=afhq
lam=0.3
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
trained_model=celebahq
task=deblurring
dataset_name=afhq
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
trained_model=celebahq
task=super_resolution
scale_factor=2
dataset_name=afhq
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor


sigma_n=0.05
part="test"
trained_model=celebahq
task=super_resolution
scale_factor=4
dataset_name=afhq
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor


### RUN WCRR, CelebaHQ to FFHQ
sigma_n=0.05
part="test"
trained_model=celebahq
task=inpainting
dataset_name=ffhq
lam=0.2
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
trained_model=celebahq
task=deblurring
dataset_name=ffhq
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam


sigma_n=0.05
part="test"
trained_model=celebahq
task=super_resolution
scale_factor=2
dataset_name=ffhq
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor


sigma_n=0.05
part="test"
trained_model=celebahq
task=super_resolution
scale_factor=4
dataset_name=ffhq
lam=0.1
python main_learned_reg.py --trained_model $trained_model --model_type=WCRR --dataset_name $dataset_name --part $part --task $task --sigma_n $sigma_n --lmbd $lam --scale_factor $scale_factor
