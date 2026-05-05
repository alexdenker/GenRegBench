#!/bin/bash

### RED-diff, CelebaHQ to CelebaHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=inpainting 
obs_weight=0.5
grad_term_weight=0.25
lr=0.01
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=deblurring 
obs_weight=0.5
grad_term_weight=0.25
lr=0.05
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=2
obs_weight=1.0
grad_term_weight=0.25
lr=0.01
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --scale_factor $scale_factor --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=4
obs_weight=2.0
grad_term_weight=0.1
lr=0.01
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --scale_factor $scale_factor --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


### RED-diff, CelebaHQ to FFHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=inpainting 
obs_weight=1.0
grad_term_weight=0.25
lr=0.01
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=deblurring 
obs_weight=1.0
grad_term_weight=0.25
lr=0.05
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=2
obs_weight=1.0
grad_term_weight=0.25
lr=0.01
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --scale_factor $scale_factor --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=4
obs_weight=2.0
grad_term_weight=0.1
lr=0.01
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --scale_factor $scale_factor --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


### RED-diff, CelebaHQ to AFHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=inpainting 
obs_weight=0.5
grad_term_weight=0.25
lr=0.01
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=deblurring 
obs_weight=1.0
grad_term_weight=0.25
lr=0.05
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=2
obs_weight=1.0
grad_term_weight=0.25
lr=0.01
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --scale_factor $scale_factor --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=4
obs_weight=2.0
grad_term_weight=0.1
lr=0.01
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --scale_factor $scale_factor --task $task --part $part --method reddiff --dataset_name $dataset --sigma_n $sigma_n --grad_term_weight $grad_term_weight --num_steps 1000 --obs_weight $obs_weight --lr $lr 
