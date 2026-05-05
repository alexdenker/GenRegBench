#!/bin/bash

### DPS, CelebAHQ to CelebAHQ 
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=2
grad_coeff=4.0
echo "Running main.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000 --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=4
grad_coeff=4.0
echo "Running main.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000 --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=inpainting
grad_coeff=4.0
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=deblurring
grad_coeff=4.0
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000


# DPS, CelebAHQ to AFHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=inpainting
grad_coeff=4.0
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=deblurring
grad_coeff=5.0
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=2
grad_coeff=4.5
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000 --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=4
grad_coeff=2.0
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000 --scale_factor $scale_factor


# DPS, CelebAHQ to FFHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=inpainting
grad_coeff=4.5
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=deblurring
grad_coeff=6.0
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=2
grad_coeff=4.5
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000 --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=4
grad_coeff=4.5
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dps --dataset_name $dataset --sigma_n $sigma_n --grad_coeff $grad_coeff --num_steps 1000 --scale_factor $scale_factor
