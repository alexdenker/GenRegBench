#!/bin/bash

### DMPlug, CelebAHQ to CelebAHQ 
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=2
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=4
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=inpainting
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=deblurring
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps


# # DMPlug, CelebAHQ to AFHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=inpainting
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=deblurring
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=2
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=4
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps --scale_factor $scale_factor


# # DMPlug, CelebAHQ to FFHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=inpainting
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=deblurring
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps


model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=2
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=4
adam_steps=1500
num_steps=4
echo "Running main.py for model $model_id, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method dmplug --dataset_name $dataset --sigma_n $sigma_n  --num_steps $num_steps --adam_steps $adam_steps --scale_factor $scale_factor
