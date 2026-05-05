#!/bin/bash


### DiffPIR, CelebaHQ to CelebaHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=inpainting 
lam=10.0
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100 

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=deblurring 
lam=5.0
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=2 
lam=5.0
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100 --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=4
lam=2.5
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100 --scale_factor $scale_factor

### DiffPIR, CelebaHQ to AFHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=inpainting 
lam=5.0
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100 

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=deblurring 
lam=5.0
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=2 
lam=5.0
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100 --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=4
lam=5.0
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100 --scale_factor $scale_factor

# ### DiffPIR, CelebaHQ to FFHQ
model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=inpainting 
lam=5.0
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100 

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=deblurring 
lam=5.0
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task, on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=2 
lam=2.5
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100 --scale_factor $scale_factor

model_id="google/ddpm-ema-celebahq-256"
sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=4
lam=2.5
zeta=0.7
echo "Running main_natural_images.py for model $model_id, task $task (scale_factor=$scale_factor), on dataset $dataset"
python main_natural_images.py --model_id $model_id --task $task --part $part --method diffpir --dataset_name $dataset --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 100 --scale_factor $scale_factor
