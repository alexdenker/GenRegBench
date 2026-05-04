#!/bin/bash


cuda_devices=1
part="test"

##############################################
#### Walnut to Walnut (in-distribution) #####   
##############################################

method="diffpir"
task="tomography_sparseview"
model_path="saved_models/diffusers/walnut/ema_model"
num_angles=8
sigma_n=0.01
zeta=0.7
lam=0.1
echo "Running $method for model $model_path, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 1


model_path="saved_models/diffusers/walnut/ema_model"
num_angles=16
sigma_n=0.01
zeta=0.7
lam=0.5
echo "Running $method for model $model_path, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 1

model_path="saved_models/diffusers/walnut/ema_model"
num_angles=32
sigma_n=0.01
zeta=0.7
lam=1.0 
echo "Running $method for model $model_path, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 1

model_path="saved_models/diffusers/walnut/ema_model"
num_angles=64
sigma_n=0.01
zeta=0.7
lam=1.0 
echo "Running $method for model $model_path, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 1

model_path="saved_models/diffusers/walnut/ema_model"
num_angles=128
sigma_n=0.01
zeta=0.7
lam=2.0 
echo "Running $method for model $model_path, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 1



##############################################
#### AAPM to Walnut (out-distribution) #####   
##############################################

method="diffpir"
task="tomography_sparseview"

model_path="saved_models/diffusers/aapm/ema_model"
num_angles=32
sigma_n=0.01
zeta=0.7
lam=0.1
echo "Running $method for model $model_path, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 5

model_path="saved_models/diffusers/aapm/ema_model"
num_angles=128
sigma_n=0.01
zeta=0.7
lam=0.5
echo "Running $method for model $model_path, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 5


##############################################
#### Ellipses to Walnut (out-distribution) #####   
##############################################

method="diffpir"
task="tomography_sparseview"

model_path="saved_models/diffusers/diskellipses/ema_model"
num_angles=32
sigma_n=0.01
zeta=0.7
lam=0.1
echo "Running $method for model $model_path, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 5

model_path="saved_models/diffusers/diskellipses/ema_model"
num_angles=128
sigma_n=0.01
zeta=0.7
lam=0.5
echo "Running $method for model $model_path, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 5



##############################################
#### CelebA to Walnut (out-distribution) #####   
##############################################

method="diffpir"
task="tomography_sparseview"

model_id="google/ddpm-ema-celebahq-256"
num_angles=32
sigma_n=0.01
zeta=0.7
lam=5.0
echo "Running $method for model $model_id, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_id $model_id --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 5

model_id="google/ddpm-ema-celebahq-256"
num_angles=128
sigma_n=0.01
zeta=0.7
lam=7.5
echo "Running $method for model $model_id, task $task, num_angles $num_angles, lam $lam, zeta $zeta"
CUDA_VISIBLE_DEVICES=$cuda_devices python main.py --model_id $model_id --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --lam $lam --zeta $zeta --num_steps 1000 --batch_size 5
