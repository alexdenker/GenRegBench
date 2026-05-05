#!/bin/bash

part="test"



###############################
###### WALNUT TO WALNUT #######
###############################

model_path="saved_models/diffusers/walnut/ema_model"
task="tomography_sparseview"
num_angles=8
obs_weight=2.0
grad_term_weight=0.01
lr=0.05
echo "Running main.py for model $model_path, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_path "$model_path" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n 0.01 \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000



model_path="saved_models/diffusers/walnut/ema_model"
task="tomography_sparseview"
num_angles=32
obs_weight=0.75
grad_term_weight=0.01
lr=0.05
echo "Running main.py for model $model_path, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_path "$model_path" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n 0.01 \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000



model_path="saved_models/diffusers/walnut/ema_model"
task="tomography_sparseview"
num_angles=64
obs_weight=0.5
grad_term_weight=0.01
lr=0.05
echo "Running main.py for model $model_path, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_path "$model_path" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n 0.01 \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000


model_path="saved_models/diffusers/walnut/ema_model"
task="tomography_sparseview"
num_angles=128
obs_weight=0.5
grad_term_weight=0.01
lr=0.01
echo "Running main.py for model $model_path, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_path "$model_path" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n 0.01 \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000

################################
###### Ellipses to Walnut ######
################################


model_path="saved_models/diffusers/diskellipses/ema_model"
task="tomography_sparseview"
num_angles=32
obs_weight=4.0
grad_term_weight=0.01
lr=0.1
sigma_n=0.01
echo "Running main.py for model $model_path, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_path "$model_path" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n $sigma_n \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000


model_path="saved_models/diffusers/diskellipses/ema_model"
task="tomography_sparseview"
num_angles=128
obs_weight=1.5
grad_term_weight=0.01
lr=0.01
sigma_n=0.01
echo "Running main.py for model $model_path, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_path "$model_path" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n $sigma_n \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000



###############################
###### CelebA TO WALNUT #######
###############################

model_id="google/ddpm-ema-celebahq-256"
task="tomography_sparseview"
num_angles=32
obs_weight=10.0
grad_term_weight=0.1
lr=0.05
sigma_n=0.01
echo "Running main.py for model $model_id, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_id "$model_id" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n $sigma_n \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000


model_id="google/ddpm-ema-celebahq-256"
task="tomography_sparseview"
num_angles=128
obs_weight=5.0
grad_term_weight=0.1
lr=0.05
sigma_n=0.01
echo "Running main.py for model $model_id, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_id "$model_id" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n $sigma_n \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000

###############################
###### AAPM TO WALNUT #######
###############################

model_path="saved_models/diffusers/aapm/ema_model"
task="tomography_sparseview"
num_angles=32
obs_weight=1.0
grad_term_weight=0.01
lr=0.05
sigma_n=0.01
echo "Running main.py for model $model_path, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_path "$model_path" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n $sigma_n \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000


model_path="saved_models/diffusers/aapm/ema_model"
task="tomography_sparseview"
num_angles=128
obs_weight=7.5
grad_term_weight=0.1
lr=0.05
sigma_n=0.01
echo "Running main.py for model $model_path, task $task, num_angles $num_angles, grad_term_weight $grad_term_weight, obs_weight $obs_weight, lr $lr"
python main.py \
    --model_path "$model_path" \
    --method reddiff \
    --task "$task" \
    --part $part \
    --num_angles $num_angles \
    --sigma_n $sigma_n \
    --grad_term_weight $grad_term_weight \
    --obs_weight $obs_weight \
    --lr $lr \
    --batch_size 1 \
    --num_steps 1000
