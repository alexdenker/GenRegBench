#!/bin/bash
part="test"
sigma_n=0.01


model_path="saved_models/diffusers/walnut/ema_model"

# DMPLUG + sparseview
method="dmplug"
task="tomography_sparseview"

num_angles=16
adam_steps=1500
num_steps=4
echo "Running $method for model $model_path, task $task, num_angles $num_angles"
python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --adam_steps $adam_steps --num_steps $num_steps --batch_size 1 

num_angles=32
adam_steps=1500
num_steps=4
echo "Running $method for model $model_path, task $task, num_angles $num_angles"
python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --adam_steps $adam_steps --num_steps $num_steps --batch_size 1 

num_angles=64
adam_steps=1500
num_steps=4
echo "Running $method for model $model_path, task $task, num_angles $num_angles"
python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --adam_steps $adam_steps --num_steps $num_steps --batch_size 1 


num_angles=128
adam_steps=1500
num_steps=4
echo "Running $method for model $model_path, task $task, num_angles $num_angles"
python main.py --model_path $model_path --method $method --task $task --part $part --num_angles $num_angles --sigma_n $sigma_n --adam_steps $adam_steps --num_steps $num_steps --batch_size 1



