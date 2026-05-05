#!/bin/bash

#######################
### RAM to CelebaHQ ###
#######################

sigma_n=0.05
part="test"
dataset=celebahq
task=inpainting 
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n 


sigma_n=0.05
part="test"
dataset=celebahq
task=deblurring 
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n 


sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=2 
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n --scale_factor $scale_factor

sigma_n=0.05
part="test"
dataset=celebahq
task=super_resolution
scale_factor=4
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n --scale_factor $scale_factor


###################
### RAM to FFHQ ###
###################

sigma_n=0.05
part="test"
dataset=ffhq
task=inpainting 
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n 


sigma_n=0.05
part="test"
dataset=ffhq
task=deblurring 
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n 


sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=2 
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n --scale_factor $scale_factor

sigma_n=0.05
part="test"
dataset=ffhq
task=super_resolution
scale_factor=4
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n --scale_factor $scale_factor


###################
### RAM to AFHQ ###
###################

sigma_n=0.05
part="test"
dataset=afhq
task=inpainting 
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n 


sigma_n=0.05
part="test"
dataset=afhq
task=deblurring 
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n 


sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=2 
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n --scale_factor $scale_factor

sigma_n=0.05
part="test"
dataset=afhq
task=super_resolution
scale_factor=4
echo "Running main_RAM.py for task $task, on dataset $dataset"
python main_RAM.py --task $task --part $part --dataset_name $dataset --sigma_n $sigma_n --scale_factor $scale_factor
