#!/bin/bash

#######################################
### FlowDPS with Stable Diffusion #####
########################################

task="tomography_sparseview"
num_angles=32
step_size=10.0
num_dc_steps=50
part="test"
echo "Running main_flow.py for model $model_path, task $task, num_angles $num_angles, step_size $step_size, num_dc_steps $num_dc_steps"
python main_flow.py --method flowdps --task "$task" --part "$part" --num_angles $num_angles --sigma_n 0.01 --step_size $step_size --num_dc_steps $num_dc_steps --efficient_memory


task="tomography_sparseview"
num_angles=128
step_size=10.0
num_dc_steps=50
part="test"
echo "Running main_flow.py for model $model_path, task $task, num_angles $num_angles, step_size $step_size, num_dc_steps $num_dc_steps"
python main_flow.py --method flowdps --task "$task" --part "$part" --num_angles $num_angles --sigma_n 0.01 --step_size $step_size --num_dc_steps $num_dc_steps --efficient_memory
