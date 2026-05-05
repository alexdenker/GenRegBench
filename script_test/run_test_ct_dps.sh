#!/bin/bash


# DPS 

task="tomography_sparseview"
model_path="saved_models/diffusers/walnut/ema_model"

# for the varying angles figure
num_angles=8
grad_coeff=1.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

# for the varying angles figure and table
num_angles=16
grad_coeff=1.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

num_angles=32
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

num_angles=64
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

num_angles=128
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

# for the varying noise level table

sigma_n=0.005
num_angles=128
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n $sigma_n --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

sigma_n=0.002
num_angles=128
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n $sigma_n --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

sigma_n=0.001
num_angles=128
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n $sigma_n --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

sigma_n=0.0
num_angles=128
grad_coeff=2.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n $sigma_n --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

# ood results
sigma_n=0.01
model_path="saved_models/diffusers/aapm/ema_model" # aapm

num_angles=32
grad_coeff=20.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

num_angles=128
grad_coeff=20.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

model_path="saved_models/diffusers/diskellipses/ema_model" # ellipses
num_angles=32
grad_coeff=5.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

num_angles=128
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000


# celebahq
num_angles=32
grad_coeff=10.0
python main.py --model_id "google/ddpm-ema-celebahq-256" --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

num_angles=128
grad_coeff=10.0
python main.py --model_id "google/ddpm-ema-celebahq-256" --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000

# misaligned angles
model_path="saved_models/diffusers/walnut/ema_model" # walnut

num_angles=32
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000 --misaligned_angles

num_angles=128
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.01 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000 --misaligned_angles

# misaligned noise

num_angles=32
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.02 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000 --misaligned_noise

num_angles=128
grad_coeff=10.0
python main.py --model_path $model_path --method dps --task "$task" --part test --num_angles $num_angles --sigma_n 0.02 --grad_coeff $grad_coeff --batch_size 1 --num_steps 1000 --misaligned_noise