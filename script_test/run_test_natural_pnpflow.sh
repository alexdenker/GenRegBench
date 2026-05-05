#!/bin/bash


### RUN PnP-flow, CelebaHQ to CelebaHQ
dataset=celebahq
sigma_n=0.05

task=inpainting
alpha=1
gamma=20
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n

task=deblurring
alpha=1
gamma=30
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n

task=super_resolution
alpha=1
gamma=30
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n --scale_factor 2

task=super_resolution
alpha=1
gamma=50
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n --scale_factor 4

### RUN PnP-Flow, CelebaHQ to CelebaHQ
dataset=ffhq
sigma_n=0.05

task=inpainting
alpha=1
gamma=20
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n

task=deblurring
alpha=1
gamma=30
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n

task=super_resolution
alpha=1
gamma=30
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n --scale_factor 2

task=super_resolution
alpha=1
gamma=50
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n --scale_factor 4

### RUN PnP-Flow, CelebaHQ to CelebaHQ
dataset=afhq
sigma_n=0.05

task=inpainting
alpha=1
gamma=20
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n

task=deblurring
alpha=1
gamma=30
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n

task=super_resolution
alpha=1
gamma=30
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n --scale_factor 2

task=super_resolution
alpha=1
gamma=50
python main_pnpflow.py --trained_model celebahq --dataset_name $dataset --part test --task $task --alpha $alpha --gamma $gamma --sigma_n $sigma_n --scale_factor 4