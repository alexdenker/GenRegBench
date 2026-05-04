import torch.utils.data.dataset
from velocity import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import torchvision
import numpy as np
from torchvision.transforms.v2 import Compose, RandomHorizontalFlip, ToDtype, ToImage
from dataset import get_dataset, TransformDataset
from diffusers import UNet2DModel
import torchvision.transforms as v2
import logging
import datetime
import argparse

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--dataset", type=str, default="celebahq",
                    choices=["walnut", "ellipses", "celebahq", "aapm"],
                    help="Dataset for training")

inp = parser.parse_args()
dataset = inp.dataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="log_training_"+str(datetime.datetime.now())+"_.log",
    level=logging.INFO,
    format="%(asctime)s: %(message)s",
)

def create_model(image_size=256, in_channels=1, out_channels=1):
    """Create a UNet2DModel using diffusers."""
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 256, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.isdir("imgs"):
    os.mkdir("imgs")

save_folder = "weights/"+str(dataset)+"/pnpflow"
os.makedirs(save_folder, exist_ok=True)


batch_size = 4
gradient_accumilation_steps = 1
epochs = 200
ot_flow_matching=False
logger.info(f"dataset: {dataset}, batch size: {batch_size}, gradient accumilation steps: {gradient_accumilation_steps}, epochs: {epochs}")


train_dataset = get_dataset(dataset, part="train")
#val_dataset = get_dataset(dataset, part="val")

if dataset=="celebahq":
    transform = None 
    channels=3
elif dataset in ["walnut", "ellipses"]:
    transform = v2.Compose([
    v2.RandomRotation(degrees=90),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5)])
    channels=1
elif dataset in ["aapm"]:
    transform=v2.RandomHorizontalFlip(p=0.5)
    channels=1

train_dataset = TransformDataset(train_dataset, transform=transform)


img_height=256
img_width=256

train_dataloader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)

def get_velocity_field():
    unet=create_model(img_height,channels,channels)
    return UNetVelocity(
        unet
    ).to(device)

v = get_velocity_field()

def set_train(model,state):
    for p in model.parameters():
        p.requires_grad_(state)

set_train(v,True)

numel=0
for p in v.parameters():
    numel+=p.data.numel()

print(f"The velocity field has {numel} parameters...")

def save_image(imgs,name,nrow=25):
    grid = make_grid(imgs,nrow=nrow,padding=1,pad_value=.5)
    plt.imsave(name,torch.clip(grid.permute(1,2,0),0,1).cpu().numpy())
    return

def save_grid(nr,size=4):
    latent_imgs=torch.randn((size**2,channels,img_width,img_height),device=device,dtype=torch.float)
    with torch.no_grad():
        imgs=flow_matching_sample(v,latent_imgs,atol=1e-3,rtol=1e-3)
        print(torch.min(imgs),torch.max(imgs))
        imgs=torch.clamp(
            imgs * 0.5 + 0.5, min=0.0, max=1.0
        )
    save_image(imgs,"imgs/img_"+dataset+"_"+str(nr)+".png",nrow=size)

save_grid(0)
optim = torch.optim.AdamW(v.parameters(), lr=1e-4, betas=[0.9, 0.95])
scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.995)
acc_counter=0

for epoch in (progress_bar := tqdm(range(epochs))):
    loss_sum=0
    for x_1 in (inner_progress_bar := tqdm(train_dataloader)):
        if acc_counter==0:
            optim.zero_grad()
        x_1=x_1.to(device)*2.0 - 1.0
        x_0=torch.randn_like(x_1)
        loss = torch.mean(flow_matching_loss(v, x_0, x_1,skewed=True))
        loss.backward()
        acc_counter+=1
        if acc_counter==gradient_accumilation_steps:
            optim.step()
            acc_counter=0
        loss_sum += loss.item()
        inner_progress_bar.set_description("{0:.4f}".format(loss.item()))
    scheduler.step()
    progress_bar.set_description(
        "Epoch {0}, Loss {1:.8E}".format(epoch, loss_sum / len(train_dataloader))
    )
    logger.info("Epoch {0}, Loss {1:.8E}".format(epoch, loss_sum / len(train_dataloader)))
    set_train(v,False)
    torch.save(v.state_dict(),save_folder+"/velocity_"+str(dataset)+"_last.pt")
    torch.save(optim.state_dict(),save_folder+"/optim_"+str(dataset)+"_last.pt")
    save_grid(epoch)
    set_train(v,True)
    
set_train(v,False)
torch.save(v.state_dict(),save_folder+"/velocity_"+str(dataset)+"_final.pt")
