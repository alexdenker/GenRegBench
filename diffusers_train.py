"""
Training script for unconditional diffusion model using HuggingFace diffusers library.
"""

import os
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as v2

from diffusers import UNet2DModel, DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from ema_pytorch import EMA
from tqdm import tqdm
import yaml

import argparse


from dataset import get_dataset

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


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


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="aapm", choices=["aapm", "ellipses", "walnut"], 
                        help="Name of the dataset to train on")
    args = parser.parse_args()

    dataset_name = args.dataset_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    # Training hyperparameters
    image_size = 256
    in_channels = 1
    batch_size = 4
    num_epochs = 1000
    learning_rate = 1e-4
    num_train_timesteps = 1000
    gradient_accumulation_steps = 4
    ema_decay = 0.999
    save_every = 50  # Save every N epochs
    
    # Create output directory
    log_dir = os.path.join("saved_models", "diffusers", dataset_name, f'{time.strftime("%d-%m-%Y-%H-%M-%S")}')
    os.makedirs(log_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(log_dir, "config.yml"), "w") as f:
        yaml.dump({
            "image_size": image_size,
            "in_channels": in_channels,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "num_train_timesteps": num_train_timesteps,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "ema_decay": ema_decay,
        }, f)
    
    # Create model
    model = create_model(image_size=image_size, in_channels=in_channels, out_channels=in_channels)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule="linear",
        beta_start=0.0001,
        beta_end=0.02,
        prediction_type="epsilon",  # Predict noise
    )
    
    dataset = get_dataset(dataset_name, part="train")

    if isinstance(dataset, torch.utils.data.IterableDataset):
        print("Using IterableDataset. DataLoader will not shuffle data.")
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)



    ### add data augmentation here 
    if dataset_name == "ellipses":
        # rotation (random angle), flipping
        transform = v2.Compose([
        v2.RandomRotation(degrees=90),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)])
    elif dataset_name == "lodopab":
        class RandomRotate90:
            def __call__(self, x):
                k = torch.randint(0, 4, (1,)).item()
                return torch.rot90(x, k, dims=(-2, -1))
        # only flipping 
        transform = v2.Compose([
        RandomRotate90(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)])
    elif dataset_name == "walnut":
        # rotation (random angle), flipping
        transform = v2.Compose([
        v2.RandomRotation(degrees=90),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5)])
    elif dataset_name == "aapm":
        # only left right flipping, since up/down flipping would not make sense for CT images
        transform = v2.RandomHorizontalFlip(p=0.5)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please implement data augmentation for this dataset.")

    # rotation (random angle), flipping

    print("Number of training batches per epoch:", len(train_loader))
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=num_epochs * len(train_loader),
    )
    
    ema = EMA(
        model,
        beta=ema_decay,
        update_after_step=100,
        update_every=10,
    )
    
    writer = SummaryWriter(log_dir=log_dir)
    
    # Training loop
    global_step = 0
    batches_per_epoch = len(dataset) // batch_size  # Manually compute since IterableDataset
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", total=batches_per_epoch)
        for batch_idx, x in enumerate(progress_bar):
            if batch_idx >= batches_per_epoch:
                break  # Stop after one epoch's worth of data
            x = x.to(device)
            x = transform(x)  # Apply data augmentation
            
            x = 2 * x - 1  # Scale to [-1, 1]

            # Sample noise
            noise = torch.randn_like(x)
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (x.shape[0],), device=device
            ).long()
            
            # Add noise to images according to noise schedule
            noisy_images = noise_scheduler.add_noise(x, noise, timesteps)
            
            # Predict noise
            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
            
            # Compute loss
            loss = F.mse_loss(noise_pred, noise)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            num_batches += 1
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                ema.update()
                global_step += 1
            
            progress_bar.set_postfix({"loss": epoch_loss / num_batches})
        
        avg_loss = epoch_loss / num_batches
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

        if epoch == 0:
            print("First epoch complete. Saving initial model checkpoint.")
            model.save_pretrained(os.path.join(log_dir, f"model_epoch_{epoch+1}"))
            # Save EMA model
            ema_model_path = os.path.join(log_dir, f"ema_model_epoch_{epoch+1}")
            os.makedirs(ema_model_path, exist_ok=True)
            ema.ema_model.save_pretrained(ema_model_path)
            
            # Save scheduler config
            noise_scheduler.save_pretrained(os.path.join(log_dir, "scheduler"))           
    
        # Save checkpoint
        if (epoch + 1) % save_every == 0 or epoch == num_epochs - 1:
            # Save model
            model.save_pretrained(os.path.join(log_dir, f"model_epoch_{epoch+1}"))
            
            # Save EMA model
            ema_model_path = os.path.join(log_dir, f"ema_model_epoch_{epoch+1}")
            os.makedirs(ema_model_path, exist_ok=True)
            ema.ema_model.save_pretrained(ema_model_path)
            
            # Save scheduler config
            noise_scheduler.save_pretrained(os.path.join(log_dir, "scheduler"))
            
            print(f"Saved checkpoint at epoch {epoch+1}")
    
    writer.close()
    print(f"Training complete! Models saved to {log_dir}")


if __name__ == "__main__":
    train()
