"""
Evaluate diffusion-based image restoration methods (DiffPIR, REDDiff, DPS, DMPlug) 
on natural image datasets (CelebA-HQ, AFHQ, FFHQ) using pre-trained diffusion models from Hugging Face Diffusers. 
"""

import os
import argparse
import json 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from diffusers import DDPMPipeline, DDIMPipeline, PNDMPipeline
from PIL import Image
import numpy as np 
import time 

from deepinv.physics import Downsampling

from reconstruction_methods.diffpir import DiffPIR
from reconstruction_methods.reddiff import REDDiff
from reconstruction_methods.dps import DPS
from reconstruction_methods.dmplug import DMPlug

from utils.eval_metrics import PSNR, SSIM, LPIPS
from dataset import get_dataset
from utils.degradation import get_forward_op 

def load_diffusers_model(model_id: str, pipeline_type: str = "ddpm", device: str = "cuda"):
    """
    Load a pre-trained diffusion model from Hugging Face Diffusers.
    
    Args:
        model_id: Model ID from Hugging Face (e.g., "google/ddpm-ema-church-256")
        pipeline_type: Type of pipeline - "ddpm", "ddim", or "pndm"
        device: Device to load model on
        
    Returns:
        pipeline: Loaded diffusion pipeline
    """
    pipeline_class = {
        "ddpm": DDPMPipeline,
        "ddim": DDIMPipeline,
        "pndm": PNDMPipeline,
    }[pipeline_type.lower()]
    
    print(f"Loading {pipeline_type.upper()} pipeline from {model_id}...")
    pipeline = pipeline_class.from_pretrained(model_id)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    
    return pipeline


def extract_scheduler_and_model(pipeline):
    """
    Extract scheduler and model from a diffusers pipeline.
    
    Args:
        pipeline: Diffusers pipeline (DDPM, DDIM, or PNDM)
        
    Returns:
        model: UNet model
        scheduler: Noise scheduler with alpha schedules
    """
    model = pipeline.unet
    scheduler = pipeline.scheduler
    
    return model, scheduler


def main():
    parser = argparse.ArgumentParser(description="DiffPIR Image Restoration")
    parser.add_argument(
                        "--model_id",
                        type=str,
                        default="google/ddpm-ema-celebahq-256",
                        help="Hugging Face model ID (from Diffusers)",
                    )
    parser.add_argument("--dataset_name", type=str, default="celebahq",
                        choices=["celebahq", "afhq", "ffhq"],
                        help="Dataset to use for testing")
    parser.add_argument("--part", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to use")
    parser.add_argument("--task", type=str, default="inpainting",
                        choices=["inpainting", "super_resolution", "deblurring","box_inpainting"],
                        help="Restoration task")
    parser.add_argument("--method", type=str, default="diffpir",
                        choices=["diffpir", "reddiff", "dps", "dmplug"],
                        help="Method to use for restoration")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sigma_n", type=float, default=0.05,
                        help="Noise level in measurement")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for reconstruction (reddiff and dmplug only support batch_size=1)")

    base_args, remaining = parser.parse_known_args()

    if base_args.method == "reddiff":
        parser.add_argument("--num_steps", type=int, default=1000,
                    help="Number of diffusion steps")
        parser.add_argument("--lr", type=float, default=0.1,
                            help="Learning rate")
        parser.add_argument("--obs_weight", type=float, default=1.0,
                            help="Weight for the observation term")
        parser.add_argument("--grad_term_weight", type=float, default=0.25,
                            help="Weight for the gradient term")
        
    elif base_args.method == "diffpir":
        parser.add_argument("--num_steps", type=int, default=100,
                            help="Number of diffusion steps")
        parser.add_argument("--lam", type=float, default=1.0,
                            help="Regularization parameter lambda")
        parser.add_argument("--zeta", type=float, default=0.3,
                            help="Noise mixing (0=deterministic, 1=stochastic)")

    elif base_args.method == "dps":
        parser.add_argument("--num_steps", type=int, default=1000,
                            help="Number of diffusion steps")
        parser.add_argument("--grad_coeff", type=float, default=0.1,
                            help="Coefficient for the gradient step")

    elif base_args.method == "dmplug":
        parser.add_argument("--num_steps", type=int, default=4,
                            help="Number of diffusion steps")
        parser.add_argument("--adam_lr", type=float, default=1e-2,
                            help="Learning rate for AdamSphere")
        parser.add_argument("--adam_steps", type=int, default=200,
                            help="Number of AdamSphere steps")

    if base_args.task == "super_resolution":
        parser.add_argument("--scale_factor", type=int, default=4,
                    help="Downsampling scale factor for super-resolution task")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    model_name = args.model_id.split("/")[-1]

    dataset_name = args.dataset_name

    # save path: save_dir / model_name_to_dataset / task / method / part/ hyperparameters
    if args.task == "super_resolution":
        task_folder = f"task_{args.task}_scale_factor={args.scale_factor}"
    else:
        task_folder = f"task_{args.task}"

    sigma_n = args.sigma_n
    save_folder = os.path.join(args.save_dir, f"{model_name}_to_{dataset_name}", task_folder, f"sigma_n={sigma_n}", args.method, "sigma_n="+str(args.sigma_n), args.part)

    if args.method == "reddiff":
        save_folder = os.path.join(save_folder, f"obs{args.obs_weight}_grad{args.grad_term_weight}_lr{args.lr}_steps{args.num_steps}")
    elif args.method == "diffpir":
        save_folder = os.path.join(save_folder, f"lam_{args.lam}_zeta_{args.zeta}_steps_{args.num_steps}")
    elif args.method == "dps":
        save_folder = os.path.join(save_folder, f"grad_coeff_{args.grad_coeff}_steps_{args.num_steps}")
    elif args.method == "dmplug":
        save_folder = os.path.join(save_folder, f"adam_lr_{args.adam_lr}_adam_steps_{args.adam_steps}_steps_{args.num_steps}")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    os.makedirs(save_folder, exist_ok=True)

    
    # Load model
    # Load diffusion pipeline
    pipeline = load_diffusers_model(args.model_id, pipeline_type="ddim", device=device)

    # Extract model and scheduler
    model, scheduler = extract_scheduler_and_model(pipeline)
    
    # Get image size from model config
    image_size = model.config.sample_size
    in_channels = model.config.in_channels

    eval_dataset = get_dataset(name=args.dataset_name, part=args.part)

    batch_size = args.batch_size
    if args.method in ["reddiff", "dmplug"] and batch_size > 1:
        print(f"Warning: {args.method} only supports batch_size=1. Setting batch_size=1.")
        batch_size = 1
    dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()

    # Setup forward operator and create measurement
    physics = get_forward_op(
        degradation_type=args.task,
        device=device,
        in_channels=in_channels,
        image_size=image_size,
        **vars(args))

    if args.method == "reddiff":
        reco_method = REDDiff(model, scheduler, device=device)
        print(f"Running RED-Diff for {args.task}...")
    elif args.method == "diffpir":
        reco_method = DiffPIR(model, scheduler, device=device)
        print(f"Running DiffPIR for {args.task}...")
    elif args.method == "dps":
        reco_method = DPS(model, scheduler, device=device)
        print(f"Running DPS for {args.task}...")
    elif args.method == "dmplug":
        reco_method = DMPlug(model, scheduler, device=device)
        print(f"Running DMPlug for {args.task}...")
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    psnr_list = []
    ssim_list = []
    lpips_list = []
    time_list = []
    data_fidelity_list = []

    global_idx = 0
    num_total = len(eval_dataset)

    for batch in dataloader:
        x_true = batch.to(device)  # (B, C, H, W)
        B = x_true.shape[0]
        print(f"Processing images {global_idx+1}-{global_idx+B}/{num_total}...")

        y = physics.A(x_true)
        y = y + args.sigma_n * torch.randn_like(y)

        noise_norm = args.sigma_n**2 * np.prod(y.shape[1:])  # Variance * number of elements per image

        start_time = time.time()
        if args.method == "reddiff":
            if isinstance(physics, Downsampling):
                x_init = physics.A_adjoint(y)
                x_init = (x_init - x_init.min()) / (x_init.max() - x_init.min())
                print("x_init min/max: ", x_init.min(), x_init.max())
            else:
                x_init = physics.A_dagger(y)

            x_restored = reco_method.sample(
                y=y,
                physics=physics,
                x_init=2 * x_init - 1,  # Scale to [-1, 1] for model
                num_inference_steps=args.num_steps,
                lr=args.lr,
                obs_weight=args.obs_weight,
                grad_term_weight=args.grad_term_weight,
                show_progress=True
            )
            base_name = f"{args.method}_{model_name}_to_{dataset_name}_{args.task}_obs{args.obs_weight}_grad{args.grad_term_weight}_lr{args.lr}_steps{args.num_steps}"

        elif args.method == "diffpir":
            x_restored = reco_method.sample(
                y=y,
                physics=physics,
                sigma_n=args.sigma_n,
                lam=args.lam,
                zeta=args.zeta,
                num_inference_steps=args.num_steps,
                use_closed_form=True,
            )
            base_name = f"{args.method}_{model_name}_to_{dataset_name}_{args.task}_lam{args.lam}_zeta{args.zeta}_steps{args.num_steps}"

        elif args.method == "dps":
            x_restored = reco_method.sample(
                y=y,
                physics=physics,
                grad_coeff=args.grad_coeff,
                num_inference_steps=args.num_steps,
                show_progress=True
            )
            base_name = f"{args.method}_{model_name}_to_{dataset_name}_{args.task}_grad_coeff{args.grad_coeff}_steps{args.num_steps}"

        elif args.method == "dmplug":
            x_restored = reco_method.sample(
                y=y,
                physics=physics,
                num_inference_steps=args.num_steps,
                adam_lr=args.adam_lr,
                adam_steps=args.adam_steps,
                show_progress=True
            )
            base_name = f"{args.method}_{model_name}_to_{dataset_name}_{args.task}_adam_lr{args.adam_lr}_adam_steps{args.adam_steps}_steps{args.num_steps}"

        end_time = time.time()
        duration = end_time - start_time

        x_restored = (x_restored + 1) / 2

        for b in range(B):
            idx = global_idx + b
            per_image_time = duration / B

            psnr_value = psnr_metric.compute(x_restored[b].cpu(), x_true[b].cpu())
            ssim_value = ssim_metric.compute(x_restored[b].cpu(), x_true[b].cpu())
            lpips_value = lpips_metric.compute(x_restored[b].cpu(), x_true[b].cpu())

            y_pred = physics.A(x_restored[b].unsqueeze(0))
            data_fidelity = torch.linalg.norm(y_pred - y[b].unsqueeze(0))**2 / noise_norm

            print(f"  [img {idx}] PSNR: {psnr_value:.2f} dB \t SSIM: {ssim_value:.4f} \t LPIPS: {lpips_value:.4f} \t Data Fidelity: {data_fidelity.item():.4f} \t Duration: {per_image_time:.4f}")

            save_name = f"img_{idx}_{base_name}.png"

            if args.method == "reddiff":
                plt.figure()
                plt.imshow(x_init[b].permute(1,2,0).cpu().numpy())
                plt.title("Initial Reconstruction (x_init)")
                plt.axis("off")
                plt.savefig(os.path.join(save_folder, f"x_init_img_{idx}.png"), dpi=150, bbox_inches="tight")
                plt.close()

            fig, axes = plt.subplots(1, 3, figsize=(16, 4))

            axes[0].imshow(x_true[b].permute(1,2,0).cpu().numpy())
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")

            axes[1].imshow(y[b].permute(1,2,0).cpu().numpy())
            axes[1].set_title(f"Measurement ({args.task})")
            axes[1].axis("off")

            axes[2].imshow(x_restored[b].permute(1,2,0).cpu().numpy())
            axes[2].set_title("Reconstruction")
            axes[2].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=150, bbox_inches="tight")
            plt.close()

            x_true_np = (np.clip(x_true[b].permute(1,2,0).cpu().numpy(), 0,1) * 255).astype("uint8")
            Image.fromarray(x_true_np).save(os.path.join(save_folder, f"x_true_img_{idx}.png"))

            x_restored_np = (np.clip(x_restored[b].permute(1,2,0).cpu().numpy(), 0, 1) * 255).astype("uint8")
            Image.fromarray(x_restored_np).save(os.path.join(save_folder, save_name.replace(".png", "_restored.png")))

            if args.task == "box_inpainting":
                # also save the masked image
                masked_image = np.clip(y[b].cpu().numpy(), 0, 1) 
                masked_image = (masked_image * 255).astype("uint8").transpose(1,2,0)
                Image.fromarray(masked_image).save(os.path.join(save_folder, f"masked_img_{idx}.png"))

            print(f"  Results saved to {os.path.join(save_folder, save_name)}")

            psnr_list.append(float(psnr_value))
            ssim_list.append(float(ssim_value))
            lpips_list.append(float(lpips_value))
            time_list.append(float(per_image_time))
            data_fidelity_list.append(float(data_fidelity.item()))

        global_idx += B

    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB \t Average SSIM: {np.mean(ssim_list):.4f} \t Average LPIPS: {np.mean(lpips_list):.4f} \t Average Data Fidelity: {np.mean(data_fidelity_list):.4f} \t Average Duration: {np.mean(time_list):.4f}")

    # Save metrics to a JSON file
    metrics_path = os.path.join(save_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "mean_psnr": float(np.mean(psnr_list)),
            "mean_ssim": float(np.mean(ssim_list)),
            "mean_lpips": float(np.mean(lpips_list)),
            "mean_data_fidelity": float(np.mean(data_fidelity_list)),
            "mean_time": float(np.mean(time_list)),
            "std_psnr": float(np.std(psnr_list)),
            "std_ssim": float(np.std(ssim_list)),
            "std_lpips": float(np.std(lpips_list)),
            "std_data_fidelity": float(np.std(data_fidelity_list)),
            "std_time": float(np.std(time_list)),
            "psnr": psnr_list,
            "ssim": ssim_list,
            "lpips": lpips_list,
            "time": time_list,
            "data_fidelity": data_fidelity_list,
        }, f, indent=4)

    print(f"\nAll metrics saved to {metrics_path}")

    ### also create a text file for each metric with the mean and the hyperparameters
    # if its not exist, if it exists append to end
    # the file should have a first line 
    # with columns psnr, zeta, lam, num_steps, model_name
    # save summary one level up from save_folder
    save_dir_summary = os.path.join(args.save_dir, f"{model_name}_to_{dataset_name}", task_folder, f"sigma_n={sigma_n}", args.method, "sigma_n="+str(args.sigma_n), args.part)
    summary_path = os.path.join(save_dir_summary, f"summary_{args.task}_{args.part}.txt")

    if not os.path.exists(summary_path):

        with open(summary_path, "w") as f:
            if args.method == "reddiff":
                f.write("psnr\tssim\tlpips\ttime\tdata_fidelity\tobs_weight\tgrad_term_weight\tlr\tnum_steps\tmodel_name\n")
            elif args.method == "diffpir":
                f.write("psnr\tssim\tlpips\ttime\tdata_fidelity\tzeta\tlam\tnum_steps\tmodel_name\n")
            elif args.method == "dps":
                f.write("psnr\tssim\tlpips\ttime\tdata_fidelity\tgrad_coeff\tnum_steps\tmodel_name\n")
            elif args.method == "dmplug":
                f.write("psnr\tssim\tlpips\ttime\tdata_fidelity\tadam_lr\tadam_steps\tnum_steps\tmodel_name\n")
    
    if args.method == "reddiff":
        with open(summary_path, "a") as f:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(time_list)}\t{np.mean(data_fidelity_list)}\t{args.obs_weight}\t{args.grad_term_weight}\t{args.lr}\t{args.num_steps}\t{model_name}\n")
    elif args.method == "dmplug":
        with open(summary_path, "a") as f:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(time_list)}\t{np.mean(data_fidelity_list)}\t{args.adam_lr}\t{args.adam_steps}\t{args.num_steps}\t{model_name}\n")
    elif args.method == "diffpir":
        with open(summary_path, "a") as f:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(time_list)}\t{np.mean(data_fidelity_list)}\t{args.zeta}\t{args.lam}\t{args.num_steps}\t{model_name}\n")
    elif args.method == "dps":
        with open(summary_path, "a") as f:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(time_list)}\t{np.mean(data_fidelity_list)}\t{args.grad_coeff}\t{args.num_steps}\t{model_name}\n")

if __name__ == "__main__":
    main()
