"""
FlowDPS: Flow-based Diffusion Posterior Sampling using Stable Diffusion 3.

Adapted from solve.py (FlowDPS reference implementation).
Uses SD3 as a generative prior for inverse problems with text-guided sampling.
"""

import os
import argparse
import json
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from deepinv.physics import Tomography

from reconstruction_methods.flow_dps import get_solver
from dataset import get_dataset
from utils.eval_metrics import PSNR, SSIM, LPIPS
from utils.degradation import get_forward_op


def get_prompt(dataset_name):
    if dataset_name == "walnut":
        return "a computed tomography image of a walnut"
    else:
        return "a computed tomography image"


def main():
    parser = argparse.ArgumentParser(description="FlowDPS Image Restoration")
    parser.add_argument("--dataset_name", type=str, default="walnut",
                        choices=["walnut", "ellipses", "aapm"],
                        help="Dataset to use for testing")
    parser.add_argument("--part", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to use")
    parser.add_argument("--task", type=str, default="tomography_sparseview",
                        choices=["tomography_sparseview", "tomography_limitedangle"],
                        help="Restoration task")
    parser.add_argument("--method", type=str, default="flowdps",
                        choices=["flowdps"],
                        help="Solver to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sigma_n", type=float, default=0.01,
                        help="Noise level in measurement")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")

    # FlowDPS-specific arguments
    parser.add_argument("--NFE", type=int, default=28,
                        help="Number of function evaluations (sampling steps)")
    parser.add_argument("--cfg_scale", type=float, default=2.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--step_size", type=float, default=15.0,
                        help="Step size for data consistency")
    parser.add_argument("--num_dc_steps", type=int, default=3,
                        help="Number of data consistency steps per sampling step")
    parser.add_argument("--img_size", type=int, default=512,
                        help="Image size for SD3 generation")
    parser.add_argument("--efficient_memory", action="store_true", default=False,
                        help="Precompute text embeddings and free text encoders to save GPU memory")

    base_args, _ = parser.parse_known_args()

    if base_args.task == "tomography_sparseview":
        parser.add_argument("--num_angles", type=int, default=60,
                            help="Number of angles for Radon transform")
    elif base_args.task == "tomography_limitedangle":
        parser.add_argument("--missing_wedge", type=int, default=30,
                            help="Missing wedge angle for limited-angle tomography (in degrees)")

    args = parser.parse_args()

    device = "cuda"
    torch.manual_seed(args.seed)

    dataset_name = args.dataset_name
    model_name = "SD3"

    # Build save path
    if args.task == "tomography_sparseview":
        task_folder = f"task_{args.task}_num_angles={args.num_angles}"
    elif args.task == "tomography_limitedangle":
        task_folder = f"task_{args.task}_missing_wedge={args.missing_wedge}"
    else:
        task_folder = f"task_{args.task}"

    save_folder = os.path.join(
        args.save_dir, f"{model_name}_to_{dataset_name}", task_folder,
        f"sigma_n={args.sigma_n}", args.method, args.part,
        f"NFE{args.NFE}_cfg{args.cfg_scale}_step{args.step_size}_num_dc_steps{args.num_dc_steps}"
    )
    os.makedirs(save_folder, exist_ok=True)

    # Load solver
    print(f"Loading {args.method} solver (SD3)...")
    solver = get_solver(args.method)

    # Set up text prompt and precompute embeddings
    prompt = get_prompt(dataset_name)
    print(f"Using prompt: '{prompt}'")

    solver.text_enc_1.to("cuda")
    solver.text_enc_2.to("cuda")
    solver.text_enc_3.to("cuda")

    with torch.no_grad():
        prompt_emb, pooled_emb = solver.encode_prompt([prompt], batch_size=1)
        null_emb, null_pooled_emb = solver.encode_prompt([""], batch_size=1)

    if args.efficient_memory:
        del solver.text_enc_1
        del solver.text_enc_2
        del solver.text_enc_3
        torch.cuda.empty_cache()
        print("Text encoders removed from GPU for memory efficiency.")

    solver.vae.to("cuda")
    solver.transformer.to("cuda")

    # Load dataset
    image_size = args.img_size
    in_channels = 1  # CT images are grayscale
    eval_dataset = get_dataset(name=dataset_name, part=args.part)

    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()

    # Setup forward operator
    physics = get_forward_op(
        degradation_type=args.task,
        device=device,
        in_channels=in_channels,
        image_size=256,  # Forward operator is designed for 256x256 images, SD3 will generate 512x512 and we will downsample before applying the forward operator   
        **vars(args),
    )

    print(f"Running {args.method} for {args.task}...")

    psnr_list = []
    ssim_list = []
    lpips_list = []
    time_list = []
    data_fidelity_list = []
    num_total = len(eval_dataset)

    for idx in range(num_total):
        print(f"Processing image {idx + 1}/{num_total}...")
        x_true = eval_dataset[idx].unsqueeze(0).to(device)  # (1, C, H, W)

        y = physics.A(x_true)
        y = y + args.sigma_n * torch.randn_like(y)

        noise_norm = args.sigma_n ** 2 * np.prod(y.shape[1:])

        start_time = time.time()

        x_restored = solver.sample(
            measurement=y,
            operator=physics,
            prompts=prompt,
            NFE=args.NFE,
            img_shape=(image_size, image_size),
            cfg_scale=args.cfg_scale,
            noise_level=noise_norm,
            step_size=args.step_size,
            max_dc_steps=args.num_dc_steps,
            task=args.task,
            prompt_emb=[prompt_emb, pooled_emb],
            null_emb=[null_emb, null_pooled_emb],
        )

        end_time = time.time()
        duration = end_time - start_time

        # SD3 output is in [-1, 1], scale to [0, 1] and take grayscale
        x_restored = (x_restored + 1) / 2
        x_restored = x_restored.mean(dim=1, keepdim=True)  # RGB -> grayscale
        x_restored = x_restored.clamp(0, 1)

        print(f"  x_restored shape: {x_restored.shape}")

        psnr_value = psnr_metric.compute(x_restored[0].cpu(), x_true[0].cpu())
        ssim_value = ssim_metric.compute(x_restored[0].cpu(), x_true[0].cpu())
        lpips_value = lpips_metric.compute(x_restored[0].cpu(), x_true[0].cpu())

        y_pred = physics.A(x_restored)
        data_fidelity = torch.linalg.norm(y_pred - y) ** 2 / noise_norm

        print(f"  [img {idx}] PSNR: {psnr_value:.2f} dB \t SSIM: {ssim_value:.4f} \t LPIPS: {lpips_value:.4f} \t Data Fidelity: {data_fidelity:.4f} \t Duration: {duration:.2f}s")

        # Build save name
        base_name = f"{args.method}_{model_name}_to_{dataset_name}_{args.task}"
        if args.task == "tomography_sparseview":
            base_name += f"_num_angles{args.num_angles}"
        elif args.task == "tomography_limitedangle":
            base_name += f"_missing_wedge{args.missing_wedge}"
        base_name += f"_NFE{args.NFE}_cfg{args.cfg_scale}_step{args.step_size}_dc{args.num_dc_steps}"
        save_name = f"img_{idx}_{base_name}.png"

        # Plot results
        if isinstance(physics, Tomography):
            fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        else:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].imshow(x_true[0, 0].cpu().numpy(), cmap="gray")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")

        axes[1].imshow(y[0, 0].cpu().numpy(), cmap="gray")
        axes[1].set_title(f"Measurement ({args.task})")
        axes[1].axis("off")

        axes[2].imshow(x_restored[0, 0].cpu().numpy(), cmap="gray")
        axes[2].set_title("Reconstruction")
        axes[2].axis("off")

        error = torch.abs(x_true - x_restored)
        axes[3].imshow(error[0, 0].cpu().numpy(), cmap="hot")
        axes[3].set_title(f"Error (MSE: {error.pow(2).mean().item():.6f})")
        axes[3].axis("off")

        if isinstance(physics, Tomography):
            x_fbp = physics.fbp(y)
            axes[4].imshow(x_fbp[0, 0].cpu().numpy(), cmap="gray")
            axes[4].set_title("FBP (Filtered Back Projection)")
            axes[4].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, save_name), dpi=150, bbox_inches="tight")
        plt.close()

        # Save npy files
        np.save(os.path.join(save_folder, save_name.replace(".png", "_x_true.npy")), x_true[0].cpu().numpy())
        np.save(os.path.join(save_folder, save_name.replace(".png", "_y.npy")), y[0].cpu().numpy())
        np.save(os.path.join(save_folder, save_name.replace(".png", "_x_restored.npy")), x_restored[0].cpu().numpy())
        if isinstance(physics, Tomography):
            np.save(os.path.join(save_folder, save_name.replace(".png", "_x_fbp.npy")), x_fbp[0].cpu().numpy())

        # Save png files
        x_true_np = (x_true[0, 0].cpu().numpy() * 255).astype("uint8")
        Image.fromarray(x_true_np).save(os.path.join(save_folder, f"x_true_img_{idx}.png"))

        x_restored_np = (np.clip(x_restored[0, 0].cpu().numpy(), 0, 1) * 255).astype("uint8")
        Image.fromarray(x_restored_np).save(os.path.join(save_folder, save_name.replace(".png", "_restored.png")))

        if isinstance(physics, Tomography):
            x_fbp_np = (np.clip(x_fbp[0, 0].cpu().numpy(), 0, 1) * 255).astype("uint8")
            Image.fromarray(x_fbp_np).save(os.path.join(save_folder, f"x_fbp_img_{idx}.png"))

        print(f"  Results saved to {os.path.join(save_folder, save_name)}")

        psnr_list.append(float(psnr_value))
        ssim_list.append(float(ssim_value))
        lpips_list.append(float(lpips_value))
        time_list.append(float(duration))
        data_fidelity_list.append(float(data_fidelity.item()))

    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB \t Average SSIM: {np.mean(ssim_list):.4f} \t Average LPIPS: {np.mean(lpips_list):.4f} \t Average Time: {np.mean(time_list):.4f}")

    # Save metrics to JSON
    metrics_path = os.path.join(save_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "mean_psnr": float(np.mean(psnr_list)),
            "mean_ssim": float(np.mean(ssim_list)),
            "mean_lpips": float(np.mean(lpips_list)),
            "mean_time": float(np.mean(time_list)),
            "mean_data_fidelity": float(np.mean(data_fidelity_list)),
            "psnr": psnr_list,
            "ssim": ssim_list,
            "lpips": lpips_list,
            "time": time_list,
            "data_fidelity": data_fidelity_list,
        }, f, indent=4)

    print(f"\nAll metrics saved to {metrics_path}")

    # Append to summary file
    save_dir_summary = os.path.join(
        args.save_dir, f"{model_name}_to_{dataset_name}", task_folder,
        f"sigma_n={args.sigma_n}", args.method, args.part
    )
    os.makedirs(save_dir_summary, exist_ok=True)
    summary_path = os.path.join(save_dir_summary, f"summary_{args.task}_{args.part}.txt")

    if not os.path.exists(summary_path):
        with open(summary_path, "w") as f:
            f.write("psnr\tssim\tlpips\tdata_fidelity\ttime\tNFE\tcfg_scale\tstep_size\tnum_dc_steps\n")

    with open(summary_path, "a") as f:
        f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(data_fidelity_list)}\t{np.mean(time_list)}\t{args.NFE}\t{args.cfg_scale}\t{args.step_size}\t{args.num_dc_steps}\n")


if __name__ == "__main__":
    main()
