"""
Total Variation baselines for image reconstruction.
We solve the optimization problem 
    x* = argmin_x 0.5 * || A(x) - y ||^2 + alpha * TV(x)
using Condat-Vu.


"""

import os
import argparse
import json 
import time

import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
from tqdm import tqdm 

import deepinv as dinv 

from utils.eval_metrics import PSNR, SSIM, LPIPS
from utils.degradation import get_forward_op
from dataset import get_dataset

def main():
    parser = argparse.ArgumentParser(description="Baseline Image Restoration")

    parser.add_argument("--dataset_name", type=str, default="walnut",
                        choices=["walnut", "ellipses", "celebahq", "afhq", "ffhq"],
                        help="Dataset to use for testing")
    parser.add_argument("--part", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to use")
    parser.add_argument("--task", type=str, default="inpainting",
                        choices=["deblurring", "inpainting", "super_resolution", "tomography_sparseview", "tomography_limitedangle"],
                        help="Restoration task")
    parser.add_argument("--method", type=str, default="tv",
                        choices=["tv"],
                        help="Method to use for restoration")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sigma_n", type=float, default=0.01,
                        help="Noise level in measurement")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--sensitivity_check", action="store_true",
                        help="Evaluatne the sensitivity wrt noise realisations")

    base_args, remaining = parser.parse_known_args()

    if base_args.method == "tv":
        parser.add_argument("--max_iter", type=int, default=20000,
                    help="Maximum number of iterations for TV baseline")
        parser.add_argument("--alpha", type=float, default=5e-5,
                    help="Regularisation parameter for TV baseline")
        
    if base_args.task == "tomography_sparseview":
        parser.add_argument("--num_angles", type=int, default=60,
                            help="Number of angles for Radon transform")
        parser.add_argument("--misaligned_angles", action="store_true",
                            help="Whether to use misaligned angles for tomography")
        parser.add_argument("--misaligned_noise", action="store_true",
                            help="Whether to add noise to the angles for tomography_sparseview task")
        # add range 

    elif base_args.task == "tomography_limitedangle":
        parser.add_argument("--missing_wedge", type=int, default=30,
                            help="Missing wedge angle for limited-angle tomography (in degrees)")
    elif base_args.task == "super_resolution":
        parser.add_argument("--scale_factor", type=int, default=4,
                            help="Scale factor for super-resolution")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    dataset_name = args.dataset_name

    # save path: save_dir / dataset_name / task / method / part/ hyperparameters
    if args.task == "tomography_sparseview":
        task_folder = f"task_{args.task}_num_angles={args.num_angles}"
        if args.misaligned_angles:
            task_folder += "_misaligned_angles"
        if args.misaligned_noise:
            task_folder += "_wrongnoise"
    elif args.task == "tomography_limitedangle":
        task_folder = f"task_{args.task}_missing_wedge={args.missing_wedge}"
    elif args.task == "super_resolution":
        task_folder = f"task_{args.task}_scale_factor={args.scale_factor}"
    else:
        task_folder = f"task_{args.task}"

    if args.sensitivity_check:
        task_folder += "_sensitivity"

    save_folder = os.path.join(args.save_dir, dataset_name, task_folder, f"sigma_n={args.sigma_n}", args.method, args.part)

    if args.method == "tv":
        save_folder = os.path.join(save_folder, f"alpha{args.alpha}_max_iter{args.max_iter}")

    os.makedirs(save_folder, exist_ok=True)

    image_size = 256
    
    dataset = get_dataset(dataset_name, part=args.part)
    in_channels = dataset[0].shape[0]  # assuming dataset returns (C, H, W)

    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()

    # Setup forward operator and create measurement
    physics = get_forward_op(args.task, 
                             image_size=image_size, 
                             in_channels=in_channels, 
                             device=device,
                             **vars(args))
    
    if args.task == "tomography_sparseview" and args.misaligned_angles:
        physics_sim = get_forward_op(args.task + "_misaligned", 
                                image_size=image_size, 
                             in_channels=in_channels, 
                             device=device,
                             **vars(args))
    else:
        physics_sim = physics

    psnr_list = []
    ssim_list = []
    lpips_list = []
    time_list = []
    data_fidelity_list = []

    for idx in range(len(dataset)):
        if args.sensitivity_check and not idx % 10 == 0:
            continue
        print(f"Processing image {idx+1}/{len(dataset)}...")
        x_true = dataset[idx].unsqueeze(0).to(device)  # (1, C, H, W)

        if idx==0:
            norm_squared = physics_sim.compute_norm(torch.rand_like(x_true), squared= True)
            print(f"Computed squared spectral norm of physics: {norm_squared:.4f}")

        if args.method == "tv":
            def TV_rec(
                y,
                x_init,
                lambd,
                sigma,
                tau,
                max_iter,
                tol=1e-6,
                track_psnr=True,
                x_gt=None,
            ):
                if track_psnr and x_gt is None:
                    raise ValueError("x_gt must be provided when track_psnr=True")

                prior = dinv.optim.prior.TVPrior(n_it_max=100)

                xk = torch.clone(x_init)
                b  = torch.zeros_like(prior.nabla(xk))
                psnr_list = []

                for i in tqdm(range(max_iter)):
                    x_old = xk.clone()

                    # --- Primal update (gradient of ½‖Ax − y‖²) ---
                    grad_f = physics.A_adjoint(physics.A(xk) - y)
                    xk = xk - tau * (grad_f + prior.nabla_adjoint(b))
                    xk = xk.clamp(min=0.)

                    # --- Dual update ---
                    b = b + sigma * prior.nabla(2 * xk - x_old)
                    b_norm = b.norm(dim=-1, keepdim=True).clamp(min=lambd)
                    b = b * (lambd / b_norm)   

                    # --- Stopping criterion ---
                    rel_err = torch.linalg.norm(x_old.flatten() - xk.flatten()) \
                            / torch.linalg.norm(xk.flatten() + 1e-12)
                    if rel_err < tol:
                        print(f"Converged at iteration {i} with relative error {rel_err:.6e}")
                        break

                    if track_psnr:
                        psnr = psnr_metric.compute(x_gt[0].cpu(), xk[0].detach().cpu())
                        psnr_list.append(psnr)
                # print(torch.norm(physics.A(xk) - y)**2 + lambd * prior.fn(xk)) energy for debugging
                if track_psnr:
                    return xk, psnr_list
                else:
                    return xk


        if args.sensitivity_check:
            psnr_values=[]
            ssim_values=[]
            lpips_values=[]
            df_values=[]
            for _ in tqdm(range(40)):
                y = physics_sim.A(x_true)
                if args.misaligned_noise:       
                    y=y+args.sigma_n*torch.sqrt(y)*torch.randn_like(y)
                else:
                    y = y + args.sigma_n * torch.randn_like(y)
                noise_norm = args.sigma_n ** 2 * torch.numel(y)
                x_init = torch.zeros_like(x_true)  # Start from zero image
                sigma = norm_squared/ (100 * 8)  # spectral norm of gradient
                tau = 1.0 / (norm_squared/2 + 8 * sigma)
                start_time = time.time()
                x_restored = TV_rec(y, x_init, float(args.alpha), sigma, tau, max_iter=args.max_iter, tol=2e-6, track_psnr=False, x_gt=x_true)
                end_time = time.time()
                duration = end_time - start_time
                psnr_values.append(psnr_metric.compute(x_restored[0].cpu(), x_true[0].cpu()))
                ssim_values.append(ssim_metric.compute(x_restored[0].cpu(), x_true[0].cpu()))
                lpips_values.append(lpips_metric.compute(x_restored[0].cpu(), x_true[0].cpu()))

                y_pred = physics_sim.A(x_restored)
                df_values.append(torch.linalg.norm(y_pred - y)**2 / noise_norm)
            psnr_value = np.array(psnr_values).std()
            ssim_value = np.array(ssim_values).std()
            lpips_value = np.array(lpips_values).std()
            data_fidelity = torch.stack(df_values).std()
        else:
            y = physics_sim.A(x_true)
            if args.misaligned_noise:       
                y=y+args.sigma_n*torch.sqrt(y)*torch.randn_like(y)
            else:
                y = y + args.sigma_n * torch.randn_like(y)
            noise_norm = args.sigma_n ** 2 * torch.numel(y)
            x_init = torch.zeros_like(x_true)  # Start from zero image
            sigma = norm_squared/ (100 * 8)  # spectral norm of gradient
            tau = 1.0 / (norm_squared/2 + 8 * sigma)
            start_time = time.time()
            x_restored = TV_rec(y, x_init, float(args.alpha), sigma, tau, max_iter=args.max_iter, tol=2e-6, track_psnr=False, x_gt=x_true)
            end_time = time.time()
            duration = end_time - start_time
            print("Shape of restored image:", x_restored.shape)

            psnr_value = psnr_metric.compute(x_restored[0].cpu(), x_true[0].cpu())
            ssim_value = ssim_metric.compute(x_restored[0].cpu(), x_true[0].cpu())
            lpips_value = lpips_metric.compute(x_restored[0].cpu(), x_true[0].cpu())

            y_pred = physics_sim.A(x_restored)
            data_fidelity = torch.linalg.norm(y_pred - y)**2 / noise_norm


        if args.task == "tomography_sparseview":
            save_name = f"img_{idx}_{args.method}_{dataset_name}_{args.task}_num_angles{args.num_angles}_alpha{args.alpha}_maxiter{args.max_iter}.png"
        elif args.task == "tomography_limitedangle":
            save_name = f"img_{idx}_{args.method}_{dataset_name}_{args.task}_missing_wedge{args.missing_wedge}_alpha{args.alpha}_maxiter{args.max_iter}.png"
        else:
            save_name = f"img_{idx}_{args.method}_{dataset_name}_{args.task}_alpha{args.alpha}_maxiter{args.max_iter}.png"


        print(f"PSNR: {psnr_value:.2f} dB \t SSIM: {ssim_value:.4f} \t LPIPS: {lpips_value:.4f} \t Data Fidelity: {data_fidelity:.4f}")

        if "tomography" in args.task:
            fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        else:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        if in_channels == 1:
            axes[0].imshow(x_true[0, 0].cpu().numpy(), cmap="gray")
            axes[1].imshow(y[0, 0].cpu().numpy(), cmap="gray")
            axes[2].imshow(x_restored[0, 0].cpu().numpy(), cmap="gray")
            error = torch.abs(x_true - x_restored)
            axes[3].imshow(error[0, 0].cpu().numpy(), cmap="hot")
            axes[3].set_title(f"Error (MSE: {error.pow(2).mean().item():.6f})")
        else:
            axes[0].imshow(x_true[0].cpu().numpy().transpose(1, 2, 0))
            axes[1].imshow(y[0].cpu().numpy().transpose(1, 2, 0))
            axes[2].imshow(x_restored[0].cpu().numpy().transpose(1, 2, 0))

        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        
        axes[1].set_title(f"Measurement ({args.task})")
        axes[1].axis("off")
        
        axes[2].set_title("Reconstruction")
        axes[2].axis("off")

        axes[3].axis("off")

        if "tomography" in args.task:
            # Show FBP
            x_fbp = physics.fbp(y)
            axes[4].imshow(x_fbp[0, 0].cpu().numpy(), cmap="gray")
            axes[4].set_title("FBP (Filtered Back Projection)")
            axes[4].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, save_name), dpi=150, bbox_inches="tight")
        plt.close()

        if in_channels == 1:
            ### also save as numpy 
            np.save(os.path.join(save_folder, f"x_restored_img_{idx}.npy"), x_restored[0, 0].cpu().numpy())

            ### save results as pngs 
            x_true = (x_true[0, 0].cpu().numpy() * 255).astype("uint8")
            x_true = Image.fromarray(x_true)
            x_true.save(os.path.join(save_folder, f"x_true_img_{idx}.png"))

            x_restored = (np.clip(x_restored[0, 0].cpu().numpy(), 0, 1) * 255).astype("uint8")
            x_restored = Image.fromarray(x_restored)
            x_restored.save(os.path.join(save_folder, save_name.replace(".png", "_restored.png")))

            if "tomography" in args.task:
                x_fbp = (np.clip(x_fbp[0, 0].cpu().numpy(), 0, 1) * 255).astype("uint8")
                x_fbp = Image.fromarray(x_fbp)
                x_fbp.save(os.path.join(save_folder, f"x_fbp_img_{idx}.png"))
        else:
            x_true = (x_true[0].cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8")
            x_true = Image.fromarray(x_true)
            x_true.save(os.path.join(save_folder, f"x_true_img_{idx}.png"))

            x_restored = (np.clip(x_restored[0].cpu().numpy().transpose(1, 2, 0), 0, 1) * 255).astype("uint8")
            x_restored = Image.fromarray(x_restored)
            x_restored.save(os.path.join(save_folder, save_name.replace(".png", "_restored.png")))

        print(f"Results saved to {os.path.join(save_folder, save_name)}")

        psnr_list.append(float(psnr_value))
        ssim_list.append(float(ssim_value))
        lpips_list.append(float(lpips_value))
        time_list.append(float(duration))

        data_fidelity_list.append(float(data_fidelity))

    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB \t Average SSIM: {np.mean(ssim_list):.4f} \t Average LPIPS: {np.mean(lpips_list):.4f} \t Average Data Fidelity: {np.mean(data_fidelity_list):.4f}")

    # Save metrics to a JSON file
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
            "data_fidelity": data_fidelity_list
        }, f, indent=4)

    print(f"\nAll metrics saved to {metrics_path}")

    ### also create a text file for each metric with the mean and the hyperparameters
    save_dir_summary = os.path.join(args.save_dir, dataset_name, task_folder, f"sigma_n={args.sigma_n}", args.method, args.part)
    summary_path = os.path.join(save_dir_summary, f"summary_{args.task}_{args.part}.txt")

    if not os.path.exists(summary_path):

        with open(summary_path, "w") as f:
            if args.method == "tv":
                f.write("psnr\tssim\tlpips\tdata_fidelity\talpha\ttime\tmax_iter\n")

    if args.method == "tv":
        with open(summary_path, "a") as f:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(data_fidelity_list)}\t{args.alpha}\t{np.mean(time_list)}\t{args.max_iter}\n")



if __name__ == "__main__":
    main()
