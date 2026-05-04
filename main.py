import os
import argparse
import json 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from diffusers import UNet2DModel, DDPMScheduler, DDIMPipeline
from PIL import Image
import numpy as np 
import time 
from tqdm import tqdm

from deepinv.physics import Tomography

from reconstruction_methods.diffpir import DiffPIR
from reconstruction_methods.reddiff import REDDiff
from reconstruction_methods.dmplug import DMPlug
from reconstruction_methods.dps import DPS 

from dataset import get_dataset
from utils.eval_metrics import PSNR, SSIM, LPIPS
from utils.degradation import get_forward_op


def main():
    parser = argparse.ArgumentParser(description="Generative Regulariser Image Restoration")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to local pretrained diffusion model")
    parser.add_argument("--model_id", type=str, default=None,
                        help="Hugging Face model ID (e.g., google/ddpm-ema-celebahq-256)")
    parser.add_argument("--dataset_name", type=str, default="walnut",
                        choices=["walnut", "ellipses", "aapm"],
                        help="Dataset to use for testing")
    parser.add_argument("--part", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to use")
    parser.add_argument("--task", type=str, default="inpainting",
                        choices=["inpainting", "super_resolution", "deblurring", "tomography_sparseview", "tomography_limitedangle"],
                        help="Restoration task")
    parser.add_argument("--method", type=str, default="diffpir",
                        choices=["diffpir", "reddiff", "dps", "dmplug"],
                        help="Method to use for restoration")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sigma_n", type=float, default=0.01,
                        help="Noise level in measurement")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for reconstruction (reddiff and dmplug only support batch_size=1)")
    parser.add_argument("--sensitivity_check", action="store_true",
                        help="Evaluatne the sensitivity wrt noise realisations")

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
        parser.add_argument("--lam", type=float, default=0.1,
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
        parser.add_argument("--patience", type=int, default=300,
                            help="Early stopping patience")
        parser.add_argument("--delta", type=float, default=0.99,
                            help="Early stopping threshold factor for new minimum")
        parser.add_argument("--w", type=int, default=50,
                            help="Window size for variance calculation")
    if base_args.task == "tomography_sparseview":
        parser.add_argument("--num_angles", type=int, default=60,
                            help="Number of angles for Radon transform")
        parser.add_argument("--misaligned_angles", action="store_true",
                            help="Whether to add noise to the angles for tomography_sparseview task")
        parser.add_argument("--misaligned_noise", action="store_true",
                            help="Whether to add noise to the angles for tomography_sparseview task")
        
    elif base_args.task == "tomography_limitedangle":
        parser.add_argument("--missing_wedge", type=int, default=30,
                            help="Missing wedge angle for limited-angle tomography (in degrees)")

    elif base_args.task == "super_resolution":
        parser.add_argument("--scale_factor", type=int, default=4,
                            help="Scale factor for super-resolution")

    args = parser.parse_args()

    device = "cuda" #"cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    if args.model_path is None and args.model_id is None:
        raise ValueError("Either --model_path or --model_id must be provided.")
    if args.model_path is not None and args.model_id is not None:
        raise ValueError("Only one of --model_path or --model_id should be provided, not both.")

    if args.model_id is not None:
        model_name = args.model_id.split("/")[-1]
    else:
        print("model_path:", args.model_path)
        model_name = args.model_path.split("/")[2]

    dataset_name = args.dataset_name

    # save path: save_dir / model_name_to_dataset / task / method / part/ hyperparameters
    if args.task == "tomography_sparseview":
        task_folder = f"task_{args.task}_num_angles={args.num_angles}"
        if args.misaligned_angles:
            task_folder += "_misaligned"
        if args.misaligned_noise:
            task_folder += "_wrongnoise"
    elif args.task == "tomography_limitedangle":
        task_folder = f"task_{args.task}_missing_wedge={args.missing_wedge}"
    else:
        task_folder = f"task_{args.task}"

    if args.sensitivity_check:
        task_folder += "_sensitivity"

    save_folder = os.path.join(args.save_dir, f"{model_name}_to_{dataset_name}", task_folder, f"sigma_n={args.sigma_n}", args.method, args.part)

    if args.method == "reddiff":
        save_folder = os.path.join(save_folder, f"obs{args.obs_weight}_grad{args.grad_term_weight}_lr{args.lr}_steps{args.num_steps}")
    elif args.method == "diffpir":
        save_folder = os.path.join(save_folder, f"lam_{args.lam}_zeta_{args.zeta}_steps_{args.num_steps}")
    elif args.method == "dps":
        save_folder = os.path.join(save_folder, f"grad_coeff_{args.grad_coeff}_steps_{args.num_steps}")
    elif args.method == "dmplug":
        save_folder = os.path.join(save_folder, f"adam_lr_{args.adam_lr}_adam_steps_{args.adam_steps}_steps_{args.num_steps}_w_{args.w}_patience_{args.patience}_delta_{args.delta}")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    os.makedirs(save_folder, exist_ok=True)

    
    # Load model
    if args.model_id is not None:
        print(f"Loading model from Hugging Face: {args.model_id}")
        pipeline = DDIMPipeline.from_pretrained(args.model_id)
        pipeline = pipeline.to(device)
        model = pipeline.unet
        scheduler = pipeline.scheduler
    else:
        print(f"Loading model from local path: {args.model_path}")
        model = UNet2DModel.from_pretrained(args.model_path)
        model.to(device)
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            prediction_type="epsilon",
        )
    model.eval()
    
    image_size = model.config.sample_size
    in_channels = model.config.in_channels
    print(f"Model config: image_size={image_size}, in_channels={in_channels}")

    eval_dataset = get_dataset(name=args.dataset_name, part=args.part)
    # Force batch_size=1 for methods that don't support batching
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
        **vars(args))  # pass all args as kwargs for specific degradation parameters

    if args.task == "tomography_sparseview" and args.misaligned_angles:
        print("Using misaligned angles for tomography_sparseview task")
        physics_sim = get_forward_op(
            degradation_type=args.task+"_misaligned",
            device=device,
            in_channels=in_channels,
            image_size=image_size,
            **vars(args)
        )
    else:
        physics_sim = physics  

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
    global_idx = 0  # track image index across batches
    num_total = len(eval_dataset)

    if args.sensitivity_check:
        assert args.batch_size == 1, "Batching not supported for the sensitivity check"

    for batch_number, batch in enumerate(dataloader):
        if args.sensitivity_check and not batch_number % 10 == 0:
            continue 
        x_true = batch.to(device)  # (B, C, H, W)
        B = x_true.shape[0]
        print(f"Processing images {global_idx+1}-{global_idx+B}/{num_total}...")

        # if isinstance(physics, Tomography):
        #     # relative noise for tomography tasks (per-image)
        #     y_clean = physics.A(x_true)
        #     rel_noise = args.sigma_n * y_clean.abs().mean(dim=tuple(range(1, y_clean.ndim)), keepdim=True)  # (B, 1, ...)
        #     y = y_clean + rel_noise * torch.randn_like(y_clean)
        #     noise = y - y_clean
        #     delta = torch.norm(noise.reshape(noise.shape[0], -1), dim=1)
        # else:

        def reco(y):
            start_time = time.time()
            if args.method == "reddiff":
                if isinstance(physics, Tomography):
                    x_init = physics.fbp(y)
                else:
                    x_init = physics.A_dagger(y)

                x_restored = reco_method.sample(
                    y=y,
                    physics=physics,
                    x_init=2 * x_init - 1,
                    num_inference_steps=args.num_steps,
                    lr=args.lr,
                    obs_weight=args.obs_weight, 
                    grad_term_weight=args.grad_term_weight,
                    show_progress=True
                )

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
            elif args.method == "dps":
                x_restored = reco_method.sample(
                    y=y,
                    physics=physics,
                    grad_coeff=args.grad_coeff,
                    num_inference_steps=args.num_steps,
                    show_progress=True
                )
            elif args.method == "dmplug":
                print("Run DMPlug with num_steps =", args.num_steps, "Adam steps =", args.adam_steps, "Adam lr =", args.adam_lr)
                x_restored = reco_method.sample(
                    y=y,
                    physics=physics,
                    num_inference_steps=args.num_steps,
                    adam_lr=args.adam_lr,
                    adam_steps=args.adam_steps,
                    patience=args.patience,
                    delta=args.delta,
                    w=args.w,
                    show_progress=True
                )
            
            end_time = time.time()
            duration = end_time - start_time

            x_restored = (x_restored + 1) / 2
            return x_restored, duration

        if not args.sensitivity_check:
            y_clean = physics_sim.A(x_true) 
            if args.misaligned_noise:       
                y=y_clean+args.sigma_n*torch.sqrt(y_clean)*torch.randn_like(y_clean)
            else:
                y = y_clean + args.sigma_n * torch.randn_like(y_clean)
            noise_norm = args.sigma_n**2 * np.prod(y.shape[1:])  # Variance * number of elements per image
            x_restored, duration = reco(y)

            print("x_restored shape:", x_restored.shape)
        # Per-image metrics and saving
        for b in range(B):
            idx = global_idx + b
            if args.sensitivity_check:
                psnr_values=[]
                ssim_values=[]
                lpips_values=[]
                df_values=[]
                for _ in tqdm(range(40)):
                    y_clean = physics_sim.A(x_true) 
                    if args.misaligned_noise:       
                        y=y_clean+args.sigma_n*torch.sqrt(y_clean)*torch.randn_like(y_clean)
                    else:
                        y = y_clean + args.sigma_n * torch.randn_like(y_clean)
                    noise_norm = args.sigma_n**2 * np.prod(y.shape[1:])  # Variance * number of elements per image
                    x_restored, duration = reco(y)

                    psnr_values.append(psnr_metric.compute(x_restored[b].cpu(), x_true[b].cpu()))
                    ssim_values.append(ssim_metric.compute(x_restored[b].cpu(), x_true[b].cpu()))
                    lpips_values.append(lpips_metric.compute(x_restored[b].cpu(), x_true[b].cpu()))
                    per_image_time = duration / B

                    y_pred = physics_sim.A(x_restored[b].unsqueeze(0))
                    df_values.append(torch.linalg.norm(y_pred - y[b].unsqueeze(0))**2 / noise_norm)
                psnr_value = np.array(psnr_values).std()
                ssim_value = np.array(ssim_values).std()
                lpips_value = np.array(lpips_values).std()
                data_fidelity = torch.stack(df_values).std()
            else:

                psnr_value = psnr_metric.compute(x_restored[b].cpu(), x_true[b].cpu())
                ssim_value = ssim_metric.compute(x_restored[b].cpu(), x_true[b].cpu())
                lpips_value = lpips_metric.compute(x_restored[b].cpu(), x_true[b].cpu())
                per_image_time = duration / B

                y_pred = physics_sim.A(x_restored[b].unsqueeze(0))
                data_fidelity = torch.linalg.norm(y_pred - y[b].unsqueeze(0))**2 / noise_norm


            print(f"  [img {idx}] PSNR: {psnr_value:.2f} dB \t SSIM: {ssim_value:.4f} \t LPIPS: {lpips_value:.4f}  \t Data Fidelity: {data_fidelity:.4f} \t Duration: {per_image_time:.4f}")

            # Build save name
            if args.method == "reddiff":
                base_name = f"{args.method}_{model_name}_to_{dataset_name}_{args.task}"
                if args.task == "tomography_sparseview":
                    base_name += f"_num_angles{args.num_angles}"
                elif args.task == "tomography_limitedangle":
                    base_name += f"_missing_wedge{args.missing_wedge}"
                base_name += f"_obs{args.obs_weight}_grad{args.grad_term_weight}_lr{args.lr}_steps{args.num_steps}"
            elif args.method == "diffpir":
                base_name = f"{args.method}_{model_name}_to_{dataset_name}_{args.task}"
                if args.task == "tomography_sparseview":
                    base_name += f"_num_angles{args.num_angles}"
                elif args.task == "tomography_limitedangle":
                    base_name += f"_missing_wedge{args.missing_wedge}"
                base_name += f"_lam{args.lam}_zeta{args.zeta}_steps{args.num_steps}"
            elif args.method == "dps":
                base_name = f"{args.method}_{model_name}_to_{dataset_name}_{args.task}"
                if args.task == "tomography_sparseview":
                    base_name += f"_num_angles{args.num_angles}"
                elif args.task == "tomography_limitedangle":
                    base_name += f"_missing_wedge{args.missing_wedge}"
                base_name += f"_grad_coeff{args.grad_coeff}_steps{args.num_steps}"
            elif args.method == "dmplug":
                base_name = f"{args.method}_{model_name}_to_{dataset_name}_{args.task}"
                if args.task == "tomography_sparseview":
                    base_name += f"_num_angles{args.num_angles}"
                elif args.task == "tomography_limitedangle":
                    base_name += f"_missing_wedge{args.missing_wedge}"
                base_name += f"_adam_lr{args.adam_lr}_adam_steps{args.adam_steps}_steps{args.num_steps}_w{args.w}_patience{args.patience}_delta{args.delta}"
            
            save_name = f"img_{idx}_{base_name}.png"

            if "tomography" in args.task:
                fig, axes = plt.subplots(1, 5, figsize=(16, 4))
            else:
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

            axes[0].imshow(x_true[b, 0].cpu().numpy(), cmap="gray")
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")
            
            if args.task == "inpainting":
                axes[1].imshow(y[b, 0].cpu().numpy(), cmap="gray")
            else:
                axes[1].imshow(y[b, 0].cpu().numpy(), cmap="gray")
            axes[1].set_title(f"Measurement ({args.task})")
            axes[1].axis("off")
            
            axes[2].imshow(x_restored[b, 0].cpu().numpy(), cmap="gray")
            axes[2].set_title("Reconstruction")
            axes[2].axis("off")
            
            error = torch.abs(x_true[b] - x_restored[b])
            axes[3].imshow(error[0].cpu().numpy(), cmap="hot")
            axes[3].set_title(f"Error (MSE: {error.pow(2).mean().item():.6f})")
            axes[3].axis("off")

            if "tomography" in args.task:
                x_fbp = physics.fbp(y[b:b+1])
                axes[4].imshow(x_fbp[0, 0].cpu().numpy(), cmap="gray")
                axes[4].set_title("FBP (Filtered Back Projection)")
                axes[4].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=150, bbox_inches="tight")
            plt.close()

            ### save results as npy for potential further analysis
            np.save(os.path.join(save_folder, save_name.replace(".png", "_x_true.npy")), x_true[b].cpu().numpy())
            np.save(os.path.join(save_folder, save_name.replace(".png", "_y.npy")), y[b].cpu().numpy())
            np.save(os.path.join(save_folder, save_name.replace(".png", "_x_restored.npy")), x_restored[b].cpu().numpy())
            if "tomography" in args.task:
                np.save(os.path.join(save_folder, save_name.replace(".png", "_x_fbp.npy")), x_fbp[0].cpu().numpy())

            ### save results as pngs 
            x_true_np = (x_true[b, 0].cpu().numpy() * 255).astype("uint8")
            Image.fromarray(x_true_np).save(os.path.join(save_folder, f"x_true_img_{idx}.png"))

            x_restored_np = (np.clip(x_restored[b, 0].cpu().numpy(), 0, 1) * 255).astype("uint8")
            Image.fromarray(x_restored_np).save(os.path.join(save_folder, save_name.replace(".png", "_restored.png")))

            if "tomography" in args.task:
                x_fbp_np = (np.clip(x_fbp[0, 0].cpu().numpy(), 0, 1) * 255).astype("uint8")
                Image.fromarray(x_fbp_np).save(os.path.join(save_folder, f"x_fbp_img_{idx}.png"))

            print(f"  Results saved to {os.path.join(save_folder, save_name)}")

            psnr_list.append(float(psnr_value))
            ssim_list.append(float(ssim_value))
            lpips_list.append(float(lpips_value))
            time_list.append(float(per_image_time))
            data_fidelity_list.append(data_fidelity.item())

        global_idx += B


    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB \t Average SSIM: {np.mean(ssim_list):.4f} \t Average LPIPS: {np.mean(lpips_list):.4f} \t Average Time: {np.mean(time_list):.4f}")

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
    # if its not exist, if it exists append to end
    # the file should have a first line 
    # with columns psnr, zeta, lam, num_steps, model_name
    # save summary one level up from save_folder
    save_dir_summary = os.path.join(args.save_dir, f"{model_name}_to_{dataset_name}", task_folder, f"sigma_n={args.sigma_n}",args.method, args.part)
    summary_path = os.path.join(save_dir_summary, f"summary_{args.task}_{args.part}.txt")

    if not os.path.exists(summary_path):

        with open(summary_path, "w") as f:
            if args.method == "reddiff":
                f.write("psnr\tssim\tlpips\tdata_fidelity\ttime\tobs_weight\tgrad_term_weight\tlr\tnum_steps\tmodel_name\n")
            elif args.method == "diffpir":
                f.write("psnr\tssim\tlpips\tdata_fidelity\ttime\tzeta\tlam\tnum_steps\tmodel_name\n")
            elif args.method == "dps":
                f.write("psnr\tssim\tlpips\tdata_fidelity\ttime\tgrad_coeff\tnum_steps\tmodel_name\n")
            elif args.method == "dmplug":
                f.write("psnr\tssim\tlpips\tdata_fidelity\ttime\tadam_lr\tadam_steps\tnum_steps\tw\tpatience\tdelta\tmodel_name\n")
            
    if args.method == "reddiff":
        with open(summary_path, "a") as f:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(data_fidelity_list)}\t{np.mean(time_list)}\t{args.obs_weight}\t{args.grad_term_weight}\t{args.lr}\t{args.num_steps}\t{model_name}\n")
    elif args.method == "dmplug":
        with open(summary_path, "a") as f:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(data_fidelity_list)}\t{np.mean(time_list)}\t{args.adam_lr}\t{args.adam_steps}\t{args.num_steps}\t{args.w}\t{args.patience}\t{args.delta}\t{model_name}\n")
    elif args.method == "diffpir":
        with open(summary_path, "a") as f:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(data_fidelity_list)}\t{np.mean(time_list)}\t{args.zeta}\t{args.lam}\t{args.num_steps}\t{model_name}\n")
    elif args.method == "dps":
        with open(summary_path, "a") as f:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(data_fidelity_list)}\t{np.mean(time_list)}\t{args.grad_coeff}\t{args.num_steps}\t{model_name}\n")
    
if __name__ == "__main__":
    main()