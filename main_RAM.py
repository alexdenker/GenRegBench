"""
Evaluation script for the Reconstruct Anything Model (RAM)

    "Reconstruct Anything Model a lightweight general model for computational imaging"
    Matthieu Terris, Samuel Hurault, Maxime Song, Julián Tachella (ICLR 2026)

"""

import os
import time  
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from PIL import Image
import json 

from deepinv.physics import TomographyWithAstra
from deepinv.models import RAM


from utils.eval_metrics import PSNR, SSIM, LPIPS
from dataset import get_dataset
from utils.degradation import get_forward_op

def main():
    parser = argparse.ArgumentParser(description="Choosing evaluation setting")
    parser.add_argument("--dataset_name", type=str, default="walnut",
                        choices=["walnut", "ellipses", "celebahq", "afhq", "ffhq"],
                        help="Dataset to use for testing")
    parser.add_argument("--part", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to use")
    parser.add_argument("--task", type=str, default="inpainting",
                        choices=["inpainting", "super_resolution", "deblurring", "tomography_sparseview", "tomography_limitedangle"],
                        help="Restoration task")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sigma_n", type=float, default=0.05,
                        help="Noise level in measurement")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    parser.add_argument("--use_finetuned_model", action="store_true",
                        help="Whether to use a finetuned model for evaluation")

    base_args, remaining = parser.parse_known_args()

    if base_args.task == "tomography_sparseview":
        parser.add_argument("--num_angles", type=int, default=60,
                            help="Number of angles for Radon transform")
        parser.add_argument("--misaligned_angles", action="store_true",
                            help="Whether to add noise to the angles for tomography_sparseview task")
        parser.add_argument("--misaligned_noise", action="store_true",
                        help="Whether to use a different noise model for simulation and reconstruction")
    elif base_args.task == "tomography_limitedangle":
        parser.add_argument("--missing_wedge", type=int, default=30,
                            help="Missing wedge angle for limited-angle tomography (in degrees)")
    elif base_args.task == "super_resolution":
        parser.add_argument("--scale_factor", type=int, default=4,
                            help="Downsampling scale factor for super-resolution task")
        
    args = parser.parse_args()

    model_type = "RAM"
    dataset_name = args.dataset_name
    part = args.part
    task = args.task
    sigma_n = args.sigma_n

    if args.task == "tomography_sparseview":
        task_folder = f"task_{args.task}_num_angles={args.num_angles}"
        if args.misaligned_angles:
            task_folder += "_misaligned"
        if args.misaligned_noise:
            task_folder += "_wrongnoise"
    elif args.task == "tomography_limitedangle":
        task_folder = f"task_{args.task}_missing_wedge={args.missing_wedge}"
    elif args.task == "super_resolution":
        task_folder = f"task_{args.task}_scale_factor={args.scale_factor}"
    else:
        task_folder = f"task_{args.task}"

    if args.use_finetuned_model:
        model_type += "_finetuned"
    save_folder = os.path.join(args.save_dir, f"{dataset_name}", task_folder, f"noise_level={sigma_n}", model_type, part)
    os.makedirs(save_folder, exist_ok=True)

    torch.random.manual_seed(args.seed)  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_size = 256
    eval_dataset = get_dataset(name=dataset_name, part=part)

    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()

    # Setup forward operator and create measurement
    physics = get_forward_op(degradation_type=task, 
                             device=device, 
                             in_channels=eval_dataset[0].shape[0], 
                             image_size=image_size, **vars(args))

    if args.task == "tomography_sparseview" and args.misaligned_angles:
        physics_sim = get_forward_op(args.task + "_misaligned", 
                                image_size=image_size, 
                                in_channels=eval_dataset[0].shape[0], 
                                device=device,
                                **vars(args))
    else:
        physics_sim = physics


    model = RAM(pretrained=True, device=device)

    if args.use_finetuned_model:
        if task == "super_resolution":
            finetuned_model_name = f"{dataset_name}_{task}_scale_factor{args.scale_factor}_sigma{sigma_n}_finetuned.pth"
        else:
            finetuned_model_name = f"{dataset_name}_{task}_sigma{sigma_n}_finetuned.pth"
        finetuned_model_path = os.path.join("finetuned_ram", finetuned_model_name)
        if os.path.exists(finetuned_model_path):
            model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
            print(f"Loaded finetuned model from {finetuned_model_path}")
        else:
            print(f"Finetuned model not found at {finetuned_model_path}. Using pretrained model instead.")

    psnr_list = []
    ssim_list = []
    lpips_list = []
    time_list = []
    data_consistency_list = []

    psnr_list_fbp = []
    ssim_list_fbp = []
    lpips_list_fbp = []
    data_consistency_list_fbp = []
    
    if isinstance(physics, TomographyWithAstra):
        class ContiguousTomographyWrapper(TomographyWithAstra):
            """Wrapper around TomographyWithAstra that ensures inputs are contiguous
            before calling A and A_adjoint, which is required by the ASTRA backend."""

            def __init__(self, base_physics):
                # Copy all attributes from the base physics object
                self.__dict__.update(base_physics.__dict__)
                self.__class__ = type(
                    "ContiguousTomographyWithAstra",
                    (TomographyWithAstra,),
                    {
                        "A": lambda self, x, **kwargs: TomographyWithAstra.A(self, x.contiguous(), **kwargs),
                        "A_adjoint": lambda self, y, **kwargs: TomographyWithAstra.A_adjoint(self, y.contiguous(), **kwargs),
                    },
                )


        physics = ContiguousTomographyWrapper(physics)

    for idx in range(len(eval_dataset)):
        print(f"Processing image {idx+1}/{len(eval_dataset)}...")
        x_true = eval_dataset[idx].unsqueeze(0).to(device)  # (1, C, H, W)

 
        y = physics_sim.A(x_true) 
        if args.task == "tomography_sparseview":    
            if args.misaligned_noise:
                y=y+sigma_n*torch.sqrt(y)*torch.randn_like(y)
            else:
                y = y + sigma_n * torch.randn_like(y)
        else:
            y = y + sigma_n * torch.randn_like(y)

        noise_norm = sigma_n **2 * np.prod(y.shape[1:])

        start_time = time.time()

        sigma_inp = sigma_n 
        with torch.no_grad():
            x_restored = model.forward(y, physics=physics, sigma=sigma_inp)

        end_time = time.time()
        duration = end_time - start_time
        
        if isinstance(physics, TomographyWithAstra):
            x_fbp = physics.fbp(y)
        else:
            x_fbp = torch.zeros_like(x_true)  # dummy FBP for non-tomography tasks
        psnr_value = psnr_metric.compute(x_restored[0], x_true[0])
        ssim_value = ssim_metric.compute(x_restored[0], x_true[0])
        lpips_value = lpips_metric.compute(x_restored[0].cpu(), x_true[0].cpu())

        y_pred = physics_sim.A(x_restored)

        data_fidelity = torch.linalg.norm(y_pred - y)**2 / noise_norm


        if isinstance(physics_sim, TomographyWithAstra):
            psnr_value_fbp = psnr_metric.compute(x_fbp[0], x_true[0])
            ssim_value_fbp = ssim_metric.compute(x_fbp[0], x_true[0])
            lpips_value_fbp = lpips_metric.compute(x_fbp[0].cpu(), x_true[0].cpu())

            y_pred_fbp = physics_sim.A(x_fbp)
            data_fidelity_fbp = torch.linalg.norm(y_pred_fbp - y)**2 / noise_norm
            
        else:
            psnr_value_fbp = 0.0
            ssim_value_fbp = 0.0
            lpips_value_fbp = 0.0
            data_fidelity_fbp = 0.0
        print(f"PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}, Time: {duration:.2f} seconds")

        if args.task == "tomography_sparseview":
            save_name = f"img_{idx}_{model_type}_{dataset_name}_{args.task}_num_angles{args.num_angles}.png"
        elif args.task == "tomography_limitedangle":
            save_name = f"img_{idx}_{model_type}_{dataset_name}_{args.task}_missing_wedge{args.missing_wedge}.png"
        else:
            save_name = f"img_{idx}_{model_type}_{dataset_name}_{args.task}.png"

        if "tomography" in args.task:
            fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        else:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        if x_true.shape[1] == 1:  # Grayscale image
                
            axes[0].imshow(x_true[0, 0].cpu().numpy().clip(0, 1), cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")
                
            if args.task == "inpainting":
                axes[1].imshow(y[0, 0].cpu().numpy().clip(0, 1), cmap="gray")
            else:
                axes[1].imshow(y[0, 0].cpu().numpy().clip(0, 1), cmap="gray")

            axes[1].set_title(f"Measurement ({args.task})")
            axes[1].axis("off")

            axes[2].imshow(x_restored[0, 0].cpu().numpy().clip(0, 1), cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Reconstruction")
            axes[2].axis("off")
            
            error = torch.abs(x_true - x_restored)
            axes[3].imshow(error[0, 0].cpu().numpy().clip(0, 1), cmap="hot")
            axes[3].set_title(f"Error (MSE: {error.pow(2).mean().item():.6f})")
            axes[3].axis("off")

            if "tomography" in args.task:
                # Show FBP
                axes[4].imshow(x_fbp[0, 0].cpu().numpy().clip(0, 1), cmap="gray")
                axes[4].set_title("FBP (Filtered Back Projection)")
                axes[4].axis("off")
        else:  # RGB image
            axes[0].imshow(x_true[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")

            if args.task == "inpainting":
                masked_measurement = y
                axes[1].imshow(masked_measurement[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            else:
                axes[1].imshow(y[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            axes[1].set_title(f"Measurement ({args.task})")
            axes[1].axis("off")

            axes[2].imshow(x_restored[0].permute(1, 2, 0).cpu().numpy().clip(0, 1))
            axes[2].set_title("Reconstruction")
            axes[2].axis("off")
            
            error = torch.abs(x_true - x_restored)
            error_image = error.pow(2).mean(dim=1)  # Average over color channels for visualization
            axes[3].imshow(error_image[0].cpu().numpy().clip(0, 1), cmap="hot")
            axes[3].set_title(f"Error (MSE: {error.pow(2).mean().item():.6f})")
            axes[3].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, save_name), dpi=150, bbox_inches="tight")
        plt.close()

        if x_true.shape[1] == 1:  # Grayscale image
                
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
        else:  # RGB image
            x_true = (x_true[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
            x_true = Image.fromarray(x_true)
            x_true.save(os.path.join(save_folder, f"x_true_img_{idx}.png"))

            x_restored = (np.clip(x_restored[0].permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype("uint8")
            x_restored = Image.fromarray(x_restored)
            x_restored.save(os.path.join(save_folder, save_name.replace(".png", "_restored.png")))


            y_image = (np.clip(y[0].permute(1, 2, 0).cpu().numpy(), 0, 1) * 255).astype("uint8")
            y_image = Image.fromarray(y_image)
            y_image.save(os.path.join(save_folder, save_name.replace(".png", "_measurement.png")))

        print(f"Results saved to {os.path.join(save_folder, save_name)}")

        psnr_list.append(float(psnr_value))
        ssim_list.append(float(ssim_value))
        lpips_list.append(float(lpips_value))
        time_list.append(float(duration))
        data_consistency_list.append(float(data_fidelity))

        psnr_list_fbp.append(float(psnr_value_fbp))
        ssim_list_fbp.append(float(ssim_value_fbp))
        lpips_list_fbp.append(float(lpips_value_fbp))   
        data_consistency_list_fbp.append(float(data_fidelity_fbp))

    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB \t Average SSIM: {np.mean(ssim_list):.4f} \t Average LPIPS: {np.mean(lpips_list):.4f} \t Data Consistency: {np.mean(data_consistency_list):.4f} \t Average Time: {np.mean(time_list):.4f}")
    print(f"Average FBP PSNR: {np.mean(psnr_list_fbp):.2f} dB \t Average FBP SSIM: {np.mean(ssim_list_fbp):.4f} \t Average FBP LPIPS: {np.mean(lpips_list_fbp):.4f} \t FBP Data Consistency: {np.mean(data_consistency_list_fbp):.4f}")
    # Save metrics to a JSON file
    metrics_path = os.path.join(save_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "mean_psnr": float(np.mean(psnr_list)),
            "mean_ssim": float(np.mean(ssim_list)),
            "mean_lpips": float(np.mean(lpips_list)),
            "mean_time": float(np.mean(time_list)),
            "mean_data_consistency": float(np.mean(data_consistency_list)),
            "std_psnr": float(np.std(psnr_list)),
            "std_ssim": float(np.std(ssim_list)),
            "std_lpips": float(np.std(lpips_list)),
            "std_time": float(np.std(time_list)),
            "std_data_consistency": float(np.std(data_consistency_list)),
            "psnr": psnr_list,
            "ssim": ssim_list,
            "lpips": lpips_list,
            "time": time_list,
            "data_consistency": data_consistency_list,
        }, f, indent=4)

    print(f"\nAll metrics saved to {metrics_path}")


    save_dir_summary = os.path.join(args.save_dir, f"{dataset_name}", task_folder, f"noise_level={sigma_n}", model_type, part)
    summary_path = os.path.join(save_dir_summary, f"summary_{task_folder}_{part}.txt")

    if not os.path.exists(summary_path):

        with open(summary_path, "w") as f:
            f.write("psnr\tssim\tlpips\ttime\tdata_consistency\tmodel_name\n")


    with open(summary_path, "a") as f:
        f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(time_list)}\t{np.mean(data_consistency_list)}\t{model_type}\n")


if __name__ == "__main__":
    main()  