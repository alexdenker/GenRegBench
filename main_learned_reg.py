import os
import sys
import time  
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
import deepinv
from PIL import Image
import json 
from tqdm import tqdm

from deepinv.optim import L2
from utils.nmAPG import reconstruct_nmAPG
from utils.LSR import LSR
from utils.wcrr import WCRR
from utils.parameter_learning_wrapper import ParameterLearningWrapper


from utils.eval_metrics import PSNR, SSIM, LPIPS
from utils.degradation import get_forward_op
from dataset import get_dataset

dataset_to_weights = {
    "walnut_WCRR": "weights/walnut/bilevel_Denoising/WCRR_bilevel_JFB_for_Denoising_walnut.pt",
    "diskellipses_WCRR": "weights/ellipses/bilevel_Denoising/WCRR_bilevel_JFB_for_Denoising_ellipses.pt",
    "celebahq_WCRR": "weights/celebahq/bilevel_Denoising/WCRR_bilevel_JFB_for_Denoising_celebahq.pt",
    "celebahq_LSR": "weights/celebahq/bilevel_Denoising/LSR_bilevel_JFB_for_Denoising_celebahq.pt",
    "diskellipses_LSR": "weights/ellipses/bilevel_Denoising/LSR_bilevel_JFB_for_Denoising_ellipses.pt",
    "walnut_LSR": "weights/walnut/bilevel_Denoising/LSR_bilevel_JFB_for_Denoising_walnut.pt",
    "aapm_LSR": "weights/aapm/bilevel_Denoising/LSR_bilevel_JFB_for_Denoising_aapm.pt",
    "aapm_WCRR": "weights/aapm/bilevel_Denoising/WCRR_bilevel_JFB_for_Denoising_aapm.pt",
    "BSD_LSR": "weights/LSR_bilevel_JFB_for_Denoising.pt",
    "CBSD_LSR": "weights/LSR_bilevel_JFB_for_Denoising_color.pt"
}

def main():
    parser = argparse.ArgumentParser(description="Choosing evaluation setting")
    parser.add_argument("--trained_model", type=str, default="diskellipses", choices=["walnut", "diskellipses", "celebahq", "aapm", "BSD", "CBSD"],
                        help="name of the pretrained model")
    parser.add_argument("--model_type", type=str, default="WCRR", choices=["CRR", "WCRR", "LSR"],
                        help="type of the regularizer")
    parser.add_argument("--dataset_name", type=str, default="walnut",
                        choices=["walnut", "ellipses", "celebahq", "afhq", "ffhq"],
                        help="Dataset to use for testing")
    parser.add_argument("--part", type=str, default="val",
                        choices=["val", "test"],
                        help="Dataset split to use")
    parser.add_argument("--task", type=str, default="inpainting",
                        choices=["inpainting", "box_inpainting" ,"super_resolution", "deblurring", "tomography_sparseview", "tomography_limitedangle"],
                        help="Restoration task")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--sigma_n", type=float, default=0.05,
                        help="Noise level in measurement")
    parser.add_argument("--save_dir", type=str, default="results",
                        help="Directory to save results")
    
    parser.add_argument("--lmbd", type=float, default=1.0,
                        help="Regularization parameter lambda")
    parser.add_argument("--sensitivity_check", action="store_true",
                        help="Evaluatne the sensitivity wrt noise realisations")

    base_args, remaining = parser.parse_known_args()

        
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

    inp = parser.parse_args()

    regularizer_name = inp.model_type
    dataset_name = inp.dataset_name
    trained_model = inp.trained_model
    part = inp.part
    task = inp.task
    sigma_n = inp.sigma_n

    if inp.task == "tomography_sparseview":
        task_folder = f"task_{inp.task}_num_angles={inp.num_angles}"
        if inp.misaligned_angles:
            task_folder += "_misaligned"
        if inp.misaligned_noise:
            task_folder += "_wrongnoise"
    elif inp.task == "tomography_limitedangle":
        task_folder = f"task_{inp.task}_missing_wedge={inp.missing_wedge}"
    elif inp.task == "super_resolution":
        task_folder = f"task_{inp.task}_scale_factor={inp.scale_factor}"
    else:
        task_folder = f"task_{inp.task}"
    
    if inp.sensitivity_check:
        task_folder += "_sensitivity"


    save_folder = os.path.join(inp.save_dir, f"{trained_model}_to_{dataset_name}", task_folder, f"sigma_n={sigma_n}", inp.model_type, inp.part)
    
    if regularizer_name == "LSR" and False:
        save_folder = os.path.join(save_folder, f"lmbd={inp.lmbd}_sigma={inp.sigma}")
    else:
        save_folder = os.path.join(save_folder, f"lmbd={inp.lmbd}")

    os.makedirs(save_folder, exist_ok=True)

    dataset = get_dataset(dataset_name, part=inp.part)
    in_channels = dataset[0].shape[0]

    torch.random.manual_seed(inp.seed)  # make results deterministic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define regularizer
    if regularizer_name == "CRR":
        reg_channels = 3 if trained_model == "celebahq" else 1
        nb_channels = [reg_channels, 4, 8, 64]
        reg = WCRR(
            sigma=0.1,
            weak_convexity=0.0,
            nb_channels=nb_channels,
        ).to(device)
    elif regularizer_name == "WCRR":
        reg_channels = 3 if trained_model == "celebahq" else 1
        nb_channels = [reg_channels, 4, 8, 64]
        reg = WCRR(
            sigma=0.1,
            weak_convexity=1.0,
            nb_channels=nb_channels,
        ).to(device)
    elif regularizer_name == "LSR":
        reg = LSR(
            channels=3 if trained_model in ["celebahq", "CBSD"] else 1,
            nc=[32, 64, 128, 256],
            deepinv=False,
            alpha=1.0,
            sigma=0.03 if inp.trained_model in ["BSD","CBSD"] else 0.05,
            act_mode="s" if inp.trained_model in ["BSD", "CBSD"] else "E",
        ).to(device)
        reg.eval()
    else:
        raise ValueError("Unknown model!")

    weights = torch.load(
        dataset_to_weights[f"{trained_model}_{regularizer_name}"],
        map_location=device,
        weights_only=True,
    )
    
    regularizer = ParameterLearningWrapper(reg, device=device)
    regularizer.load_state_dict(weights)


    data_fidelity = L2(sigma=1.0)
    step_size = 1e-1  # step size in the solver
    max_iter = 1000  # maximum number of iterations in the solver
    tol = 1e-4  # tolerance for the relative error (stopping criterion)

    image_size = 256

    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()

    # Setup forward operator and create measurement
    physics = get_forward_op(inp.task, 
                             image_size=image_size, 
                             in_channels=in_channels, 
                             device=device,
                             **vars(inp))
    

    if inp.task == "tomography_sparseview" and inp.misaligned_angles:
        print("Using misaligned angles for tomography_sparseview task")
        physics_sim = get_forward_op(
            degradation_type=inp.task+"_misaligned",
            device=device,
            in_channels=in_channels,
            image_size=image_size,
            **vars(inp)
        )
    else:
        physics_sim = physics  

    psnr_list = []
    ssim_list = []
    lpips_list = []
    time_list = []
    data_fidelity_list = []

    
    for idx in range(len(dataset)):
        if inp.sensitivity_check and not idx % 10 == 0:
            continue
        print(f"Processing image {idx+1}/{len(dataset)}...")
        x_true = dataset[idx].unsqueeze(0).to(device)  # (1, C, H, W)

        if inp.sensitivity_check:
            psnr_values=[]
            ssim_values=[]
            lpips_values=[]
            df_values=[]
            for _ in tqdm(range(40)):
                y = physics_sim.A(x_true) 
                if inp.misaligned_noise:       
                    y=y+sigma_n*torch.sqrt(y)*torch.randn_like(y)
                else:
                    y = y + sigma_n * torch.randn_like(y)

                start_time = time.time()
                x_init=physics.A_adjoint(y)
                if inp.task == "super_resolution":
                    x_init=x_init*inp.scale_factor**2
                x_restored, stats = reconstruct_nmAPG(
                    y,
                    physics,
                    data_fidelity,
                    regularizer,
                    inp.lmbd,
                    step_size,
                    max_iter,
                    tol,
                    x_init=x_init,
                    return_stats=True,
                    verbose=False,
                )
                end_time = time.time()
                duration = end_time - start_time

                noise_norm = inp.sigma_n**2 * np.prod(y.shape[1:])
                y_pred = physics_sim.A(x_restored)
                df_values.append(torch.linalg.norm(y_pred - y)**2 / noise_norm)

                psnr_values.append(psnr_metric.compute(x_restored[0], x_true[0]))
                ssim_values.append(ssim_metric.compute(x_restored[0], x_true[0]))
                lpips_values.append(lpips_metric.compute(x_restored[0].cpu(), x_true[0].cpu()))

            psnr_value = np.array(psnr_values).std()
            ssim_value = np.array(ssim_values).std()
            lpips_value = np.array(lpips_values).std()
            data_fidelity_val = torch.stack(df_values).std()
            print(f"PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}, Data Fidelity: {data_fidelity_val.item():.4f}, Time: {duration:.2f} seconds")
        else:
            y = physics_sim.A(x_true) 
            if inp.task=="tomography_sparseview" and inp.misaligned_noise:       
                y=y+sigma_n*torch.sqrt(y)*torch.randn_like(y)
            else:
                y = y + sigma_n * torch.randn_like(y)

            start_time = time.time()
            x_init=physics.A_adjoint(y)
            if inp.task == "inpainting" and trained_model == "CBSD":
                masked = x_init
                x_init1 = torch.nn.functional.max_pool2d(masked, 3, padding=1, stride=1)
                x_init = torch.nn.functional.max_pool2d(masked, 5, padding=2, stride=1)
                x_init[x_init1 != 0.0] = x_init1[x_init1 != 0.0]
                x_init[masked != 0.0] = masked[masked != 0.0]
            if inp.task == "super_resolution":
                x_init=x_init*inp.scale_factor**2
            x_restored, stats = reconstruct_nmAPG(
                y,
                physics,
                data_fidelity,
                regularizer,
                inp.lmbd,
                step_size,
                max_iter,
                tol,
                x_init=x_init,
                return_stats=True,
                verbose=False,
            )
            end_time = time.time()
            duration = end_time - start_time

            noise_norm = inp.sigma_n**2 * np.prod(y.shape[1:])
            y_pred = physics_sim.A(x_restored)
            data_fidelity_val = torch.linalg.norm(y_pred - y)**2 / noise_norm

            psnr_value = psnr_metric.compute(x_restored[0], x_true[0])
            ssim_value = ssim_metric.compute(x_restored[0], x_true[0])
            lpips_value = lpips_metric.compute(x_restored[0].cpu(), x_true[0].cpu())

            print(f"PSNR: {psnr_value:.4f}, SSIM: {ssim_value:.4f}, LPIPS: {lpips_value:.4f}, Data Fidelity: {data_fidelity_val.item():.4f}, Time: {duration:.2f} seconds")

        if inp.task == "tomography_sparseview":
            save_name = f"img_{idx}_{inp.model_type}_{trained_model}_to_{dataset_name}_{inp.task}_num_angles{inp.num_angles}_lmbd{inp.lmbd}.png"
        elif inp.task == "tomography_limitedangle":
            save_name = f"img_{idx}_{inp.model_type}_{trained_model}_to_{dataset_name}_{inp.task}_missing_wedge{inp.missing_wedge}_lmbd{inp.lmbd}.png"
        else:
            save_name = f"img_{idx}_{inp.model_type}_{trained_model}_to_{dataset_name}_{inp.task}_lmbd{inp.lmbd}.png"

        if "tomography" in inp.task:
            fig, axes = plt.subplots(1, 5, figsize=(16, 4))
        else:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        if in_channels == 1:
            axes[0].imshow(x_true[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")
            
            if inp.task == "inpainting":
                axes[1].imshow(y[0, 0].cpu().numpy(), cmap="gray")
            else:
                axes[1].imshow(y[0, 0].cpu().numpy(), cmap="gray")
            axes[1].set_title(f"Measurement ({inp.task})")
            axes[1].axis("off")

            axes[2].imshow(x_restored[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Reconstruction")
            axes[2].axis("off")
            
            error = torch.abs(x_true - x_restored)
            axes[3].imshow(error[0, 0].cpu().numpy(), cmap="hot")
            axes[3].set_title(f"Error (MSE: {error.pow(2).mean().item():.6f})")
            axes[3].axis("off")

            if "tomography" in inp.task:
                # Show FBP
                x_fbp = physics.fbp(y)
                axes[4].imshow(x_fbp[0, 0].cpu().numpy(), cmap="gray")
                axes[4].set_title("FBP (Filtered Back Projection)")
                axes[4].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=150, bbox_inches="tight")
            plt.close()
        else:
            axes[0].imshow(x_true[0].cpu().permute(1,2,0).numpy())
            axes[0].set_title("Ground Truth")
            axes[0].axis("off")
            

            axes[1].imshow(y[0].cpu().permute(1,2,0).numpy())
            axes[1].set_title(f"Measurement ({inp.task})")
            axes[1].axis("off")

            axes[2].imshow(x_restored[0].cpu().permute(1, 2, 0).numpy())
            axes[2].set_title("Reconstruction")
            axes[2].axis("off")
            
            axes[3].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(save_folder, save_name), dpi=150, bbox_inches="tight")
            plt.close()

        ### save results as pngs 
        if in_channels == 1:
            ### also save as npy numpy 
            np.save(os.path.join(save_folder, save_name.replace(".png", "_x_true.npy")), x_true.cpu().numpy())
            np.save(os.path.join(save_folder, save_name.replace(".png", "_x_restored.npy")), x_restored.cpu().numpy())
            if "tomography" in inp.task:
                np.save(os.path.join(save_folder, save_name.replace(".png", "_x_fbp.npy")), x_fbp.cpu().numpy())

            x_true = (x_true[0, 0].cpu().numpy() * 255).astype("uint8")
            x_true = Image.fromarray(x_true)
            x_true.save(os.path.join(save_folder, f"x_true_img_{idx}.png"))

            x_restored = (np.clip(x_restored[0, 0].cpu().numpy(), 0, 1) * 255).astype("uint8")
            x_restored = Image.fromarray(x_restored)
            x_restored.save(os.path.join(save_folder, save_name.replace(".png", "_restored.png")))

            if "tomography" in inp.task:
                x_fbp = (np.clip(x_fbp[0, 0].cpu().numpy(), 0, 1) * 255).astype("uint8")
                x_fbp = Image.fromarray(x_fbp)
                x_fbp.save(os.path.join(save_folder, f"x_fbp_img_{idx}.png"))
        else: 
            # save images as RGB pngs
            x_true = (x_true[0].cpu().permute(1,2,0).numpy() * 255).astype("uint8")
            print(x_true.shape)
            x_true = Image.fromarray(x_true)
            x_true.save(os.path.join(save_folder, f"x_true_img_{idx}.png"))

            x_restored = (np.clip(x_restored[0].cpu().permute(1, 2, 0).numpy(), 0, 1) * 255).astype("uint8")
            x_restored = Image.fromarray(x_restored)
            x_restored.save(os.path.join(save_folder, save_name.replace(".png", "_restored.png")))
            
        print(f"Results saved to {os.path.join(save_folder, save_name)}")

        psnr_list.append(float(psnr_value))
        ssim_list.append(float(ssim_value))
        lpips_list.append(float(lpips_value))
        time_list.append(float(duration))
        data_fidelity_list.append(float(data_fidelity_val.item()))


    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB \t Average SSIM: {np.mean(ssim_list):.4f} \t Average LPIPS: {np.mean(lpips_list):.4f} \t Average Data Fidelity: {np.mean(data_fidelity_list):.4f} \t Average Time: {np.mean(time_list):.4f}")

    # Save metrics to a JSON file
    metrics_path = os.path.join(save_folder, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "mean_psnr": float(np.mean(psnr_list)),
            "mean_ssim": float(np.mean(ssim_list)),
            "mean_lpips": float(np.mean(lpips_list)),
            "mean_time": float(np.mean(time_list)),
            "mean_data_fidelity": float(np.mean(data_fidelity_list)),
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
    save_dir_summary = os.path.join(inp.save_dir, f"{trained_model}_to_{dataset_name}", task_folder, f"sigma_n={sigma_n}", inp.model_type, inp.part)
    summary_path = os.path.join(save_dir_summary, f"summary_{task_folder}_{inp.part}.txt")

    if not os.path.exists(summary_path):

        with open(summary_path, "w") as f:
            if inp.model_type == "LSR":
                f.write("psnr\tssim\tlpips\tdata_fidelity\ttime\tlmbd\tsigma\n")
            else:
                f.write("psnr\tssim\tlpips\tdata_fidelity\ttime\tlmbd\n")

    with open(summary_path, "a") as f:
        if inp.model_type == "LSR" and False:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(data_fidelity_list)}\t{np.mean(time_list)}\t{inp.lmbd}\t{inp.sigma}\n")
        else:
            f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(data_fidelity_list)}\t{np.mean(time_list)}\t{inp.lmbd}\n")


if __name__ == "__main__":
    main()  
