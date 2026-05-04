import os
import sys
import time  
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
from PIL import Image
import json 

from deepinv.optim import L2
from tqdm import tqdm


from utils.eval_metrics import PSNR, SSIM, LPIPS
from utils.degradation import get_forward_op
from dataset import get_dataset
from diffusers import UNet2DModel
import deepinv

parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--trained_model", type=str, default="diskellipses", choices=["walnut", "diskellipses", "celebahq", "aapm"],
                    help="name of the pretrained model")
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

parser.add_argument("--alpha", type=float, default=1.0,
                    help="Regularization parameter schedule")
parser.add_argument("--gamma", type=float, default=1.0,
                    help="Regularization parameter")
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

dataset_name = inp.dataset_name
trained_model = inp.trained_model
part = inp.part
task = inp.task
alpha = inp.alpha
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

model_type = "pnpflow"


save_folder = os.path.join(inp.save_dir, f"{trained_model}_to_{dataset_name}", task_folder, f"sigma_n={sigma_n}", model_type, inp.part)
save_folder = os.path.join(save_folder, f"alpha={inp.alpha}_gamma={inp.gamma}")

os.makedirs(save_folder, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

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

class UNetVelocity(torch.nn.Module):
    # wrapper to swap arguments of the forward of the unet
    def __init__(self,model,channels):
        super(UNetVelocity,self).__init__()
        self.unet=model
        self.add_module("UNet", self.unet)
        self.channels=channels
        

    def forward(self,t,x,labels=None):
        if len(t.shape)==0:
            t=torch.ones((x.shape[0],),dtype=x.dtype,device=x.device)*t
        if x.shape[1]==1 and self.channels>1:
            x=x.tile(1,self.channels,1,1)
            return self.unet(x,t).sample.mean(1,keepdim=True)
        return self.unet(x,t).sample

def reconstruct_pnp_flow(y,velocity,physics, data_fidelity,n_steps,alpha,sigma_sq,x_acc=5,implicit=True):
    y=2*y-physics.A(torch.ones_like(physics.A_adjoint(y)))
    x = physics.A_adjoint(y)
    for iteration in range(n_steps):
        t=torch.tensor(iteration/n_steps).to(y)
        gamma_schedule = (1 - t)**alpha
        gamma_base = inp.gamma #sparse angle 16 and 128 200. # inpainting explicit 6 implicit 12 # deblurring 30 # superres 2 30
        gamma=gamma_base*gamma_schedule
        if implicit:
            #arg = gamma*physics.A_adjoint(y)+ x
            #mat = lambda x: gamma*physics.A_adjoint(physics.A(x))+ x
            z = deepinv.optim.linear.least_squares(physics.A,physics.A_adjoint,y,x,gamma=gamma)
        else:
            z = x-gamma * data_fidelity.grad(x, y, physics)
        xs=[]
        for _ in range(x_acc):
            tilde_z = (1-t) * torch.randn_like(z) + t * z
            x = tilde_z + (1-t) * velocity(t,tilde_z)
            xs.append(x)
        xs=torch.stack(xs,0)
        x=torch.mean(xs,0)
    x=(x+1)/2
    return x

dataset = get_dataset(dataset_name, part=inp.part)
in_channels = dataset[0].shape[0]


channels= 3 if trained_model=="celebahq" else in_channels
image_size = 256
unet=create_model(image_size,channels,channels)
velocity = UNetVelocity(unet,channels=channels).to(device)
for p in velocity.parameters():
    p.requires_grad_(False)

wname = inp.trained_model
if wname == "diskellipses":
    wname = "ellipses"
velocity.load_state_dict(torch.load(f"weights/{wname}/pnpflow/velocity_{wname}_final.pt"))

data_fidelity = L2(sigma=1.0)


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
            x_restored = reconstruct_pnp_flow(y,velocity,physics,data_fidelity,100,alpha,sigma_n**2)
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
        if inp.misaligned_noise:
            y=y+sigma_n*torch.sqrt(y)*torch.randn_like(y)
        else:
            y = y + sigma_n * torch.randn_like(y)

        start_time = time.time()
        x_restored = reconstruct_pnp_flow(y,velocity,physics,data_fidelity,100,alpha,sigma_n**2)
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
        save_name = f"img_{idx}_{model_type}_{trained_model}_to_{dataset_name}_{inp.task}_num_angles{inp.num_angles}_alpha{inp.alpha}_gamma{inp.gamma}.png"
    elif inp.task == "tomography_limitedangle":
        save_name = f"img_{idx}_{model_type}_{trained_model}_to_{dataset_name}_{inp.task}_missing_wedge{inp.missing_wedge}_alpha{inp.alpha}_gamma{inp.gamma}.png"
    else:
        save_name = f"img_{idx}_{model_type}_{trained_model}_to_{dataset_name}_{inp.task}_alpha{inp.alpha}_gamma{inp.gamma}.png"

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
save_dir_summary = os.path.join(inp.save_dir, f"{trained_model}_to_{dataset_name}", task_folder, f"sigma_n={sigma_n}", model_type, inp.part)
summary_path = os.path.join(save_dir_summary, f"summary_{task_folder}_{inp.part}.txt")

if not os.path.exists(summary_path):

    with open(summary_path, "w") as f:
        f.write("psnr\tssim\tlpips\tdata_fidelity\ttime\talpha\tgamma\n")

with open(summary_path, "a") as f:
    f.write(f"{np.mean(psnr_list)}\t{np.mean(ssim_list)}\t{np.mean(lpips_list)}\t{np.mean(data_fidelity_list)}\t{np.mean(time_list)}\t{inp.alpha}\t{inp.gamma}\n")

print(np.mean(psnr_list))