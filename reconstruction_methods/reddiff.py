"""
Mardani et al. A Variational Perspective on Solving Inverse Problems with Diffusion Models (2023)

"""


from typing import Callable, Optional

import torch
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

from deepinv.physics import Physics, Inpainting


class REDDiff:
    """
    REDDiff: A Variational Perspective on Solving Inverse Problems with Diffusion Models 
    """
    
    def __init__(
        self,
        model: UNet2DModel,
        scheduler: DDPMScheduler,
        device: str = "cuda",
    ):
        """
        Args:
            model: Pretrained diffusion model (noise predictor)
            scheduler: DDPM scheduler with alpha schedules
            device: Device to run on
        """
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
        # Get alpha schedule from scheduler
        self.alphas_cumprod = scheduler.alphas_cumprod.to(device)
        self.num_timesteps = scheduler.config.num_train_timesteps
    
    def predict_x0(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x0 from x_t using the diffusion model.
        
        x_0 = (x_t - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
        
        Or equivalently using score:
        x_0 = (x_t + (1 - alpha_t) * score) / sqrt(alpha_t)
        """
        with torch.no_grad():
            # Predict noise
            noise_pred = self.model(x_t, t, return_dict=False)[0]
        
        # Get alpha values
        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Predict x0: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        x0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        return x0_pred, noise_pred
    
    def sample(
        self,
        y: torch.Tensor,
        physics: Physics,
        x_init: Optional[torch.Tensor] = None,
        lr: float = 0.01,
        sigma_x0: float = 0.0001,
        obs_weight: float = 1.0,
        grad_term_weight: float = 0.25,
        denoise_term_weight: str = "linear",
        num_inference_steps: int = 1000,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Run RED-diff sampling.


        mu = argmin_mu ||y - H(mu)||^2 + rho * E_{t,noise}[||eps(mu + noise_t) - noise_t||^2]

        Args:
            y: Measurement (degraded image)
            H: Forward operator
            x_init: Optional initial image estimate (if None, start from zeros)
            lr: Learning rate for optimizing mu
            sigma_x0: Noise level for sampling x_t from mu (should be small, e.g., 0.0001)
            obs_weight: Weight for the data consistency term ||y - H(mu)||^2
            grad_term_weight: Weight for the denoising term E_{t,noise}[||eps(mu + noise_t) - noise_t||^2]
            denoise_term_weight: How to weight the denoising term based on noise level (options: "linear", "sqrt", "square", "log", "trunc_linear", "power2over3", "const")
            num_inference_steps: Number of diffusion steps
            show_progress: Show progress bar
            
        Returns:
            x0: Restored image
        """
        self.model.eval()
        
        # Get image dimensions from model config
        batch_size = y.shape[0]
        image_size = self.model.config.sample_size
        in_channels = self.model.config.in_channels
        
        # Step 1: Initialize the clean image estimate (mu) # this should be in range [-1,1]
        if x_init is not None:
            mu_tensor = x_init.clone().to(self.device)
        else:
            mu_tensor = torch.zeros(batch_size, in_channels, image_size, image_size, device=self.device)

        
        # Setup timesteps (from T to 1)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        mu = torch.autograd.Variable(mu_tensor, requires_grad=True)

        # Noise schedule variances for rho calculation
        # sigma_bar_t^2 = (1 - alpha_t) / alpha_t (variance of x_t given x_0)
        optimizer = torch.optim.Adam(
            [mu], lr=lr, betas=(0.9, 0.99), weight_decay=0.0
        )

        iterator = tqdm(timesteps, desc="RED-diff Reconstruction") if show_progress else timesteps
        
        print("MU: ", mu.shape)
        img_channels = mu.shape[1]
        for i, t in enumerate(iterator):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            #print(t, timesteps[i + 1] if i < len(timesteps) - 1 else "final")
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            
            # Corrupt mu to get x_t
            noise_x0 = torch.randn_like(mu)
            noise_xt = torch.randn_like(mu)

            x0_pred = mu + sigma_x0 * noise_x0
            xt = torch.sqrt(alpha_t) * x0_pred + torch.sqrt(1 - alpha_t) * noise_xt
            
            # Step 1: Predict noise from x_t
            with torch.no_grad():
                # if the input channels are 3, blow up the single channel x0_pred to 3 channels by repeating it
                if in_channels == 3 and img_channels == 1:
                    xt = xt.repeat(1, 3, 1, 1)
                _, noise_pred = self.predict_x0(xt, t_tensor)

                if in_channels == 3 and img_channels == 1:
                    # take the mean of the 3 channels to get back to single channel
                    noise_pred = noise_pred.mean(dim=1, keepdim=True)

            # Important: The diffusion models works with images in [-1, 1]. 
            # Our observations y are in the original scale (e.g., for images in [0,1]).
            # So we need to scale x0_pred to [0,1] before applying data consistency, and then scale back to [-1, 1] after.
            x0_pred_scaled = (x0_pred + 1) / 2  # Scale to [0, 1]

            e_obs = y - physics.A(x0_pred_scaled)
            loss_obs = (e_obs ** 2).sum() / 2

            loss_noise = torch.mul((noise_pred - noise_xt).detach(), x0_pred).sum()

            snr_inv = torch.sqrt(1 - alpha_t) / torch.sqrt(alpha_t)
            
            if denoise_term_weight == "linear":
                snr_inv_weight = snr_inv
            elif denoise_term_weight == "sqrt":
                snr_inv_weight = torch.sqrt(snr_inv)
            elif denoise_term_weight == "square":
                snr_inv_weight = torch.square(snr_inv)
            elif denoise_term_weight == "log":
                snr_inv_weight = torch.log(snr_inv + 1.0)
            elif denoise_term_weight == "trunc_linear":
                snr_inv_weight = torch.clip(snr_inv, max=1.0)
            elif denoise_term_weight == "power2over3":
                snr_inv_weight = torch.pow(snr_inv, 2 / 3)
            elif denoise_term_weight == "const":
                snr_inv_weight = torch.pow(snr_inv, 0.0)
            else:
                snr_inv_weight = snr_inv

            w_t = grad_term_weight * snr_inv_weight
            v_t = obs_weight

            # Total Energy Loss
            loss = w_t * loss_noise + v_t * loss_obs

            # Step 5: Optimization Step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Return the optimized clean image estimate
        return mu.detach()

