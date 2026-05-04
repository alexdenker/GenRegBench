"""
DiffPIR: Denoising Diffusion Models for Plug-and-Play Image Restoration

Implementation based on:
"Denoising Diffusion Models for Plug-and-Play Image Restoration" (CVPR 2023)

Algorithm:
1. Predict x0_hat from noisy x_t using diffusion model
2. Make x0_hat data consistent via proximal optimization
3. Add noise back to get x_{t-1}
"""

import math 

import torch
from tqdm import tqdm
from diffusers import UNet2DModel, DDPMScheduler

from deepinv.physics import Physics, Inpainting

class DiffPIR:
    """
    DiffPIR: Diffusion model based Plug-and-Play Image Restoration.
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
    
    def data_consistency_step(
        self,
        x0_pred: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        rho: float,
        num_iters: int = 50,
        step_size: float = 0.1,
    ) -> torch.Tensor:
        """
        Solve the proximal subproblem:
        x0_hat = argmin_x ||y - H(x)||^2 + rho * ||x - x0_pred||^2
        
        Using gradient descent.
        
        Args:
            x0_pred: Predicted x0 from diffusion model
            y: Measurement
            H: Forward operator
            rho: Regularization weight (lambda * sigma_n^2 / sigma_t^2)
            num_iters: Number of optimization iterations
            step_size: Gradient descent step size
            
        Returns:
            x0_hat: Data-consistent x0
        """
        
        x = x0_pred.clone().requires_grad_(True)

        for _ in range(num_iters):
            # Data fidelity: ||y - H(x)||^2
            residual = y - physics.A(x)
            data_term = torch.sum(residual ** 2)
            
            # Regularization: rho * ||x - x0_pred||^2
            reg_term = rho * torch.sum((x - x0_pred) ** 2)
            
            # Total loss
            loss = data_term + reg_term
            
            # Gradient descent
            grad = torch.autograd.grad(loss, x)[0]
            x = x - step_size * grad
            x = x.detach().requires_grad_(True)

        return x.detach()
    
    def data_consistency_step_cg(
        self,
        x0_pred: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        rho: float,
        num_iters: int = 20,
    ) -> torch.Tensor:
        """
        Solve the proximal subproblem using Conjugate Gradient:
        x0_hat = argmin_x ||y - H(x)||^2 + rho * ||x - x0_pred||^2
        
        This is equivalent to solving the linear system:
        (H^T H + rho I) x = H^T y + rho * x0_pred
        
        Using conjugate gradient with a fixed number of iterations.
        
        Args:
            x0_pred: Predicted x0 from diffusion model
            y: Measurement
            H: Forward operator
            rho: Regularization weight (lambda * sigma_n^2 / sigma_t^2)
            num_iters: Number of CG iterations (fixed)
            
        Returns:
            x0_hat: Data-consistent x0
        """
        # Right-hand side: b = H^T y + rho * x0_pred
        b = physics.A_adjoint(y) + rho * x0_pred
        
        # Define the linear operator A = H^T H + rho I
        def apply_A(x: torch.Tensor) -> torch.Tensor:
            return physics.A_adjoint(physics.A(x)) + rho * x
        
        # Initialize CG with x0_pred as starting point
        x = x0_pred.clone()
        
        # Initial residual: r = b - A(x)
        r = b - apply_A(x)
        p = r.clone()
        
        # r^T r (flattened dot product)
        r_dot_r = torch.sum(r * r)
        
        for _ in range(num_iters):
            # Compute A @ p
            Ap = apply_A(p)
            
            # Step size: alpha = (r^T r) / (p^T A p)
            p_dot_Ap = torch.sum(p * Ap)
            alpha = r_dot_r / (p_dot_Ap + 1e-10)
            
            # Update solution: x = x + alpha * p
            x = x + alpha * p
            
            # Update residual: r = r - alpha * A p
            r = r - alpha * Ap
            
            # New r^T r
            r_dot_r_new = torch.sum(r * r)
            
            # Update direction: beta = (r_new^T r_new) / (r^T r)
            beta = r_dot_r_new / (r_dot_r + 1e-10)
            
            # New search direction: p = r + beta * p
            p = r + beta * p
            
            # Update r^T r for next iteration
            r_dot_r = r_dot_r_new
        
        return x
    
    def data_consistency_closed_form(
        self,
        x0_pred: torch.Tensor,
        y: torch.Tensor,
        physics: Physics,
        rho: float,
    ) -> torch.Tensor:
        """
        Closed-form solution for simple operators (e.g., inpainting, identity).
        
        For H = mask (diagonal):
        x0_hat = (H^T y + rho * x0_pred) / (H^T H + rho)
        
        This is faster than iterative optimization.
        """
        # For diagonal operators: (H^T H + rho I)^{-1} (H^T y + rho x0_pred)
        Hty = physics.A_adjoint(y)
        HtH_diag = physics.A_adjoint(physics.A(torch.ones_like(x0_pred)))
        
        x0_hat = (Hty + rho * x0_pred) / (HtH_diag + rho + 1e-8)
        
        return x0_hat
    
    
    def sample(
        self,
        y: torch.Tensor,
        physics: Physics,
        sigma_n: float = 0.05,
        lam: float = 1.0,
        zeta: float = 0.3,
        num_inference_steps: int = 100,
        use_closed_form: bool = True,
        first_order_approx: bool = False,
        show_progress: bool = True,
    ) -> torch.Tensor:
        """
        Run DiffPIR sampling.

        1. Predict x0_hat from x_t using diffusion model
        2. Make x0_hat data consistent via proximal optimization
                argmin_x ||y - H(x)||^2 + rho * ||x - x0_pred||^2
                where rho = lambda * sigma_n^2 / sigma_t^2
        3. Add noise back to get x_{t-1}

        The authors do not minimise the data consistency term exactly at each step, 
        but rather use a single gradient step (first-order approximation):
        x0_hat = x0_pred - 1/(2 rho) * grad_x0_pred (data consistency gradient)


        Args:
            y: Measurement (degraded image)
            H: Forward operator
            sigma_n: Noise level in measurement
            lam: Regularization parameter lambda
            zeta: Noise mixing parameter (0 = deterministic, 1 = stochastic)
            num_inference_steps: Number of diffusion steps
            use_closed_form: Use closed-form DC for diagonal operators
            first_order_approx: Use first-order approximation for DC step (no inner optimization)
            show_progress: Show progress bar
            
        Returns:
            x0: Restored image
        """
        self.model.eval()
        
        # Get image dimensions from model config
        batch_size = y.shape[0]
        image_size = self.model.config.sample_size
        model_in_channels = self.model.config.in_channels
        data_in_channels = y.shape[1]

        # Initialize x_T ~ N(0, I)
        x_t = torch.randn(batch_size, model_in_channels, image_size, image_size, device=self.device)
        
        # Setup timesteps (from T to 1)
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
                
        iterator = tqdm(timesteps, desc="DiffPIR Reconstruction") if show_progress else timesteps
        for i, t in enumerate(iterator):
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            #print(t, timesteps[i + 1] if i < len(timesteps) - 1 else "final")
            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i < len(timesteps) - 1 else torch.tensor(1.0)
            
            # Calculate rho_t = lambda * sigma_n^2 / sigma_bar_t^2
            # sigma_bar_t^2 = (1 - alpha_t) / alpha_t
            sigma_bar_t_sq = (1 - alpha_t) / alpha_t
            rho_t = lam * (sigma_n ** 2) / (sigma_bar_t_sq + 1e-8)
            with torch.no_grad():
                # Step 1: Predict x0 from x_t
                
                x0_pred, _ = self.predict_x0(x_t, t_tensor)
                                    
            # Important: The diffusion models works with images in [-1, 1]. 
            # Our observations y are in the original scale (e.g., for images in [0,1]).
            # So we need to scale x0_pred to [0,1] before applying data consistency, and then scale back to [-1, 1] after.
            x0_pred_scaled = (x0_pred + 1) / 2  # Scale to [0, 1]

            # do data consistency on the correct channels 
            # in particular, if the model is RGB, but data is grayscale, we first to the mean over the channels 
            if model_in_channels > data_in_channels:
                x0_pred_scaled = x0_pred_scaled.mean(dim=1, keepdim=True)  # Convert to grayscale by averaging channels
            
            # Step 2: Data consistency - make x0_pred consistent with measurement y
            # argmin_x ||y - H(x)||^2 + rho_t * ||x - x0_pred||^2
            # rho_t = lambda * sigma_n^2 / sigma_bar_t^2
            # sigma_bar_t^2 = (1 - alpha_t) / alpha_t is the variance of the noise in x_t given x_0, which determines how much we trust the diffusion model's prediction at this step.

            if use_closed_form and isinstance(physics, Inpainting):
                x0_hat = self.data_consistency_closed_form(x0_pred_scaled, y, physics, rho_t.item())
            else:
                # TODO: Does not work, is super unstable. rho_t is nearly zero at early steps? 
                # In the DiffPIR, they do not use the squared norm and scale the gradient by the norm
                # which is both not in the paper, only hidden in the code! 
                if first_order_approx:
                    # First-order approximation: single gradient step
                    with torch.enable_grad():
                        x0_pred_scaled = x0_pred_scaled.clone().detach().requires_grad_(True)
                        e_obs = y - physics.A(x0_pred_scaled)
                        grad = -physics.A_adjoint(e_obs)  # Gradient of data consistency term w.r.t. x0_pred

                    x0_hat = x0_pred_scaled - grad / (2 * rho_t.item() + 1e-8)  # Update with scaled gradient step
                    x0_hat = x0_hat.detach()  # Detach to prevent gradients from flowing back into the diffusion model
                else:
                    x0_hat = self.data_consistency_step_cg(
                            x0_pred_scaled, y, physics, rho_t.item(), num_iters=4
                        )
            x0_hat = x0_hat * 2 - 1  # Scale back to [-1, 1]

            # map back to the model's input channels if needed
            if model_in_channels > data_in_channels:
                x0_hat = x0_hat.repeat(1, model_in_channels, 1, 1)  # Repeat channels to match model input

            # Step 3: Calculate effective noise estimate
            # epsilon_hat = (x_t - sqrt(alpha_t) * x0_hat) / sqrt(1 - alpha_t)
            alpha_t_val = alpha_t.view(-1, 1, 1, 1)
            epsilon_hat = (x_t - torch.sqrt(alpha_t_val) * x0_hat) / torch.sqrt(1 - alpha_t_val)
            
            # Step 4: Sample fresh noise
            epsilon_t = torch.randn_like(x_t)
            
            # Step 5: Compute x_{t-1}
            # x_{t-1} = sqrt(alpha_{t-1}) * x0_hat + sqrt(1 - alpha_{t-1}) * (sqrt(1-zeta) * epsilon_hat + sqrt(zeta) * epsilon_t)
            alpha_prev_val = alpha_prev.view(-1, 1, 1, 1)
            
            # No noise at final step
            if i == len(timesteps) - 1:
                x_t = x0_hat
            else:
                noise_component = math.sqrt(1 - zeta) * epsilon_hat + math.sqrt(zeta) * epsilon_t
                x_t = torch.sqrt(alpha_prev_val) * x0_hat + torch.sqrt(1 - alpha_prev_val) * noise_component
        
        # again take the mean over the channels if the model is RGB but data is grayscale
        if model_in_channels > data_in_channels:
            x_t = x_t.mean(dim=1, keepdim=True)  # Convert to grayscale by averaging channels


        return x_t

