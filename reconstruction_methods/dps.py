
import torch
from tqdm import tqdm

from deepinv.physics import Physics
from diffusers import UNet2DModel, DDPMScheduler


class DPS:
    """
    Diffusion Posterior Sampling for inverse problems.
    
    Based on: "Diffusion Posterior Sampling for General Noisy Inverse Problems"
    Implements gradient-based posterior guidance during reverse diffusion.
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
            scheduler: Diffusion scheduler with alpha schedules
            H: Forward operator for inverse problem
            device: Device to run on
            grad_term_weight: Weight for gradient term in posterior guidance
            eta: Stochasticity parameter (0=deterministic, 1=stochastic)
            cond_awd: Whether to use conditional adaptive weighting
            original: Whether to use original gradient weighting scheme
        """
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
        # Get alpha schedule from scheduler
        self.alphas_cumprod = scheduler.alphas_cumprod.to(device)
        self.num_timesteps = scheduler.config.num_train_timesteps
        
        # DPS-specific parameters
        self.eta = 1.0
        self.cond_awd = False
        self.original = True

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get cumulative product of alphas for timesteps t.
        
        Args:
            t: Tensor of timestep indices
            
        Returns:
            Alpha values for given timesteps
        """
        return self.alphas_cumprod[t]
    
    def predict_x0(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict x0 from x_t using the diffusion model.
        
        x_0 = (x_t - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
        
        Args:
            x_t: Noisy sample at timestep t
            t: Timestep indices
            
        Returns:
            x0_pred: Predicted clean sample
        """
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
        grad_coeff: float = 1.0,
        num_inference_steps: int = 1000,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Generate posterior sample using DPS with gradient guidance.
        
        Uses DDIM sampling with posterior guidance from forward operator.
        
        DDIM update rule with posterior guidance:
        x_{t-1} = sqrt(alpha_{t-1}) * x0_pred 
                + sqrt(1 - alpha_{t-1} - sigma_t^2) * eps_pred
                + sigma_t * noise
                - grad_coeff * ∇_x_t ||y - H(x0)||^2
        
        Args:
            x: Initial noise (batch_size, C, H, W)
            y: Measurement (batch_size, m)
            ts: Timesteps to traverse (descending, e.g., [999, 799, 599, ...])
            
        Returns:
            x_final: Final sample (batch_size, C, H, W)
        """
        
        # Get image dimensions from model config
        batch_size = y.shape[0]
        image_size = self.model.config.sample_size
        model_in_channels = self.model.config.in_channels
        data_in_channels = y.shape[1]
        
        # Initialize x_T ~ N(0, I)
        x_t = torch.randn(batch_size, model_in_channels, image_size, image_size, device=self.device)
        
        # Start with initial noise
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        iterator = tqdm(timesteps, desc="DPS Sampling") if show_progress else timesteps
        
        for i, t in enumerate(iterator):
            # Current and previous timestep alphas
            t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            alpha_t = self.alphas_cumprod[t]
            
            # Get previous timestep (or 1.0 at final step)
            if i < len(timesteps) - 1:
                t_prev_idx = timesteps[i + 1]
                alpha_prev = self.alphas_cumprod[t_prev_idx].view(1, 1, 1, 1)
            else:
                alpha_prev = torch.tensor(1.0, device=self.device).view(1, 1, 1, 1)
            
            # Prepare for gradient computation
            x_t = x_t.clone().requires_grad_(True)
            
            # Predict noise from model
            x0_pred, eps_pred = self.predict_x0(x_t, t_tensor)
            
            
            # ========== POSTERIOR GUIDANCE ==========
            # Compute gradient of data fidelity term
            x0_pred_scaled = (x0_pred + 1) / 2  # Scale to [0, 1]
            if model_in_channels > data_in_channels:
                x0_pred_scaled = x0_pred_scaled.mean(dim=1, keepdim=True)  # Convert to grayscale by averaging channels
            residual = y - physics.A(x0_pred_scaled)
            data_term = (residual.reshape(batch_size, -1) ** 2).sum()
            
            # Compute gradient w.r.t. x_t
            grad_guidance = torch.autograd.grad(data_term, x_t, retain_graph=True)[0]
            
            x_t = x_t.detach()
            eps_pred = eps_pred.detach()
            x0_pred = x0_pred.detach()
            grad_guidance = grad_guidance.detach()
            
            # Compute gradient weighting coefficient
            if self.original:
                # Scale by norm of residual
                residual_norm = (residual.reshape(batch_size, -1) ** 2).sum(dim=1).sqrt().detach()
                grad_coeff_term = grad_coeff / (residual_norm.reshape(-1, 1, 1, 1) + 1e-8)
            else:
                # Alternative scaling
                grad_coeff_term = torch.tensor(grad_coeff, device=self.device).view(1, 1, 1, 1)
            
            # ========== DDIM UPDATE ==========
            # Calculate variance for stochastic DDIM
            if self.eta > 0 and i < len(timesteps) - 1:
                variance = (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                sigma_t = self.eta * torch.sqrt(torch.clamp(variance, min=0))
            else:
                sigma_t = torch.tensor(0.0, device=self.device)
            
            # Coefficient for noise direction
            pred_dir_coef = torch.sqrt(torch.clamp(1 - alpha_prev - sigma_t**2, min=0))

            # DDIM step: x_{t-1} = sqrt(alpha_{t-1}) * x0 + sqrt(1-alpha_{t-1}-sigma_t^2) * eps_pred + sigma_t * noise
            x_t_next = torch.sqrt(alpha_prev) * x0_pred + pred_dir_coef * eps_pred
            
            # Add stochastic noise
            if sigma_t > 0:
                noise = torch.randn_like(x_t_next)
                x_t_next = x_t_next + sigma_t * noise
            
            # Apply posterior guidance (subtract gradient)
            x_t = x_t_next - grad_coeff_term * grad_guidance
        
        if model_in_channels > data_in_channels:
            x_t = x_t.mean(dim=1, keepdim=True)  # Convert to grayscale by averaging channels


        return x_t
