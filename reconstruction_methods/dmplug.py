import torch
from torch.optim import Adam
from tqdm import tqdm
import numpy as np 
from diffusers import UNet2DModel, DDPMScheduler

from deepinv.physics import Physics

def early_stopping(patience, delta, w, variance_list, es_dict):
    """
    Early stopping based on the running variance of the prediction. 
    Variance calculated using a sliding window approach.
    
    patience: number of iterations to wait after last minimum before stopping
    delta: threshold factor for new minimum (e.g. 0.99 means new minimum must be at least 1% lower)
    w: window size for variance calculation
    variance_list: list to store variance values
    es_dict: dict to store {'value': 0, 'index': 0, 'output': None}
    """

    g_min = float('inf')
    i_min = 0 # early stopping index 
    stopped = False
    x_buffer = []

    def callback(i, x_pred):
        nonlocal x_buffer, g_min, i_min, stopped
        
        x = x_pred.detach().cpu().numpy().ravel()
        x_buffer.append(x)
        if len(x_buffer) > w:
            x_buffer.pop(0)
        
        running_var = np.mean(np.var(x_buffer, axis=0))
        variance_list.append(running_var)
        if len(x_buffer) == w and not stopped:
            g_i = running_var
            if g_i < delta * g_min:
                #print("New Best Index:", i)
                g_min = g_i
                i_min = i
                es_dict['index'] = i_min
                es_dict['reco'] = x_pred.detach().cpu().clone()

            if i >= i_min + patience:
                #print("Early stopping at index:", i)
                stopped = True 

        es_dict['stopped'] = stopped
        
    return callback




class DMPlug:
    """
    Optimizing the initial noise vector z 

    Based on: "DMPlug: A Plug-in Method for Solving Inverse Problems with Diffusion Models" (https://arxiv.org/pdf/2405.16749)
    """
    
    def __init__(
        self,
        model: UNet2DModel,
        scheduler: DDPMScheduler,
        device: str = "cuda",
    ):
        
        self.model = model
        self.scheduler = scheduler
        self.device = device
        
        # Get alpha schedule from scheduler
        self.alphas_cumprod = scheduler.alphas_cumprod.to(device)
        self.num_timesteps = scheduler.config.num_train_timesteps


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
        num_inference_steps: int = 4,
        adam_lr: float = 1e-2,
        adam_steps: int = 200,
        show_progress: bool = False,
        patience: int = 300,
        delta: float = 0.99,
        w: int = 50,
    ) -> torch.Tensor:
        """
        Optimize initial noise z using Adam, with gradients from measurement consistency.
        Runs K steps of DDIM, then updates z with Adam based on measurement loss.
        Args:
            y: Measurement (batch_size, ...)
            H: ForwardOperator
            num_inference_steps: Number of DDIM steps
            adam_lr: Learning rate for Adam
            adam_steps: Maximum number of Adam steps
            show_progress: Show progress bar
            patience: Early stopping — steps to wait after last variance minimum
            delta: Early stopping — threshold factor for a new minimum (e.g. 0.99 means 1% lower)
            w: Early stopping — sliding window size for running variance
        Returns:
            x_final: Final sample (batch_size, C, H, W)
        """
        batch_size = y.shape[0]
        # Assume model input shape is (B, C, H, W)
        model_input_channels = self.model.config.in_channels
        data_input_channels = y.shape[1]
        
        shape = (batch_size, self.model.config.in_channels, self.model.config.sample_size, self.model.config.sample_size)
        device = self.device

        z = torch.randn(shape, device=device, requires_grad=True)
        optimizer = Adam([z], lr=adam_lr)

        # Timesteps for DDIM (descending)
        timesteps = torch.linspace(self.num_timesteps-1, 0, num_inference_steps, dtype=torch.long, device=device)

        # Early stopping setup
        variance_list = []
        es_dict = {'value': 0, 'index': 0, 'reco': None, 'stopped': False}
        es_callback = early_stopping(patience, delta, w, variance_list, es_dict)

        iterator = tqdm(range(adam_steps), desc="DMPlug") if show_progress else range(adam_steps)
        for step in iterator:
            optimizer.zero_grad()
            x = z
            # DDIM sampling trajectory
            for j in range(num_inference_steps):
                t = timesteps[j].expand(batch_size)
                x0_pred, noise_pred = self.predict_x0(x, t)
                # DDIM update
                if j < num_inference_steps - 1:
                    t_prev = timesteps[j+1].expand(batch_size)
                else:
                    t_prev = torch.zeros_like(t)

                alpha_prev = self.alphas_cumprod[t_prev]
                sigma_t = 0  # Deterministic DDIM
                # DDIM update rule
                x = (
                    torch.sqrt(alpha_prev).view(-1, 1, 1, 1) * x0_pred +
                    torch.sqrt(1 - alpha_prev - sigma_t**2).view(-1, 1, 1, 1) * noise_pred
                )

            x_final = x
            if model_input_channels > data_input_channels:
                # If model has more channels than data, take the mean
                x_final = x_final.mean(dim=1, keepdim=True)
            x_final_rescale = (x_final + 1) / 2  # Rescale from [-1,1] to [0,1]
            y_pred = physics.A(x_final_rescale)
            loss = torch.nn.functional.mse_loss(y_pred, y)
            loss.backward()
            optimizer.step()

            es_callback(step, x_final.detach())
            if es_dict['stopped']:
                break

        return es_dict['reco'].to(device)
