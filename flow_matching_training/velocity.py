import torch
from torchdiffeq import odeint

device = "cuda" if torch.cuda.is_available() else "cpu"

class UNetVelocity(torch.nn.Module):
    # wrapper to swap arguments of the forward of the unet
    def __init__(self,model):
        super(UNetVelocity,self).__init__()
        self.unet=model
        self.add_module("UNet", self.unet)
        

    def forward(self,t,x,labels=None):
        if len(t.shape)==0:
            t=torch.ones((x.shape[0],),dtype=x.dtype,device=x.device)*t
        return self.unet(x,t).sample

def flow_matching_sample(
    velocity_field, samples, reverse=False, T=1.0, rtol=1e-5, atol=1e-5, solver="dopri5", step_size=0.05,
):
    if reverse:
        integration_times = torch.tensor([T, 0.0], device=device, dtype=torch.float)
    else:
        integration_times = torch.tensor([0.0, T], device=device, dtype=torch.float)
    if solver=="rk4" or solver=="euler":
        options=dict(step_size=step_size)
    else:
        options={}

    state = odeint(
        velocity_field, samples, integration_times, atol=atol, rtol=rtol, method=solver, options=options
    )[-1]
    return state


def flow_matching_path(
    velocity_field, samples, n_steps, T=1.0, rtol=1e-5, atol=1e-5, solver="dopri5"
):
    integration_times = torch.linspace(0, T, n_steps + 1).to(device)
    if solver=="rk4":
        options=dict(step_size=0.05)
    else:
        options={}
    states = odeint(
        velocity_field, samples, integration_times, atol=atol, rtol=rtol, method=solver, options=options
    )
    return states


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time

def flow_matching_loss(velocity_field, x_0, x_1, skewed=False):
    if skewed:
        t = skewed_timestep_sample(x_0.shape[0],device=x_0.device)
    else:
        t = torch.rand((x_0.shape[0]),dtype=x_0.dtype,device=x_0.device)
    t_ = t
    while len(t_.shape) < len(x_0.shape):
        t_=t_.unsqueeze(-1)
    x_t = (1-t_) * x_0 + t_ * x_1
    dx_t = x_1 - x_0
    loss = torch.mean(((velocity_field(t.squeeze(), x_t) - dx_t) ** 2).view(x_t.shape[0], -1), 1)
    return loss



