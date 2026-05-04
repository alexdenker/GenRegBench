import torch
import torch.nn as nn
from deepinv.optim import Prior
from deepinv.models.GSPnP import GSDRUNet
from deepinv.models import EquivariantDenoiser

class LSR(Prior):
    def __init__(
        self,
        channels=1, 
        device="cpu",
        pretrained=None, 
        deepinv=True,
        nc=(
            64,
            128,
            256,
            512,
        ),
        alpha=1.0,
        sigma=0.05,
        act_mode="E",
    ):
        super(LSR, self).__init__()

        self.model = GSDRUNet(
            alpha=alpha,
            in_channels=channels,
            out_channels=channels,
            nb=2,
            nc=nc,
            act_mode=act_mode,
            pretrained="download" if deepinv else None,
            device=device,
        )
        
        self.model.detach = False
        self.channels = channels

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained, map_location=device))

        self.sigma = sigma

    def grad(self, x, sigma=None):
        if sigma is None:
            sigma = self.sigma
        if self.channels>1 and x.shape[1]==1:
            x=x.tile(1,self.channels,1,1)
            return self.model.potential_grad(x, sigma).sum(1,keepdim=True)
        return self.model.potential_grad(x, sigma)

    def g(self, x, sigma=None):
        if sigma is None:
            sigma = self.sigma
        if self.channels>1 and x.shape[1]==1:
            x=x.tile(1,self.channels,1,1)
        return self.model.potential(x, sigma)
