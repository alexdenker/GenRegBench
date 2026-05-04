# Evaluates a TV baseline on the test problems.

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from deepinv.utils.plotting import plot
from deepinv.models import TVDenoiser
from deepinv.physics import LinearPhysics
from deepinv.loss.metric import PSNR
import numpy as np
from tqdm import tqdm
import argparse
from dataset import get_dataset
from operators import get_evaluation_setting

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("device: ", device)
torch.random.manual_seed(0)  # make results deterministic

############################################################

# Problem selection
parser = argparse.ArgumentParser(description="Choosing evaluation setting")
parser.add_argument("--problem", type=str, default="Denoising")
inp = parser.parse_args()

problem = inp.problem  # Select problem setups, which we consider.
only_first = False  # just evaluate on the first image of the dataset for test purposes

############################################################

# reconstruction hyperparameters, might be problem dependent
if problem == "CT":
    lmbd_choices = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    ]  # choices to test for the regularization parameter
    Lip = 1e5  # squared spectral norm of physics
    tau = 1.0 / Lip
    sigma = Lip / (2 * 8)  # spectral norm of gradient
elif problem == "Denoising":
    lmbd_choices = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
    Lip = 2.0
    tau = 1.0 / Lip
    sigma = Lip / (2 * 8)  # spectral norm of gradient

#############################################################
############# Problem setup and evaluation ##################
#############################################################

# Define forward operator
dataset, physics, data_fidelity = get_evaluation_setting(problem, device)

# validation set for parameter fitting
if problem == "CT":
    validation_dataset = get_dataset("LoDoPaB_val")
elif problem == "Denoising":
    validation_dataset = torch.utils.data.Subset(
        get_dataset("BSDS500_gray", test=False), range(5)
    )


class TV_Reconstruction:
    def __init__(
        self,
        lambd,
        steps,
        sigma,
        tau,
        stopping_criterion=1e-5,
    ):
        self.lambd = lambd
        self.steps = steps
        self.sigma = sigma
        self.tau = tau
        self.stopping_criterion = stopping_criterion
        self.physics = LinearPhysics(
            A=TVDenoiser.nabla, A_adjoint=TVDenoiser.nabla_adjoint
        )

        def gradient_data_fidelity(x, y):
            return data_fidelity.grad(x, y, physics)

        self.f_grad = gradient_data_fidelity

    def optimize(self, x_init, y):
        # reconstrction with Condat-Vu primal-dual hybrid gradient algorithm
        x = x_init
        b = torch.zeros_like(self.physics.A(x))

        for step in range(self.steps):
            x_old = x.clone()

            # primal update
            grad_f = self.f_grad(x, y)
            x = x - self.tau * (grad_f + self.physics.A_adjoint(b))

            # Dual update
            b = b + self.sigma * self.physics.A(2 * x - x_old)
            b = b.clip(min=-self.lambd, max=self.lambd)

            rel_err = torch.linalg.norm(
                x_old.flatten() - x.flatten()
            ) / torch.linalg.norm(x.flatten() + 1e-12)
            if (rel_err < self.stopping_criterion) & (step > 10):
                print("Converged at iteration:", step)
                break

        return x


def eval_TV(lmbd, dataset, plot_example=False):

    optimizer = TV_Reconstruction(
        lmbd,
        20000,
        sigma,
        tau,
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    if problem == "CT":
        psnr = PSNR(max_pixel=None)
        dagger_name = "FBP"
    elif problem == "Denoising":
        psnr = PSNR()
        dagger_name = "Noisy"

    psnrs = []
    psnrs_dagger = []
    for i, x in (progress_bar := tqdm(enumerate(dataloader))):
        if device == "mps":
            x = x.to(torch.float32).to(device)
        else:
            x = x.to(device).to(torch.float32)
        y = physics(x)
        x_init = physics.A_dagger(y)
        psnrs_dagger.append(psnr(x_init, x).squeeze().item())
        recon = optimizer.optimize(x_init, y)
        psnrs.append(psnr(recon, x).squeeze().item())
        progress_bar.set_description(
            "Mean so far: {0:.2f}, Last: {1:.2f}, {2} so far: {3:.2f}, Last {4:.2f}".format(
                np.mean(psnrs),
                psnrs[-1],
                dagger_name,
                np.mean(psnrs_dagger),
                psnrs_dagger[-1],
            )
        )
        if i == 0:
            y_out = y
            x_out = x
            recon_out = recon
        if only_first:
            break
    mean_psnr = np.mean(psnrs)
    mean_psnr_dagger = np.mean(psnrs_dagger)
    print("Mean PSNR over the test set: {0:.2f}".format(mean_psnr))
    print(
        "Mean PSNR "
        + dagger_name
        + " over the test set: {0:.2f}".format(mean_psnr_dagger)
    )

    # plot ground truth, observation and reconstruction for the first image from the test dataset
    if plot_example:
        plot([x_out, y_out, recon_out])
    return mean_psnr


best_lmbd = -1
best_psnr = -1000
for lmbd in lmbd_choices:
    mean_psnr = eval_TV(lmbd, validation_dataset, plot_example=False)
    if mean_psnr > best_psnr:
        best_lmbd = lmbd
        best_psnr = mean_psnr
print(
    "Best lambda: {0}, best PSNR on validation set: {1:.2f}".format(
        best_lmbd, best_psnr
    )
)

eval_TV(best_lmbd, dataset, plot_example=True)
