"""
Evaluation metrics for image reconstruction tasks.

For all metrics the inputs should be in range [0,1] and of shape (C, H, W) or (H, W) for grayscale.

"""

import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim

try:
    import lpips
    _lpips_available = True
except ImportError:
    _lpips_available = False


class PSNR:
    def __init__(self, data_range=1.0):
        self.data_range = data_range

    def compute(self, img1, img2):
        """
        Compute PSNR between two images. Images are expected to be in range [0,1] and of shape (C, H, W) or (1, H, W) for grayscale.
        """
        assert img1.shape == img2.shape, "Input images must have the same shape"
        assert len(img1.shape) == 3, "Input images must have 3 dimensions (C, H, W)"
        assert len(img2.shape) == 3, "Input images must have 3 dimensions (C, H, W)"

        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        img1 = np.squeeze(img1)
        img2 = np.squeeze(img2)
        return compare_psnr(img1, img2, data_range=self.data_range)


class SSIM:
    def __init__(self, data_range=1.0):
        self.data_range = data_range

    def compute(self, img1, img2):
        """
        Compute SSIM between two images. Images are expected to be in range [0,1] and of shape (C, H, W) or (1, H, W) for grayscale.
        """
        assert img1.shape == img2.shape, "Input images must have the same shape"
        assert len(img1.shape) == 3, "Input images must have 3 dimensions (C, H, W)"
        assert len(img2.shape) == 3, "Input images must have 3 dimensions (C, H, W)"

        if isinstance(img1, torch.Tensor):
            img1 = img1.detach().cpu().numpy()
        if isinstance(img2, torch.Tensor):
            img2 = img2.detach().cpu().numpy()
        img1 = np.squeeze(img1)
        img2 = np.squeeze(img2)
        multichannel = img1.ndim == 3 and img1.shape[0] in [1, 3]
        if multichannel:
            img1 = np.transpose(img1, (1, 2, 0))
            img2 = np.transpose(img2, (1, 2, 0))
        return compare_ssim(img1, img2, data_range=self.data_range, channel_axis=-1 if multichannel else None)


class LPIPS:
    def __init__(self, net='alex'):
        if not _lpips_available:
            raise ImportError("lpips library is not installed. Please install it with 'pip install lpips'.")
        self.loss_fn = lpips.LPIPS(net=net)

    def compute(self, img1, img2):
        """
        Compute LPIPS between two images. Images are expected to be in range [0,1] and of shape (C, H, W) or (1, H, W) for grayscale.
        If grayscale (C=1), the channel is repeated to create an RGB image.
        """
        assert img1.shape == img2.shape, "Input images must have the same shape"
        assert len(img1.shape) == 3, "Input images must have 3 dimensions (C, H, W)"
        assert len(img2.shape) == 3, "Input images must have 3 dimensions (C, H, W)"

        if img1.ndim == 3:
            img1 = img1.unsqueeze(0)
        if img2.ndim == 3:
            img2 = img2.unsqueeze(0)
        # Repeat grayscale channel to RGB if needed
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
        if img2.shape[1] == 1:
            img2 = img2.repeat(1, 3, 1, 1)
        with torch.no_grad():
            d = self.loss_fn(img1 * 2 - 1, img2 * 2 - 1)
        return d.item()


if __name__ == "__main__":

    img1 = torch.rand(1, 128, 128)  
    img2 = torch.rand(1, 128, 128)  

    psnr_metric = PSNR()
    ssim_metric = SSIM()
    lpips_metric = LPIPS()
    print("PSNR:", psnr_metric.compute(img1, img2))
    print("SSIM:", ssim_metric.compute(img1, img2))
    print("LPIPS:", lpips_metric.compute(img1, img2))
