

import torch
from deepinv.physics import Inpainting, Downsampling, Tomography, BlurFFT, TomographyWithAstra
from deepinv.utils import load_degradation

def get_forward_op(degradation_type: str, device: str, in_channels: int, image_size: int, **kwargs):
    """
    degradatation_type are:
    - "inpainting": InpaintingOperator (random mask, random fixed with seed)
    - "super_resolution": SuperResolutionOperator (downsampling), with scale factor specified in kwargs (e.g., scale_factor=4)
    - "deblurring": DeblurringOperator (motion blur)
    - "tomography_sparseview": Radon Transform with number of angles specified in kwargs (e.g., num_angles=30)
    - "tomography_limitedangle": Radon Transform with limited angle specified in kwargs (e.g., missing_wedge=10)
    
    device: "cuda" or "cpu"
    in_channels: number of input channels (e.g., 1 for grayscale, 3 for RGB)
    image_size: height and width of the input image (assumed square)
    kwargs: additional parameters for specific degradation types
        scale_factor: for super-resolution
        num_angles: for tomography_sparseview
        missing_wedge: for tomography_limitedangle

    """
    assert degradation_type in ["inpainting", "box_inpainting", "super_resolution", "deblurring", "tomography_sparseview", "tomography_limitedangle", "tomography_sparseview_misaligned"], f"Unsupported degradation type: {degradation_type}"

    if degradation_type == "inpainting": # 60% random mask with fixed seed for reproducibility
        print("Use inpainting with random mask (seed=42)")
        mask_seed = 42 
        mask = torch.rand(1, 1, 256, 256, generator=torch.Generator().manual_seed(mask_seed)) > 0.6
        physics = Inpainting(mask=mask.to(device).float(), img_size=(in_channels, image_size, image_size), device=device)
    elif degradation_type == "box_inpainting": 
        mask = torch.ones(1, 1, 256, 256)
        box_size = 90
        start = (image_size - box_size) // 2
        mask[:, :, start:start+box_size, start:start+box_size] = 0
        physics = Inpainting(mask=mask.to(device).float(), img_size=(in_channels, image_size, image_size), device=device)
    elif degradation_type == "super_resolution":
        scale_factor = kwargs.get("scale_factor", 4)
        print("Use super-resolution with scale factor:", scale_factor)

        physics = Downsampling(factor=scale_factor, 
                               img_size=(in_channels, image_size, image_size),
                               filter='gaussian',
                               device=device)

    elif degradation_type == "tomography_sparseview":
        num_angles = kwargs.get("num_angles", 30)
        print("Use tomography with sparse view, num angles:", num_angles)
        angles = torch.linspace(0, 180, steps=num_angles + 1, device=device)[:-1].to(device)
        # physics = Tomography(
        #     angles=angles,
        #     img_width=image_size,
        #     circle=False,
        #     device=device,
        #     normalize=True
        # )
        physics = TomographyWithAstra(
            angles=angles,
            img_size=(image_size, image_size),
            device=device,
            normalize=True
        )
    elif degradation_type == "tomography_sparseview_misaligned":
        num_angles = kwargs.get("num_angles", 30)
        angle_noise_range = 0.7 # hardcoded for now, can be added to kwargs if needed
        angles = torch.linspace(0, 180, steps=num_angles + 1, device=device)[:-1].to(device)
        generator = torch.Generator(device=device).manual_seed(42)  # for reproducibility
        angle_noise = torch.rand(angles.shape, generator=generator, device=device) * angle_noise_range * 2 - angle_noise_range
        print(angle_noise)
        # angle_noise is uniformly distributed in the range of [-angle_noise_range, angle_noise_range]
        angles_noisy = angles + angle_noise

        physics = TomographyWithAstra(
            angles=angles_noisy,
            img_size=(image_size, image_size),
            device=device,
            normalize=True
        )
    # add tomography_sparseview_misaligned

    elif degradation_type == "tomography_limitedangle":
        missing_wedge = kwargs.get("missing_wedge", 10)
        print("Use tomography with limited angle, missing wedge:", missing_wedge)
        angles = torch.linspace(missing_wedge//2, 180 - missing_wedge//2, steps=101, device=device)[:-1].to(device)
        # physics = Tomography(
        #     angles=angles,
        #     img_width=image_size,
        #     circle=False,
        #     device=device,
        #     normalize=True
        # )
        physics = TomographyWithAstra(
            angles=angles,
            img_size=(image_size, image_size),
            device=device,
            normalize=True
        )
    elif degradation_type == "deblurring":
        kernel_index = 1  # which kernel to chose among the 8 motion kernels from 'Levin09.mat'
        kernel_torch = load_degradation(
            name ="Levin09.npy", index=kernel_index, download =True
        ).to(torch.float32)
        kernel_torch = kernel_torch.unsqueeze(0).unsqueeze(
            0
        )  # add batch and channel dimensions
        physics = BlurFFT(
        img_size=(in_channels, image_size, image_size),
        filter=kernel_torch,
        device=device,
        )

    else:
        raise ValueError(f"Unsupported degradation type: {degradation_type}")

    
    return physics