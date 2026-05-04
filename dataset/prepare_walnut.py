

import os 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.transform import resize

base_path = "walnuts/Walnut3/Reconstructions"

file_list = sorted(os.listdir(base_path))

# compute normalisation constant 
x_min = float("inf")
x_max = float("-inf")
for file in file_list:
    img_path = os.path.join(base_path, file)
    img = Image.open(img_path)  # Convert to grayscale
    img_array = np.array(img)   
    x_min = min(x_min, img_array.min())
    x_max = max(x_max, img_array.max())

print(f"Normalization constants: min={x_min}, max={x_max}")
print(file_list)
idxs = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]
for i, file in enumerate(file_list):
    if i not in idxs:
        continue
    print(f"Processing {file}...")
    img_path = os.path.join(base_path, file)
    img = Image.open(img_path)  # Convert to grayscale
    #img = img.resize((256, 256), Image.Resampling.LANCZOS)
    img_array = np.array(img)   

    img_array = (img_array - x_min) / (x_max - x_min)  # Normalize to [0, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.set_title(f"Slice {i}, name: {file}")
    ax1.imshow(img_array, cmap="gray")
    ax1.axis("off")

    # center crop to 400x400
    h, w = img_array.shape
    top = (h - 400) // 2
    left = (w - 400) // 2
    img_array = img_array[top:top+400, left:left+400]

    # bilinear interpolation to 256x256
    img_array = resize(img_array, (256, 256), order=1, mode="reflect", anti_aliasing=True)

    ax2.set_title("cropped to 400x400 and then resized to 256x256")
    ax2.imshow(img_array, cmap="gray")
    ax2.axis("off")

    plt.show()