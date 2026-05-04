import os
import numpy as np
from tqdm import tqdm
import pydicom
import h5py
from skimage.transform import resize

data_path = os.path.join('aapm', 'full_1mm')
train_subdir = 'full_1mm'
test_subdir = 'full_1mm'
output_dir = 'aapm'
train_h5_path = os.path.join(output_dir, 'aapm_train.h5')
test_h5_path = os.path.join(output_dir, 'aapm_test.h5')
train_folders = ['L067', 'L096', 'L109', 'L192', 'L286', 'L291', 'L310', 'L333']
test_folders = ['L506']

# Step 1: Collect global min and max across all volumes
global_min, global_max = np.inf, -np.inf

for folder in train_folders:
    cur_folder = os.path.join(data_path, folder, train_subdir)
    files = os.listdir(cur_folder)
    #print("files:", files)
    files.sort()
    for file in files:
        img = pydicom.dcmread(os.path.join(cur_folder, file)).pixel_array.astype(np.float32)
        global_min = min(global_min, img.min())
        global_max = max(global_max, img.max())

for folder in test_folders:
    cur_folder = os.path.join(data_path, folder, test_subdir)
    files = os.listdir(cur_folder)
    files.sort()
    for file in files:
        img = pydicom.dcmread(os.path.join(cur_folder, file)).pixel_array.astype(np.float32)
        global_min = min(global_min, img.min())
        global_max = max(global_max, img.max())


train_imgs = []
test_imgs = []



def normalize_to_unit_interval(img, min_val, max_val):
    if max_val == min_val:
        return np.zeros_like(img, dtype=np.float32)
    return (img - min_val) / (max_val - min_val)


def collect_images(folders, subdir, desc):
    imgs = []
    for folder in tqdm(folders, desc=desc):
        cur_folder = os.path.join(data_path, folder, subdir)
        files = os.listdir(cur_folder)
        files.sort()
        for file in files:
            img = pydicom.dcmread(os.path.join(cur_folder, file)).pixel_array.astype(np.float32)
            img_norm = normalize_to_unit_interval(img, global_min, global_max)
            img_norm = resize(
                img_norm,
                (256, 256),
                order=1,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            ).astype(np.float32)
            

            imgs.append(img_norm)
    return imgs


# Step 2: Normalize using global min/max and save to HDF5
train_imgs = collect_images(train_folders, train_subdir, 'Processing Train')
test_imgs = collect_images(test_folders, test_subdir, 'Processing Test')

print("Number of train images:", len(train_imgs))
print("Number of test images:", len(test_imgs))

if len(train_imgs) == 0 or len(test_imgs) == 0:
    raise RuntimeError("No images found for train or test split.")

with h5py.File(train_h5_path, 'w') as f:
    f.create_dataset('images', data=np.stack(train_imgs), dtype='float32')

with h5py.File(test_h5_path, 'w') as f:
    f.create_dataset('images', data=np.stack(test_imgs), dtype='float32')

print(f"Saved train H5 to {train_h5_path}")
print(f"Saved test H5 to {test_h5_path}")