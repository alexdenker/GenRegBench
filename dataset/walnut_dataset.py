import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import yaml
import h5py

class WalnutDataset(Dataset):
    def __init__(self, base_dir, crop_size=400, resize_size=256, split="train"):
        assert split in ["train", "val", "test"], "split must be 'train', 'val', or 'test'"
        self.base_dir = base_dir
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.samples = []
        self.norm_constants = {}
        self.split = split.lower()

        # every walnut has 500 slices, we want to skip the first and last 100 slices to avoid artefacts at the beginning and end of the scan.
        self.skip_slices_start = 100
        self.skip_slices_end = 100

        self._prepare_dataset()

    def _prepare_dataset(self):
        norm_path = os.path.join(self.base_dir, "walnut_norm_constants.yaml")
        walnut_folders = [f for f in os.listdir(self.base_dir) if f.startswith('Walnut') and os.path.isdir(os.path.join(self.base_dir, f))]
        walnut_folders = sorted(walnut_folders)  # Ensure consistent order
        print(f"Found walnut folders: {walnut_folders}")
        if os.path.exists(norm_path):
            print(f"Loading normalization constants from {norm_path}")
            with open(norm_path, 'r') as f:
                self.norm_constants = yaml.safe_load(f)
        else:
            for walnut in walnut_folders:
                recon_dir = os.path.join(self.base_dir, walnut, 'Reconstructions')
                if not os.path.isdir(recon_dir):
                    continue
                file_list = sorted(os.listdir(recon_dir))
                file_list = file_list[self.skip_slices_start:-self.skip_slices_end]
                x_min = float('inf')
                x_max = float('-inf')
                for file in file_list:
                    img_path = os.path.join(recon_dir, file)
                    img = Image.open(img_path)
                    img_array = np.array(img)
                    x_min = min(x_min, np.percentile(img_array, 1))  # Use 1st percentile to avoid outliers
                    x_max = max(x_max, np.percentile(img_array, 99))  # Use 99th percentile to avoid outliers
                self.norm_constants[walnut] = (float(x_min), float(x_max))
            with open(norm_path, 'w') as f:
                yaml.safe_dump(self.norm_constants, f)
            print(f"Saved normalization constants to {norm_path}")
        
        # Split logic
        if self.split == "test":
            split_folders = [w for w in walnut_folders if w.lower() == "walnut1"]
        elif self.split == "val":
            split_folders = [w for w in walnut_folders if w.lower() == "walnut2"]
        else:
            split_folders = [w for w in walnut_folders if w.lower() != "walnut1" and w.lower() != "walnut2"]
        for walnut in split_folders:
            recon_dir = os.path.join(self.base_dir, walnut, 'Reconstructions')
            if not os.path.isdir(recon_dir):
                continue
            file_list = sorted(os.listdir(recon_dir))
            file_list = file_list[self.skip_slices_start:-self.skip_slices_end]
            for file in file_list:
                self.samples.append((walnut, file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        walnut, file = self.samples[idx]
        recon_dir = os.path.join(self.base_dir, walnut, 'Reconstructions')
        img_path = os.path.join(recon_dir, file)
        img = Image.open(img_path)
        img_array = np.array(img)
        x_min, x_max = self.norm_constants[walnut]
        img_array = (img_array - x_min) / (x_max - x_min)
        img_array = np.clip(img_array, 0, 1)  # Ensure values are in [0, 1]
        h, w = img_array.shape
        top = (h - self.crop_size) // 2
        left = (w - self.crop_size) // 2
        img_array = img_array[top:top+self.crop_size, left:left+self.crop_size]
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0).unsqueeze(0)  # shape: (1, 1, H, W)
        img_tensor = F.interpolate(img_tensor, size=(self.resize_size, self.resize_size), mode="bilinear", align_corners=False)
        img_tensor = img_tensor.squeeze(0)  # shape: (1, H, W)
        return img_tensor, walnut, file


def create_h5_from_walnut(dataset, h5_path):
    """Save all images and metadata from a WalnutDataset to an HDF5 file."""
    with h5py.File(h5_path, 'w') as f:
        n = len(dataset)
        img_shape = dataset[0][0].shape
        imgs = f.create_dataset('images', shape=(n, *img_shape), dtype='float32')
        walnuts = f.create_dataset('walnuts', shape=(n,), dtype=h5py.string_dtype())
        files = f.create_dataset('files', shape=(n,), dtype=h5py.string_dtype())
        for i in range(n):
            img_tensor, walnut, file = dataset[i]
            imgs[i] = img_tensor.numpy()
            walnuts[i] = walnut
            files[i] = file
        print(f"Saved {n} images to {h5_path}")


class H5WalnutDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, 'r')
        self.imgs = self.h5['images']
        self.walnuts = self.h5['walnuts']
        self.files = self.h5['files']

        self.transform = transform 

    def __len__(self):
        return self.imgs.shape[0]
    
    def __getitem__(self, idx):
        img = torch.from_numpy(self.imgs[idx]).float()

        if self.transform:
            img = self.transform(img)

        #walnut = self.walnuts[idx].decode() if hasattr(self.walnuts[idx], 'decode') else self.walnuts[idx]
        #file = self.files[idx].decode() if hasattr(self.files[idx], 'decode') else self.files[idx]
        return img


if __name__ == "__main__":
    dataset = WalnutDataset(base_dir="dataset/walnuts", split="train")
    print(f"Dataset length (train): {len(dataset)}")
    create_h5_from_walnut(dataset, "walnut_train.h5")
    dataset = WalnutDataset(base_dir="dataset/walnuts", split="val")
    print(f"Dataset length (val): {len(dataset)}")
    create_h5_from_walnut(dataset, "walnut_val.h5")
    dataset = WalnutDataset(base_dir="dataset/walnuts", split="test")
    print(f"Dataset length (test): {len(dataset)}")
    create_h5_from_walnut(dataset, "walnut_test.h5")


    #create_h5_from_walnut(dataset, "walnut_test.h5")
    dataset_train = H5WalnutDataset("walnut_train.h5")
    dataset_test = H5WalnutDataset("walnut_test.h5")
    print(f"Dataset length (train): {len(dataset_train)}")
    print(f"Dataset length (test): {len(dataset_test)}")

    min_intensity = float('inf')
    max_intensity = float('-inf')
    for i in range(len(dataset_train)):
        img_tensor = dataset_train[i]
        min_intensity = min(min_intensity, img_tensor.min().item())
        max_intensity = max(max_intensity, img_tensor.max().item())
    print(f"Intensity range in training set: ({min_intensity}, {max_intensity})")

    min_intensity = float('inf')
    max_intensity = float('-inf')
    for i in range(len(dataset_test)):
        img_tensor = dataset_test[i]
        min_intensity = min(min_intensity, img_tensor.min().item())
        max_intensity = max(max_intensity, img_tensor.max().item())
    print(f"Intensity range in test set: ({min_intensity}, {max_intensity})")

    # img_tensor = dataset_train[len(dataset_train) // 2]  # Get a sample from the middle of the dataset
    # print(f"Sample shape: {img_tensor.shape}")

    # # Visualize the image tensor
    # import matplotlib.pyplot as plt
    # plt.imshow(img_tensor[0].detach().numpy(), cmap="gray", vmin=0, vmax=1)
    # plt.axis("off")
    # plt.show()

    from scipy.linalg import sqrtm

    def compute_fid(features_train, features_test):
        """Compute Fréchet distance between two sets of features."""
        mu1, sigma1 = np.mean(features_train, axis=0), np.cov(features_train, rowvar=False)
        mu2, sigma2 = np.mean(features_test, axis=0), np.cov(features_test, rowvar=False)
        
        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid

    # Use flattened images (or PCA-reduced features for efficiency)
    from sklearn.decomposition import PCA

    all_train = np.stack([dataset_train[i].numpy().flatten() for i in range(len(dataset_train))])
    all_test = np.stack([dataset_test[i].numpy().flatten() for i in range(len(dataset_test))])

    # Reduce dimensionality with PCA
    pca = PCA(n_components=256)
    pca.fit(all_train)
    print("Fitted PCA, explained variance ratio:", np.sum(pca.explained_variance_ratio_))
    train_features = pca.transform(all_train)
    # compute FID on two random subsets of 2000 samples from the train set
    idx_train = np.random.choice(len(train_features), size=2000, replace=False)
    fid = compute_fid(train_features[idx_train[:300]], train_features[idx_train[300:600]])
    print(f"FID (random subsets of train): {fid:.4f}")

    train_features = pca.transform(all_train)
    test_features = pca.transform(all_test)

    fid = compute_fid(train_features[0:300], test_features)
    print(f"FID (PCA features): {fid:.4f}")

    fid = compute_fid(train_features[0:300], train_features[600:900])
    print(f"FID (two train walnuts): {fid:.4f}")

    from skimage.metrics import structural_similarity as ssim
    from tqdm import tqdm 
    # For each test image, find average SSIM to a random subset of train images
    avg_ssims = []
    for i in tqdm(range(len(dataset_test))):
        test_img = dataset_test[i].squeeze(0).numpy()  # (H, W)
        ssim_values = []
        indices = np.random.choice(len(dataset_train), size=min(50, len(dataset_train)), replace=False)
        for j in indices:
            train_img = dataset_train[j].squeeze(0).numpy()  # (H, W)
            ssim_values.append(ssim(test_img, train_img, data_range=1.0))
        avg_ssims.append(max(ssim_values))  # best match

    print(f"Average best SSIM (test to train): {np.mean(avg_ssims):.4f} ± {np.std(avg_ssims):.4f}")