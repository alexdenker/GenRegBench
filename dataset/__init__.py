import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TransformDataset(Dataset):
    """Wraps an existing dataset and applies a transform to each sample."""

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

class DatasetFromNumpy(Dataset):
    """Dataset wrapper for loading samples from a numpy array."""

    def __init__(self, numpy_array):
        self.numpy_array = numpy_array

    def __len__(self):
        return len(self.numpy_array)

    def __getitem__(self, idx):
        sample = self.numpy_array[idx]
        sample = torch.from_numpy(sample)#.float()  # Convert to PyTorch tensor
        return sample


def get_dataset(name, part):
    """
    Load a dataset by name and split.
    The datasets are:
        - "ellipses": Synthetic dataset of random ellipses.
        - "walnut": Walnut CT dataset 
        - "celebahq": CelebA-HQ face dataset
        - "afhq": AFHQ animal faces dataset

    Parameters
    ----------
    name : str
        Dataset name. One of: "ellipses", "walnut", "celebahq", "afhq".
    part : str
        Dataset split. One of: "train", "val", "test".

    Returns
    -------
    torch.utils.data.Dataset
    """
    assert part in ["train", "val", "test"], f"Invalid part: '{part}'. Must be 'train', 'val', or 'test'."
    assert name in ["ellipses", "walnut", "celebahq", "afhq", "ffhq", "aapm"], f"Unknown dataset: '{name}'. Available: 'ellipses', 'walnut', 'celebahq', 'afhq', 'ffhq', 'aapm'."

    if os.path.exists(f"dataset/{name}_{part}.npy"):
        print(f"Loading {name} ({part}) from numpy file.")
        dataset_numpy = np.load(f"dataset/{name}_{part}.npy")
        return DatasetFromNumpy(dataset_numpy)

    if name == "ellipses":
        from .generate_ellipses import DiskDistributedEllipsesDataset
        seed = {"train": 1, "val": 2, "test": 3}[part]
        length = {"train": 10000, "val": 10, "test": 100}[part]

        dataset = DiskDistributedEllipsesDataset(
            fold=part,
            shape=(256, 256),
            length=length,
            diameter=0.4745,
            fixed_seed=seed,
            max_n_ellipse=70,
        )

        return dataset

    elif name == "walnut":
        from .walnut_dataset import H5WalnutDataset
        if part == "train":
            h5_dataset_path = "dataset/walnut_train.h5"
        elif part == "val":
            h5_dataset_path = "dataset/walnut_val.h5"
        elif part == "test":
            h5_dataset_path = "dataset/walnut_test.h5"
        else:
            raise ValueError(f"Invalid part: '{part}'. Must be 'train', 'val', or 'test'.")

        assert os.path.exists(h5_dataset_path), f"H5 dataset not found at {h5_dataset_path}. Please create using create_h5_from_walnut or download."

        if part == "train":
            return H5WalnutDataset(h5_dataset_path)

        dataset_h5 = H5WalnutDataset(h5_dataset_path)

        if part == "val":
            val_indices = list(np.linspace(0, len(dataset_h5) - 1, 10, dtype=int)) 
            return torch.utils.data.Subset(dataset_h5, indices=val_indices)
        
        if part == "test":
            test_indices = list(np.linspace(0, len(dataset_h5) - 1, 100, dtype=int))
            return torch.utils.data.Subset(dataset_h5, indices=test_indices)
      
    elif name == "celebahq":
        from .celebahq import get_celeba_hq_datasets
        train_set, val_set, test_set = get_celeba_hq_datasets(data_root="dataset")
        return {"train": train_set, "val": val_set, "test": test_set}[part]

    elif name == "afhq":
        from .afhq import get_afhq_datasets
        train_set, val_set, test_set = get_afhq_datasets(data_root="dataset")
        return {"train": train_set, "val": val_set, "test": test_set}[part]

    elif name == "ffhq":
        from .ffhq import FFHQDataset
        assert part in ["test", "val"], f"FFHQ dataset only has 'test' and 'val' splits, but got '{part}'."
        return FFHQDataset(part=part)
    elif name == "aapm":
        from .aapm import AAPMDataset
        assert part in ["train", "test", "val"], f"AAPM dataset only has 'train', 'test', and 'val' splits, but got '{part}'."

        if part == "train":
            return AAPMDataset('dataset/aapm/aapm_train.h5')
        elif part == "val":
            
            dataset = AAPMDataset('dataset/aapm/aapm_train.h5')
            # get a validation subset of 10 random samples from the training set
            val_indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42)).tolist()[:10]
            return torch.utils.data.Subset(dataset, indices=val_indices)
        else:
            return AAPMDataset('dataset/aapm/aapm_test.h5')
    
    else:
        raise ValueError(
            f"Unknown dataset: '{name}'. "
            "Available: 'ellipses', 'walnut', 'celebahq', 'afhq', 'ffhq', 'aapm'"
        )


