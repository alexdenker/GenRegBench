
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import yaml
import h5py



class AAPMDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.h5 = h5py.File(h5_path, 'r')
        
        self.transform = transform 

    def __len__(self):
        return self.h5['images'].shape[0]
    
    def __getitem__(self, idx):
        img = torch.from_numpy(self.h5['images'][idx]).float().unsqueeze(0)

        if self.transform:
            img = self.transform(img)

        return img


if __name__ == "__main__":
    dataset = AAPMDataset('dataset/aapm/aapm_train.h5')
    dataset_test = AAPMDataset('dataset/aapm/aapm_test.h5')
    print("Dataset length:", len(dataset))
    print("Test Dataset length:", len(dataset_test))
    sample = dataset[0]
    print("Sample shape:", sample.shape)

    import matplotlib.pyplot as plt 

    plt.figure()
    plt.imshow(sample.squeeze(0).numpy(), cmap='gray')
    plt.title("Sample Image from AAPM Dataset")
    plt.colorbar()
    plt.savefig("aapm_sample.png")
    plt.close()