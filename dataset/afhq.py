import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class AFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in sorted(os.listdir(root_dir))
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def get_afhq_datasets(
    data_root,
    val_len=10,
    test_len=100,
    seed=42,
    transform=None
):
    dataset = AFHQDataset(
        root_dir=os.path.join(data_root, 'cat'),
        transform=transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC),
        ])
    )
    total_len = len(dataset)
    train_len = total_len - val_len - test_len
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)

    return train_set, val_set, test_set

if __name__ == "__main__":
    data_root = "dataset"
    train_set, val_set, test_set = get_afhq_datasets(data_root)
    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2,4, figsize=(12,6))
    for idx, ax in enumerate(axes.flatten()):
        img = train_set[idx]
        print(img)
        ax.imshow(img.permute(1, 2, 0).cpu().numpy())
        ax.axis("off")
    plt.tight_layout()
    plt.show()