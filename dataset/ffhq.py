
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset


class FFHQDataset(Dataset):
    """FFHQ dataset loaded via HuggingFace, downsampled to 256x256.

    test: first 100 images (indices 0-99)
    val:  images 100-109    (indices 100-109)
    """

    SPLITS = {
        "test": (0, 100),
        "val": (100, 110),
    }

    def __init__(self, part: str = "test"):
        assert part in self.SPLITS, f"part must be one of {list(self.SPLITS.keys())}"
        start, end = self.SPLITS[part]

        # Load only the slice we need via streaming to avoid downloading all 70k images
        ds = load_dataset("marcosv/ffhq-dataset", split="train", streaming=True)
        self.images = []
        for i, sample in enumerate(ds):
            if i >= end:
                break
            if i >= start:
                self.images.append(sample["image"])

        # Lanczos downsampling to 256x256 (high-quality, anti-aliased)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.transform(self.images[idx])


if __name__ == "__main__":
    for part in ("test", "val"):
        ds = FFHQDataset(part=part)
        print(f"{part}: {len(ds)} images, first sample shape: {ds[0].shape}")


    import matplotlib.pyplot as plt
    ds = FFHQDataset(part="test")
    img = ds[0].permute(1, 2, 0).numpy()
    plt.figure()
    plt.imshow(img)
    plt.title("First image from FFHQ test set")
    plt.axis("off")
    plt.savefig("ffhq_sample_test.png")

    ds = FFHQDataset(part="val")
    img = ds[0].permute(1, 2, 0).numpy()
    plt.figure()
    plt.imshow(img)
    plt.title("First image from FFHQ validation set")
    plt.axis("off")
    plt.savefig("ffhq_sample_val.png")

