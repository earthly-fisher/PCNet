from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BinaryDataset(Dataset):
    """data/{train,validation,test}/{image,vessel}/ with identical filenames."""
    def __init__(self, root, split):
        folder = Path(root) / split
        self.labels = folder / 'vessel'
        self.images = sorted(x for x in (folder / 'image').iterdir()
                             if x.is_file() and x.suffix.lower() in {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.gif'})
        if not self.images:
            raise ValueError(f'No images in {folder / "image"}')
        for image in self.images:
            if not (self.labels / image.name).is_file():
                raise FileNotFoundError(self.labels / image.name)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as source:
            image = np.array(source.convert('L'), dtype=np.float32) / 255
        with Image.open(self.labels / path.name) as source:
            label = np.array(source.convert('L'), dtype=np.float32) / 255
        if image.shape != label.shape:
            raise ValueError(f'Image and mask dimensions differ: {path.name}')
        if any(size < 16 or size % 16 for size in image.shape):
            raise ValueError(f'Image dimensions must be multiples of 16: {path.name}')
        return torch.from_numpy(image).unsqueeze(0), torch.from_numpy((label > 0.5).astype(np.float32)).unsqueeze(0), path.name
