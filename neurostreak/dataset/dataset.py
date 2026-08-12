from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import torch

class NeuroStreakDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_path = self.root_dir  / 'data'
        self.labels_path = self.root_dir / 'groundtruth.npy'
        self.template_path = self.root_dir / 'template.npy'

        self.len = len(list(Path(self.image_path).glob('*.tif')) )
        self.dim = 2048
        self.cashe = dict()

    def __len__(self):
        return self.len * self.dim

    def image(self, idx):
        if idx in self.cashe:
            return self.cashe[idx]
        else:
            filename = idx + '.tif'
            img = np.array(Image.open(self.image_path /filename))
            self.cashe[idx] = img
            return self.cashe[idx]


    def __getitem__(self, idx):
        img_n, row = idx // self.dim + 1, idx % self.dim

        template = np.load(self.template_path)
        labels = np.load(self.labels_path)[row, img_n-1]

        img = self.image(str(img_n).zfill(3))

        signal = img[row]

        return torch.tensor(signal), torch.tensor(template) , torch.tensor(labels).to(dtype=torch.float32)
