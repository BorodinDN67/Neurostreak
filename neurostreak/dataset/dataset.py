from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image

class NeuroStreakDataset(Dataset):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image = self.root_dir  / 'data'
        self.labels = self.root_dir / 'groundtruth.npy'
        self.template = self.root_dir / 'templates.npy'

        self.len = len(list(Path(root_dir).iterdir()))
        self.dim = 2048
        self.cashe = dict()

    def __len__(self):
        return self.len * self.dim

    def image(self, idx):
        if idx in self.cashe:
            return self.cashe[idx]
        else:
            img = np.array(Image.open(self.image / idx/'.tif'))
            self.cashe[idx] = img
            return self.cashe[idx]


    def __getitem__(self, idx):
        img_n, row = idx // self.dim, idx % self.dim

        template = np.load(self.template)
        labels = np.load(self.labels)

        img = self.image(str(img_n).zfill(3))

        signal = img[row]

        return (signal, template) , labels
