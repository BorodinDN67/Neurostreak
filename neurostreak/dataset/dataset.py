from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import torch

class NeuroStreakDataset(Dataset):
    def __init__(self, root_dir, groundtruth = True   ):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.image_path = self.root_dir  / 'data'
        self.labels_path = self.root_dir / 'groundtruth.npy'
        self.template_path = self.root_dir / 'template.npy'
        self.groundtruth = groundtruth

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

        labels = None

        if self.groundtruth:
            labels = np.load(self.labels_path)[row, img_n-1]

        img = self.image(str(img_n).zfill(3))

        signal = img[row]

        if self.groundtruth:
            return (torch.tensor(signal).to(dtype=torch.float32),
                    torch.tensor(template).to(dtype=torch.float32),
                    torch.tensor(labels).to(dtype=torch.float32)
                )
        else:
            return (
                torch.tensor(signal).to(dtype=torch.float32),
                torch.tensor(template).to(dtype=torch.float32)
            )

class NeuroStreakConcatDataset(Dataset):
    def __init__(self, root_dirs: str | list | Path, groundtruth = True   ):
        super().__init__()


        if not isinstance(root_dirs, list):
            self.root_dir = [Path(root_dirs)]
        else:
            self.root_dir = [Path(root_dir) for root_dir in root_dirs]

        self.datasets = []
        for root_dir in self.root_dir:
            self.datasets.append(NeuroStreakDataset(root_dir, groundtruth))

        self.len = sum([dataset.__len__ for dataset in self.datasets])

        self.lens = [dataset.__len__ for dataset in self.datasets]



    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        number_dataset = 0
        for i, l in enumerate(self.lens):
            if idx < l:
                idx -= l
            else:
                number_dataset = i
                break

        dataset = self.datasets[number_dataset]

        return dataset.__getitem__(idx)

