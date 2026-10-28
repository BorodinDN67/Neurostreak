import pathlib

from src.config.config import Config
from .dataset import NeuroStreakDataset, NeuroStreakConcatDataset
import yaml
import numpy as np
from torch.utils.data import Subset, ConcatDataset, DataLoader


def split_dataset(
    dataset: NeuroStreakConcatDataset,
    train_scheme: str | pathlib.Path,
    validation_scheme: str | pathlib.Path,
    test_scheme: str | pathlib.Path,
    config_dl: Config
):
    """
пример конфига выборок:

sub_datasets:
  - "clean_water_10m"
  - "clean_water_13m"
  - "clean_water_15m"
  - "clean_water_20m"

config:
  -
    sub_idx: 0
    row_slice: [750, 1260]
    col_slice: [220, 300]
  -
    sub_idx: 1
    row_slice: [750, 1240]
    col_slice: [165, 262]
  -
    sub_idx: 2
    row_slice: [750, 1240]
    col_slice: [145, 225]
  -
    sub_idx: 3
    row_slice: [750, 1190]
    col_slice: [130, 201]

    """

    #Загрузили конфиги
    configs = []

    for config in [train_scheme, validation_scheme, test_scheme]:
        with open(config) as f:
            loaded_config = yaml.safe_load(f)
            configs.append(loaded_config['config'])

    configs = {
        'train': configs[0],
        'validation': configs[1],
        'test': configs[2],
    }
    # поехали
    datasets = dataset.datasets
    dim = dataset.dim
    subsets = {
        'train': [],
        'validation': [],
        'test': [],
    }

    for split in ['train', 'validation', 'test']:
        for sub_config in configs[split]:
            idx = sub_config['sub_idx']
            row_slice  = configs[split][idx]['row_slice']
            col_slice = configs[split][idx]['col_slice']
            tmp_dataset = datasets[idx]

            row_slice = np.arange(*row_slice)
            col_slice = np.arange(*col_slice)

            subsets[split].append(
                Subset(
                    dataset = tmp_dataset,
                    indices = (col_slice[:, None] * dim +  row_slice[None, :]).ravel()
                )
            )

    train_dataset = ConcatDataset(subsets['train'])
    validation_dataset = ConcatDataset(subsets['validation'])
    test_dataset = ConcatDataset(subsets['test'])

    train_dl = DataLoader(dataset=train_dataset, batch_size=config_dl.batch_size, shuffle=config_dl.shuffle, num_workers=config_dl.num_workers)
    validation_dl = DataLoader(dataset=validation_dataset, batch_size=config_dl.batch_size, shuffle=config_dl.shuffle, num_workers=config_dl.num_workers)
    test_dl = DataLoader(dataset=test_dataset, batch_size=config_dl.batch_size, num_workers=config_dl.num_workers)
    return train_dl, validation_dl, test_dl
