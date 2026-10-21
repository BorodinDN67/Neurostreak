from dataset import NeuroStreakDataset, NeuroStreakConcatDataset
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


CURRENT_DIR = Path(__file__).parent
DATASET_DIR = CURRENT_DIR.parent.parent / 'data'
DATASETS_DIR = [
    DATASET_DIR / 'clean_water_10m',
    DATASET_DIR / 'clean_water_13m',
    DATASET_DIR / 'clean_water_15m',
    DATASET_DIR / 'clean_water_20m',
]

datasets = NeuroStreakConcatDataset(DATASETS_DIR)

print(datasets.len)
print(datasets.lens)
print(datasets.root_dir)
print(datasets.datasets)

returned = datasets.__getitem__(819200)
for elem in returned:
    print(elem.shape)
    print(elem)
    print('='*64)
