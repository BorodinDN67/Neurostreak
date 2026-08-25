from torch.utils.data import DataLoader, ConcatDataset, random_split

from neurostreak.trainer import TrainerNeuroStreak
from src.config.config import Config
from neurostreak.dataset import NeuroStreakDataset
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import numpy as np
PROJECT_ROOT = Path()
CONFIG_PATH = PROJECT_ROOT /  Path('src/config/config.yaml')
CONFIG_TRAIN = PROJECT_ROOT /  Path('src/config/config_train.yaml')
DATASET_PATHS = [
    # PROJECT_ROOT / Path('data/clean_water_10m'), # 147183 819200 - > 5.56
    # PROJECT_ROOT / Path('data/clean_water_13m'), # 79484 714752 -> 8.99
    # PROJECT_ROOT / Path('data/clean_water_15m'), #82742 614400 -> 7.42
    PROJECT_ROOT / Path('data/clean_water_20m'), #37250 509566 -> 14.67
]
WEIGHTS_PATH = PROJECT_ROOT / Path('scripts/save/24_08_26_neurostreak/model_checkpoin_epoch_5.pt')


if len(DATASET_PATHS) == 1:
    dataset = NeuroStreakDataset(DATASET_PATHS[0])
else:
    datasets = [NeuroStreakDataset(dataset_path) for dataset_path in DATASET_PATHS]
    dataset = ConcatDataset(datasets)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2],
                                          generator=torch.Generator().manual_seed(42))
val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=0)

config = Config()
config.load_model_config(CONFIG_PATH)
config.load_train_config(CONFIG_TRAIN)
model = config.get_model_from_config().to('cuda')
weights = torch.load(WEIGHTS_PATH)
model.load_state_dict(weights)



trainer = config.get_trainer_from_config(TrainerNeuroStreak)

trainer.validation(val_dataloader, model)
