from torch.utils.data import DataLoader, ConcatDataset

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
    PROJECT_ROOT / Path('data/clean_water_10m'), # 147183 819200 - > 5.56
    # PROJECT_ROOT / Path('data/clean_water_13m'), # 79484 714752 -> 8.99
    # PROJECT_ROOT / Path('data/clean_water_15m'), #82742 614400 -> 7.42
    # PROJECT_ROOT / Path('data/clean_water_20m'), #37250 509566 -> 14.67
]
WEIGHTS_PATH = PROJECT_ROOT / Path('scripts/save/24_08_26_neurostreak/model_checkpoin_epoch_5.pt')

if len(DATASET_PATHS) == 1:
    dataset = NeuroStreakDataset(DATASET_PATHS[0])
else:
    datasets = [NeuroStreakDataset(dataset_path) for dataset_path in DATASET_PATHS]
    dataset = ConcatDataset(datasets)

config = Config()
config.load_model_config(CONFIG_PATH)
config.load_train_config(CONFIG_TRAIN)
model = config.get_model_from_config().to('cuda')
weights = torch.load(WEIGHTS_PATH)
model.load_state_dict(weights)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)


batch = next(iter(dataloader))

trainer = config.get_trainer_from_config(TrainerNeuroStreak)

trainer.validation(dataloader, model)
print(batch)
