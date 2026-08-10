import pathlib

import torch
import torch.optim as optim

from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
from pathlib import Path

from src.config.config import Config
from neurostreak.trainer import TrainerNeuroStreak
from torch.optim.lr_scheduler import StepLR
from neurostreak.dataset import NeuroStreakDataset
from src.data import StreakImageDataset

CURRENT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_ROOT.parent.parent
CONFIG_PATH = PROJECT_ROOT /  Path('src/config/config.yaml')
CONFIG_TRAIN = PROJECT_ROOT /  Path('src/config/config_train.yaml')
DATASET_PATH = PROJECT_ROOT / Path('data/clean_water_10m')

config = Config()
config.load_model_config(CONFIG_PATH)
config.load_train_config(CONFIG_TRAIN)

dataset = StreakImageDataset(DATASET_PATH)
dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0)
model = config.get_model_from_config().to('cuda')
loss  = nn.CrossEntropyLoss()
optimizer = config.get_optimizer_from_config(optim.Adam, model )
scheduler = config.get_scheduler_from_config(StepLR, optimizer)
trainer = config.get_trainer_from_config(TrainerNeuroStreak)


model.train()
trainer.fit(model,dataloader, loss, optimizer, scheduler)
trainer.save_model(model, epoch = config.epochs)

