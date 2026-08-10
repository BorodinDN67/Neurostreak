import pathlib

import torch
import torch.optim as optim

from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split

from src.config.config import Config
from neurostreak.trainer import TrainerNeuroStreak
from neurostreak.dataset import NeuroStreakDataset
from src.data import StreakImageDataset

CURRENT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_ROOT.parent.parent
CONFIG_PATH = PROJECT_ROOT /  Path('src/config/config.yaml')
CONFIG_TRAIN = PROJECT_ROOT /  Path('src/config/config_train.yaml')
DATASET_PATH = PROJECT_ROOT / Path('data/clean_water_10m')



def main():
    config = Config()
    config.load_model_config(CONFIG_PATH)
    config.load_train_config(CONFIG_TRAIN)

    dataset = NeuroStreakDataset(DATASET_PATH)
    train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * config.train_size), len(dataset) - int(len(dataset) * config.train_size)])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=int(len(dataset) * config.train_size), shuffle=False, num_workers=0)
    model = config.get_model_from_config().to('cuda')
    loss  = nn.BCEWithLogitsLoss()
    optimizer = config.get_optimizer_from_config(optim.Adam, model )
    scheduler = config.get_scheduler_from_config(StepLR, optimizer)
    trainer = config.get_trainer_from_config(TrainerNeuroStreak)


    model.train()
    trainer.fit(model,train_dataloader,val_dataloader, loss, optimizer, scheduler)
    trainer.save_model(model, epoch = config.epochs)

if __name__ == '__main__':
    main()