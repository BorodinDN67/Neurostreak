import pathlib

import torch
import torch.optim as optim

from torch import nn
from torch.cuda import device
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from loguru import logger

from src.config.config import Config
from neurostreak.trainer import TrainerNeuroStreak
from neurostreak.dataset import NeuroStreakDataset
from src.data import StreakImageDataset

CURRENT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_ROOT.parent.parent
CONFIG_PATH = PROJECT_ROOT /  Path('src/config/config.yaml')
CONFIG_TRAIN = PROJECT_ROOT /  Path('src/config/config_train.yaml')
DATASET_PATHS =[ PROJECT_ROOT / Path('data/clean_water_20m')]



def main():
    config = Config()
    config.load_model_config(CONFIG_PATH)
    config.load_train_config(CONFIG_TRAIN)
    #Получаем датасет ИЛИ, если несколько датасетов, склеиваем их в один
    if len(DATASET_PATHS) == 1:
        dataset = NeuroStreakDataset(DATASET_PATHS[0])

    else:
        datasets = [NeuroStreakDataset(dataset_path) for dataset_path in DATASET_PATHS]
        dataset = ConcatDataset(datasets)

    train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * config.train_size), len(dataset) - int(len(dataset) * config.train_size)], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=0)
    model = config.get_model_from_config().to('cuda')

    pos_weights = torch.tensor(
        [15.0],
        dtype=torch.float32,
        device='cuda'
    )
    loss  = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = config.get_optimizer_from_config(optim.Adam, model )
    scheduler = config.get_scheduler_from_config(StepLR, optimizer)
    trainer = config.get_trainer_from_config(TrainerNeuroStreak)


    model.train()
    trainer.fit(model,train_dataloader,val_dataloader, loss, optimizer, scheduler)
    trainer.save_model(model, epoch = config.epochs)
    logger.success(f'Обучение закончено и веса сохранены в {config.checkpoint_path}')
if __name__ == '__main__':
    main()