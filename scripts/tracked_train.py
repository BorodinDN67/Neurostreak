import random

import mlflow
import mlflow.pytorch
from src.config.config import Config, flatten_dict
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import random_split
from loguru import logger
from src.config.config import Config
from neurostreak.trainer import TrainerNeuroStreak
from neurostreak.dataset import NeuroStreakDataset
import torch
import torch.optim as optim
import numpy as np
from torch import nn

CURRENT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_ROOT.parent.parent
CONFIG_PATH = PROJECT_ROOT /  Path('src/config/config.yaml')
CONFIG_TRAIN = PROJECT_ROOT /  Path('src/config/config_train.yaml')
DATASET_PATHS =[ PROJECT_ROOT / Path('data/clean_water_20m')]

mlflow.set_tracking_uri("sqlite:///C:/Users/supervisor.LTCPC-280426-01/PycharmProjects/NeuroStreak/mlflow.db")

rndm = 8810002
random.seed(rndm)
torch.manual_seed(rndm)
np.random.seed(rndm)
torch.cuda.manual_seed_all(rndm)

def tracked_train():
    config = Config()
    config.load_model_config(CONFIG_PATH)
    config.load_train_config(CONFIG_TRAIN)

    train_config = config.train_config
    model_config = config.config

    mlflow.set_experiment("NeuroStreak_FFT_&_CWT_1_0")
    with mlflow.start_run(run_name=config.exp_name):
        mlflow.log_params(flatten_dict(train_config))
        mlflow.log_params(flatten_dict(model_config))

        #Получение основных объектов и модели
        model = config.get_model_from_config().to('cuda')
        dataset = NeuroStreakDataset(DATASET_PATHS[0])
        pos_weights = torch.tensor(
            [15.0],
            dtype=torch.float32,
            device='cuda'
        )
        train_dataset, val_dataset = random_split(dataset, [int(len(dataset) * config.train_size),
                                                            len(dataset) - int(len(dataset) * config.train_size)],
                                                  generator=torch.Generator().manual_seed(42))
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=0)

        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
        optimizer = config.get_optimizer_from_config(optim.Adam, model)
        scheduler = config.get_scheduler_from_config(StepLR, optimizer)
        trainer = config.get_trainer_from_config(TrainerNeuroStreak)

        #Очевидный запуск обучения
        model.train()
        trainer.fit(model, train_dataloader, val_dataloader, loss, optimizer, scheduler)

        #Логгируем
        with open("model_summary.txt", "w") as f:
            f.write(str(model))
        mlflow.log_artifact("model_summary.txt")

        mlflow.pytorch.log_model(
            model,
            f"model_{config.exp_name}",
            metadata={"architecture_summary": str(model)},
            serialization_format='pickle'
        )


if __name__ == '__main__':
    tracked_train()
