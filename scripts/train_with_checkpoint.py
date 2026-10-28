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
from neurostreak.dataset import NeuroStreakDataset, NeuroStreakConcatDataset, split_dataset
import torch
import torch.optim as optim
import numpy as np
from torch import nn

CURRENT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_ROOT.parent.parent
CONFIG_PATH = PROJECT_ROOT /  Path('src/config/config.yaml')
CONFIG_TRAIN = PROJECT_ROOT /  Path('src/config/config_train.yaml')
MLFLOW_PATH = PROJECT_ROOT / 'mlruns'

mlflow.set_tracking_uri("sqlite:///C:/Users/supervisor.LTCPC-280426-01/PycharmProjects/NeuroStreak/mlflow.db")

#Пожалуйста, пока что важно в порядке возрастания их заполнять ((
DATASET_PATHS =[
    PROJECT_ROOT / Path('data/clean_water_10m'),
    PROJECT_ROOT / Path('data/clean_water_13m'),
    PROJECT_ROOT / Path('data/clean_water_15m'),
    PROJECT_ROOT / Path('data/clean_water_20m'),
]
CONFIG_DATA_PATH = PROJECT_ROOT / 'data'


rndm = 8810002
random.seed(rndm)
torch.manual_seed(rndm)
np.random.seed(rndm)
torch.cuda.manual_seed_all(rndm)

def main():
    config = Config()
    config.load_model_config(CONFIG_PATH)
    config.load_train_config(CONFIG_TRAIN)

    train_config = config.train_config
    model_config = config.config

    mlflow.set_experiment("NeuroStreak_FFT_&_CWT_1_0")
    RUN_ID = train_config.RUN_ID

    with mlflow.start_run(run_name=config.exp_name, run_id=RUN_ID):
        mlflow.log_params(flatten_dict(train_config))
        mlflow.log_params(flatten_dict(model_config))

        # Получение основных объектов и модели; загрузка чекпоинта
        model = config.get_model_from_config().to('cuda')
        optimizer = config.get_optimizer_from_config(optim.Adam, model)
        scheduler = config.get_scheduler_from_config(StepLR, optimizer)

        start_epoch, global_step = config.get_chechpoint(
            path = MLFLOW_PATH,
            model = model,
            optimizer = optimizer,
            scheduler = scheduler,
        )
        dataset = NeuroStreakConcatDataset(DATASET_PATHS)

        train_dataloader, validation_dataloader, test_dataloader = split_dataset(
            dataset=dataset,
            train_scheme=CONFIG_DATA_PATH / 'test_config.yaml',
            validation_scheme=CONFIG_DATA_PATH / 'valid_config.yaml',
            test_scheme=CONFIG_DATA_PATH / 'test_config.yaml',
            config_dl=config
        )

        pos_weights = torch.tensor(
            [1.5],
            dtype=torch.float32,
            device='cuda'
        )

        loss = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        trainer = config.get_trainer_from_config(TrainerNeuroStreak, start_epoch = start_epoch, global_step = global_step)

        logger.success(f'Начинаем файн-тюнинг модели с id: {config.model_id}\n')
        model.train()
        trainer.fit(model, train_dataloader, validation_dataloader, loss, optimizer, scheduler)
        with open(f"model_summary_{config.exp_name}.txt", "w") as f:
            f.write(str(model))
        mlflow.log_artifact("model_summary.txt")

        mlflow.pytorch.log_model(
            model,
            f"model_{config.exp_name}",
            metadata={"architecture_summary": str(model)},
            serialization_format='pickle'
        )
        logger.success(f'Обучение закончено и веса сохранены в {MLFLOW_PATH}')
if __name__ == '__main__':
    main()