import pathlib
import yaml

from neurostreak.model import NeurostreakArch
from neurostreak.model import NeuroStreakHead, NeuroStreakBackbone, NeuroStreakEmbedding
from torch.utils.data import DataLoader


class Config:
    def __init__(self):
        self.config = None
        self.train_config = None


    def load_model_config(self, path: str | pathlib.Path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config
        for key, value in config.items():
            print(key, value)
            setattr(self, key, value)
        print('//////////////////////////////////////////////////////////////////////')
        print()
        print('Конфиг модели успешно загружен')


    def load_train_config(self, path: str | pathlib.Path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.train_config = config
        for key_, value_ in config.items():
            for key, value in value_.items():
                print(key, value)
                setattr(self, key, value)
        print('//////////////////////////////////////////////////////////////////////')
        print()
        print('Конфиг обучения успешно загружен')

    def get_model_from_config(self):
        if self.config is None:
            print('Пожалуйста, загрузите модель с помощью метода load данного объекта')
            return None
        head = NeuroStreakHead(
            spectral_dim = self.head['spectral_dim'],
            wavelet_dim  = self.head['wavelet_dim'],
            embedding_dim = self.head['embedding_dim'],
        )

        backbone = NeuroStreakBackbone(
            hidden_dim = self.backbone['hidden_dim'],
            num_heads=self.backbone['num_heads'],
            depth=self.backbone['depth'],
        )




        embedding = NeuroStreakEmbedding(
            max_amp= self.embedding['max_amp'],
            step = self.embedding['step'],
            wavelet = self.embedding['wavelet'],
        )



        model = NeurostreakArch(head = head,backbone =  backbone,embedding =  embedding,config =  self.config)
        return model
    def get_optimizer_from_config(self, optimizer, model):
        return optimizer(model.parameters(), lr = float(self.lr))

    def get_scheduler_from_config(self, scheduler, optimizer):
        return scheduler(optimizer, gamma = float(self.gamma), step_size = int(self.step_size))

    def get_trainer_from_config(self, trainer):
        return trainer(self.epochs,self.checkpoint_path )

    def get_dataloaders_from_config(self, dataset):
        return DataLoader(
            dataset = dataset,
            batch_size = self.batch_size,
            shuffle = self.shuffle,
        )