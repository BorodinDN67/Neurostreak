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
            hidden_2d_dim_1 = self.backbone['hidden_2d_dim_1'],
            hidden_2d_dim_2= self.backbone['hidden_2d_dim_2'],
            kernel_size=self.backbone['kernel_size'],
            padding=self.backbone['padding'],
            num_heads=self.backbone['num_heads'],
            embed_dim=self.backbone['embed_dim'],
            depth=self.backbone['depth'],
            max_pool_kernel_size_1=self.backbone['max_pool_kernel_size_1'],
            max_pool_stride_1=self.backbone['max_pool_stride_1'],
            max_pool_kernel_size_2=self.backbone['max_pool_kernel_size_2'],
            max_pool_stride_2=self.backbone['max_pool_stride_2'],
            max_pool_kernel_size_3=self.backbone['max_pool_kernel_size_3'],
            max_pool_stride_3=self.backbone['max_pool_stride_3'],
            num_scales=self.backbone['num_scales']
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