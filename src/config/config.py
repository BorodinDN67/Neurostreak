import pathlib
import yaml

from neurostreak.model import NeurostreakArch
from neurostreak.model import NeuroStreakHead, NeuroStreakBackbone, NeuroStreakEmbedding

class Config:
    def __init__(self):
        self.config = None
    def load(self, path: str | pathlib.Path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config
        for key, value in config.items():
            print(key, value)
            setattr(self, key, value)
        print('//////////////////////////////////////////////////////////////////////')
        print()
        print('Конфиг успешно загружен')

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