import pathlib
import yaml


class Config:
    def __init__(self):
        pass
    def load(self, path: str | pathlib.Path):
        with open(path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
