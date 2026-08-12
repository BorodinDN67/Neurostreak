import torch
from openpyxl.styles.builtins import output
from torch.profiler import profile, ProfilerActivity
from src.config.config import Config
from neurostreak.trainer import TrainerNeuroStreak
from neurostreak.dataset import NeuroStreakDataset
from pathlib import Path
import torch.nn as nn
from torch.utils.data import random_split, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


CURRENT_ROOT = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_ROOT.parent.parent
CONFIG_PATH = PROJECT_ROOT /  Path('src/config/config.yaml')
CONFIG_TRAIN = PROJECT_ROOT /  Path('src/config/config_train.yaml')
DATASET_PATH = PROJECT_ROOT / Path('data/clean_water_20m')

config = Config()
config.load_model_config(CONFIG_PATH)
config.load_train_config(CONFIG_TRAIN)
model = config.get_model_from_config()
model = model.to(device = 'cuda')
loss = nn.BCEWithLogitsLoss()
dataset = NeuroStreakDataset(DATASET_PATH)
train_dataset, val_dataset = random_split(
    dataset,
[
        int(len(dataset) * config.train_size),
        len(dataset) - int(len(dataset) * config.train_size)
    ]
)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False, num_workers=0)

optimizer = config.get_optimizer_from_config(optim.Adam, model)
scheduler = config.get_scheduler_from_config(StepLR, optimizer)
trainer = config.get_trainer_from_config(TrainerNeuroStreak)

import time
import torch


class ModuleProfiler:
    def __init__(self, model):
        self.model = model
        self.stats = {}
        self.handles = []
        self.starts = {}

        # только leaf-модули:
        # Conv2d, Linear, MultiheadAttention, MaxPool2d и т.д.
        for name, module in model.named_modules():

            if name == "":
                continue

            # если у модуля есть дочерние nn.Module — пропускаем
            if len(list(module.children())) > 0:
                continue

            self.handles.append(
                module.register_forward_pre_hook(
                    self._make_pre_hook(name)
                )
            )

            self.handles.append(
                module.register_forward_hook(
                    self._make_post_hook(name)
                )
            )

    def _make_pre_hook(self, name):
        def hook(module, inputs):

            if torch.cuda.is_available():
                torch.cuda.synchronize()

                allocated_before = torch.cuda.memory_allocated()

                torch.cuda.reset_peak_memory_stats()

            else:
                allocated_before = 0

            self.starts[name] = {
                "time": time.perf_counter(),
                "memory": allocated_before,
            }

        return hook

    def _make_post_hook(self, name):
        def hook(module, inputs, output):

            if torch.cuda.is_available():
                # CUDA асинхронная, поэтому ждём завершения kernels
                torch.cuda.synchronize()

                allocated_after = torch.cuda.memory_allocated()
                peak = torch.cuda.max_memory_allocated()

            else:
                allocated_after = 0
                peak = 0

            elapsed = (
                time.perf_counter()
                - self.starts[name]["time"]
            )

            before = self.starts[name]["memory"]

            delta = allocated_after - before
            peak_extra = peak - before

            self.stats[name] = {
                "type": module.__class__.__name__,
                "time_ms": elapsed * 1000,
                "delta_mb": delta / 1024**2,
                "peak_mb": peak_extra / 1024**2,
                "output": self._shape(output),
            }

        return hook

    @staticmethod
    def _shape(x):

        if isinstance(x, torch.Tensor):
            return tuple(x.shape)

        if isinstance(x, (tuple, list)):
            return [
                tuple(t.shape)
                if isinstance(t, torch.Tensor)
                else type(t).__name__
                for t in x
            ]

        if isinstance(x, dict):
            return {
                k: tuple(v.shape)
                if isinstance(v, torch.Tensor)
                else type(v).__name__
                for k, v in x.items()
            }

        return type(x).__name__

    def print_report(self, sort_by="peak_mb"):
        stats = sorted(
            self.stats.items(),
            key=lambda x: x[1][sort_by],
            reverse=True,
        )

        print(
            f"{'Module':45} "
            f"{'Type':22} "
            f"{'Time ms':>10} "
            f"{'Δ VRAM MB':>12} "
            f"{'Peak MB':>12} "
            f"Output"
        )

        print("-" * 130)

        for name, s in stats:
            print(
                f"{name:45} "
                f"{s['type']:22} "
                f"{s['time_ms']:10.3f} "
                f"{s['delta_mb']:12.2f} "
                f"{s['peak_mb']:12.2f} "
                f"{s['output']}"
            )

    def remove(self):
        for handle in self.handles:
            handle.remove()


profiler = ModuleProfiler(model)
signal, template, target = next(iter(train_dataloader))
signal = signal.to(device = 'cuda')
template = template.to(device = 'cuda')

output = model(signal = signal, template = template)
profiler.print_report(sort_by="peak_mb")
profiler.remove()