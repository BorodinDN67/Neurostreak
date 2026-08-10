from loguru import logger
from pathlib import Path
import torch

class TrainerNeuroStreak:
    def __init__(
        self,
        epochs,
        checkpoin_path=None
    ):
        self.epochs = epochs
        self.checkpoint_path = checkpoin_path

    def fit(self,model, dataloader, loss, optimizer, scheduler=None):
        logger.info(f"Начали обучение на {self.epochs + 1} эпохах")
        for epoch in range(self.epochs):
            for batch in dataloader:
                signal, template, target = batch

                res = model(signal = signal, template = template)
                print(res)
                res_loss = loss(res, target)
                print(res_loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()


    def save_model(self,model, epoch = None ):
        path = Path(self.checkpoint_path)
        filename = f'model_checkpoin_epoch_{epoch}.pt' if epoch else f'model_checkpoin.pt'
        torch.save(model.state_dict(), path / filename)

