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

    def fit(self,model, train_dataloader, val_dataloader, loss, optimizer, scheduler=None):
        logger.info(f"Начали обучение на {self.epochs} эпохах")
        device = next(model.parameters()).device
        for epoch in range(self.epochs):
            cnt_batch = 0
            logger.info(f'Эпоха: {epoch + 1}')
            total_loss = 0
            for batch in train_dataloader:
                signal, template, target = batch
                signal = signal.to(device)
                template = template.to(device)
                target = target.to(device).unsqueeze(-1)

                res = model(signal = signal, template = template)
                res_loss = loss(res, target)
                total_loss += res_loss
                res_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cnt_batch += 1
                if cnt_batch % 100 == 0:
                    logger.info(f'BCELoss: {total_loss / 100}', )
                    total_loss = 0
            if epoch % 5 == 0:
                self.validation(val_dataloader, model)
        if scheduler is not None:
            scheduler.step()


    def save_model(self,model, epoch = None ):
        path = Path(self.checkpoint_path)
        filename = f'model_checkpoin_epoch_{epoch}.pt' if epoch else f'model_checkpoin.pt'
        torch.save(model.state_dict(), path / filename)

    def validation(self, val_loader, model):
        labels = None
        outputs = None
        for signal, template, label in val_loader:
            model.eval()
            with torch.no_grad():
                signal = signal.to('cuda')
                template = template.to('cuda')
                label = label.to('cuda')
                output = torch.sigmoid(model(signal, template))

                if labels is not None:
                    labels = torch.concat([labels, label.flatten(start_dim=0)], dim=0)
                    outputs = torch.concat([outputs, output.flatten(start_dim=0)], dim=0)

        TP = sum(outputs[labels == 1] == 1)
        TN = sum(outputs[labels == 0] == 0)
        FP = sum(outputs[labels == 0] == 1)
        FN = sum(outputs[labels == 1] == 0)

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)

        print('//////////////////////////////////////////////////////////////////////////////////')
        print('Результаты вадилации: ')
        print("TP: ", TP)
        print("TN: ", TN)
        print("FP: ", FP)
        print("FN: ", FN)
        print()
        print("Accuracy: ", accuracy)
        print("Recall: ", recall)
        print("Precision: ", precision)