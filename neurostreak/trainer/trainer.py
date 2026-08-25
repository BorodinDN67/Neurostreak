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
                with torch.no_grad():
                    total_loss += res_loss
                res_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                cnt_batch += 1
                if cnt_batch % 250 == 0:
                    logger.info(f'BCELoss: {total_loss / 250}', )

                    total_loss = 0
            self.validation(val_dataloader, model)
            self.save_model(model, epoch)
            logger.success(f'Обучение закончено и веса сохранены в {self.checkpoint_path}')

        if scheduler is not None:
            scheduler.step()


    def save_model(self,model, epoch = None, filename_user = None ):
        path = Path(self.checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)
        filename = f'model_checkpoin_epoch_{epoch}.pt' if epoch is not None else f'model_checkpoin.pt'
        if filename_user is not None:
            filename = filename_user +  f'_{0 if not epoch else epoch}_' + '.pt'
        torch.save(model.state_dict(), path / filename)

    def validation(self, val_loader, model, threshold=0.5):
        model.eval()

        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for signal, template, label in val_loader:
                signal = signal.to('cuda')
                template = template.to('cuda')
                label = label.to('cuda')

                output = torch.sigmoid(model(signal, template))

                all_labels.append(label.flatten())
                all_outputs.append(output.flatten())

        labels = torch.cat(all_labels)
        outputs = torch.cat(all_outputs)

        preds = (outputs > threshold).float()

        tp = torch.sum((preds == 1) & (labels == 1)).item()
        tn = torch.sum((preds == 0) & (labels == 0)).item()
        fp = torch.sum((preds == 1) & (labels == 0)).item()
        fn = torch.sum((preds == 0) & (labels == 1)).item()

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        print('//////////////////////////////////////////////////////////////////////////////////')
        print('Результаты валидации:')
        print(f"TP: {tp}")
        print(f"TN: {tn}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")
        print()
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Precision: {precision:.4f}")
        print('//////////////////////////////////////////////////////////////////////////////////')
