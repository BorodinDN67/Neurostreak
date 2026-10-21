from loguru import logger
from pathlib import Path
import torch
import mlflow
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef, brier_score_loss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc

class TrainerNeuroStreak:
    def __init__(
        self,
        epochs,
    ):
        self.epochs = epochs
        self.checkpoint_path = 'checkpoints_lol'

    def fit(self,model, train_dataloader, val_dataloader, loss, optimizer, scheduler=None):
        logger.info(f"Начали обучение на {self.epochs} эпохах")
        device = next(model.parameters()).device
        validation_results = []
        step_for_logs = 0
        debug_stats = []

        for epoch in range(self.epochs):
            model.train()
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

                # Проверка численной стабильности
                if not torch.isfinite(res).all():
                    raise RuntimeError(f"NaN/Inf в выходе модели, batch={cnt_batch}")

                if not torch.isfinite(res_loss):
                    raise RuntimeError(f"NaN/Inf в loss, batch={cnt_batch}")

                total_loss += res_loss.item()

                res_loss.backward()

                # Проверяем градиенты, не изменяя их
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=float("inf"),
                    error_if_nonfinite=True
                )

                # Снимок весов в начале интервала
                if not debug_stats:
                    weights_before = [
                        p.detach().cpu().clone()
                        for p in model.parameters() if p.requires_grad
                    ]

                # Собираем статистики текущего батча
                with torch.no_grad():
                    probs = torch.sigmoid(res.detach())
                    pos = target == 1
                    neg = target == 0

                    debug_stats.append({
                        "loss": res_loss.item(),
                        "grad": grad_norm.item(),
                        "n": probs.numel(),
                        "prob_sum": probs.sum().item(),
                        "prob_sq_sum": probs.square().sum().item(),
                        "pos_sum": probs[pos].sum().item(),
                        "neg_sum": probs[neg].sum().item(),
                        "n_pos": pos.sum().item(),
                        "n_neg": neg.sum().item(),
                    })

                optimizer.step()
                optimizer.zero_grad()

                cnt_batch += 1
                step_for_logs += 1

                # Агрегирование и логирование каждые 25 батчей
                if len(debug_stats) == 25:
                    n = sum(s["n"] for s in debug_stats)
                    n_pos = sum(s["n_pos"] for s in debug_stats)
                    n_neg = sum(s["n_neg"] for s in debug_stats)

                    avg_loss = np.mean([s["loss"] for s in debug_stats])
                    avg_grad = np.mean([s["grad"] for s in debug_stats])
                    max_grad = max(s["grad"] for s in debug_stats)

                    prob_mean = sum(s["prob_sum"] for s in debug_stats) / n
                    prob_sq_mean = sum(s["prob_sq_sum"] for s in debug_stats) / n
                    prob_std = max(prob_sq_mean - prob_mean ** 2, 0.0) ** 0.5

                    pos_prob = (
                        sum(s["pos_sum"] for s in debug_stats) / n_pos
                        if n_pos else None
                    )
                    neg_prob = (
                        sum(s["neg_sum"] for s in debug_stats) / n_neg
                        if n_neg else None
                    )

                    # Изменение всех обучаемых параметров за 25 шагов
                    with torch.no_grad():
                        params = [p for p in model.parameters() if p.requires_grad]

                        delta_sq = sum(
                            (p.detach().cpu() - w).square().sum().item()
                            for p, w in zip(params, weights_before)
                        )
                        norm_sq = sum(w.square().sum().item() for w in weights_before)

                    weight_delta = delta_sq ** 0.5
                    weight_delta_rel = weight_delta / max(norm_sq ** 0.5, 1e-12)

                    metrics = {
                        "train/loss": float(avg_loss),
                        "train/grad_norm_mean": float(avg_grad),
                        "train/grad_norm_max": float(max_grad),
                        "train/weight_delta": weight_delta,
                        "train/weight_delta_rel": weight_delta_rel,
                        "train/prob_mean": prob_mean,
                        "train/prob_std": prob_std,
                        "train/positive_ratio": n_pos / n,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }

                    if pos_prob is not None:
                        metrics["train/positive_prob"] = pos_prob
                    if neg_prob is not None:
                        metrics["train/negative_prob"] = neg_prob

                    mlflow.log_metrics(metrics, step=step_for_logs)

                    logger.info(
                        f"\n[DEBUG] Batch {cnt_batch}"
                        f"\n  Loss:           {avg_loss:.6f}"
                        f"\n  Grad mean/max:  {avg_grad:.6f} / {max_grad:.6f}"
                        f"\n  Weight delta:   {weight_delta:.8f}"
                        f"\n  Relative delta: {weight_delta_rel:.8f}"
                        f"\n  Prob mean/std:  {prob_mean:.5f} / {prob_std:.5f}"
                        f"\n  Positive prob:  {pos_prob if pos_prob is not None else 'N/A'}"
                        f"\n  Negative prob:  {neg_prob if neg_prob is not None else 'N/A'}"
                        f"\n  Positive ratio: {n_pos / n:.4f}"
                        f"\n  LR:             {metrics['train/lr']:.8f}"
                    )

                    debug_stats.clear()

                step_for_logs += 1


            metrics = self.validation(model = model,val_loader = val_dataloader, threshold=0.5, epoch=epoch)
            mlflow.log_param("threshold", metrics['threshold'])
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if k not in ('epoch', 'threshold')},
                step=epoch
            )
            validation_results.append(metrics)
            self.save_checkpoint(model, epoch = epoch, validation_result = validation_results)
            logger.success(f'Обучение закончено и веса сохранены в артефактах')
            debug_stats.clear()
            if scheduler is not None:
                scheduler.step()

    def save_checkpoint(self, model, optimizer=None, scheduler=None,epoch=None, filename_user=None, validation_result=None, config_trainer=None):
        # --- локальный полный чекпоинт ---
        path = Path(__file__).parent.parent.parent / Path(self.checkpoint_path)
        path.mkdir(parents=True, exist_ok=True)

        filename = f"{filename_user}.pt" if filename_user else f"checkpoint_epoch_{epoch}.pt"

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'validation_result': validation_result,
            'config_model': getattr(model, "config", None),
            'config_trainer': config_trainer,
        }
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        full_path = path / filename
        torch.save(checkpoint, full_path)

        # --- в MLflow ---
        mlflow.log_artifact(str(full_path), artifact_path="checkpoints")
        mlflow.pytorch.log_model(model, artifact_path=f"model_{epoch}", serialization_format = 'pickle' )


    def validation(self, val_loader, model, threshold=0.5, epoch = None):
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
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = (recall + specificity) / 2

        # порого-независимые и калибровочные — через numpy
        y_true = labels.cpu().numpy()
        y_prob = outputs.cpu().numpy()
        y_pred = preds.cpu().numpy()

        # защита от случая, когда в батче только один класс
        if len(np.unique(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
            pr_auc = average_precision_score(y_true, y_prob)
        else:
            roc_auc = float('nan')
            pr_auc = float('nan')

        mcc = matthews_corrcoef(y_true, y_pred)
        brier = brier_score_loss(y_true, y_prob)

        # --- ROC Curve ---
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)

        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        ax_roc.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc_val:.4f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', label='Random')
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title(f'ROC Curve (Epoch {epoch})')
        ax_roc.legend(loc='lower right')
        ax_roc.grid(True, alpha=0.3)

        # --- Precision-Recall Curve ---
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_prob)
        pr_auc_val = auc(recall_arr, precision_arr)

        fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
        ax_pr.plot(recall_arr, precision_arr, label=f'PR (AUC = {pr_auc_val:.4f})')
        ax_pr.set_xlabel('Recall')
        ax_pr.set_ylabel('Precision')
        ax_pr.set_title(f'Precision-Recall Curve (Epoch {epoch})')
        ax_pr.legend(loc='lower left')
        ax_pr.grid(True, alpha=0.3)

        # --- Логируем как артефакты MLflow ---
        mlflow.log_figure(fig_roc, f"plots/roc_curve_epoch_{epoch}.png")
        mlflow.log_figure(fig_pr, f"plots/pr_curve_epoch_{epoch}.png")

        # Закрываем фигуры, чтобы не копились в памяти
        plt.close(fig_roc)
        plt.close(fig_pr)

        result = {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'recall': recall,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'mcc': mcc,
            'brier': brier,
            'epoch': epoch,
            'threshold': threshold,
        }

        GREEN = '\033[92m'; YELLOW = '\033[93m'; CYAN = '\033[96m'; RESET = '\033[0m'

        print('=' * 60)
        print(f'{CYAN} РЕЗУЛЬТАТЫ ВАЛИДАЦИИ (epoch {epoch}){RESET}')
        print('=' * 60)
        print(f'  {GREEN}TP:{RESET} {tp:>5}   {GREEN}TN:{RESET} {tn:>5}')
        print(f'  {YELLOW}FP:{RESET} {fp:>5}   {YELLOW}FN:{RESET} {fn:>5}')
        print('-' * 60)
        print(f'  Accuracy:    {accuracy:.4f}   Balanced: {balanced_acc:.4f}')
        print(f'  Recall:      {recall:.4f}   Specificity: {specificity:.4f}')
        print(f'  Precision:   {precision:.4f}   F1: {f1:.4f}')
        print(f'  ROC-AUC:     {roc_auc:.4f}   PR-AUC: {pr_auc:.4f}')
        print(f'  MCC:         {mcc:.4f}   Brier: {brier:.4f}')
        print('=' * 60)

        return result