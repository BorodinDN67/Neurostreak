import torch
import torch.functional as F

def metrics(val_loader, model):
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
                labels = torch.concat([labels, label.flatten(start_dim = 0)], dim = 0)
                outputs = torch.concat([outputs, output.flatten(start_dim = 0)], dim = 0)

    TP  = sum(outputs[labels == 1] == 1)
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