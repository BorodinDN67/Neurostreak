import torch.nn as nn
import torch


class NeuroStreakHead(nn.Module):
    def __init__(self):
        super(NeuroStreakHead, self).__init__()
        pass
    def forward(self, x):
        pass


class StreakNetImagingHead(nn.Module):
    def __init__(self, width=1.0, act='silu', loss='crossloss', len=2):
        super(StreakNetImagingHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(round(512 * width) * len, 2)
        self.act = self.get_activation(act, inplace=False)
        self.losses = self.get_loss(loss)

    def forward(self, x, labels=None):
        flatten = self.flatten(x)
        pred = self.act(self.fc(flatten))
        if self.training:
            return self.get_losses(pred, labels)
        else:
            pred = torch.argmax(pred, 1)
            return pred

    def get_losses(self, x, labels):
        loss_dict = self.losses(x, labels)
        return loss_dict

    def get_loss(self, name="streakloss"):
        if name == "crossloss":
            module = CrossLoss()
        else:
            raise AttributeError("Unsupported loss type: {}".format(name))
        return module

    def get_activation(self, name="silu", inplace=False):
        if name == "silu":
            module = nn.SiLU(inplace=inplace)
        elif name == "relu":
            module = nn.ReLU(inplace=inplace)
        elif name == "lrelu":
            module = nn.LeakyReLU(0.1, inplace=inplace)
        else:
            raise AttributeError("Unsupported act type: {}".format(name))
        return module

class CrossLoss(nn.Module):
    def __init__(self):
        super(CrossLoss, self).__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        labels = labels.reshape(-1)
        cls_loss = self.cls_loss(preds, labels)
        loss_dict = {
            "total_loss": cls_loss,
            "cls_loss": cls_loss
        }
        return loss_dict