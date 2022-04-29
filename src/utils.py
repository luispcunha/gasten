import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def binary_accuracy(y_pred, y_true, avg=True, threshold=0.5):
    correct = (y_pred > threshold) == y_true

    return correct.sum() if avg is False else correct.type(torch.float32).mean()


def multiclass_accuracy(y_pred, y_true, avg=True, threshold=0.5):
    pred = y_pred.max(1, keepdim=True)[1]

    correct = pred.eq(y_true.view_as(pred))

    return correct.sum() if avg is False else correct.type(torch.float32).mean()
