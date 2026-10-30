import torch
from torch import nn


def dice_loss(prediction, label):
    return 1 - (2 * (prediction * label).sum() + 1e-6) / (prediction.sum() + label.sum() + 1e-6)


class IGLoss(nn.Module):
    def forward(self, predictions, label):
        return sum((nn.functional.binary_cross_entropy(pred, label) + 0.1 * dice_loss(pred, label))
                   / (len(predictions) - i) for i, pred in enumerate(predictions))


class MultiLoss(nn.Module):
    def forward(self, predictions, prediction, label):
        return 0.3 * IGLoss()(predictions, label) + 0.7 * (
            nn.functional.binary_cross_entropy(prediction, label) + 0.5 * dice_loss(prediction, label))
