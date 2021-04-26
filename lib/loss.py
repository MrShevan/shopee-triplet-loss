import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    def __init__(self, margin):
        """
        Triplet loss.
        Takes embeddings of an anchor sample, a positive sample and a negative sample

        Args:
            margin:
        """
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor):
        """
        Compute loss

        Args:
            anchor:
            positive:
            negative:
        """
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean(), distance_positive.mean(), distance_negative.mean()
