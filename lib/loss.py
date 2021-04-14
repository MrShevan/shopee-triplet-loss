import torch.nn as nn
import torch.nn.functional as F


class OnlineTripletLoss(nn.Module):
    def __init__(self, margin, triplet_selector):
        """
        Online Triplets loss
        Takes a batch of embeddings and corresponding labels.
        Triplets are generated using triplet_selector object that take embeddings and targets
        and return indices of triplets
        """
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def _log_triplets(self):
        """Log triplets to TensorBoard"""
        pass

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        distance_positive = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        distance_negative = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)

        return losses.mean(), distance_positive.mean(), distance_negative.mean()
