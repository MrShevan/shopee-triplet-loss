from itertools import combinations

import numpy as np
import torch


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """
    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class RandomTripletSelector(TripletSelector):
    def __init__(self, cpu=True):
        super(RandomTripletSelector, self).__init__()
        self.cpu = cpu

    def get_triplets(self, embeddings, labels):
        pass


class HardTripletSelector(TripletSelector):
    def __init__(self, cpu=True):
        super(HardTripletSelector, self).__init__()
        self.cpu = cpu

    def get_triplets(self, embeddings, labels):
        with torch.no_grad():
            if self.cpu:
                embeddings = embeddings.cpu()

            distance_matrix = torch.cdist(embeddings, embeddings, p=2.0)
            distance_matrix = distance_matrix.cpu()

            labels = labels.cpu().data.numpy()
            triplets = []

            for i, (distances, label) in enumerate(zip(distance_matrix, labels)):
                label_mask = (labels == label)

                positive_indices = np.argwhere(label_mask).squeeze()
                negative_indices = np.argwhere(np.logical_not(label_mask)).squeeze()

                anchor_positives_dists = distances[positive_indices]
                anchor_negatives_dists = distances[negative_indices]

                positive = positive_indices[np.argmax(anchor_positives_dists)]
                negative = negative_indices[np.argmin(anchor_negatives_dists)]

                triplets.append([i, positive, negative])

            triplets = np.array(triplets)

        return torch.LongTensor(triplets)


class SemiHardTripletSelector(TripletSelector):
    def __init__(self, cpu=True):
        super(SemiHardTripletSelector, self).__init__()
        self.cpu = cpu

    @staticmethod
    def _minimum_hard_negative(an_distances, ap_distance):
        # make dists lower than ap_distances as max_distance, higher as zero
        lower_idxs = (ap_distance > an_distances) * an_distances.max()

        # make dists lower than ap_distances as zero, higher without changes
        higher_idxs = (ap_distance < an_distances) * an_distances

        return np.argmin(lower_idxs + higher_idxs)

    def get_triplets(self, embeddings, labels):
        with torch.no_grad():
            if self.cpu:
                embeddings = embeddings.cpu()

            distance_matrix = torch.cdist(embeddings, embeddings, p=2.0)
            distance_matrix = distance_matrix.cpu()

            labels = labels.cpu().data.numpy()
            triplets = []

            for label in set(labels):
                label_mask = (labels == label)
                label_indices = np.where(label_mask)[0]
                if len(label_indices) < 2:
                    continue

                # All anchor-positive pairs
                anchor_positives = list(combinations(label_indices, 2))
                anchor_positives = np.array(anchor_positives)

                negative_indices = np.where(np.logical_not(label_mask))[0]

                # all anchor-positive distances
                ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
                for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):

                    # all anchor-negative distances
                    an_distances = distance_matrix[
                        torch.LongTensor(np.array([anchor_positive[0]])),
                        torch.LongTensor(negative_indices)
                    ]

                    hard_negative = self._minimum_hard_negative(an_distances, ap_distance)

                    triplets.append(
                        [anchor_positive[0], anchor_positive[1], negative_indices[hard_negative]]
                    )

            triplets = np.array(triplets)

        return torch.LongTensor(triplets)
