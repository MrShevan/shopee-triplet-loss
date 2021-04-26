from itertools import combinations

import numpy as np
import torch

from lib.distances import pdist_cosine, pdist_l2


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def balanced_negative(loss_values, margin, high_bound):
    a_p_n = np.argwhere(np.logical_and(loss_values > 0, loss_values < margin)).T[0]
    a_n_p_low = np.argwhere(np.logical_and(loss_values > margin, loss_values < 2 * margin)).T[0]
    a_n_p_high = np.argwhere(np.logical_and(loss_values > 2 * margin, loss_values < high_bound)).T[0]

    non_empty_groups = list(filter(lambda x: len(x) > 0, [a_p_n, a_n_p_low, a_n_p_high]))

    if len(non_empty_groups) == 0:
        return None

    indices = np.concatenate(non_empty_groups)
    counts = np.concatenate([np.repeat(len(group), len(group)) for group in non_empty_groups])
    probs = 1 / (len(non_empty_groups) * counts)

    return np.random.choice(indices, p=probs)


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """
    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """
    def __init__(self, margin, negative_selection_fn, distance='l2', cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

        self.distance = distance
        self.possible_distances = ['l2', 'cosine']

        if self.distance not in self.possible_distances:
            Exception(f'Not supported distance. Use from {self.possible_distances}')

    def get_triplets(self, embeddings, labels):
        with torch.no_grad():
            if self.cpu:
                embeddings = embeddings.cpu()

            if self.distance == 'l2':
                distance_matrix = pdist_l2(embeddings)

            if self.distance == 'cosine':
                distance_matrix = pdist_cosine(embeddings)

            distance_matrix = distance_matrix.cpu()

            labels = labels.cpu().data.numpy()
            triplets = []

            for label in set(labels):
                label_mask = (labels == label)
                label_indices = np.where(label_mask)[0]
                if len(label_indices) < 2:
                    continue
                negative_indices = np.where(np.logical_not(label_mask))[0]
                anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
                anchor_positives = np.array(anchor_positives)

                ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
                for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                    loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                    loss_values = loss_values.data.cpu().numpy()

                    hard_negative = self.negative_selection_fn(loss_values)
                    if hard_negative is not None:
                        hard_negative = negative_indices[hard_negative]
                        triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

            if len(triplets) == 0:
                triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

            triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(loss_margin, distance, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=loss_margin,
        negative_selection_fn=hardest_negative,
        distance=distance,
        cpu=cpu
    )


def RandomNegativeTripletSelector(loss_margin, distance, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=loss_margin,
        negative_selection_fn=random_hard_negative,
        distance=distance,
        cpu=cpu
    )


def SemihardNegativeTripletSelector(loss_margin, selector_margin, distance, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=loss_margin,
        negative_selection_fn=lambda x: semihard_negative(x, margin=selector_margin),
        distance=distance,
        cpu=cpu
    )


def BalancedNegativeTripletSelector(loss_margin, high_bound, distance, cpu=False):
    return FunctionNegativeTripletSelector(
        margin=loss_margin,
        negative_selection_fn=lambda x: balanced_negative(x, margin=loss_margin, high_bound=high_bound),
        distance=distance,
        cpu=cpu
    )
