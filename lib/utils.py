import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

import logging
import numpy as np
import matplotlib.pyplot as plt


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix


def compute_f1_score(dist_matrix, targets, thr=1.0):
    preds_matches = [np.where(row < thr)[0] for row in dist_matrix]
    labels_group = [np.where(targets == label)[0] for label in targets]

    f1_scores = []
    for labels_row, matches_row in zip(labels_group, preds_matches):
        n = len(np.intersect1d(labels_row, matches_row))
        f1_scores.append(2 * n / (len(labels_row) + len(matches_row)))

    return np.mean(f1_scores)


def load_checkpoint(filepath: str, multi_gpu: bool = True):
    """
    Example of using:
    model = load_checkpoint('checkpoint.pth')

    Args:
        filepath: path to file
        multi_gpu:
    """
    checkpoint = torch.load(filepath)
    model = checkpoint['model']

    if multi_gpu:
        model.module.load_state_dict(checkpoint['state_dict'])

    else:
        model.load_state_dict(checkpoint['state_dict'])

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def save_model(
    epoch,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.StepLR,
    filepath: str,
    multi_gpu: bool = True
):
    """
    Save model, states, and optimizer

    Args:
        epoch:
        model: torch model
        optimizer:
        scheduler:
        filepath:
        multi_gpu:
    """
    checkpoint = {
        'epoch': epoch,
        'model': model,
        'state_dict': model.module.state_dict() if multi_gpu else model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }

    torch.save(checkpoint, filepath)


def imshow(
    inp: torch.Tensor,
    title: str = None,
    denormalize: bool = True,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
):
    """
    Matplotlib Imshow for Tensor

    Args:
        inp: image data
        title:
        denormalize:
        mean: return denormalized image
        std: return denormalized image
    """
    inp = inp.numpy().transpose((1, 2, 0))

    if denormalize:
        inp = np.array(std) * inp + np.array(mean)
        inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated
