import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

import numpy as np
import matplotlib.pyplot as plt

from lib.distances import pdist_cosine, pdist_l2


def compute_f1_score(dist_matrix, targets, thr=0.8):
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
        model = model.module

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
    std: tuple = (0.229, 0.224, 0.225),
    show_axis: bool = True
):
    """
    Matplotlib Imshow for Tensor

    Args:
        inp: image data
        title:
        denormalize:
        mean: return denormalized image
        std: return denormalized image
        show_axis: if show axis on image
    """
    inp = inp.numpy().transpose((1, 2, 0))

    if denormalize:
        inp = np.array(std) * inp + np.array(mean)
        inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    plt.axis('on' if show_axis else 'off')
    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated


def imshow_triplet(
    anchor_image,
    anchor_embedding,
    positive_image,
    positive_embedding,
    negative_image,
    negative_embedding,
    denormalize: bool = True,
    mean: tuple = (0.485, 0.456, 0.406),
    std: tuple = (0.229, 0.224, 0.225)
):
    with torch.no_grad():
        embeddings = torch.cat(
            (
                anchor_embedding.unsqueeze(dim=0),
                positive_embedding.unsqueeze(dim=0),
                negative_embedding.unsqueeze(dim=0)
            ), dim=0
        )

        result_l2 = pdist_l2(embeddings).cpu().numpy()
        result_cosine = pdist_cosine(embeddings).cpu().numpy()

        # define figure
        fig = plt.figure(figsize=(15, 5))

        # Anchor image
        ax = fig.add_subplot(1, 3, 1)
        anchor = anchor_image.cpu().numpy().transpose((1, 2, 0))
        if denormalize:
            anchor = np.array(std) * anchor + np.array(mean)
            anchor = np.clip(anchor, 0, 1)
        ax.imshow(anchor)
        ax.axis('off')
        ax.set_title("Anchor", size=12)

        # Positive image
        ax = fig.add_subplot(1, 3, 2)
        positive = positive_image.cpu().numpy().transpose((1, 2, 0))
        if denormalize:
            positive = np.array(std) * positive + np.array(mean)
            positive = np.clip(positive, 0, 1)
        ax.imshow(positive)
        ax.axis('off')
        ax.set_title(
            "Positive\nl2 distance: {:.2f}\ncosine distance: {:.2}".format(result_l2[0][1], result_cosine[0][1]),
            color=("green" if result_l2[0][1] < result_l2[0][2] else "red"),
            size=12
        )

        # Negative image
        ax = fig.add_subplot(1, 3, 3)
        negative = negative_image.cpu().numpy().transpose((1, 2, 0))
        if denormalize:
            negative = np.array(std) * negative + np.array(mean)
            negative = np.clip(negative, 0, 1)
        ax.imshow(negative)
        ax.axis('off')
        ax.set_title(
            "Negative\nl2 distance : {:.2f}\ncosine distance: {:.2}".format(result_l2[0][2], result_cosine[0][2]),
            color=("red" if result_l2[0][1] < result_l2[0][2] else "green"),
            size=12
        )

    return fig
