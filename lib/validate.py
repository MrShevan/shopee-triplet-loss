import logging
import numpy as np

import torch
from torch import nn
from torch.nn import functional as f
from torch.utils.data import dataloader
from torch.utils import tensorboard

from lib.distances import knn
from lib.utils import f1_score


def validate_model(
    val_loader: dataloader.DataLoader,
    model: nn.Module,
    n_neighbors: int,
    device: torch.device,
    log_interval: int,
    writer: tensorboard.SummaryWriter = None
):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss

    """
    # Val stage
    logging.info('Val Epoch')

    val_f1, thr_f1 = val_epoch(
        val_loader=val_loader,
        model=model,
        n_neighbors=n_neighbors,
        device=device,
        log_interval=log_interval
    )

    logging.info(f'Val epoch F1: {val_f1:.4f}')
    logging.info(f'Threshold: {thr_f1:.4f}')

    if writer:
        writer.add_scalar(f'Val F1', val_f1, 1)


def val_epoch(
    val_loader: dataloader.DataLoader,
    model: nn.Module,
    n_neighbors: int,
    device: torch.device,
    log_interval: int
):
    """
    Test stage

    Args:
        test_loader:
        triplet_selector:
        model:
        loss_fn:
        device:
        log_interval:
    """
    with torch.no_grad():
        model.eval()

        embeddings = []
        targets = []

        for batch_idx, (data, target) in enumerate(val_loader):
            # prepare tensors
            if device.type == 'cuda':
                data = data.to(device)
                if target is not None:
                    target = target.to(device)

            # forward
            outputs = model(data)

            # l2 normalization
            outputs = f.normalize(outputs, dim=-1, p=2)

            # save embeddings and targets
            embeddings.append(outputs)
            targets.append(target)

            if batch_idx % log_interval == 0:

                logging.info('[{:3d}/{:3d}] {:3d}%'.format(
                        batch_idx, len(val_loader), int(100. * batch_idx / len(val_loader))
                    )
                )

    # compute distances and concat targets
    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)

    distances, indices = knn(embeddings, n_neighbors, device)

    # compute f1 score
    targets = targets.detach().cpu().numpy()
    labels = [np.where(targets == label)[0] for label in targets]
    x, y = [], []

    bins = np.arange(0.05, 1.0, 0.05)

    logging.info('Threshold searching...')
    for i, thr in enumerate(bins):
        predictions = []
        for k in range(embeddings.shape[0]):
            idx = np.where(distances[k,] < thr)[0]
            ids = indices[k, idx]

            predictions.append(ids)

        logging.info('[{:3d}/{:3d}] {:3d}%'.format(
            i + 1, len(bins), int(100. * (i + 1) / len(bins))
        ))

        y.append(f1_score(labels, predictions))
        x.append(thr)

    threshold = x[np.argmax(y)]
    score = y[np.argmax(y)]

    return score, threshold
