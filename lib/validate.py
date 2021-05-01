import os
import logging
import numpy as np

import torch
from torch import nn
from torch.nn import functional as f
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import dataloader
from torch.utils import tensorboard

from lib.distances import pdist_l2
from lib.utils import save_model
from lib.utils import compute_f1_score
from lib.utils import imshow_triplet


def validate_model(
    val_loader: dataloader.DataLoader,
    model: nn.Module,
    f1_thr: float,
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

    val_f1 = val_epoch(
        val_loader=val_loader,
        model=model,
        f1_thr=f1_thr,
        device=device,
        log_interval=log_interval
    )

    logging.info(f'Val epoch F1: {val_f1:.4f}')

    if writer:
        writer.add_scalar(f'Val F1', val_f1, 1)


def val_epoch(
    val_loader: dataloader.DataLoader,
    model: nn.Module,
    f1_thr: float,
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

    dist_matrix = pdist_l2(embeddings).cpu().detach().numpy()
    targets = targets.cpu().numpy()

    # metrics
    f1_score = compute_f1_score(dist_matrix, targets, f1_thr)
    return f1_score
