import os
import logging
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils.data import dataloader
from torch.optim import lr_scheduler

from lib.utils import save_model
from lib.utils import pdist
from lib.utils import compute_f1_score


def train_model(
    train_loader: dataloader.DataLoader,
    test_loader: dataloader.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.StepLR,
    n_epochs: int,
    device: torch.device,
    multi_gpu: bool,
    log_interval: int,
    model_save_interval: int,
    start_epoch: int = 0,
    path_to_save: str = ''
):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss

    Args:
        train_loader: pytorch loader for train dataset
        test_loader: pytorch loader for val dataset
        model: pytorch model
        loss_fn: loss function
        optimizer:
        scheduler:
        n_epochs:
        device:
        multi_gpu:
        log_interval:
        start_epoch:
        path_to_save:
    """
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        logging.info(f'Train Epoch: {epoch + 1}/{n_epochs}')

        model, train_loss, f1_score = train_epoch(
            train_loader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            log_interval=log_interval
        )

        logging.info(f'Train epoch Loss: {train_loss:.4f}')
        logging.info(f'Train epoch F1: {f1_score:.4f}')

        if (epoch + 1) % model_save_interval == 0:
            modelpath = os.path.join(path_to_save, f'model_epoch_{epoch}.pth')
            save_model(epoch, model, optimizer, scheduler, modelpath, multi_gpu)

        # Test stage
        logging.info(f'Test Epoch: {epoch + 1}/{n_epochs}')

        val_loss, f1_score = test_epoch(
            test_loader=test_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            log_interval=log_interval
        )

        logging.info(f'Test epoch Loss: {val_loss:.4f}')
        logging.info(f'Test epoch F1: {f1_score:.4f}')


def train_epoch(
    train_loader: dataloader.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int
):
    """
    Train stage

    Args:
        train_loader:
        model:
        loss_fn:
        optimizer:
        device:
        log_interval:
    """
    model.train()

    losses = []
    dists_pos = []
    dists_neg = []

    embeddings = []
    targets = []

    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        # prepare tensors
        if device.type == 'cuda':
            data = data.to(device)
            if target is not None:
                target = target.to(device)

        # forward
        optimizer.zero_grad()
        outputs = model(data)

        # compute loss, backward gradients and make step
        loss, dist_pos, dist_neg = loss_fn(outputs, target)

        losses.append(loss.item())
        dists_pos.append(dist_pos.item())
        dists_neg.append(dist_neg.item())

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # save embeddings and targets
        embeddings.append(outputs)
        targets.append(target)

        if batch_idx % log_interval == 0:
            running_loss = np.mean(losses)
            running_pos_dists = np.mean(dists_pos)
            running_neg_dists = np.mean(dists_neg)

            logging.info(
                ' | '.join([
                    '[{:3d}/{:3d}] {:3d}%',
                    'Loss: {:.3f}',
                    'Avg positive dist.: {:.2f}',
                    'Avg negative dist.: {:.2f}'
                ]).format(
                    batch_idx, len(train_loader), int(100. * batch_idx / len(train_loader)),
                    running_loss,
                    running_pos_dists,
                    running_neg_dists
                )
            )

            losses = []
            dists_pos = []
            dists_neg = []

    # compute distances and cocncat targets
    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)

    dist_matrix = pdist(embeddings).cpu().detach().numpy()
    targets = targets.cpu().numpy()

    # metrics
    total_loss /= (batch_idx + 1)
    f1_score = compute_f1_score(dist_matrix, targets)
    return model, total_loss, f1_score


def test_epoch(
    test_loader: dataloader.DataLoader,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    log_interval: int
):
    """
    Test stage

    Args:
        val_loader:
        model:
        loss_fn:
        device:
        log_interval:
    """
    with torch.no_grad():
        model.eval()

        losses = []
        dists_pos = []
        dists_neg = []

        embeddings = []
        targets = []

        val_loss = 0

        for batch_idx, (data, target) in enumerate(test_loader):
            # prepare tensors
            if device.type == 'cuda':
                data = data.to(device)
                if target is not None:
                    target = target.to(device)

            # forward
            outputs = model(data)

            # compute loss, backward gradients and make step
            loss, dist_pos, dist_neg = loss_fn(outputs, target)

            losses.append(loss.item())
            dists_pos.append(dist_pos.item())
            dists_neg.append(dist_neg.item())

            losses.append(loss.item())
            val_loss += loss.item()

            # save embeddings and targets
            embeddings.append(outputs)
            targets.append(target)

            if batch_idx % log_interval == 0:
                running_loss = np.mean(losses)
                running_pos_dists = np.mean(dists_pos)
                running_neg_dists = np.mean(dists_neg)

                logging.info(
                    ' | '.join([
                        '[{:3d}/{:3d}] {:3d}%',
                        'Loss: {:.3f}',
                        'Avg positive dist.: {:.2f}',
                        'Avg negative dist.: {:.2f}'
                    ]).format(
                        batch_idx, len(test_loader), int(100. * batch_idx / len(test_loader)),
                        running_loss,
                        running_pos_dists,
                        running_neg_dists
                    )
                )

                losses = []
                dists_pos = []
                dists_neg = []

    # compute distances and cocncat targets
    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)

    dist_matrix = pdist(embeddings).cpu().detach().numpy()
    targets = targets.cpu().numpy()

    # metrics
    val_loss /= (batch_idx + 1)
    f1_score = compute_f1_score(dist_matrix, targets)
    return val_loss, f1_score
