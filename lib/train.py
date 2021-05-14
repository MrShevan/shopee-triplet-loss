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

from lib.distances import knn
from lib.utils import save_model
from lib.utils import f1_score
from lib.utils import imshow_triplet

from lib.triplet_selector import TripletSelector


def train_model(
    train_loader: dataloader.DataLoader,
    test_loader: dataloader.DataLoader,
    triplet_selector: TripletSelector,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler.StepLR,
    n_epochs: int,
    device: torch.device,
    multi_gpu: bool,
    n_neighbors: int,
    log_interval: int,
    model_save_interval: int,
    writer: tensorboard.SummaryWriter,
    start_epoch: int = 1,
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
        triplet_selector:
        model: pytorch model
        loss_fn: loss function
        optimizer:
        scheduler:
        n_epochs:
        device:
        multi_gpu:
        n_neighbors:
        log_interval:
        model_save_interval:
        writer:
        start_epoch:
        path_to_save:
    """
    for epoch in range(1, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        logging.info(f'Train Epoch: {epoch}/{n_epochs}')

        model, train_loss, train_f1, threshold = train_epoch(
            train_loader=train_loader,
            triplet_selector=triplet_selector,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            n_neighbors=n_neighbors,
            log_interval=log_interval,
            writer=writer
        )

        logging.info(f'Train epoch Loss: {train_loss:.4f}')
        logging.info(f'Train epoch F1: {train_f1:.4f}')
        logging.info(f'Train Threshold: {threshold:.4f}')

        writer.add_scalar(f'Train Loss', train_loss, epoch)
        writer.add_scalar(f'Train F1', train_f1, epoch)

        if (epoch + 1) % model_save_interval == 0:
            modelpath = os.path.join(path_to_save, f'model_epoch_{epoch}.pth')
            save_model(epoch, model, optimizer, scheduler, modelpath, multi_gpu)

        # Test stage
        logging.info(f'Test Epoch: {epoch}/{n_epochs}')

        test_loss, test_f1 = test_epoch(
            test_loader=test_loader,
            triplet_selector=triplet_selector,
            model=model,
            loss_fn=loss_fn,
            device=device,
            threshold=threshold,
            n_neighbors=n_neighbors,
            log_interval=log_interval
        )

        logging.info(f'Test epoch Loss: {test_loss:.4f}')
        logging.info(f'Test epoch F1: {test_f1:.4f}')

        writer.add_scalar(f'Test Loss', test_loss, epoch)
        writer.add_scalar(f'Test F1', test_f1, epoch)


def train_epoch(
    train_loader: dataloader.DataLoader,
    triplet_selector: TripletSelector,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_neighbors: int,
    log_interval: int,
    writer: tensorboard.SummaryWriter
):
    """
    Train stage

    Args:
        train_loader:
        triplet_selector:
        model:
        loss_fn:
        optimizer:
        device:
        log_interval:
        writer:
    """
    model.train()

    losses = []

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

        # l2 normalization
        outputs = f.normalize(outputs, dim=-1, p=2)

        # choose triplets indices
        triplets_indices = triplet_selector.get_triplets(outputs, target)

        if device.type == 'cuda':
            triplets_indices = triplets_indices.to(device)

        anchor = outputs[triplets_indices[:, 0]]
        positive = outputs[triplets_indices[:, 1]]
        negative = outputs[triplets_indices[:, 2]]

        # compute loss, backward gradients and make step
        loss = loss_fn(anchor, positive, negative)

        losses.append(loss.item())
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        # save embeddings and targets
        embeddings.append(outputs)
        targets.append(target)

        # logging
        if batch_idx % log_interval == 0:
            logging.info(
                '[{:3d}/{:3d}] {:3d}% | Loss: {:.3f}'.format(
                    batch_idx, len(train_loader), int(100. * batch_idx / len(train_loader)),
                    np.mean(losses)
                )
            )

            writer.add_figure(
                'Train Triplet',
                imshow_triplet(
                    data[triplets_indices[0][0]], anchor[0],
                    data[triplets_indices[0][1]], positive[0],
                    data[triplets_indices[0][2]], negative[0]
                ),
                batch_idx
            )

            losses = []

    # concat targets and compute distances
    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)

    distances, indices = knn(embeddings, n_neighbors, device)

    # finding threshold and compute f1 score
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

    total_loss /= (batch_idx + 1)
    return model, total_loss, score, threshold


def test_epoch(
    test_loader: dataloader.DataLoader,
    triplet_selector: TripletSelector,
    model: nn.Module,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float,
    n_neighbors: int,
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
        threshold:
        n_neighbors:
        log_interval:
    """
    with torch.no_grad():
        model.eval()

        losses = []

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

            # l2 normalization
            outputs = f.normalize(outputs, dim=-1, p=2)

            # choose triplets
            triplets_indices = triplet_selector.get_triplets(outputs, target)

            if outputs.is_cuda:
                triplets_indices = triplets_indices.cuda()

            anchor = outputs[triplets_indices[:, 0]]
            positive = outputs[triplets_indices[:, 1]]
            negative = outputs[triplets_indices[:, 2]]

            # compute loss, backward gradients and make step
            loss = loss_fn(anchor, positive, negative)
            losses.append(loss.item())

            val_loss += loss.item()

            # save embeddings and targets
            embeddings.append(outputs)
            targets.append(target)

            # logging
            if batch_idx % log_interval == 0:
                logging.info(
                    '[{:3d}/{:3d}] {:3d}% | Loss: {:.3f}'.format(
                        batch_idx, len(test_loader), int(100. * batch_idx / len(test_loader)),
                        np.mean(losses)
                    )
                )

                losses = []

    # concat targets and compute distances
    embeddings = torch.cat(embeddings, dim=0)
    targets = torch.cat(targets, dim=0)

    distances, indices = knn(embeddings, n_neighbors, device)

    # compute f1 score
    targets = targets.detach().cpu().numpy()
    labels = [np.where(targets == label)[0] for label in targets]

    predictions = []
    for k in range(embeddings.shape[0]):
        idx = np.where(distances[k,] < threshold)[0]
        ids = indices[k, idx]

        predictions.append(ids)

    score = f1_score(labels, predictions)

    val_loss /= (batch_idx + 1)
    return val_loss, score
