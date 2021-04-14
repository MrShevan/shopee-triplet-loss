import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler

from torchvision import transforms

from lib.datasets import ShopeeDataset
from lib.model import EmbeddingNet, TripletNet
from lib.loss import OnlineTripletLoss
from lib.sampler import BalancedBatchSampler
from lib.triplet_selector import HardestNegativeTripletSelector, \
    SemihardNegativeTripletSelector


def get_dataset(dataset_name: str, transform: transforms.Compose, params: dict):
    if dataset_name == 'shopee':
        return ShopeeDataset(
            transform=transform,
            **params
        )

    else:
        Exception('Not implemented dataset!')


def get_loader(dataset: ShopeeDataset, sampler: dict, params: dict):
    if sampler:
        if sampler["sampler_name"] == 'BatchSampler':
            batch_sampler = BalancedBatchSampler(
                dataset.labels,
                **sampler["params"]
            )

        else:
            Exception('Not implemented sampler!')

        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **params
        )

    return DataLoader(
        dataset,
        **params
    )


def get_loss(loss_name: str, params: dict):
    if loss_name == 'online_triplet_loss':
        return OnlineTripletLoss(
            params['margin'],
            SemihardNegativeTripletSelector(params['margin'])
        )

    else:
        Exception('Not implemented loss!')


def get_model(model_name: str):
    if model_name == 'EmbeddingNet':
        return EmbeddingNet()

    if model_name == 'TripletNet':
        embedding_net = EmbeddingNet()
        return TripletNet(embedding_net)

    else:
        Exception('Not implemented model!')


def get_optimizer(optimizer_name: str, model: nn.Module, params: dict):
    if optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), **params)

    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), **params)

    else:
        Exception('Not implemented optimizer!')


def get_scheduler(scheduler_name: str, optimizer: optim.Optimizer, params: dict):
    if scheduler_name == 'step_lr':
        return lr_scheduler.StepLR(optimizer, **params)

    else:
        Exception('Not implemented scheduler')


def get_device_name(cuda: bool):
    """
    Extract correct cuda name

    Args:
        cuda: use GPU or not
    """
    device = torch.device("cpu")
    multi_gpu = False

    if cuda:
        if torch.cuda.is_available():
            cuda_name = f'cuda:{torch.cuda.current_device()}'
            device = torch.device(cuda_name)

            if torch.cuda.device_count() > 1:
                multi_gpu = True
                logging.info("Use " + str(torch.cuda.device_count()) + " GPUs")
        else:
            logging.info("GPU not available, training on CPU")

    logging.info("Training on device: {}".format(device))
    return device, multi_gpu
