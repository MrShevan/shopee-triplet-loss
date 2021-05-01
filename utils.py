import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from torch.optim import lr_scheduler

from torchvision import transforms, models

from lib.datasets import ShopeeDataset
from lib.sampler import BalancedBatchSampler
from lib.triplet_selector import HardTripletSelector, SemiHardTripletSelector


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
    if loss_name == 'triplet_margin_loss':
        return nn.TripletMarginLoss(**params)

    else:
        Exception('Not implemented loss!')


def get_triplet_selector(selector_name: str, params: dict):
    if selector_name == 'hard_triplet_selector':
        return HardTripletSelector(**params)

    if selector_name == 'semi_hard_triplet_selector':
        return SemiHardTripletSelector(**params)

    else:
        Exception('Not implemented triplet selector!')


def get_model(model_name: str, pretrained: bool, finetune: bool, embedding_size: int = 224):
    if model_name == 'resnet50':
        embedding_net = models.resnet50(pretrained=pretrained)
        if finetune:
            for param in embedding_net.parameters():
                param.requires_grad = False
        num_ftrs = embedding_net.fc.in_features
        embedding_net.fc = nn.Linear(num_ftrs, embedding_size)

        return embedding_net

    else:
        Exception('Not implemented model!')


def get_optimizer(optimizer_name: str, model: nn.Module, params: dict):
    params_to_update = []
    params_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            params_names.append(name)

    logging.info("Params to learn: {}".format('\t'.join(params_names)))

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
