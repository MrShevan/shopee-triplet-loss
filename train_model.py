import os
import logging
import json
import numpy as np
import argparse

import torch
import torch.nn as nn

from torch.utils import tensorboard

from torchvision import transforms
import lib.transforms as mytransforms

from lib.train import train_model

from utils import get_dataset
from utils import get_loader
from utils import get_loss
from utils import get_triplet_selector
from utils import get_model
from utils import get_optimizer
from utils import get_scheduler
from utils import get_device_name

import warnings
warnings.filterwarnings("ignore")

# FIX SEED
SEED = 1329784
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# TEACHER MODEL DUMP
torch.nn.Module.dump_patches = True

# TRANSFORMS
transforms_pipeline = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        mytransforms.RandomWatermark(p=0.5, watermarks_dir='/app/data/watermarks'),
        mytransforms.RandomText(p=0.5, fonts_dir='/app/data/fonts'),
        mytransforms.RandomBound(p=0.3),
        transforms.Resize((220, 220)),
        mytransforms.RandomResizeWithPad(p=0.7, scale=(0.8, 1), fill_value=256),  # does not change size
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ]
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--configs",
        type=str,
        default='configs/train_config.json'
    )
    args = parser.parse_args()

    with open(args.configs, 'r') as json_file:
        settings = json.load(json_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s::%(levelname)s::%(name)s::%(message)s",
        handlers=[logging.FileHandler(settings['log_file']), logging.StreamHandler()]
    )

    # Make Dataset
    train_dataset = get_dataset(
        dataset_name=settings["train_dataset"]["dataset_name"],
        transform=transforms_pipeline,
        params=settings["train_dataset"]["params"]
    )

    test_dataset = get_dataset(
        dataset_name=settings["test_dataset"]["dataset_name"],
        transform=transforms_pipeline,
        params=settings["test_dataset"]["params"]
    )

    # Make Dataloader
    train_loader = get_loader(
        dataset=train_dataset,
        sampler=settings["train_loader"]["sampler"],
        params=settings["train_loader"]["params"]
    )

    test_loader = get_loader(
        dataset=test_dataset,
        sampler=settings["test_loader"]["sampler"],
        params=settings["test_loader"]["params"]
    )

    # Make model
    loss_fn = get_loss(
        loss_name=settings['loss']['loss_name'],
        params=settings['loss']['params']
    )

    triplet_selector = get_triplet_selector(
        selector_name=settings['triplet_selector']['selector_name'],
        params=settings['triplet_selector']['params']
    )

    # Prepare model
    device, multi_gpu = get_device_name(cuda=settings['cuda'])

    model = get_model(
            model_name=settings['model']['model_name'],
            pretrained=settings['model']['pretrained'],
            finetune=settings['model']['finetune'],
            embedding_size=settings['model']['embedding_size']
    )

    if multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Get Optimizer
    optimizer = get_optimizer(
        optimizer_name=settings['optimizer']['optimizer_name'],
        model=model,
        params=settings['optimizer']['params']
    )

    # Get Scheduler
    scheduler = get_scheduler(
        scheduler_name=settings['scheduler']['scheduler_name'],
        optimizer=optimizer,
        params=settings['scheduler']['params']
    )

    writer = tensorboard.SummaryWriter(
        log_dir=os.path.join(settings['tensorboard_dir'], settings['config_name'])
    )

    # Train Parameters
    n_epochs = settings['num_epochs']
    start_epoch = settings['start_epoch']
    log_interval = settings['log_interval']
    model_save_interval = settings['model_save_interval']

    # Model Directory
    path_to_save = os.path.join(settings["models_dir"], settings["config_name"])
    if not os.path.exists(path_to_save):
        os.mkdir(path_to_save)

    # Load checkpoint
    if os.listdir(path_to_save):
        model2epoch = {model: int(model.split('_')[-1].split('.')[0]) for model in
                       os.listdir(path_to_save)}
        last_model = max(model2epoch, key=lambda k: model2epoch[k])

        logging.info(f'Load model: {os.path.join(path_to_save, last_model)}')

        checkpoint = torch.load(os.path.join(path_to_save, last_model))

        if multi_gpu:
            model.module.load_state_dict(checkpoint['state_dict'])

        else:
            model.load_state_dict(checkpoint['state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Train model
    train_model(
        train_loader,
        test_loader,
        triplet_selector=triplet_selector,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        n_epochs=n_epochs,
        start_epoch=start_epoch,
        device=device,
        multi_gpu=multi_gpu,
        log_interval=log_interval,
        model_save_interval=model_save_interval,
        path_to_save=path_to_save,
        writer=writer
    )
