import os
import json
import argparse
import logging

import pandas as pd
import numpy as np

from torchvision import transforms

from utils import get_dataset
from utils import get_loader
from utils import get_device_name

from lib.validate import validate_model

from lib.utils import load_checkpoint
from lib.utils import f1_score

# TRANSFORMS
transforms_pipeline = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
    ]
)


def compute_phash_baseline(df: pd.DataFrame):
    tmp = df.reset_index().groupby('image_phash').index.agg('unique').to_dict()
    preds_matches = df.image_phash.map(tmp).values

    labels = df.label_group.values
    labels_group = [np.where(labels == label)[0] for label in labels]

    return f1_score(labels_group, preds_matches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--configs",
        type=str,
        default='configs/val_config.json'
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
    val_dataset = get_dataset(
        dataset_name=settings["val_dataset"]["dataset_name"],
        transform=transforms_pipeline,
        params=settings["val_dataset"]["params"]
    )

    val_loader = get_loader(
        dataset=val_dataset,
        sampler=settings["val_loader"]["sampler"],
        params=settings["val_loader"]["params"]
    )

    # Prepare model
    device, multi_gpu = get_device_name(cuda=settings['cuda'])

    # Model Directory
    path_to_save = os.path.join(settings["models_dir"], settings["config_name"])
    model2epoch = {model: int(model.split('_')[-1].split('.')[0]) for model in
                   os.listdir(path_to_save)}
    last_model = max(model2epoch, key=lambda k: model2epoch[k])

    logging.info(f'Load model: {os.path.join(path_to_save, last_model)}')

    model = load_checkpoint(
        os.path.join(path_to_save, last_model),
        multi_gpu
    )

    log_interval = settings['log_interval']
    n_neighbors = settings['n_neighbors']

    # Validate model
    validate_model(
        val_loader,
        model=model,
        n_neighbors=n_neighbors,
        device=device,
        log_interval=log_interval
    )

    # Phash baseline
    phash_score = compute_phash_baseline(val_dataset.dataset)
    logging.info(f'Phash baseline: {phash_score:.4f}')
