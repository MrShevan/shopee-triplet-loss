import os
import json
import argparse
import logging

from torchvision import transforms

from utils import get_dataset
from utils import get_loader
from utils import get_device_name

from lib.validate import validate_model

from lib.utils import load_checkpoint

# TRANSFORMS
transforms_pipeline = transforms.Compose(
    [
        transforms.Resize((220, 220)),
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

    logging.info(f'F1 Score Threshold: {settings["f1_thr"]}')
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
    f1_thr = settings['f1_thr']

    # Validate model
    validate_model(
        val_loader,
        model=model,
        f1_thr=f1_thr,
        device=device,
        log_interval=log_interval
    )
