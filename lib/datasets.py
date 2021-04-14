import os
import logging

import numpy as np
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

from torchvision import transforms


class ShopeeDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        images_dir: str,
        transform: transforms.Compose = None
    ):
        """
        Read Dataset for Shopee Competition

        Args:
            dataset_path: path to dataset file
            images_dir: path to images directory
            transform: torch transforms for images
        """
        self.dataset = self._dataset_prepare(dataset_path, images_dir)
        self.transform = transform

        self.filepath = self.dataset.filepath.values
        self.labels = self.dataset.label_group.values
        self.labels_set = set(self.labels)

        self.label_to_indices = {
            label: np.where(self.labels == label)[0]
            for label in self.labels_set
        }

    @staticmethod
    def _dataset_prepare(path: str, images_dir: str):
        """
        Read dataset and make dataset preparation

        Args:
            path: path to dataset
            images_dir: path to images directory
        """
        dataset = pd.read_csv(path)
        dataset['filepath'] = dataset['image'].apply(
            lambda x: os.path.join(images_dir, x)
        )

        return dataset

    @staticmethod
    def _image_read(path: str):
        """
        Safe read by path with PIL

        Args:
            path: path to image
        """
        try:
            image = Image.open(path)

        except (FileNotFoundError, OSError) as e:
            logging.error(f'{e.__class__.__name__}: {path}')
            raise Exception(e)

        return image

    def _get_neighbours(self, label: int):
        """
        Get images indixes with same label_group

         Args:
            label: target label_group
        """
        return self.label_to_indices[label]

    def __getitem__(self, index: int):
        """
        Get image and label by index

        Args:
            index:
        """
        image_path, label = self.filepath[index], self.labels[index]
        image = self._image_read(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """Get dataset length"""
        return len(self.dataset)
