import os
from typing import Tuple, Any

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import logging
import logs_handler


def load_dataset(path: str, batch_size=32, download=True) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    """
    Load the CIFAR10 dataset, save it to [path] and return it.
    """

    logger = logging.getLogger("data_handler:load_dataset")

    logger.debug("Creating transform")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    logger.debug("Downloading dataset")
    train_set = torchvision.datasets.CIFAR10(root=path, train=True, download=download, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=path, train=False, download=download, transform=transform)

    logger.debug("Creating DataLoaders")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    logger.info("Dataset successfully loaded")

    return train_loader, test_loader

if __name__ == '__main__':
    DATA_DIR = "./data"

    logging.basicConfig(filename='logs.log', level=logs_handler.get_log_level())
    logger = logging.getLogger("data_handler:main")

    if not os.path.exists(DATA_DIR):
        logger.debug("Creating %s folder", DATA_DIR)

        os.mkdir(DATA_DIR)

        logger.info("Directory %s created successfully", DATA_DIR)

    load_dataset(DATA_DIR)
