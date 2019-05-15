import yaml
from io import open
from easydict import EasyDict
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import Compose, ToTensor, Resize


def get_yaml_config(path, verbose=False):
    with open(path, encoding="utf-8") as h:
        config = EasyDict(yaml.safe_load(h))
    if verbose:
        print("Config values:")
        print("==============")
        for k, v in config.items():
            print(f"{k}: {v}")
    return config


def get_dataloader(image_type, image_path, image_size, batch_size):
    train_path = image_path / image_type
    test_path = image_path / f"test_{image_type}"
    transform = Compose([Resize(image_size), ToTensor()])
    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return train_dataloader, test_dataloader


def set_seed(seed):
    # https://pytorch.org/docs/stable/notes/randomness.html
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    # cudnn
    # [WARN] Deterministic mode can have a performance impact, depending on your model.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
