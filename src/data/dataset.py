"""
Dataset loader class
"""

import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from typing import Callable
from data.transforms import IJEPATrainTransform, IJEPAEvalTransform

def get_dataset(
    dataset_name : str,
    data_path : str,
    transform : Callable,
    split : str = 'train',
) -> Dataset:
    if dataset_name == "imagenet":
        split_folder = "train" if split == "train" else "val"
        return datasets.ImageFolder(
            os.path.join(data_path, split_folder),
            transform=transform,
        )
    elif dataset_name == "cifar10":
        return datasets.CIFAR10(
            root=data_path,
            train=(split == "train"),
            download=True,
            transform=transform,
        )
    elif dataset_name == "cifar100":
        return datasets.CIFAR100(
            root=data_path,
            train=(split == "train"),
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def create_dataloader(
    dataset_name : str,
    data_path : str,
    batch_size : int,
    img_size : int = 224,
    split : str = 'train',
    num_workers : int = 4,
    use_color_jitter : bool = True,
    use_gaussian_blur : bool = True,
    use_random_resized_crop : bool = True,
    use_random_horizontal_flip : bool = True,
    distributed : bool = False,
):
    if split == 'train':
        transform = IJEPATrainTransform(
            img_size=img_size,
            use_color_jitter=use_color_jitter,
            use_gaussian_blur=use_gaussian_blur,
            use_random_resized_crop=use_random_resized_crop,
            use_random_horizontal_flip=use_random_horizontal_flip
        )
    else:
        transform = IJEPAEvalTransform(
            img_size=img_size
        )
    dataset = get_dataset(
        dataset_name=dataset_name,
        data_path=data_path,
        transform=transform,
        split=split
    )
    sampler = None
    if distributed:
        if split == 'train':
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=(split == 'train')
    )

    return dataloader