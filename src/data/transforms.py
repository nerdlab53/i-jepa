"""
Class to define transforms for datasets
"""

import torch
import torchvision.transforms as transforms
from typing import Optional, Tuple, List, Dict


class IJEPATrainTransform:
    """
    Transformations for I-JEPA training.
    """
    def __init__(
        self,
        img_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        use_color_jitter: bool = True,
        use_gaussian_blur: bool = True,
        use_random_resized_crop: bool = True,
        use_random_horizontal_flip: bool = True,
    ):
        transforms_list = []

        if use_random_resized_crop:
            transforms_list.append(
                transforms.RandomResizedCrop(
                    img_size, scale=(0.2, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BICUBIC
                )
            )
        else:
            transforms_list.append(
                transforms.Resize(
                    img_size, interpolation=transforms.InterpolationMode.BICUBIC
                )
            )
            transforms_list.append(
                transforms.CenterCrop(img_size)
            )
        
        if use_random_horizontal_flip:
            transforms_list.append(transforms.RandomHorizontalFlip())
        
        if use_color_jitter:
            transforms_list.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8
                )
            )
        
        if use_gaussian_blur:
            transforms_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                    p=0.5
                )
            )
        
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize(mean=mean, std=std))
        
        self.transform = transforms.Compose(transforms_list)
    
    def __call__(self, img):
        return self.transform(img)


class IJEPAEvalTransform:
    def __init__(
        self,
        img_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        self.transform = transforms.Compose([
            transforms.Resize(
                int(img_size * (256/224)), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, img):
        return self.transform(img) 