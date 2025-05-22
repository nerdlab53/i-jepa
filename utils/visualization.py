"""
Visualization utilities for I-JEPA.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from einops import rearrange


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Denormalize a tensor image with mean and standard deviation.
    
    Args:
        tensor: Tensor image of size (C, H, W) to be denormalized.
        mean: Mean used for normalization.
        std: Standard deviation used for normalization.
        
    Returns:
        Denormalized tensor image.
    """
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def visualize_context_target_masks(
    images,
    context_masks,
    target_masks,
    grid_size,
    num_images=4,
    figsize=(15, 10),
):
    """
    Visualize context and target masks on images.
    
    Args:
        images: Tensor of shape (B, C, H, W).
        context_masks: Boolean tensor of shape (B, N) where N is the number of patches.
        target_masks: Boolean tensor of shape (B, N) where N is the number of patches.
        grid_size: Size of the grid (H/patch_size, W/patch_size).
        num_images: Number of images to visualize.
        figsize: Figure size.
    """
    # Select a subset of images
    images = images[:num_images].detach().cpu()
    context_masks = context_masks[:num_images].detach().cpu()
    target_masks = target_masks[:num_images].detach().cpu()
    
    # Denormalize images
    images = torch.stack([denormalize(img) for img in images])
    
    # Create figure
    fig, axes = plt.subplots(num_images, 3, figsize=figsize)
    
    for i in range(num_images):
        # Original image
        axes[i, 0].imshow(rearrange(images[i], 'c h w -> h w c').numpy())
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis("off")
        
        # Image with context mask
        img_with_context = images[i].clone()
        mask_2d = rearrange(context_masks[i], '(h w) -> h w', h=grid_size, w=grid_size)
        mask_size = img_with_context.shape[1] // grid_size
        
        # Create a mask overlay
        for h in range(grid_size):
            for w in range(grid_size):
                if mask_2d[h, w]:
                    # Add a colored border to context patches
                    h_start = h * mask_size
                    h_end = (h + 1) * mask_size
                    w_start = w * mask_size
                    w_end = (w + 1) * mask_size
                    
                    # Add green border
                    border_width = 2
                    img_with_context[:, h_start:h_start+border_width, w_start:w_end] = rearrange(torch.tensor([0, 1, 0]), 'c -> c 1 1')
                    img_with_context[:, h_end-border_width:h_end, w_start:w_end] = rearrange(torch.tensor([0, 1, 0]), 'c -> c 1 1')
                    img_with_context[:, h_start:h_end, w_start:w_start+border_width] = rearrange(torch.tensor([0, 1, 0]), 'c -> c 1 1')
                    img_with_context[:, h_start:h_end, w_end-border_width:w_end] = rearrange(torch.tensor([0, 1, 0]), 'c -> c 1 1')
        
        axes[i, 1].imshow(rearrange(img_with_context, 'c h w -> h w c').numpy())
        axes[i, 1].set_title("Context Regions")
        axes[i, 1].axis("off")
        
        # Image with target mask
        img_with_target = images[i].clone()
        mask_2d = rearrange(target_masks[i], '(h w) -> h w', h=grid_size, w=grid_size)
        
        # Create a mask overlay
        for h in range(grid_size):
            for w in range(grid_size):
                if mask_2d[h, w]:
                    # Add a colored border to target patches
                    h_start = h * mask_size
                    h_end = (h + 1) * mask_size
                    w_start = w * mask_size
                    w_end = (w + 1) * mask_size
                    
                    # Add red border
                    border_width = 2
                    img_with_target[:, h_start:h_start+border_width, w_start:w_end] = rearrange(torch.tensor([1, 0, 0]), 'c -> c 1 1')
                    img_with_target[:, h_end-border_width:h_end, w_start:w_end] = rearrange(torch.tensor([1, 0, 0]), 'c -> c 1 1')
                    img_with_target[:, h_start:h_end, w_start:w_start+border_width] = rearrange(torch.tensor([1, 0, 0]), 'c -> c 1 1')
                    img_with_target[:, h_start:h_end, w_end-border_width:w_end] = rearrange(torch.tensor([1, 0, 0]), 'c -> c 1 1')
        
        axes[i, 2].imshow(rearrange(img_with_target, 'c h w -> h w c').numpy())
        axes[i, 2].set_title("Target Regions")
        axes[i, 2].axis("off")
    
    plt.tight_layout()
    return fig


def plot_training_progress(
    log_file,
    metrics=["loss", "mse", "cos_sim"],
    figsize=(12, 8),
):
    """
    Plot training progress from log file.
    
    Args:
        log_file: Path to log file.
        metrics: List of metrics to plot.
        figsize: Figure size.
    """
    # Parse log file
    epochs = []
    metrics_values = {metric: [] for metric in metrics}
    
    with open(log_file, "r") as f:
        for line in f:
            if "Epoch:" in line and "|" in line:
                parts = line.strip().split("|")
                epoch_part = parts[0].strip()
                epoch = int(epoch_part.split(":")[1].strip())
                epochs.append(epoch)
                
                for metric in metrics:
                    for part in parts[1:]:
                        if metric.lower() in part.lower():
                            value = float(part.split(":")[1].strip())
                            metrics_values[metric].append(value)
                            break
    
    # Plot metrics
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    
    if len(metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(metrics):
        axes[i].plot(epochs, metrics_values[metric])
        axes[i].set_ylabel(metric)
        axes[i].grid(True)
    
    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    
    return fig