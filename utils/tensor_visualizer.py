"""
Tensor visualization utilities for I-JEPA.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
import tabulate
import matplotlib.pyplot as plt
from collections import OrderedDict


def tensor_info(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
    """
    Extract useful information from a tensor.
    
    Args:
        tensor: The tensor to analyze
        name: Name identifier for the tensor
        
    Returns:
        Dictionary containing tensor information
    """
    info = OrderedDict()
    info["Name"] = name
    info["Shape"] = tuple(tensor.shape)
    info["Dimensions"] = len(tensor.shape)
    info["Dtype"] = tensor.dtype
    info["Device"] = tensor.device
    info["Min"] = tensor.min().item() if tensor.numel() > 0 else None
    info["Max"] = tensor.max().item() if tensor.numel() > 0 else None
    info["Mean"] = tensor.mean().item() if tensor.numel() > 0 else None
    info["Std"] = tensor.std().item() if tensor.numel() > 0 else None
    info["Memory (MB)"] = tensor.element_size() * tensor.numel() / (1024 * 1024)
    info["Requires Grad"] = tensor.requires_grad
    info["Has NaN"] = torch.isnan(tensor).any().item() if tensor.numel() > 0 else None
    info["Has Inf"] = torch.isinf(tensor).any().item() if tensor.numel() > 0 else None
    return info


def print_tensor_info(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Print a formatted summary of tensor information.
    
    Args:
        tensor: The tensor to analyze
        name: Name identifier for the tensor
    """
    info = tensor_info(tensor, name)
    headers = ["Property", "Value"]
    table = [[k, v] for k, v in info.items()]
    print(tabulate.tabulate(table, headers=headers, tablefmt="grid"))


def visualize_tensor_stats(tensors: Dict[str, torch.Tensor], figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Visualize statistics of multiple tensors in a grouped bar chart.
    
    Args:
        tensors: Dictionary mapping tensor names to tensors
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    tensor_stats = OrderedDict()
    
    for name, tensor in tensors.items():
        if tensor.numel() > 0:
            tensor_stats[name] = {
                "Min": tensor.min().item(),
                "Max": tensor.max().item(),
                "Mean": tensor.mean().item(),
                "Std": tensor.std().item()
            }
    
    if not tensor_stats:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    stats = list(list(tensor_stats.values())[0].keys())
    tensor_names = list(tensor_stats.keys())
    
    x = np.arange(len(stats))
    width = 0.8 / len(tensor_names)
    
    for i, name in enumerate(tensor_names):
        values = [tensor_stats[name][stat] for stat in stats]
        offset = width * i - width * len(tensor_names) / 2 + width / 2
        ax.bar(x + offset, values, width, label=name)
    
    ax.set_ylabel('Value')
    ax.set_title('Tensor Statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(stats)
    ax.legend()
    
    plt.tight_layout()
    
    return fig


def batch_tensor_info(batch: Dict[str, Any], prefix: str = "") -> None:
    """
    Print information for all tensors in a batch dictionary.
    
    Args:
        batch: Dictionary containing tensors
        prefix: Prefix string for tensor names
    """
    for name, item in batch.items():
        if isinstance(item, torch.Tensor):
            print_tensor_info(item, f"{prefix}{name}")
        elif isinstance(item, dict):
            batch_tensor_info(item, f"{prefix}{name}.")


def model_tensor_stats(model: torch.nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Extract statistics about model parameters and buffers.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing parameter statistics
    """
    stats = OrderedDict()
    
    # Parameters
    for name, param in model.named_parameters():
        stats[f"param.{name}"] = {
            "shape": tuple(param.shape),
            "size": param.numel(),
            "dtype": param.dtype,
            "device": param.device,
            "requires_grad": param.requires_grad,
            "min": param.min().item() if param.numel() > 0 else None,
            "max": param.max().item() if param.numel() > 0 else None,
            "mean": param.mean().item() if param.numel() > 0 else None,
            "std": param.std().item() if param.numel() > 0 else None,
            "has_nan": torch.isnan(param).any().item() if param.numel() > 0 else None,
            "has_inf": torch.isinf(param).any().item() if param.numel() > 0 else None,
        }
    
    # Buffers
    for name, buffer in model.named_buffers():
        stats[f"buffer.{name}"] = {
            "shape": tuple(buffer.shape),
            "size": buffer.numel(),
            "dtype": buffer.dtype,
            "device": buffer.device,
            "requires_grad": buffer.requires_grad,
            "min": buffer.min().item() if buffer.numel() > 0 else None,
            "max": buffer.max().item() if buffer.numel() > 0 else None,
            "mean": buffer.mean().item() if buffer.numel() > 0 else None,
            "std": buffer.std().item() if buffer.numel() > 0 else None,
            "has_nan": torch.isnan(buffer).any().item() if buffer.numel() > 0 else None,
            "has_inf": torch.isinf(buffer).any().item() if buffer.numel() > 0 else None,
        }
    
    return stats


def print_model_summary(model: torch.nn.Module, max_depth: int = 3) -> None:
    """
    Print a summary of the model architecture with parameters and shapes.
    
    Args:
        model: PyTorch model
        max_depth: Maximum depth of nested modules to display
    """
    def get_module_summary(module, depth=0, parent_name=""):
        if depth > max_depth:
            return []
        
        rows = []
        name = parent_name + ("." if parent_name else "") + module.__class__.__name__
        
        # Count parameters
        params_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        params_non_trainable = sum(p.numel() for p in module.parameters() if not p.requires_grad)
        
        # Get input/output shapes if available
        input_shape = getattr(module, "input_shape", "-")
        output_shape = getattr(module, "output_shape", "-")
        
        rows.append([
            "  " * depth + name,
            params_trainable,
            params_non_trainable,
            input_shape,
            output_shape
        ])
        
        # Process child modules
        for child_name, child in module.named_children():
            child_rows = get_module_summary(
                child, 
                depth + 1, 
                parent_name + ("." if parent_name else "") + child_name
            )
            rows.extend(child_rows)
        
        return rows
    
    rows = get_module_summary(model)
    headers = ["Layer", "Trainable Params", "Non-trainable Params", "Input Shape", "Output Shape"]
    print(tabulate.tabulate(rows, headers=headers, tablefmt="grid"))
    
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_non_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = total_trainable + total_non_trainable
    
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {total_trainable:,}")
    print(f"Non-trainable parameters: {total_non_trainable:,}")
    print(f"Parameter size: {total * 4 / (1024 * 1024):.2f} MB (assuming float32)") 