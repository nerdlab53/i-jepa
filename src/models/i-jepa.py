"""
Implement I-JEPA here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np
from einops import rearrange

from models.vit import VisionTransformer

class PredictorMLP(nn.Module):

    def __init__(
        self, 
        embed_dim : int,
        hidden_dim : int,
        num_layers : int,
        norm_layer : nn.Module = nn.LayerNorm
    ):
        super().__init__()
        layers = []
        # 1st layer
        layers.append(nn.Linear(embed_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(norm_layer(hidden_dim))
        # Middle layer
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(norm_layer(hidden_dim))
        # Last layer
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.layers = nn.Sequential(*layers)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class IJEPA(nn.Module):

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        encoder_depth: int = 12,
        encoder_num_heads: int = 12,
        encoder_mlp_ratio: float = 4.,
        predictor_hidden_dim: int = 2048,
        predictor_num_layers: int = 3,
        target_momentum: float = 0.996,
        use_cls_token: bool = True,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        # online encoder
        self.encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_cls_token=use_cls_token,
        )

        # target encoder
        self.target_encoder = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=encoder_mlp_ratio,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_cls_token=use_cls_token,
        )

        # disabling gradient update for the target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # predictor
        self.predictor = PredictorMLP(
            embed_dim=embed_dim,
            hidden_dim=predictor_hidden_dim,
            num_layers=predictor_num_layers,
            norm_layer=norm_layer,
        )

        self.target_momentum = target_momentum
        self.copy_params()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.use_cls_token = use_cls_token

    def copy_params(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.copy_(param_q.data)
    
    def momentum_update(self):
        for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_k.data = param_k.data * self.target_momentum + param_q.data * (1 - self.target_momentum)
    
    