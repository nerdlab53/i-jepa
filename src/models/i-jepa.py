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

    def get_context_target_masks(
            self,
            batch_size : int,
            num_context_blocks : int, 
            num_target_blocks : int,
            context_block_size : int,
            target_block_size : int,
    ):
        device = next(self.parameters()).device()
        num_patches = self.grid_size ** 2

        context_masks = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)
        target_masks = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)

        for b in range(batch_size):
            context_positions = []
            for _ in range(num_context_blocks):
                while True:
                    row = np.random.randint(0, self.grid_size-context_block_size+1)
                    col = np.random.randint(0, self.grid_size-context_block_size+1)

                    overlap = False
                    for pos in context_positions:
                        if(abs(row - pos[0]) < context_block_size and
                           abs(col - pos[1]) < context_block_size
                           ):
                            overlap = True
                            break
                    if not overlap:
                        context_positions.append((row, col))
                        break
            target_positions = []
            for _ in range(num_target_blocks):
                while True:
                    row = np.random.randint(0, self.grid_size-target_block_size+1)
                    col = np.random.randint(0, self.grid_size-target_block_size+1)
                    overlap = False
                    for pos in context_positions:
                        if(
                            row < pos[0] + context_block_size and
                            row + target_block_size > pos[0] and
                            col < pos[1] + context_block_size and 
                            col + target_block_size > pos[1]
                        ):
                            overlap = True
                            break
                    for pos in target_positions:
                        if(
                            abs(row - pos[0]) < target_block_size and
                            abs(col - pos[1]) < target_block_size
                        ):
                            overlap = True
                            break
                    if not overlap:
                        target_positions.append((row, col))
                        break
            
            for row, col in context_positions:
                for i in range(context_block_size):
                    for j in range(context_block_size):
                        patch_idx = (row + i) * self.grid_size + (col + j)
                        context_masks[b, patch_idx] = True
            for row, col in target_positions:
                for i in range(target_block_size):
                    for j in range(target_block_size):
                        patch_idx = (row + i) * self.grid_size + (col + j)
                        target_masks[b, patch_idx] = True
        
        return context_masks, target_masks

    def extract_masked_features(
            self, 
            x : torch.Tensor,
            masks : torch.Tensor,
            encoder : nn.Module
    ):
        features = encoder(x)
        if self.use_cls_token:
            features = features[:, 1:]
        
        batch_indices = torch.arange(features.shape[0], device=features.device).unsqueeze(1).expand_as(masks)
        patch_indices = torch.arange(features.shape[1], device=features.device).unsqueeze(0).expand_as(masks)

        batch_idx = rearrange(batch_indices[masks], 'm -> m 1')
        patch_idx = rearrange(patch_indices[masks], 'm -> m 1')

        masked_features = features[batch_idx, patch_idx].squeeze(1)

        # Reshape to (B, M_per_batch, D)
        # Count number of masked patches per batch
        num_masked_per_batch = masks.sum(dim=1)  # (B,)
        max_masked = num_masked_per_batch.max().item()
        
        # Create output tensor
        batch_size = x.shape[0]
        embed_dim = features.shape[-1]
        output = torch.zeros(batch_size, max_masked, embed_dim, device=x.device)
        
        # Fill output tensor
        start_idx = 0
        for b in range(batch_size):
            num_masked = num_masked_per_batch[b].item()
            output[b, :num_masked] = masked_features[start_idx:start_idx + num_masked]
            start_idx += num_masked
            
        return output, num_masked_per_batch



    """

    a simple run down :

    1. IMAGE PATCHES
    +---+---+---+---+    Original image (4×4 grid example)
    | 0 | 1 | 2 | 3 |    Each number is a patch index
    +---+---+---+---+
    | 4 | 5 | 6 | 7 |
    +---+---+---+---+
    | 8 | 9 |10 |11 |
    +---+---+---+---+
    |12 |13 |14 |15 |
    +---+---+---+---+

    2. CONTEXT & TARGET MASKS
    +---+---+---+---+    +---+---+---+---+
    |   |   |   |   |    |   |   |   |   |
    +---+---+---+---+    +---+---+---+---+
    |   |[C]|[C]|   |    |   |   |   |   |
    +---+---+---+---+    +---+---+---+---+
    |   |[C]|[C]|   |    |   |   |[T]|[T]|
    +---+---+---+---+    +---+---+---+---+
    |   |   |   |   |    |   |   |[T]|[T]|
    +---+---+---+---+    +---+---+---+---+
    Context mask          Target mask
    (C = True)            (T = True)

    3. MASK TENSORS (Flattened)
    Context: [0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0]
    Target:  [0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1]

    4. INDICES EXPANSION
    Batch indices (batch_size=2):
    [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],   After unsqueeze(1)
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]]   and expand_as(masks)

    Patch indices:
    [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],   After unsqueeze(0)
        [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]   and expand_as(masks)

    5. MASKED EXTRACTION
    When masks are applied:
    Context mask selects: 
    - Batch 0: patches 5,6,9,10 (coordinates from 2D grid)
    - Batch 1: different patches (not shown)

    Gathering masked features:
    [features[0,5], features[0,6], features[0,9], features[0,10], features[1,x], ...]

    6. RESHAPE INTO BATCH STRUCTURE
    +----------------+----------------+
    | features[0,5]  | features[0,6]  |   Batch 0's
    +----------------+----------------+   context features
    | features[0,9]  | features[0,10] |
    +----------------+----------------+
    | features[1,x]  | features[1,y]  |   Batch 1's 
    +----------------+----------------+   context features
    | features[1,z]  | padding(zeros) |
    +----------------+----------------+

    7. PREDICTION PROCESS
                                +-----------------+
    Context Features         |                 |
    +--------------+  -----> |   Predictor    | -----> Predicted Target
    |              |         |     (MLP)      |        Features
    +--------------+         |                 |
                                +-----------------+
                                        |
                                        | Compare with
                                        v
                                +----------------+
                                | Target Features|
                                | from Target    |
                                | Encoder        |
                                +----------------+

    """

    def forward(
        self,
        x : torch.Tensor,
        batch_size : int,
        num_context_blocks : int,
        num_target_blocks : int,
        context_block_size : int,
        target_block_size : int
    ) -> Dict[str, torch.Tensor] :
        batch = x.shape[0]

        context_masks, target_masks = self.get_context_target_masks(
            batch_size=batch_size,
            num_context_blocks=num_context_blocks,
            num_target_blocks=num_target_blocks,
            context_block_size=context_block_size,
            target_block_size=target_block_size
        )

        context_features, _ = self.extract_masked_features(
            x=x,
            masks=context_masks,
            encoder=self.encoder,
        )

        with torch.no_grad():
            target_features, num_target_per_batch = self.extract_masked_features(
                x=x,
                masks=target_masks,
                encoder=self.target_encoder,
            )
        
        pred_target_features = self.predictor(
            context_features
        )

        return {
            "pred_target_features" : pred_target_features,
            "target_features" : target_features,
            "context_masks" : context_masks,
            "target_masks" : target_masks,
            "num_target_per_batch" : num_target_per_batch
        }
