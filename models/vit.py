"""
Implementation of ViT
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional

## Code taken as it is
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
## Code taken as it is

# Patch Embedding
class PatchEmbed(nn.Module):

    def __init__(
            self,
            img_size : int = 224, 
            patch_size : int = 16,
            in_channels : int = 3,
            embed_dim : int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = self.img_size // self.patch_size

        self.projection = nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=self.embed_dim,
            kernel = self.patch_size,
            stride = self.patch_size,
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert h == w == self.img_size, f"Input image size '{h}x{w}' does not match '{self.img_size}x{self.img_size}'"
        x = rearrange(self.projection(x), 'b c h w -> b (hw) c')
        return x


# Attention Module
class Attention(nn.Module):

    def __init__(
            self, 
            dim : int = 768,
            num_heads : int = 8,
            qkv_bias : bool = True,
            attn_drop : float = 0.,
            proj_drop : float = 0.,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, qkv_bias = qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        qkv = rearrange(self.qkv(x), 'b n (three h d) -> three b h n d', three=3, h= self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = rearrange(attn @ v, 'b h n d -> b n (h d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
# MLP
class MLP(nn.Module):

    def __init__(
            self,
            in_features : int,
            hidden_features : Optional[int],
            out_features : Optional[int],
            activation_layer : nn.Module = nn.GELU(),
            drop : float = 0.,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = activation_layer()
        self.drop = nn.Dropout(p=drop)
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))
        return x
    

# Block 
class Block(nn.Module):
    def __init__(
            self,
            dim : int,
            num_heads : int,
            mlp_ratio : float = 4.,
            qkv_bias : bool = True,
            drop : float = 0.,
            drop_path : float = 0.,
            attn_drop : float = 0.,
            proj_drop : float = 0.,
            act_layer : nn.Module = nn.GELU(),
            norm_layer : nn.Module = nn.LayerNorm()
    ):
        super().__init__()
        self.norm1 = norm_layer()
        self.attn = Attention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )
        self.norm2 = norm_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = MLP(
            in_features=dim, hidden_features=int(dim * mlp_ratio), activation_layer=act_layer, drop=drop
        )
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisionTransformer(nn.Module):
    
    def __init__(
            self,
            img_size: int = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            use_cls_token: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.use_cls_token = use_cls_token
        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, in_channels=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1 if use_cls_token else 0, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # Define the blocks
        blocks = nn.ModuleList([
            Block(
                dim=embed_dim,num_heads=num_heads,mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,drop=drop_rate,attn_drop=attn_drop_rate,
                drop_path=dpr[i],norm_layer=norm_layer
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

    def initialize_weights(self):
        # Initialize patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # Initialize positional embedding with small random values
        nn.init.normal_(self.pos_embed, std=0.02)
        # Initialize class token
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=0.02)
        # Initialize all other linear layers and LayerNorms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def f(self, x : torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        return x
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.f(x)
        return x
    
    def get_grid_pos(self) -> torch.Tensor:
        grid_size = self.img_size // self.patch_size
        y = torch.arange(grid_size)
        x = torch.arange(grid_size)
        grid = torch.stack(torch.meshgrid(y, x, indexing='ij', dim=-1))
        return rearrange(grid, 'h w c -> (h w) c')