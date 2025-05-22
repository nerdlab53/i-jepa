import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from einops import rearrange


class IJEPALoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "smoothl1",
        normalize: bool = True,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
        
        if loss_type == "smoothl1":
            self.loss_fn = nn.SmoothL1Loss(reduction="none")
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss(reduction="none")
        elif loss_type == "cosine":
            self.loss_fn = lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=-1)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        pred_target_feats: torch.Tensor,
        target_feats: torch.Tensor,
        num_target_per_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, max_targets, embed_dim = pred_target_feats.shape
        
        mask = torch.zeros(batch_size, max_targets, dtype=torch.bool, device=pred_target_feats.device)
        for b in range(batch_size):
            mask[b, :num_target_per_batch[b]] = True
        
        if self.normalize:
            pred_target_feats = F.normalize(pred_target_feats, dim=-1)
            target_feats = F.normalize(target_feats, dim=-1)
        
        if self.loss_type in ["smoothl1", "mse"]:
            loss_per_element = self.loss_fn(pred_target_feats, target_feats)  # (B, M, D)
            
            loss_per_patch = loss_per_element.mean(dim=-1)  # (B, M)
            
            loss_per_batch = (loss_per_patch * mask.float()).sum(dim=1) / num_target_per_batch.float()  # (B,)
            loss = loss_per_batch.mean()  # scalar
            
        elif self.loss_type == "cosine":
            loss_per_patch = self.loss_fn(pred_target_feats, target_feats)  # (B, M)
            
            loss_per_batch = (loss_per_patch * mask.float()).sum(dim=1) / num_target_per_batch.float()  # (B,)
            loss = loss_per_batch.mean()  # scalar
        
        with torch.no_grad():
            mse = ((pred_target_feats - target_feats) ** 2).mean(dim=-1)  # (B, M)
            mse = (mse * mask.float()).sum(dim=1) / num_target_per_batch.float()  # (B,)
            mse = mse.mean() 
            
            cos_sim = F.cosine_similarity(pred_target_feats, target_feats, dim=-1)  # (B, M)
            cos_sim = (cos_sim * mask.float()).sum(dim=1) / num_target_per_batch.float()  # (B,)
            cos_sim = cos_sim.mean()  # scalar
        
        loss_dict = {
            "loss": loss,
            "mse": mse,
            "cos_sim": cos_sim,
        }
        
        return loss, loss_dict 