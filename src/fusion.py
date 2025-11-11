"""
Multimodal Fusion Architectures for Sensor Integration

This module implements three fusion strategies:
1. Early Fusion: Concatenate features before processing
2. Late Fusion: Independent processing, combine predictions
3. Hybrid Fusion: Cross-modal attention + learned weighting
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from attention import CrossModalAttention


class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate encoder outputs and process jointly.
    
    Pros: Joint representation learning across modalities
    Cons: Requires temporal alignment, sensitive to missing modalities
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
                          Example: {'video': 512, 'imu': 64}
            hidden_dim: Hidden dimension for fusion network
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()

        # TODO: Implement early fusion architecture
        # Hint: Concatenate all modality features, pass through MLP
        # Architecture suggestion:

        self.modality_names = list(modality_dims.keys())
        self.modality_dims = modality_dims
        concat_dim = sum(modality_dims[m] for m in self.modality_names)

        self.net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
            
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with early fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor shape: (batch_size, feature_dim)
            modality_mask: Binary mask (batch_size, num_modalities)
                          1 = available, 0 = missing
                          
        Returns:
            logits: (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Extract features for each modality from dict
        #   2. Handle missing modalities (use zeros or learned embeddings)
        #   3. Concatenate all features
        #   4. Pass through fusion network
        
        B = next(iter(modality_features.values())).size(0)
        device = next(iter(modality_features.values())).device

        fused_feats = []
        if modality_mask is None:
            # simple concat
            for m in self.modality_names:
                fused_feats.append(modality_features[m])
        else:
            # mask out missing modalities with zeros
            for i, m in enumerate(self.modality_names):
                feat = modality_features[m]
                mask_col = modality_mask[:, i].view(B, 1).to(device)
                feat = feat * mask_col + (1 - mask_col) * torch.zeros_like(feat)
                fused_feats.append(feat)
        
        x = torch.cat(fused_feats, dim=-1)
        logits = self.net(x)
        return logits
    

class LateFusion(nn.Module):
    """
    Late fusion: Independent classifiers per modality, combine predictions.
    
    Pros: Handles asynchronous sensors, modular per-modality training
    Cons: Limited cross-modal interaction, fusion only at decision level
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for per-modality classifiers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_classes = num_classes
        
        # TODO: Create separate classifier for each modality
        # Hint: Use nn.ModuleDict to store per-modality classifiers
        # Each classifier: Linear(modality_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)
        
        # TODO: Learn fusion weights (how to combine predictions)
        # Option 1: Learnable weights (nn.Parameter)
        # Option 2: Attention over predictions
        # Option 3: Simple averaging

        self.heads = nn.ModuleDict()
        for m, d in modality_dims.items():
            self.heads[m] = nn.Sequential(
                nn.Linear(d, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        
        # optional learned global weights per modality
        self.modality_logits = nn.Parameter(torch.ones(len(self.modality_names)))

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ):
        """
        Forward pass with late fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            
        Returns:
            logits: (batch_size, num_classes) - fused predictions
            per_modality_logits: Dict of individual modality predictions
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Get predictions from each modality classifier
        #   2. Handle missing modalities (mask out or skip)
        #   3. Combine predictions using fusion weights
        #   4. Return both fused and per-modality predictions
        
        B = next(iter(modality_features.values())).size(0)
        device = next(iter(modality_features.values())).device

        per_mod_logits: Dict[str, torch.Tensor] = {}
        weighted = []
        weights = []

        for i, m in enumerate(self.modality_names):
            logits_m = self.heads[m](modality_features[m]) 
            per_mod_logits[m] = logits_m
            
            if modality_mask is not None:
                valid = modality_mask[:, i].float().view(B, 1).to(device)
            else:
                valid = torch.ones(B, 1, device=device)
            
            # learned global weight for modality m
            w_m = torch.relu(self.modality_logits[i])
            w_m = w_m * valid  
            
            weighted.append(logits_m * w_m)
            weights.append(w_m)
        
        sum_logits = torch.stack(weighted, dim=0).sum(dim=0)  
        sum_weights = torch.stack(weights, dim=0).sum(dim=0)  
        fused_logits = sum_logits / (sum_weights + 1e-6)

        return fused_logits, per_mod_logits
    

class HybridFusion(nn.Module):
    """
    Hybrid fusion: Cross-modal attention + learned fusion weights.
    
    Pros: Rich cross-modal interaction, robust to missing modalities
    Cons: More complex, higher computation cost
    
    This is the main focus of the assignment!
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for fusion
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # TODO: Project each modality to common hidden dimension
        # Hint: Use nn.ModuleDict with Linear layers per modality
        # project all modalities to common size
        self.proj = nn.ModuleDict(
            {
                m: nn.Linear(dim, hidden_dim)
                for m, dim in modality_dims.items()
            }
        )

        # TODO: Implement cross-modal attention
        # Use CrossModalAttention from attention.py
        # Each modality should attend to all other modalities
        # in __init__
        self.cross_attn = nn.ModuleDict()
        for m in self.modality_names:
            self.cross_attn[m] = nn.ModuleDict()
            for n in self.modality_names:
                if m == n:
                    continue
                self.cross_attn[m][n] = CrossModalAttention(
                    query_dim=hidden_dim,
                    key_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )

        # TODO: Learn adaptive fusion weights based on modality availability
        # Hint: Small MLP that takes modality mask and outputs weights
        self.weight_mlp = nn.Sequential(
            nn.Linear(self.num_modalities * 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_modalities),
        )


        # TODO: Final classifier
        # we fuse to a single (B, D) vector then classify
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        """
        Forward pass with hybrid fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            return_attention: If True, return attention weights for visualization
            
        Returns:
            logits: (batch_size, num_classes)
            attention_info: Optional dict with attention weights and fusion weights
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Project all modalities to common hidden dimension
        #   2. Apply cross-modal attention between modality pairs
        #   3. Compute adaptive fusion weights based on modality_mask
        #   4. Fuse attended representations with learned weights
        #   5. Pass through final classifier
        #   6. Optionally return attention weights for visualization

        device = next(self.parameters()).device
        B = next(iter(modality_features.values())).size(0)
        
        if modality_mask is None:
            modality_mask = torch.ones(B, self.num_modalities, device=device, dtype=torch.bool)

         # 1) project to common dim
        proj_feats: Dict[str, torch.Tensor] = {
            m: self.proj[m](modality_features[m]) for m in self.modality_names
        }

        # 2) cross-modal attention per modality
        attended: Dict[str, torch.Tensor] = {}
        raw_attn = {}
        
        for i, m in enumerate(self.modality_names):
            q = proj_feats[m]  # (B, D)
            incoming = []
            for j, n in enumerate(self.modality_names):
                if m == n:
                    continue
                attn_mod = self.cross_attn[m][n]
                out_mn, attn_mn = attn_mod(q, proj_feats[n], proj_feats[n],
                                           mask=modality_mask[:, j])
                incoming.append(out_mn)
                raw_attn[(m, n)] = attn_mn  
            
            if len(incoming) == 0:
                # no other modalities, keep itself
                attended[m] = q
            else:
                # average incoming and add residual to preserve self info
                cross_avg = torch.stack(incoming, dim=0).mean(dim=0)
                attended[m] = 0.5 * q + 0.5 * cross_avg

        # 3) adaptive weighting
        weights = self.compute_adaptive_weights(attended, modality_mask) 

        # 4) fuse to a single vector (weighted sum over modalities)
        stacked = torch.stack([attended[m] for m in self.modality_names], dim=1)  
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1) 

        # safety: kill NaNs/infs so they don't blow up loss
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4)

        # 5) classify
        logits = self.classifier(fused)  

        if return_attention:
            # build a simple (M, M) attention matrix averaged over batch & heads
            M = self.num_modalities
            attn_matrix = torch.zeros(M, M, device=device)
            name_to_idx = {m: i for i, m in enumerate(self.modality_names)}
            
            for (m, n), val in raw_attn.items():
                i = name_to_idx[m]
                j = name_to_idx[n]
                v = val.mean(dim=0).mean()  
                attn_matrix[i, j] = v
            
            for i in range(M):
                attn_matrix[i, i] = 1.0
            
            return logits, {
                "fusion_weights": weights,
                "attention_matrix": attn_matrix,
                "raw_attn": raw_attn,
            }
        
        return logits
        
    def compute_adaptive_weights(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive fusion weights based on modality availability.
        
        Args:
            modality_features: Dict of modality features
            modality_mask: (batch_size, num_modalities) binary mask
            
        Returns:
            weights: (batch_size, num_modalities) normalized fusion weights
        """
        # TODO: Implement adaptive weighting
        # Ideas:
        #   1. Learn weight predictor from modality features + mask
        #   2. Higher weights for more reliable/informative modalities
        #   3. Ensure weights sum to 1 (softmax) and respect mask
        
        # stack in fixed order
        feats = torch.stack(
            [modality_features[m] for m in self.modality_names], dim=1
        )  
        
        # summary = mean over feature dim
        summary = feats.mean(dim=-1) 
        
        # concat summary and mask
        x = torch.cat([summary, modality_mask.float()], dim=-1) 
        raw = self.weight_mlp(x)
        
        # mask out missing modalities BEFORE softmax
        very_neg = torch.finfo(raw.dtype).min + 1
        raw = raw.masked_fill(modality_mask == 0, very_neg)
        
        weights = torch.softmax(raw, dim=-1)  
        return weights

# Helper functions

def build_fusion_model(
    fusion_type: str,
    modality_dims: Dict[str, int],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to build fusion models.
    
    Args:
        fusion_type: One of ['early', 'late', 'hybrid']
        modality_dims: Dictionary mapping modality names to dimensions
        num_classes: Number of output classes
        **kwargs: Additional arguments for fusion model
        
    Returns:
        Fusion model instance
    """
    fusion_classes = {
        'early': EarlyFusion,
        'late': LateFusion,
        'hybrid': HybridFusion,
    }
    
    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion_classes[fusion_type](
        modality_dims=modality_dims,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Simple test to verify implementation
    print("Testing fusion architectures...")
    
    # Test configuration
    modality_dims = {'video': 512, 'imu': 64}
    num_classes = 11
    batch_size = 4
    
    # Create dummy features
    features = {
        'video': torch.randn(batch_size, 512),
        'imu': torch.randn(batch_size, 64)
    }
    mask = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]])  # Different availability patterns
    
    # Test each fusion type
    for fusion_type in ['early', 'late', 'hybrid']:
        print(f"\nTesting {fusion_type} fusion...")
        try:
            model = build_fusion_model(fusion_type, modality_dims, num_classes)
            
            if fusion_type == 'late':
                logits, per_mod_logits = model(features, mask)
            else:
                logits = model(features, mask)
            
            assert logits.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
            print(f"✓ {fusion_type} fusion working! Output shape: {logits.shape}")
            
        except NotImplementedError:
            print(f"✗ {fusion_type} fusion not implemented yet")
        except Exception as e:
            print(f"✗ {fusion_type} fusion error: {e}")

