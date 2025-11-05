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
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)

        # TODO: Implement early fusion architecture
        # Hint: Concatenate all modality features, pass through MLP
        # Architecture suggestion:
        concat_dim = sum(modality_dims.values())
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
        
        feats = []
        for idx, m in enumerate(self.modality_names):
            f = modality_features[m]
            if modality_mask is not None:
                mask_col = modality_mask[:, idx].view(-1, 1)
                f = f * mask_col  # zero missing
            feats.append(f)
        x = torch.cat(feats, dim=1)
        return self.net(x)
    

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
        self.num_modalities = len(self.modality_names)
        
        # TODO: Create separate classifier for each modality
        # Hint: Use nn.ModuleDict to store per-modality classifiers
        # Each classifier: Linear(modality_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)
        
        # TODO: Learn fusion weights (how to combine predictions)
        # Option 1: Learnable weights (nn.Parameter)
        # Option 2: Attention over predictions
        # Option 3: Simple averaging

        self.classifiers = nn.ModuleDict()
        for m, dim in modality_dims.items():
            self.classifiers[m] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        # global learnable weights (one per modality)
        self.fusion_logits = nn.Linear(self.num_modalities, self.num_modalities)
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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
        per_mod_logits = {}
        stacked = []
        for idx, m in enumerate(self.modality_names):
            logits_m = self.classifiers[m](modality_features[m])  # (B, C)
            per_mod_logits[m] = logits_m
            stacked.append(logits_m.unsqueeze(1))
        stacked = torch.cat(stacked, dim=1)  # (B, M, C)

        # compute weights
        base = torch.ones(B, self.num_modalities, device=stacked.device)
        if modality_mask is not None:
            base = modality_mask.float()
        w = self.fusion_logits(base)  # (B, M)
        # mask missing
        if modality_mask is not None:
            w = w.masked_fill(modality_mask == 0, -1e9)
        w = torch.softmax(w, dim=-1).unsqueeze(-1)  # (B, M, 1)

        fused = (stacked * w).sum(dim=1)  # (B, C)
        return fused, per_mod_logits
    

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
        self.projections = nn.ModuleDict()
        for m, dim in modality_dims.items():
            self.projections[m] = nn.Linear(dim, hidden_dim)

        # TODO: Implement cross-modal attention
        # Use CrossModalAttention from attention.py
        # Each modality should attend to all other modalities
        self.cross_attn = nn.ModuleDict()
        for m in self.modality_names:
            # each modality can be a query
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
        self.weight_net = nn.Sequential(
            nn.Linear(self.num_modalities, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.num_modalities),
        )

        # TODO: Final classifier
        # Takes fused representation -> num_classes logits
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
            
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
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
            modality_mask = torch.ones(B, self.num_modalities, device=device)

        # 1) project
        proj_feats = {}
        for i, m in enumerate(self.modality_names):
            f = self.projections[m](modality_features[m])  # (B, D)
            f = f * modality_mask[:, i].unsqueeze(-1)       # zero if missing
            proj_feats[m] = f

        # 2) cross-modal attention:
        # for each query modality m, let it attend to all other available modalities n
        attended = {}
        attn_maps = {}
        for i, m in enumerate(self.modality_names):
            q_feat = proj_feats[m]
            incoming = []
            for j, n in enumerate(self.modality_names):
                if m == n:
                    continue
                # if key modality is missing for this sample, we'll pass its mask
                mask_n = modality_mask[:, j]  # (B,)
                attn_layer = self.cross_attn[m][n]
                attended_mn, attn_w = attn_layer(q_feat, proj_feats[n], proj_feats[n], mask_n)
                incoming.append(attended_mn)
                attn_maps[f"{m}_to_{n}"] = attn_w
            if len(incoming) == 0:
                attended[m] = q_feat
            else:
                # average over attended views
                stacked = torch.stack(incoming, dim=0).mean(dim=0)  # (B, D)
                attended[m] = stacked

        # 3) fuse with adaptive weights
        weights = self.compute_adaptive_weights(modality_mask)  # (B, M)
        fused = 0
        for i, m in enumerate(self.modality_names):
            fused = fused + attended[m] * weights[:, i].unsqueeze(-1)

        # 4) classify
        logits = self.classifier(fused)

        if return_attention:
            return logits, {
                "fusion_weights": weights,
                "attn_maps": attn_maps,
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
        
        # modality_mask: (B, M)
        raw = self.weight_net(modality_mask.float())  # (B, M)
        # don't give weight to missing modalities
        raw = raw.masked_fill(modality_mask == 0, -1e9)
        weights = torch.softmax(raw, dim=1)           # (B, M)
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

