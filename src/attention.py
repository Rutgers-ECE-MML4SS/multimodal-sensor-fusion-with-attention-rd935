"""
Attention Mechanisms for Multimodal Fusion

Implements:
1. CrossModalAttention: Attention between different modalities
2. TemporalAttention: Attention over time steps in sequences
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: Modality A attends to Modality B.
    
    Example: Video features attend to IMU features to incorporate
    relevant motion information at each timestep.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of query modality features
            key_dim: Dimension of key/value modality features  
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # TODO: Implement multi-head attention projections
        # Hint: Use nn.Linear for Q, K, V projections
        # Query from modality A, Key and Value from modality B
        
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** 0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            query: (batch_size, query_dim) - features from modality A
            key: (batch_size, key_dim) - features from modality B
            value: (batch_size, key_dim) - features from modality B
            mask: Optional (batch_size,) - binary mask for valid keys
            
        Returns:
            attended_features: (batch_size, hidden_dim) - query attended by key/value
            attention_weights: (batch_size, num_heads, 1, 1) - attention scores
        """
        batch_size = query.size(0)
        
        # TODO: Implement multi-head attention computation
        # Steps:
        #   1. Project query, key, value to (batch, num_heads, seq_len, head_dim)
        #   2. Compute attention scores: Q @ K^T / sqrt(head_dim)
        #   3. Apply mask if provided (set masked positions to -inf before softmax)
        #   4. Apply softmax to get attention weights
        #   5. Apply attention to values: attn_weights @ V
        #   6. Reshape and project back to hidden_dim

        q = self.q_proj(query).view(batch_size, self.num_heads, 1, self.head_dim)
        k = self.k_proj(key).view(batch_size, self.num_heads, 1, self.head_dim)
        v = self.v_proj(value).view(batch_size, self.num_heads, 1, self.head_dim)

        scores = (q * k).sum(-1, keepdim=True) / self.scale

        if mask is not None:
            # mask: (B,)
            mask = mask.view(batch_size, 1, 1, 1)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = attn * v  # (B, H, 1, D_head)
        context = context.reshape(batch_size, -1)  # (B, hidden_dim)
        out = self.out_proj(context)
        return out, attn  # attn for viz

class TemporalAttention(nn.Module):
    """
    Temporal attention: Attend over sequence of time steps.
    
    Useful for: Variable-length sequences, weighting important timesteps
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        # TODO: Implement self-attention over temporal dimension
        # Hint: Similar to CrossModalAttention but Q, K, V from same modality

        self.q_proj = nn.Linear(feature_dim, hidden_dim)
        self.k_proj = nn.Linear(feature_dim, hidden_dim)
        self.v_proj = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention.
        
        Args:
            sequence: (batch_size, seq_len, feature_dim) - temporal sequence
            mask: Optional (batch_size, seq_len) - binary mask for valid timesteps
            
        Returns:
            attended_sequence: (batch_size, seq_len, hidden_dim) - attended features
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # TODO: Implement temporal self-attention
        # Steps:
        #   1. Project sequence to Q, K, V
        #   2. Compute self-attention over sequence length
        #   3. Apply mask for variable-length sequences
        #   4. Return attended sequence and weights
        
        B, T, _ = sequence.shape

        scale = (self.head_dim ** 0.5)

        q = self.q_proj(sequence).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D)
        k = self.k_proj(sequence).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(sequence).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # (B, H, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,T)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)  # (B, H, T, D)
        context = context.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, hidden_dim)
        out = self.out_proj(context)
        return out, attn
    
    def pool_sequence(
        self,
        sequence: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation using attention weights.
        
        Args:
            sequence: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            pooled: (batch_size, hidden_dim) - fixed-size representation
        """
        # TODO: Implement attention-based pooling
        # Option 1: Weighted average using mean attention weights
        # Option 2: Learn pooling query vector
        # Option 3: Take output at special [CLS] token position

        # attention_weights: (B, H, T, T) -> we want weights over time for each query timestep.
        # simplest: average heads and take attention of the first timestep as global
        attn_mean = attention_weights.mean(1)  # (B, T, T)
        # take attention of CLS-like position (0), or average over queries
        global_weights = attn_mean.mean(1)  # (B, T)
        global_weights = torch.softmax(global_weights, dim=-1)  # (B, T)
        pooled = torch.bmm(global_weights.unsqueeze(1), sequence).squeeze(1)  # (B, D)
        return pooled

class PairwiseModalityAttention(nn.Module):
    """
    Pairwise attention between all modality combinations.
    
    For M modalities, computes M*(M-1)/2 pairwise attention operations.
    Example: {video, audio, IMU} -> {video<->audio, video<->IMU, audio<->IMU}
    """
    
    def __init__(
        self,
        modality_dims: dict,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dict mapping modality names to feature dimensions
                          Example: {'video': 512, 'audio': 128, 'imu': 64}
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # TODO: Create CrossModalAttention for each modality pair
        # Hint: Use nn.ModuleDict with keys like "video_to_audio"
        # For each pair (A, B), create attention A->B and B->A
        
        self.attn_layers = nn.ModuleDict()
        for i, mi in enumerate(self.modality_names):
            for j, mj in enumerate(self.modality_names):
                if i == j:
                    continue
                self.attn_layers[f"{mi}_to_{mj}"] = CrossModalAttention(
                    query_dim=modality_dims[mi],
                    key_dim=modality_dims[mj],
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
        
    def forward(
        self,
        modality_features: dict,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict]:
        """
        Apply pairwise attention between all modalities.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor: (batch_size, feature_dim)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            attended_features: Dict of {modality_name: attended_features}
            attention_maps: Dict of {f"{mod_a}_to_{mod_b}": attention_weights}
        """
        # TODO: Implement pairwise attention
        # Steps:
        #   1. For each modality pair (A, B):
        #      - Apply attention A->B (A attends to B)
        #      - Apply attention B->A (B attends to A)
        #   2. Aggregate attended features (options: sum, concat, gating)
        #   3. Handle missing modalities using mask
        #   4. Return attended features and attention maps for visualization
    
        B = next(iter(modality_features.values())).size(0)
        M = self.num_modalities

        # we'll collect all incoming attended vectors for each modality
        collected = {m: [] for m in self.modality_names}
        attention_maps = {}

        for i, mi in enumerate(self.modality_names):
            for j, mj in enumerate(self.modality_names):
                if i == j:
                    continue

                layer_key = f"{mi}_to_{mj}"
                attn_layer = self.attn_layers[layer_key]

                q = modality_features[mi]  # (B, D_mi)
                k = modality_features[mj]  # (B, D_mj)
                v = modality_features[mj]

                # if we have a mask, we only want to attend to mj when mj is present
                mask_ij = None
                if modality_mask is not None:
                    # modality_mask: (B, M)
                    # for this attention, we care if mj exists
                    mask_ij = modality_mask[:, j]  # (B,)

                attended_vec, attn_weights = attn_layer(q, k, v, mask_ij)
                # store attention map for visualization
                attention_maps[layer_key] = attn_weights  # (B, H, 1, 1) in our CrossModalAttention

                # collect this "view" of mi (mi reading from mj)
                collected[mi].append(attended_vec)

        # now aggregate per modality
        attended_features = {}
        for m in self.modality_names:
            if len(collected[m]) == 0:
                # no incoming attention (shouldn't happen for M>1), just keep original
                attended_features[m] = modality_features[m]
            else:
                # average all attended versions for this modality
                stacked = torch.stack(collected[m], dim=0)  # (num_sources, B, hidden_dim)
                fused = stacked.mean(dim=0)  # (B, hidden_dim)
                attended_features[m] = fused

        # if we have a mask, zero out missing modalities in the output
        if modality_mask is not None:
            for i, m in enumerate(self.modality_names):
                attended_features[m] = attended_features[m] * modality_mask[:, i].unsqueeze(-1)

        return attended_features, attention_maps


def visualize_attention(
    attention_weights: torch.Tensor,
    modality_names: list,
    save_path: str = None
) -> None:
    """
    Visualize attention weights between modalities.
    
    Args:
        attention_weights: (num_heads, num_queries, num_keys) or similar
        modality_names: List of modality names for labeling
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # TODO: Implement attention visualization
    # Create heatmap showing which modalities attend to which
    # Useful for understanding fusion behavior

    attn = attention_weights
    if isinstance(attn, torch.Tensor):
        attn = attn.detach().cpu()

    # 2) normalize shape
    # cases:
    # (B, H, Q, K) -> avg over B and H
    # (H, Q, K)     -> avg over H
    # (Q, K)        -> use as is
    if attn.ndim == 4:
        # (B, H, Q, K)
        attn = attn.mean(dim=0).mean(dim=0)  # -> (Q, K)
    elif attn.ndim == 3:
        # (H, Q, K)
        attn = attn.mean(dim=0)              # -> (Q, K)
    elif attn.ndim == 2:
        # already (Q, K)
        pass
    else:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    attn = attn.numpy()  # (Q, K)

    # 3) make sure labels match
    # if Q != len(modality_names) or K != len(modality_names),
    # we still plot, but trimming/padding could be added here
    fig, ax = plt.subplots(figsize=(4 + 0.4 * attn.shape[0],
                                    4 + 0.4 * attn.shape[1]))
    im = ax.imshow(attn, cmap="viridis")

    ax.set_xticks(np.arange(attn.shape[1]))
    ax.set_yticks(np.arange(attn.shape[0]))

    # if lengths mismatch, just use indices
    if len(modality_names) == attn.shape[1]:
        ax.set_xticklabels(modality_names, rotation=45, ha="right")
    else:
        ax.set_xticklabels([f"k{i}" for i in range(attn.shape[1])], rotation=45, ha="right")

    if len(modality_names) == attn.shape[0]:
        ax.set_yticklabels(modality_names)
    else:
        ax.set_yticklabels([f"q{i}" for i in range(attn.shape[0])])

    ax.set_xlabel("Key / attended modality")
    ax.set_ylabel("Query / attending modality")
    ax.set_title("Cross-Modal Attention")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        # show inline if running in a notebook/script
        plt.show()
    


if __name__ == '__main__':
    # Simple test
    print("Testing attention mechanisms...")
    
    batch_size = 4
    query_dim = 512  # e.g., video features
    key_dim = 64     # e.g., IMU features
    hidden_dim = 256
    num_heads = 4
    
    # Test CrossModalAttention
    print("\nTesting CrossModalAttention...")
    try:
        attn = CrossModalAttention(query_dim, key_dim, hidden_dim, num_heads)
        
        query = torch.randn(batch_size, query_dim)
        key = torch.randn(batch_size, key_dim)
        value = torch.randn(batch_size, key_dim)
        
        attended, weights = attn(query, key, value)
        
        assert attended.shape == (batch_size, hidden_dim)
        print(f"✓ CrossModalAttention working! Output shape: {attended.shape}")
        
    except NotImplementedError:
        print("✗ CrossModalAttention not implemented yet")
    except Exception as e:
        print(f"✗ CrossModalAttention error: {e}")
    
    # Test TemporalAttention
    print("\nTesting TemporalAttention...")
    try:
        seq_len = 10
        feature_dim = 128
        
        temporal_attn = TemporalAttention(feature_dim, hidden_dim, num_heads)
        sequence = torch.randn(batch_size, seq_len, feature_dim)
        
        attended_seq, weights = temporal_attn(sequence)
        
        assert attended_seq.shape == (batch_size, seq_len, hidden_dim)
        print(f"✓ TemporalAttention working! Output shape: {attended_seq.shape}")
        
    except NotImplementedError:
        print("✗ TemporalAttention not implemented yet")
    except Exception as e:
        print(f"✗ TemporalAttention error: {e}")

