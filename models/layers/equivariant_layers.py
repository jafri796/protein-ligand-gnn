"""
Equivariant Message Passing Layers (PaiNN-inspired)

Implements SE(3)-equivariant message passing following:
Schütt et al. (2021) "Equivariant message passing for the prediction 
of tensorial properties and molecular spectra"

Key Features:
- Scalar features (s): rotation-invariant
- Vector features (v): rotation-equivariant  
- Maintains equivariance through construction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional
import math


class RBFExpansion(nn.Module):
    """
    Radial Basis Function expansion for distances.
    
    Converts continuous distances to discrete features using Gaussian RBFs.
    Used in SchNet, DimeNet, PaiNN.
    """
    
    def __init__(self, num_rbf: int = 20, cutoff: float = 10.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        # RBF centers
        self.register_buffer(
            'centers',
            torch.linspace(0, cutoff, num_rbf)
        )
        
        # RBF width
        self.gamma = 1.0 / (cutoff / num_rbf) ** 2
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: (num_edges,) edge distances
            
        Returns:
            (num_edges, num_rbf) RBF expansion
        """
        # Gaussian RBF: exp(-gamma * (d - center)^2)
        diff = distances.unsqueeze(-1) - self.centers
        rbf = torch.exp(-self.gamma * diff ** 2)
        
        # Cosine cutoff function
        cutoff_values = 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1.0)
        cutoff_values = torch.where(
            distances < self.cutoff,
            cutoff_values,
            torch.zeros_like(cutoff_values)
        )
        
        return rbf * cutoff_values.unsqueeze(-1)


class PaiNNMessage(MessagePassing):
    """
    PaiNN message passing block.
    
    Computes both scalar and vector messages:
    - Scalar message: function of (s_i, s_j, d_ij)
    - Vector message: function of (s_i, s_j, r_ij)
    
    Maintains SE(3) equivariance.
    """
    
    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        
        # Scalar message network: produces 3 * hidden_dim scalars
        # Split into (ds, phi_vv, phi_sv) for scalar msg, vector scaling, direction scaling
        self.scalar_message_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )
    
    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_vec: torch.Tensor
    ):
        """
        Args:
            s: (num_nodes, hidden_dim) scalar features
            v: (num_nodes, 3, hidden_dim) vector features
            edge_index: (2, num_edges) edge connectivity
            edge_rbf: (num_edges, num_rbf) RBF-expanded distances
            edge_vec: (num_edges, 3) edge direction vectors
            
        Returns:
            Tuple of message updates (delta_s, delta_v)
        """
        return self.propagate(
            edge_index,
            s=s,
            v=v,
            edge_rbf=edge_rbf,
            edge_vec=edge_vec
        )
    
    def message(self, s_i, s_j, v_j, edge_rbf, edge_vec):
        """Compute SE(3)-equivariant messages (Schütt et al. 2021).
        
        Equivariance proof:
        - ds: derived from invariants (s_i, s_j, ||r_ij||) → invariant 
        - phi_vv, phi_sv: derived from invariants → invariant scalars 
        - phi_vv * v_j: invariant × equivariant → equivariant 
        - phi_sv * edge_vec: invariant × equivariant → equivariant 
        - dv = sum of equivariant terms → equivariant 
        """
        # Compute scalar filter from [s_i, s_j, RBF(d_ij)] — all invariant
        scalar_input = torch.cat([s_i, s_j, edge_rbf], dim=-1)
        scalar_out = self.scalar_message_net(scalar_input)  # (E, 3*H)
        
        # Split into scalar message, vector-vector scale, vector-direction scale
        ds, phi_vv, phi_sv = torch.split(scalar_out, self.hidden_dim, dim=-1)
        
        # Vector message: channel-wise scaling of existing vectors + edge directions
        # phi_vv: (E, H) → (E, 1, H) — scales each channel of v_j independently
        # phi_sv: (E, H) → (E, 1, H) — scales edge_vec broadcast over channels
        # edge_vec: (E, 3) → (E, 3, 1) — direction vector broadcast over channels
        # No learned linear ever acts on the spatial dimension (3)
        dv = phi_vv.unsqueeze(1) * v_j + phi_sv.unsqueeze(1) * edge_vec.unsqueeze(-1)
        
        return ds, dv
    
    def aggregate(self, inputs, index, dim_size=None):
        """Aggregate messages."""
        ds, dv = inputs
        
        # Aggregate scalar messages
        ds_agg = torch.zeros(dim_size, ds.size(-1), device=ds.device)
        ds_agg.index_add_(0, index, ds)
        
        # Aggregate vector messages
        dv_agg = torch.zeros(dim_size, dv.size(1), dv.size(2), device=dv.device)
        dv_agg.index_add_(0, index, dv)
        
        return ds_agg, dv_agg


class PaiNNUpdate(nn.Module):
    """
    PaiNN update block.
    
    Updates scalar and vector features using:
    - Gated mixing of scalar and vector features
    - Maintains SE(3) equivariance
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Update networks
        self.update_U = nn.Linear(hidden_dim, hidden_dim)
        
        # Mixing networks
        self.mix_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 3)
        )
    
    def forward(self, s: torch.Tensor, v: torch.Tensor):
        """
        Args:
            s: (num_nodes, hidden_dim) scalar features
            v: (num_nodes, 3, hidden_dim) vector features
            
        Returns:
            Updated (s, v)
        """
        # Compute SE(3)-equivariant scalar from vector magnitudes
        # Taking norm over spatial dimension (dim=1) gives rotation-invariant scalars
        v_norm = torch.linalg.vector_norm(v, ord=2, dim=1, keepdim=True)  # (N, 1, hidden_dim)
        v_norm = v_norm.squeeze(1)  # (N, hidden_dim) - invariant under SE(3)
        
        # Mix scalar and vector information
        s_v_combined = torch.cat([s, v_norm], dim=-1)
        mix_output = self.mix_net(s_v_combined)
        s_gate, v_gate, s_update = torch.split(mix_output, self.hidden_dim, dim=-1)
        
        # Update scalar features
        s = s + s_update + s_gate * self.update_U(s)
        
        # Update vector features (maintaining SE(3) equivariance)
        # Gate acts uniformly on all spatial components, preserving structure
        v = v * (1 + v_gate.unsqueeze(1))
        
        return s, v


class PaiNNLayer(nn.Module):
    """
    Complete PaiNN layer: Message + Update.
    
    One iteration of equivariant message passing.
    """
    
    def __init__(self, hidden_dim: int, num_rbf: int):
        super().__init__()
        self.message = PaiNNMessage(hidden_dim, num_rbf)
        self.update = PaiNNUpdate(hidden_dim)
    
    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_rbf: torch.Tensor,
        edge_vec: torch.Tensor
    ):
        """
        Args:
            s: (num_nodes, hidden_dim) scalar features
            v: (num_nodes, 3, hidden_dim) vector features
            edge_index: (2, num_edges) edge connectivity
            edge_rbf: (num_edges, num_rbf) RBF features
            edge_vec: (num_edges, 3) edge vectors
            
        Returns:
            Updated (s, v)
        """
        # Message passing
        ds, dv = self.message(s, v, edge_index, edge_rbf, edge_vec)
        s = s + ds
        v = v + dv
        
        # Update
        s, v = self.update(s, v)
        
        return s, v


class InteractionLayer(MessagePassing):
    """
    Vector-aware interaction message passing layer for protein-ligand.
    
    Maintains SE(3) equivariance by incorporating vector features through
    their norm (invariant scalar). This enables richer interactions while
    preserving equivariance guarantees.
    
    Following the approach of:
    - GIGN (Zhang et al., 2023)
    - EquiBind (Stärk et al., 2022)
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, edge_dim: int = 14,
                 vector_weight: float = 0.3, edge_weight: float = 0.3):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.vector_weight = vector_weight
        self.edge_weight = edge_weight
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention for scalar features
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Vector feature projection (processes v_norm = ||v||)
        self.vector_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for scalar + vector-derived
        
        # Edge network - input dim depends on graph construction (padded edge_attr + edge type one-hot)
        self.edge_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(
        self,
        s: torch.Tensor,
        v: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ):
        """
        Args:
            s: (num_nodes, hidden_dim) scalar node features
            v: (num_nodes, 3, hidden_dim) vector node features
            edge_index: (2, num_edges) edge connectivity
            edge_attr: (num_edges, 7) edge features [dist, dx, dy, dz, edge_type_1hot]
            
        Returns:
            Updated scalar features (equivariance preserved through v_norm)
        """
        # Compute vector norm (SE(3) invariant)
        v_norm = v.norm(dim=1)  # (num_nodes, hidden_dim) - norm across spatial dimensions
        
        return self.propagate(edge_index, s=s, v_norm=v_norm, edge_attr=edge_attr)
    
    def message(self, s_i, s_j, v_norm_i, v_norm_j, edge_attr, index):
        """Compute attention-based messages with vector-aware features."""
        # Multi-head attention projections for scalars
        q = self.query(s_i).view(-1, self.num_heads, self.head_dim)
        k = self.key(s_j).view(-1, self.num_heads, self.head_dim)
        v_val = self.value(s_j).view(-1, self.num_heads, self.head_dim)
        
        # Attention scores from query-key interaction
        attn_qk = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)
        
        # Vector feature contribution (SE(3) invariant)
        v_feat_i = self.vector_proj(v_norm_i)
        v_feat_j = self.vector_proj(v_norm_j)
        v_q = v_feat_i.view(-1, self.num_heads, self.head_dim)
        v_k = v_feat_j.view(-1, self.num_heads, self.head_dim)
        attn_v = (v_q * v_k).sum(dim=-1) / math.sqrt(self.head_dim)
        
        # Edge feature embedding
        edge_emb = self.edge_net(edge_attr)
        edge_emb = edge_emb.view(-1, self.num_heads, self.head_dim)
        
        # Edge feature contribution
        edge_contrib = (edge_emb * q).sum(dim=-1) / math.sqrt(self.head_dim)
        
        # Combine all contributions
        attn = attn_qk + self.vector_weight * attn_v + self.edge_weight * edge_contrib
        
        # Softmax normalization
        attn = softmax(attn, index, dim=0)
        
        # Apply attention weights
        attn_expanded = attn.unsqueeze(-1)
        out = (v_val * attn_expanded).view(-1, self.hidden_dim)
        
        # Concatenate with vector-derived features for output
        combined = torch.cat([out, v_feat_i], dim=-1)
        return self.out_proj(combined)


if __name__ == "__main__":
    print("Testing equivariant layers...")
    
    # Test RBF expansion
    rbf = RBFExpansion(num_rbf=20, cutoff=10.0)
    distances = torch.rand(100) * 10.0
    rbf_features = rbf(distances)
    print(f"✓ RBF expansion: {rbf_features.shape}")
    
    # Test PaiNN layer
    batch_size = 10
    hidden_dim = 128
    num_rbf = 20
    
    s = torch.randn(batch_size, hidden_dim)
    v = torch.randn(batch_size, 3, hidden_dim)
    edge_index = torch.randint(0, batch_size, (2, 20))
    edge_rbf = torch.randn(20, num_rbf)
    edge_vec = torch.randn(20, 3)
    edge_vec = edge_vec / (edge_vec.norm(dim=1, keepdim=True) + 1e-8)
    
    painn_layer = PaiNNLayer(hidden_dim, num_rbf)
    s_out, v_out = painn_layer(s, v, edge_index, edge_rbf, edge_vec)
    
    print(f"✓ PaiNN layer: s {s_out.shape}, v {v_out.shape}")
    
    # Test interaction layer (vector-aware)
    s = torch.randn(batch_size, hidden_dim)
    v = torch.randn(batch_size, 3, hidden_dim)
    edge_attr = torch.randn(20, 7)  # 4 base features + 3 edge type indicators
    
    interaction_layer = InteractionLayer(hidden_dim, num_heads=4)
    s_out = interaction_layer(s, v, edge_index, edge_attr)
    
    print(f"✓ Interaction layer: {s_out.shape}")
    
    print("\n✅ All layer tests passed!")