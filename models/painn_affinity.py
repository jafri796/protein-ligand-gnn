"""
PaiNN-Based Binding Affinity Prediction Model

Complete architecture integrating:
1. Equivariant ligand encoder (PaiNN layers)
2. Protein encoder (GAT or simpler layers)
3. Interaction modeling (cross-attention)
4. Global pooling and readout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool, GATConv
from typing import Dict, Optional

from .layers.equivariant_layers import (
    PaiNNLayer,
    InteractionLayer,
    RBFExpansion
)


class PaiNNAffinityPredictor(nn.Module):
    """
    Main model for binding affinity prediction.
    
    Architecture:
    - Ligand: PaiNN equivariant layers (5 layers)
    - Protein: GAT layers (3 layers)
    - Interaction: Cross-attention layer
    - Readout: Global pooling + MLP → affinity (pKd)
    
    Args:
        config: Configuration dictionary with hyperparameters
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Hyperparameters
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_painn_layers = config.get('num_message_passing_layers', 5)
        self.num_protein_layers = config.get('num_protein_layers', 3)
        self.num_rbf = config.get('num_rbf', 20)
        self.cutoff = config.get('cutoff', 10.0)
        self.dropout = config.get('dropout', 0.1)
        
        # Input dimensions (from featurization)
        ligand_feat_dim = 49  # Atom features
        protein_feat_dim = 31  # Residue features
        
        # RBF expansion for distances
        self.rbf = RBFExpansion(num_rbf=self.num_rbf, cutoff=self.cutoff)
        
        # ================== LIGAND ENCODER ==================
        # Embedding
        self.ligand_embedding_s = nn.Linear(ligand_feat_dim, self.hidden_dim)
        # Vector features initialized to zero
        
        # PaiNN layers
        self.ligand_layers = nn.ModuleList([
            PaiNNLayer(self.hidden_dim, self.num_rbf)
            for _ in range(self.num_painn_layers)
        ])
        
        # ================== PROTEIN ENCODER ==================
        # Embedding
        self.protein_embedding = nn.Linear(protein_feat_dim, self.hidden_dim)
        
        # GAT layers (simpler than PaiNN for efficiency)
        self.protein_layers = nn.ModuleList([
            GATConv(
                self.hidden_dim,
                self.hidden_dim,
                heads=4,
                concat=False,
                dropout=self.dropout
            )
            for _ in range(self.num_protein_layers)
        ])
        
        # ================== INTERACTION LAYER ==================
        self.interaction_layer = InteractionLayer(
            self.hidden_dim,
            num_heads=4
        )
        
        # ================== READOUT ==================
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: PyG Data/Batch object with:
                - x: Node features (ligand + protein concatenated)
                - pos: 3D coordinates
                - edge_index: All edges (intra-ligand, intra-protein, inter)
                - edge_attr: Edge features
                - node_type: 0 for ligand, 1 for protein
                - batch: Batch assignment vector
                
        Returns:
            Predicted binding affinity (pKd)
        """
        # Separate ligand and protein nodes
        is_ligand = (batch.node_type == 0)
        is_protein = (batch.node_type == 1)
        
        ligand_idx = torch.where(is_ligand)[0]
        protein_idx = torch.where(is_protein)[0]
        
        # ================== LIGAND ENCODING ==================
        # Initialize scalar and vector features
        s_ligand = self.ligand_embedding_s(batch.x[is_ligand])
        v_ligand = torch.zeros(
            s_ligand.size(0), 3, self.hidden_dim,
            device=s_ligand.device
        )
        
        # Get ligand edges
        ligand_mask = is_ligand[batch.edge_index[0]] & is_ligand[batch.edge_index[1]]
        ligand_edge_index = batch.edge_index[:, ligand_mask]
        ligand_edge_attr = batch.edge_attr[ligand_mask]
        
        # Remap edge indices to local ligand indexing
        ligand_idx_map = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)
        ligand_idx_map[ligand_idx] = torch.arange(len(ligand_idx), device=batch.x.device)
        ligand_edge_index_local = ligand_idx_map[ligand_edge_index]
        
        # Compute edge vectors and RBF features
        row, col = ligand_edge_index_local
        edge_vec = batch.pos[ligand_edge_index[0]] - batch.pos[ligand_edge_index[1]]
        edge_dist = torch.norm(edge_vec, dim=1)
        edge_vec_norm = edge_vec / (edge_dist.unsqueeze(-1) + 1e-8)
        edge_rbf = self.rbf(edge_dist)
        
        # PaiNN message passing
        for layer in self.ligand_layers:
            s_ligand, v_ligand = layer(
                s_ligand, v_ligand,
                ligand_edge_index_local,
                edge_rbf,
                edge_vec_norm
            )
        
        # ================== PROTEIN ENCODING ==================
        h_protein = self.protein_embedding(batch.x[is_protein])
        
        # Get protein edges
        protein_mask = is_protein[batch.edge_index[0]] & is_protein[batch.edge_index[1]]
        protein_edge_index = batch.edge_index[:, protein_mask]
        
        # Remap edge indices
        protein_idx_map = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)
        protein_idx_map[protein_idx] = torch.arange(len(protein_idx), device=batch.x.device)
        protein_edge_index_local = protein_idx_map[protein_edge_index]
        
        # GAT message passing
        for layer in self.protein_layers:
            h_protein = layer(h_protein, protein_edge_index_local)
            h_protein = F.relu(h_protein)
        
        # ================== INTERACTION ==================
        # Get interaction edges
        inter_mask = (is_ligand[batch.edge_index[0]] & is_protein[batch.edge_index[1]]) | \
                     (is_protein[batch.edge_index[0]] & is_ligand[batch.edge_index[1]])
        inter_edge_index = batch.edge_index[:, inter_mask]
        inter_edge_attr = batch.edge_attr[inter_mask]
        
        if inter_edge_index.size(1) > 0:
            # Combine ligand and protein features
            # For ligand: use scalar features (invariant)
            all_features = torch.zeros(batch.num_nodes, self.hidden_dim, device=batch.x.device)
            all_features[ligand_idx] = s_ligand
            all_features[protein_idx] = h_protein
            
            # Apply interaction layer
            interaction_features = self.interaction_layer(
                all_features,
                inter_edge_index,
                inter_edge_attr
            )
            
            # Update features with interaction information
            s_ligand = s_ligand + interaction_features[ligand_idx]
            h_protein = h_protein + interaction_features[protein_idx]
        
        # ================== GLOBAL POOLING ==================
        # Map back to batch indexing
        ligand_batch = batch.batch[ligand_idx]
        protein_batch = batch.batch[protein_idx]
        
        # Pool ligand and protein separately
        ligand_global = global_mean_pool(s_ligand, ligand_batch)
        protein_global = global_mean_pool(h_protein, protein_batch)
        
        # Concatenate
        global_features = torch.cat([ligand_global, protein_global], dim=1)
        
        # ================== READOUT ==================
        affinity = self.readout_mlp(global_features)
        
        return affinity.squeeze(-1)
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    @staticmethod
    def from_config(config_path: str):
        """Load model from config file."""
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return PaiNNAffinityPredictor(config['model'])


if __name__ == "__main__":
    print("Testing PaiNN affinity model...")
    
    # Test configuration
    config = {
        'hidden_dim': 128,
        'num_message_passing_layers': 3,
        'num_protein_layers': 2,
        'num_rbf': 20,
        'cutoff': 10.0,
        'dropout': 0.1
    }
    
    model = PaiNNAffinityPredictor(config)
    print(f"✓ Model created with {model.get_num_params():,} parameters")
    
    # Create dummy batch
    from torch_geometric.data import Data, Batch
    
    # Create two dummy complexes
    data_list = []
    for _ in range(2):
        num_ligand = 10
        num_protein = 20
        num_nodes = num_ligand + num_protein
        
        x = torch.randn(num_nodes, 49)  # Placeholder features
        x[num_ligand:, :] = torch.randn(num_protein, 31)  # Different dim for protein
        x = torch.cat([
            torch.randn(num_ligand, 49),
            torch.randn(num_protein, 31)
        ], dim=0)
        # Pad to same dimension
        x = torch.cat([
            F.pad(torch.randn(num_ligand, 49), (0, 31-49)),
            torch.randn(num_protein, 31)
        ], dim=0)
        
        pos = torch.randn(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 40))
        edge_attr = torch.randn(40, 10)
        node_type = torch.cat([
            torch.zeros(num_ligand, dtype=torch.long),
            torch.ones(num_protein, dtype=torch.long)
        ])
        y = torch.tensor([7.5])
        
        data = Data(
            x=x, pos=pos, edge_index=edge_index,
            edge_attr=edge_attr, node_type=node_type, y=y
        )
        data_list.append(data)
    
    batch = Batch.from_data_list(data_list)
    
    # Test forward pass
    try:
        output = model(batch)
        print(f"✓ Forward pass successful: output shape {output.shape}")
        print(f"  Predicted affinities: {output.detach().numpy()}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✅ Model tests completed!")