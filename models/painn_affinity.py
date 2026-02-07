"""
PaiNN-Based Binding Affinity Prediction Model

Complete SE(3)-equivariant architecture:
1. Equivariant ligand encoder (PaiNN layers)
2. Equivariant protein encoder (PaiNN layers) 
3. Interaction modeling (cross-attention)
4. Global pooling and readout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from typing import Dict, Optional

from .layers.equivariant_layers import (
    PaiNNLayer,
    InteractionLayer,
    RBFExpansion
)
from data.featurization import (
    LIGAND_ATOM_FEATURE_DIM,
    LIGAND_BOND_FEATURE_DIM,
    PROTEIN_RESIDUE_FEATURE_DIM
)


class PaiNNAffinityPredictor(nn.Module):
    """
    SE(3)-equivariant binding affinity prediction model.
    
    Architecture:
    - Ligand: PaiNN equivariant layers (scalar + vector features)
    - Protein: PaiNN equivariant layers (scalar + vector features)
    - Interaction: Cross-attention layer between ligand and protein
    - Readout: Global pooling + MLP → affinity (pKd)
    
    Both ligand and protein encoders are fully SE(3)-equivariant, ensuring
    the model respects rotational and translational symmetries.
    
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
        # construct_complex_graph pads both to max(ligand, protein) dim
        ligand_feat_dim = LIGAND_ATOM_FEATURE_DIM  # 49
        protein_feat_dim = PROTEIN_RESIDUE_FEATURE_DIM  # 31
        padded_feat_dim = max(ligand_feat_dim, protein_feat_dim)
        
        # RBF expansion for distances
        self.rbf = RBFExpansion(num_rbf=self.num_rbf, cutoff=self.cutoff)
        
        # ================== LIGAND ENCODER ==================
        # Embedding (input is padded_feat_dim after graph construction)
        self.ligand_embedding_s = nn.Linear(padded_feat_dim, self.hidden_dim)
        # Vector features initialized to zero
        
        # PaiNN layers
        self.ligand_layers = nn.ModuleList([
            PaiNNLayer(self.hidden_dim, self.num_rbf)
            for _ in range(self.num_painn_layers)
        ])
        
        # ================== PROTEIN ENCODER ==================
        # Embedding (input is padded_feat_dim after graph construction)
        self.protein_embedding_s = nn.Linear(padded_feat_dim, self.hidden_dim)
        # Vector features initialized to zero (will be learned)
        
        # PaiNN layers for protein (SE(3)-equivariant)
        self.protein_layers = nn.ModuleList([
            PaiNNLayer(self.hidden_dim, self.num_rbf)
            for _ in range(self.num_protein_layers)
        ])
        
        # ================== INTERACTION LAYER ==================
        # edge_attr dim from construct_complex_graph:
        #   max(ligand_bond_feat+1_dist, protein_4, inter_4) + 3_edge_type_onehot
        interaction_edge_dim = LIGAND_BOND_FEATURE_DIM + 1 + 3  # bond_feats + dist + edge_type
        self.interaction_layer = InteractionLayer(
            self.hidden_dim,
            num_heads=4,
            edge_dim=interaction_edge_dim
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
        # Initialize scalar and vector features for protein
        s_protein = self.protein_embedding_s(batch.x[is_protein])
        v_protein = torch.zeros(
            s_protein.size(0), 3, self.hidden_dim,
            device=s_protein.device
        )
        
        # Get protein edges
        protein_mask = is_protein[batch.edge_index[0]] & is_protein[batch.edge_index[1]]
        protein_edge_index = batch.edge_index[:, protein_mask]
        
        # Remap edge indices
        protein_idx_map = torch.zeros(batch.num_nodes, dtype=torch.long, device=batch.x.device)
        protein_idx_map[protein_idx] = torch.arange(len(protein_idx), device=batch.x.device)
        protein_edge_index_local = protein_idx_map[protein_edge_index]
        
        # Compute edge vectors and RBF for protein
        if protein_edge_index.size(1) > 0:
            row, col = protein_edge_index_local
            prot_edge_vec = batch.pos[protein_edge_index[0]] - batch.pos[protein_edge_index[1]]
            prot_edge_dist = torch.norm(prot_edge_vec, dim=1)
            prot_edge_vec_norm = prot_edge_vec / (prot_edge_dist.unsqueeze(-1) + 1e-8)
            prot_edge_rbf = self.rbf(prot_edge_dist)
            
            # PaiNN message passing for protein
            for layer in self.protein_layers:
                s_protein, v_protein = layer(
                    s_protein, v_protein,
                    protein_edge_index_local,
                    prot_edge_rbf,
                    prot_edge_vec_norm
                )
        
        # Use scalar features for downstream tasks
        h_protein = s_protein
        
        # ================== INTERACTION ==================
        # Get interaction edges
        inter_mask = (is_ligand[batch.edge_index[0]] & is_protein[batch.edge_index[1]]) | \
                     (is_protein[batch.edge_index[0]] & is_ligand[batch.edge_index[1]])
        inter_edge_index = batch.edge_index[:, inter_mask]
        inter_edge_attr = batch.edge_attr[inter_mask]
        
        if inter_edge_index.size(1) > 0:
            # Combine ligand and protein features for vector-aware interaction
            all_s = torch.zeros(batch.num_nodes, self.hidden_dim, device=batch.x.device)
            all_s[ligand_idx] = s_ligand
            all_s[protein_idx] = h_protein
            
            # Combine vector features
            all_v = torch.zeros(batch.num_nodes, 3, self.hidden_dim, device=batch.x.device)
            all_v[ligand_idx] = v_ligand
            all_v[protein_idx] = v_protein
            
            # Apply vector-aware interaction layer
            interaction_s = self.interaction_layer(
                all_s,
                all_v,
                inter_edge_index,
                inter_edge_attr
            )
            
            # Update features with interaction information
            s_ligand = s_ligand + interaction_s[ligand_idx]
            h_protein = h_protein + interaction_s[protein_idx]
        
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
        """Load model from config file.
        
        Args:
            config_path: Path to YAML config file
            
        Returns:
            PaiNNAffinityPredictor instance configured from file
        """
        import yaml
        from pathlib import Path
        
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
        
        # Extract model config section
        if 'model' in full_config:
            model_config = full_config['model']
        else:
            # Assume entire config is model config
            model_config = full_config
        
        # Create model
        model = PaiNNAffinityPredictor(model_config)
        
        return model


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
    
    # Create dummy batch with PROPER feature dimensions
    from torch_geometric.data import Data, Batch
    
    # Create two dummy complexes
    data_list = []
    for _ in range(2):
        num_ligand = 10
        num_protein = 20
        num_nodes = num_ligand + num_protein
        
        # Create features with correct dimensions
        x_ligand = torch.randn(num_ligand, 49)  # Placeholder features
        x_protein = torch.randn(num_protein, 31)  # Different dim for protein
        x = torch.cat([x_ligand, x_protein], dim=0)
        
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