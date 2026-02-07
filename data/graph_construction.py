"""
Graph Construction Module

Constructs PyTorch Geometric Data objects for:
1. Ligand graphs (covalent bonds + 3D geometry)
2. Protein graphs (residue-level with k-NN or distance cutoff)
3. Heterogeneous protein-ligand interaction graphs

Based on IGN/GIGN papers and PyG best practices.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from torch_geometric.nn import knn_graph, radius_graph
from typing import Tuple, Optional, Dict


def construct_ligand_graph(
    atom_features: np.ndarray,
    atom_coords: np.ndarray,
    bond_indices: np.ndarray,
    bond_features: np.ndarray,
    add_self_loops: bool = True
) -> Data:
    """
    Construct PyG Data object for ligand molecule.
    
    Args:
        atom_features: (num_atoms, feat_dim) atom feature matrix
        atom_coords: (num_atoms, 3) 3D coordinates
        bond_indices: (2, num_bonds) edge connectivity
        bond_features: (num_bonds, feat_dim) bond features
        add_self_loops: Whether to add self-loops
        
    Returns:
        PyG Data object with ligand graph
    """
    # Convert to torch tensors
    x = torch.from_numpy(atom_features).float()
    pos = torch.from_numpy(atom_coords).float()
    edge_index = torch.from_numpy(bond_indices).long()
    edge_attr = torch.from_numpy(bond_features).float()
    
    # Add self-loops if requested (useful for message passing)
    if add_self_loops:
        num_atoms = x.size(0)
        self_loop_index = torch.arange(num_atoms).unsqueeze(0).repeat(2, 1)
        # Create zero features for self-loops
        self_loop_attr = torch.zeros(num_atoms, edge_attr.size(1))
        
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)
        edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
    
    # Compute edge distances for message passing
    row, col = edge_index
    edge_vec = pos[row] - pos[col]  # Edge vectors
    edge_dist = torch.norm(edge_vec, dim=1, keepdim=True)  # Distances
    
    # Append distance to edge features
    edge_attr = torch.cat([edge_attr, edge_dist], dim=1)
    
    # Create Data object
    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.size(0)
    )
    
    return data


def construct_protein_graph(
    residue_features: np.ndarray,
    residue_coords: np.ndarray,
    method: str = 'knn',
    k: int = 10,
    cutoff: float = 10.0
) -> Data:
    """
    Construct PyG Data object for protein.
    
    Args:
        residue_features: (num_residues, feat_dim) residue features
        residue_coords: (num_residues, 3) Cα coordinates
        method: 'knn' or 'radius' for edge construction
        k: Number of nearest neighbors (for knn)
        cutoff: Distance cutoff in Angstroms (for radius)
        
    Returns:
        PyG Data object with protein graph
    """
    # Convert to torch tensors
    x = torch.from_numpy(residue_features).float()
    pos = torch.from_numpy(residue_coords).float()
    
    # Construct edges based on method
    if method == 'knn':
        # k-NN graph (bidirectional)
        edge_index = knn_graph(pos, k=k, loop=False)
    elif method == 'radius':
        # Radius graph (all pairs within cutoff)
        edge_index = radius_graph(pos, r=cutoff, loop=False)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'knn' or 'radius'.")
    
    # Compute edge features
    row, col = edge_index
    edge_vec = pos[row] - pos[col]
    edge_dist = torch.norm(edge_vec, dim=1, keepdim=True)
    
    # Edge attributes: distance + normalized direction
    edge_attr = torch.cat([
        edge_dist,
        edge_vec / (edge_dist + 1e-8)  # Normalized direction
    ], dim=1)
    
    # Create Data object
    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.size(0)
    )
    
    return data


def construct_interaction_edges(
    ligand_coords: torch.Tensor,
    protein_coords: torch.Tensor,
    cutoff: float = 5.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Construct protein-ligand interaction edges based on distance cutoff.
    
    Args:
        ligand_coords: (num_ligand_atoms, 3) ligand coordinates
        protein_coords: (num_protein_residues, 3) protein coordinates
        cutoff: Distance cutoff for interactions (Angstroms)
        
    Returns:
        Tuple of:
        - edge_index: (2, num_interactions) interaction edges
        - edge_attr: (num_interactions, 4) edge features [distance, direction]
    """
    # Compute pairwise distances
    # Shape: (num_ligand, num_protein)
    dist_matrix = torch.cdist(ligand_coords, protein_coords, p=2)
    
    # Find pairs within cutoff
    ligand_idx, protein_idx = torch.where(dist_matrix <= cutoff)
    
    if len(ligand_idx) == 0:
        # No interactions found - return empty tensors
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float)
        )
    
    # Compute edge features for forward direction (ligand → protein)
    ligand_pos = ligand_coords[ligand_idx]
    protein_pos = protein_coords[protein_idx]
    
    edge_vec_fwd = protein_pos - ligand_pos
    edge_dist = torch.norm(edge_vec_fwd, dim=1, keepdim=True)
    edge_dir_fwd = edge_vec_fwd / (edge_dist + 1e-8)
    
    # Reverse direction (protein → ligand): negate direction vectors
    edge_dir_rev = -edge_dir_fwd
    
    # Create bidirectional edges
    # Forward: ligand(local) → protein(local)
    # Reverse: protein(local) → ligand(local)
    edge_index = torch.stack([
        torch.cat([ligand_idx, protein_idx]),
        torch.cat([protein_idx, ligand_idx])
    ], dim=0)
    
    # Edge attributes with correct direction for each direction
    edge_attr_fwd = torch.cat([edge_dist, edge_dir_fwd], dim=1)
    edge_attr_rev = torch.cat([edge_dist, edge_dir_rev], dim=1)
    edge_attr = torch.cat([edge_attr_fwd, edge_attr_rev], dim=0)
    
    return edge_index, edge_attr


def construct_complex_graph(
    ligand_data: Data,
    protein_data: Data,
    interaction_cutoff: float = 5.0,
    use_heterogeneous: bool = False
) -> Data:
    """
    Construct complete protein-ligand complex graph.
    
    This creates a single graph with:
    - Ligand nodes (0 to n_lig-1)
    - Protein nodes (n_lig to n_lig+n_prot-1)
    - Intra-ligand edges
    - Intra-protein edges
    - Inter-molecular interaction edges
    
    Args:
        ligand_data: PyG Data object for ligand
        protein_data: PyG Data object for protein
        interaction_cutoff: Distance cutoff for interactions
        use_heterogeneous: If True, return HeteroData instead
        
    Returns:
        Combined PyG Data object (or HeteroData if requested)
    """
    if use_heterogeneous:
        return _construct_heterogeneous_complex(
            ligand_data, protein_data, interaction_cutoff
        )
    
    # Homogeneous graph construction
    num_ligand = ligand_data.num_nodes
    num_protein = protein_data.num_nodes
    
    # Pad node features to same dimension before concatenation
    ligand_feat_dim = ligand_data.x.size(1)
    protein_feat_dim = protein_data.x.size(1)
    max_feat_dim = max(ligand_feat_dim, protein_feat_dim)
    
    if ligand_feat_dim < max_feat_dim:
        ligand_x = F.pad(ligand_data.x, (0, max_feat_dim - ligand_feat_dim))
    else:
        ligand_x = ligand_data.x
    
    if protein_feat_dim < max_feat_dim:
        protein_x = F.pad(protein_data.x, (0, max_feat_dim - protein_feat_dim))
    else:
        protein_x = protein_data.x
    
    x = torch.cat([ligand_x, protein_x], dim=0)
    pos = torch.cat([ligand_data.pos, protein_data.pos], dim=0)
    
    # Adjust protein edge indices
    protein_edge_index = protein_data.edge_index + num_ligand
    
    # Construct interaction edges
    inter_edge_index, inter_edge_attr = construct_interaction_edges(
        ligand_data.pos,
        protein_data.pos,
        cutoff=interaction_cutoff
    )
    
    # Remap interaction edge indices from local to global index space.
    # construct_interaction_edges returns bidirectional edges:
    #   Forward half (first N_inter edges): src=ligand(local), dst=protein(local)
    #   Reverse half (last N_inter edges):  src=protein(local), dst=ligand(local)
    # We must add num_ligand ONLY to protein node indices.
    n_inter = inter_edge_index.size(1) // 2
    # Forward: row0 = ligand (no offset), row1 = protein (+ num_ligand)
    inter_edge_index[1, :n_inter] += num_ligand
    # Reverse: row0 = protein (+ num_ligand), row1 = ligand (no offset)
    inter_edge_index[0, n_inter:] += num_ligand
    
    # Combine all edges
    edge_index = torch.cat([
        ligand_data.edge_index,
        protein_edge_index,
        inter_edge_index
    ], dim=1)
    
    # Pad edge attributes to same dimension and add edge type indicators
    max_edge_dim = max(
        ligand_data.edge_attr.size(1),
        protein_data.edge_attr.size(1),
        inter_edge_attr.size(1)
    )
    
    def pad_edge_attr(attr, target_dim, edge_type_id):
        """Pad edge attributes and add edge type indicator.
        
        Args:
            attr: Edge attributes tensor
            target_dim: Target dimension after padding
            edge_type_id: 0=ligand, 1=protein, 2=interaction
        """
        num_edges = attr.size(0)
        
        # Pad features to target dimension
        if attr.size(1) < target_dim:
            padding = torch.zeros(num_edges, target_dim - attr.size(1))
            attr_padded = torch.cat([attr, padding], dim=1)
        else:
            attr_padded = attr
        
        # Add edge type one-hot indicator (3 types)
        edge_type = torch.zeros(num_edges, 3)
        edge_type[:, edge_type_id] = 1.0
        
        return torch.cat([attr_padded, edge_type], dim=1)
    
    # Concatenate with edge type indicators
    # Edge types: 0=ligand-ligand, 1=protein-protein, 2=ligand-protein interaction
    edge_attr_ligand = pad_edge_attr(ligand_data.edge_attr, max_edge_dim, 0)
    edge_attr_protein = pad_edge_attr(protein_data.edge_attr, max_edge_dim, 1)
    edge_attr_inter = pad_edge_attr(inter_edge_attr, max_edge_dim, 2)
    
    edge_attr = torch.cat([
        edge_attr_ligand,
        edge_attr_protein,
        edge_attr_inter
    ], dim=0)
    
    # Create batch indices for pooling
    ligand_batch = torch.zeros(num_ligand, dtype=torch.long)
    protein_batch = torch.zeros(num_protein, dtype=torch.long)
    batch = torch.cat([ligand_batch, protein_batch], dim=0)
    
    # Node type indicators (0: ligand, 1: protein)
    node_type = torch.cat([
        torch.zeros(num_ligand, dtype=torch.long),
        torch.ones(num_protein, dtype=torch.long)
    ], dim=0)
    
    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
        node_type=node_type,
        num_ligand_nodes=num_ligand,
        num_protein_nodes=num_protein
    )
    
    return data


def _construct_heterogeneous_complex(
    ligand_data: Data,
    protein_data: Data,
    interaction_cutoff: float = 5.0
) -> HeteroData:
    """
    Construct heterogeneous graph for complex.
    
    This explicitly separates ligand and protein as different node types,
    following PyG's HeteroData format.
    """
    data = HeteroData()
    
    # Add ligand nodes
    data['ligand'].x = ligand_data.x
    data['ligand'].pos = ligand_data.pos
    data['ligand', 'bonded_to', 'ligand'].edge_index = ligand_data.edge_index
    data['ligand', 'bonded_to', 'ligand'].edge_attr = ligand_data.edge_attr
    
    # Add protein nodes
    data['protein'].x = protein_data.x
    data['protein'].pos = protein_data.pos
    data['protein', 'connected_to', 'protein'].edge_index = protein_data.edge_index
    data['protein', 'connected_to', 'protein'].edge_attr = protein_data.edge_attr
    
    # Add interaction edges
    inter_edge_index, inter_edge_attr = construct_interaction_edges(
        ligand_data.pos,
        protein_data.pos,
        cutoff=interaction_cutoff
    )
    
    if inter_edge_index.size(1) > 0:
        # construct_interaction_edges returns bidirectional edges:
        #   First half: ligand(local) → protein(local)
        #   Second half: protein(local) → ligand(local)
        n_inter = inter_edge_index.size(1) // 2
        
        # Ligand to protein (forward half)
        data['ligand', 'interacts_with', 'protein'].edge_index = inter_edge_index[:, :n_inter]
        data['ligand', 'interacts_with', 'protein'].edge_attr = inter_edge_attr[:n_inter]
        
        # Protein to ligand (reverse half)
        data['protein', 'interacts_with', 'ligand'].edge_index = inter_edge_index[:, n_inter:]
        data['protein', 'interacts_with', 'ligand'].edge_attr = inter_edge_attr[n_inter:]
    
    return data


def radial_basis_expansion(
    distances: torch.Tensor,
    num_rbf: int = 20,
    cutoff: float = 10.0
) -> torch.Tensor:
    """
    Expand distances using radial basis functions (Gaussian).
    
    Used in SchNet, DimeNet, PaiNN for continuous distance representation.
    This is a standalone function for convenience. For model training,
    use RBFExpansion class from models.layers.equivariant_layers for consistency.
    
    Args:
        distances: (num_edges,) or (num_edges, 1) distance values
        num_rbf: Number of radial basis functions
        cutoff: Cutoff distance
        
    Returns:
        (num_edges, num_rbf) expanded features
    """
    distances = distances.squeeze(-1) if distances.dim() > 1 else distances
    
    # Gaussian RBF centers
    centers = torch.linspace(0, cutoff, num_rbf, device=distances.device)
    gamma = 1.0 / (cutoff / num_rbf) ** 2
    
    # RBF expansion: exp(-gamma * (d - center)^2)
    diff = distances.unsqueeze(-1) - centers.unsqueeze(0)
    rbf = torch.exp(-gamma * diff ** 2)
    
    # Cosine cutoff function to smoothly go to 0 at cutoff distance
    cutoff_values = 0.5 * (torch.cos(np.pi * distances / cutoff) + 1.0)
    cutoff_values = torch.where(
        distances < cutoff,
        cutoff_values,
        torch.zeros_like(cutoff_values)
    )
    
    # Apply cutoff to RBF features
    rbf = rbf * cutoff_values.unsqueeze(-1)
    
    return rbf


if __name__ == "__main__":
    # Test graph construction
    print("Testing graph construction module...")
    
    # Create dummy ligand
    num_atoms = 10
    atom_features = np.random.randn(num_atoms, 49).astype(np.float32)
    atom_coords = np.random.randn(num_atoms, 3).astype(np.float32)
    bond_indices = np.array([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64)
    bond_features = np.random.randn(4, 9).astype(np.float32)
    
    ligand_graph = construct_ligand_graph(
        atom_features, atom_coords, bond_indices, bond_features
    )
    print(f"✓ Ligand graph: {ligand_graph}")
    
    # Create dummy protein
    num_residues = 50
    residue_features = np.random.randn(num_residues, 31).astype(np.float32)
    residue_coords = np.random.randn(num_residues, 3).astype(np.float32)
    
    protein_graph = construct_protein_graph(
        residue_features, residue_coords, method='knn', k=10
    )
    print(f"✓ Protein graph: {protein_graph}")
    
    # Create complex graph
    complex_graph = construct_complex_graph(
        ligand_graph, protein_graph, interaction_cutoff=5.0
    )
    print(f"✓ Complex graph: {complex_graph}")
    
    # Test heterogeneous version
    hetero_graph = construct_complex_graph(
        ligand_graph, protein_graph, use_heterogeneous=True
    )
    print(f"✓ Heterogeneous graph: {hetero_graph}")
    
    print("\n✅ All tests passed!")