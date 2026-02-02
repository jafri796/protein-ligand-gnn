"""
Data processing module for protein-ligand binding affinity prediction.

Provides featurization, graph construction, dataset loading, and data splitting utilities.
"""

from .featurization import (
    get_atom_features,
    get_bond_features,
    get_residue_features,
    featurize_ligand,
    featurize_complex,
    identify_binding_pocket,
)
from .graph_construction import (
    construct_ligand_graph,
    construct_protein_graph,
    construct_complex_graph,
    construct_interaction_edges,
)
from .dataset import ProteinLigandDataset, InMemoryProteinLigandDataset
from .splits import (
    create_lp_pdbbind_splits,
    create_external_test_set,
    compute_ligand_similarity,
    compute_sequence_similarity,
)

__all__ = [
    # Featurization
    'get_atom_features',
    'get_bond_features', 
    'get_residue_features',
    'featurize_ligand',
    'featurize_complex',
    'identify_binding_pocket',
    # Graph construction
    'construct_ligand_graph',
    'construct_protein_graph',
    'construct_complex_graph',
    'construct_interaction_edges',
    # Dataset
    'ProteinLigandDataset',
    'InMemoryProteinLigandDataset',
    # Splits
    'create_lp_pdbbind_splits',
    'create_external_test_set',
    'compute_ligand_similarity',
    'compute_sequence_similarity',
]
