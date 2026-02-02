"""
Molecular Featurization Module

Implements scientifically justified feature extraction for:
- Ligands: Atomic features following GraphDTA, IGN best practices
- Proteins: Residue-level features with secondary structure
- 3D geometry: Coordinates, distances, angles

All features grounded in chemistry/biology literature.
"""

from typing import Tuple, Dict, List, Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from Bio.PDB import PDBParser, PPBuilder, DSSP
import warnings

# Suppress specific RDKit and BioPython warnings that are noisy but not actionable
warnings.filterwarnings('ignore', category=DeprecationWarning, module='rdkit')
warnings.filterwarnings('ignore', message='.*PDBConstructionWarning.*')


# =============================================================================
# LIGAND FEATURIZATION
# =============================================================================

def get_atom_features(atom: Chem.Atom) -> np.ndarray:
    """
    Extract atom features following GraphDTA and IGN conventions.
    
    Features (79-dimensional):
    - Atomic number (1-hot, 44 atoms: H, C, N, O, F, P, S, Cl, Br, I, + others)
    - Degree (1-hot, 0-5)
    - Formal charge (1-hot, -2 to +2)
    - Hybridization (1-hot, SP, SP2, SP3, SP3D, SP3D2)
    - Aromaticity (binary)
    - Number of hydrogen atoms (1-hot, 0-4)
    - Chirality (1-hot, R, S, unspecified)
    - Is in ring (binary)
    
    Justification:
    - Duvenaud et al. (2015) Neural Fingerprints
    - GraphDTA, IGN papers use similar featurization
    - Captures electronic, geometric, and topological properties
    
    Args:
        atom: RDKit atom object
        
    Returns:
        Feature vector (79-dim)
    """
    # Atomic number (1-hot encoding of common atoms)
    allowable_atoms = [
        'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I',
        'B', 'Si', 'Se', 'As', 'Al', 'Mg', 'Ca', 'Fe', 'Zn'
    ]
    atom_symbol = atom.GetSymbol()
    atom_encoding = [int(atom_symbol == a) for a in allowable_atoms]
    atom_encoding.append(int(atom_symbol not in allowable_atoms))  # "Other"
    
    # Degree (0-5, then 6+)
    degree = atom.GetDegree()
    degree_encoding = [int(degree == i) for i in range(6)]
    degree_encoding.append(int(degree >= 6))
    
    # Formal charge (-2 to +2, then other)
    formal_charge = atom.GetFormalCharge()
    charge_encoding = [int(formal_charge == i) for i in range(-2, 3)]
    charge_encoding.append(int(formal_charge < -2 or formal_charge > 2))
    
    # Hybridization
    hybridization = atom.GetHybridization()
    hybrid_encoding = [
        int(hybridization == Chem.HybridizationType.SP),
        int(hybridization == Chem.HybridizationType.SP2),
        int(hybridization == Chem.HybridizationType.SP3),
        int(hybridization == Chem.HybridizationType.SP3D),
        int(hybridization == Chem.HybridizationType.SP3D2),
    ]
    
    # Aromaticity
    is_aromatic = [int(atom.GetIsAromatic())]
    
    # Number of hydrogen atoms (0-4, then 5+)
    num_hs = atom.GetTotalNumHs()
    hs_encoding = [int(num_hs == i) for i in range(5)]
    hs_encoding.append(int(num_hs >= 5))
    
    # Chirality (R, S, unspecified)
    chirality = atom.GetChiralTag()
    chiral_encoding = [
        int(chirality == Chem.ChiralType.CHI_TETRAHEDRAL_CW),   # R
        int(chirality == Chem.ChiralType.CHI_TETRAHEDRAL_CCW),  # S
        int(chirality == Chem.ChiralType.CHI_UNSPECIFIED),
    ]
    
    # Is in ring
    is_in_ring = [int(atom.IsInRing())]
    
    # Concatenate all features
    features = (
        atom_encoding +         # 20
        degree_encoding +       # 7
        charge_encoding +       # 6
        hybrid_encoding +       # 5
        is_aromatic +          # 1
        hs_encoding +          # 6
        chiral_encoding +      # 3
        is_in_ring             # 1
    )  # Total: 49 dimensions
    
    return np.array(features, dtype=np.float32)


def get_dihedral_angle(mol: Chem.Mol, bond: Chem.Bond) -> float:
    """
    Compute dihedral (torsion) angle for a bond.
    
    For rotatable bonds, computes the dihedral angle between 
    the four atoms: atom_i - atom_j - atom_k - atom_l
    where bond connects atom_j and atom_k.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        bond: Bond to compute dihedral for
        
    Returns:
        Dihedral angle in radians [-π, π]
    """
    try:
        if mol.GetNumConformers() == 0:
            return 0.0
        
        conf = mol.GetConformer()
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        # Get neighbors of begin atom
        begin_neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(begin_idx).GetNeighbors() 
                          if a.GetIdx() != end_idx]
        
        # Get neighbors of end atom
        end_neighbors = [a.GetIdx() for a in mol.GetAtomWithIdx(end_idx).GetNeighbors() 
                        if a.GetIdx() != begin_idx]
        
        # Need at least one neighbor on each side
        if not begin_neighbors or not end_neighbors:
            return 0.0
        
        # Use first neighbor on each side
        i = begin_neighbors[0]
        j = begin_idx
        k = end_idx
        l = end_neighbors[0]
        
        # Get coordinates
        pos_i = conf.GetAtomPosition(i)
        pos_j = conf.GetAtomPosition(j)
        pos_k = conf.GetAtomPosition(k)
        pos_l = conf.GetAtomPosition(l)
        
        # Compute vectors
        b1 = np.array([pos_j.x - pos_i.x, pos_j.y - pos_i.y, pos_j.z - pos_i.z])
        b2 = np.array([pos_k.x - pos_j.x, pos_k.y - pos_j.y, pos_k.z - pos_j.z])
        b3 = np.array([pos_l.x - pos_k.x, pos_l.y - pos_k.y, pos_l.z - pos_k.z])
        
        # Compute normal vectors
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # Normalize
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-6 or n2_norm < 1e-6:
            return 0.0
        
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # Compute dihedral angle
        cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Determine sign
        sign = np.sign(np.dot(np.cross(n1, n2), b2 / np.linalg.norm(b2)))
        
        return sign * angle
    
    except Exception:
        return 0.0


def get_bond_features(bond: Chem.Bond, mol: Chem.Mol = None) -> np.ndarray:
    """
    Extract bond features.
    
    Features (13-dimensional):
    - Bond type (1-hot: single, double, triple, aromatic) - 4 dims
    - Conjugation (binary) - 1 dim
    - Is in ring (binary) - 1 dim
    - Stereochemistry (1-hot: E, Z, none) - 3 dims
    - Rotatable (binary) - 1 dim
    - Dihedral angle (sine and cosine) - 2 dims
    
    Args:
        bond: RDKit bond object
        mol: RDKit molecule object (optional, for dihedral computation)
        
    Returns:
        Feature vector (13-dim with 3D info, or 9-dim without)
    """
    # Bond type
    bond_type = bond.GetBondType()
    bond_type_encoding = [
        int(bond_type == Chem.BondType.SINGLE),
        int(bond_type == Chem.BondType.DOUBLE),
        int(bond_type == Chem.BondType.TRIPLE),
        int(bond_type == Chem.BondType.AROMATIC),
    ]
    
    # Conjugation
    is_conjugated = [int(bond.GetIsConjugated())]
    
    # Is in ring
    is_in_ring = [int(bond.IsInRing())]
    
    # Stereochemistry
    stereo = bond.GetStereo()
    stereo_encoding = [
        int(stereo == Chem.BondStereo.STEREOE),
        int(stereo == Chem.BondStereo.STEREOZ),
        int(stereo == Chem.BondStereo.STEREONONE),
    ]
    
    # Rotatable bond (single, not in ring)
    rotatable = [int(bond_type == Chem.BondType.SINGLE and not bond.IsInRing())]
    
    features = (
        bond_type_encoding +   # 4
        is_conjugated +        # 1
        is_in_ring +           # 1
        stereo_encoding +      # 3
        rotatable             # 1
        # Note: 3D distance added separately in graph construction
    )  # Total: 10 dimensions (+ 1 distance = 11, + 2 dihedral = 13)
    
    # Add 3D dihedral features if molecule is provided with 3D coordinates
    if mol is not None and mol.GetNumConformers() > 0:
        dihedral = get_dihedral_angle(mol, bond)
        features = np.append(features, [np.sin(dihedral), np.cos(dihedral)])
    
    return np.array(features, dtype=np.float32)


def featurize_ligand(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Featurize a ligand molecule.
    
    Args:
        mol: RDKit molecule with 3D coordinates
        
    Returns:
        Tuple of:
        - atom_features: (num_atoms, 49) array
        - atom_coords: (num_atoms, 3) array of 3D coordinates
        - bond_indices: (2, num_bonds) array of edge indices
        - bond_features: (num_bonds, 9) array
    """
    # Ensure molecule has 3D coordinates
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
    
    # Atom features and coordinates
    num_atoms = mol.GetNumAtoms()
    atom_features = np.zeros((num_atoms, 49), dtype=np.float32)
    atom_coords = np.zeros((num_atoms, 3), dtype=np.float32)
    
    for i, atom in enumerate(mol.GetAtoms()):
        atom_features[i] = get_atom_features(atom)
        pos = conf.GetAtomPosition(i)
        atom_coords[i] = [pos.x, pos.y, pos.z]
    
    # Bond features and indices
    num_bonds = mol.GetNumBonds()
    bond_indices = np.zeros((2, num_bonds * 2), dtype=np.int64)  # Bidirectional
    
    # First pass: determine feature dimension (includes dihedral if 3D coords available)
    first_bond = list(mol.GetBonds())[0]
    sample_feat = get_bond_features(first_bond, mol)
    bond_feat_dim = len(sample_feat)
    bond_features = np.zeros((num_bonds * 2, bond_feat_dim), dtype=np.float32)
    
    for i, bond in enumerate(mol.GetBonds()):
        start_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        
        # Add both directions (undirected graph)
        bond_indices[0, 2*i] = start_idx
        bond_indices[1, 2*i] = end_idx
        bond_indices[0, 2*i+1] = end_idx
        bond_indices[1, 2*i+1] = start_idx
        
        bond_feat = get_bond_features(bond, mol)  # Pass molecule for 3D features
        bond_features[2*i] = bond_feat
        bond_features[2*i+1] = bond_feat
    
    return atom_features, atom_coords, bond_indices, bond_features


# =============================================================================
# PROTEIN FEATURIZATION
# =============================================================================

# Standard amino acid one-letter codes
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")  # 20 standard AAs

# Secondary structure types (DSSP)
SECONDARY_STRUCTURES = ['H', 'B', 'E', 'G', 'I', 'T', 'S', '-']  # Helix, Bridge, Strand, etc.


def get_residue_features(residue, secondary_structure: Optional[str] = None,
                        is_n_terminus: bool = False, is_c_terminus: bool = False) -> np.ndarray:
    """
    Extract residue-level features.
    
    Features (31-dimensional):
    - Amino acid type (1-hot, 20 AAs + 1 unknown)  [21 dims]
    - Secondary structure (1-hot, 8 types from DSSP) [8 dims]
    - Is N-terminus (binary)                         [1 dim]
    - Is C-terminus (binary)                         [1 dim]
    
    Total: 31 dimensions
    
    Scientific Rationale:
    - Amino acid identity crucial for chemical properties
    - Secondary structure indicates local fold geometry
    - Terminus flags identify chain boundaries, important for binding pocket edges
    
    Args:
        residue: BioPython residue object
        secondary_structure: DSSP secondary structure annotation (optional)
        is_n_terminus: Whether this is the N-terminus of a chain
        is_c_terminus: Whether this is the C-terminus of a chain
        
    Returns:
        Feature vector (31-dim)
    """
    # Amino acid type (1-hot)
    if residue is None:
        # Default to unknown amino acid for testing
        aa_1letter = 'X'
    else:
        resname = residue.get_resname()
        # Convert 3-letter to 1-letter code
        aa_dict = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
            'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
            'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
            'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
        }
        aa_1letter = aa_dict.get(resname, 'X')  # X for unknown
    
    aa_encoding = [int(aa_1letter == aa) for aa in AMINO_ACIDS]
    aa_encoding.append(int(aa_1letter not in AMINO_ACIDS))  # Unknown
    
    # Secondary structure (1-hot)
    if secondary_structure is not None:
        ss_encoding = [int(secondary_structure == ss) for ss in SECONDARY_STRUCTURES]
    else:
        ss_encoding = [0] * len(SECONDARY_STRUCTURES)  # Unknown
    
    # Terminus flags (properly computed from chain context)
    n_term_flag = [1 if is_n_terminus else 0]
    c_term_flag = [1 if is_c_terminus else 0]
    
    features = (
        aa_encoding +          # 21
        ss_encoding +          # 8
        n_term_flag +          # 1
        c_term_flag            # 1
    )  # Total: 31 dimensions
    
    return np.array(features, dtype=np.float32)


def featurize_protein(
    pdb_file: str,
    binding_pocket_residues: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Featurize a protein structure at residue level.
    
    Args:
        pdb_file: Path to PDB file
        binding_pocket_residues: List of residue IDs in binding pocket (optional)
                                If provided, only these residues are used
        
    Returns:
        Tuple of:
        - residue_features: (num_residues, 31) array
        - residue_coords: (num_residues, 3) array of Cα coordinates
        - metadata: Dictionary with additional info
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Use first model
    
    # Extract secondary structure using DSSP (if available)
    secondary_structure_dict = {}
    try:
        dssp = DSSP(model, pdb_file, dssp='mkdssp')
        for key in dssp.property_keys:
            residue_id = key[1][1]  # Get residue number
            ss = dssp[key][2]  # Secondary structure
            secondary_structure_dict[residue_id] = ss
    except Exception:
        # DSSP not available or failed
        pass
    
    # Extract residues with terminus detection
    residues = []
    residue_list_per_chain = {}
    
    for chain in model:
        chain_residues = []
        for residue in chain:
            # Skip non-standard residues (hetero atoms, water)
            if residue.get_id()[0] == ' ':  # Standard residue
                res_id = residue.get_id()[1]
                
                # Check if has Cα atom
                if 'CA' in residue:
                    chain_residues.append((res_id, residue))
        
        # Store for chain-specific terminus detection
        if chain_residues:
            residue_list_per_chain[chain.get_id()] = chain_residues
            residues.extend(chain_residues)
    
    num_residues = len(residues)
    residue_features = np.zeros((num_residues, 31), dtype=np.float32)
    residue_coords = np.zeros((num_residues, 3), dtype=np.float32)
    
    # Build residue feature matrix with terminus detection
    res_idx = 0
    for chain_id, chain_residues in residue_list_per_chain.items():
        for i, (res_id, residue) in enumerate(chain_residues):
            # Determine if N-terminus or C-terminus
            is_n_term = (i == 0)
            is_c_term = (i == len(chain_residues) - 1)
            
            # Filter by binding pocket if specified (do this AFTER collecting all residues)
            if binding_pocket_residues is not None:
                if res_id not in binding_pocket_residues:
                    continue
            
            ss = secondary_structure_dict.get(res_id, '-')
            residue_features[res_idx] = get_residue_features(
                residue, ss,
                is_n_terminus=is_n_term,
                is_c_terminus=is_c_term
            )
            ca_atom = residue['CA']
            residue_coords[res_idx] = ca_atom.get_coord()
            res_idx += 1
    
    # Trim to actual size (if binding pocket filtering was applied)
    residue_features = residue_features[:res_idx]
    residue_coords = residue_coords[:res_idx]
    
    metadata = {
        'num_residues': num_residues,
        'has_secondary_structure': len(secondary_structure_dict) > 0,
        'pdb_file': pdb_file,
    }
    
    return residue_features, residue_coords, metadata


def identify_binding_pocket(
    protein_coords: np.ndarray,
    ligand_coords: np.ndarray,
    cutoff: float = 10.0
) -> np.ndarray:
    """
    Identify binding pocket residues within cutoff distance of ligand.
    
    Following PLAIG approach: focus on residues within 10Å of ligand
    to reduce computation while preserving relevant interactions.
    
    Args:
        protein_coords: (num_residues, 3) protein Cα coordinates
        ligand_coords: (num_atoms, 3) ligand atom coordinates
        cutoff: Distance cutoff in Angstroms (default: 10.0)
        
    Returns:
        Boolean array indicating which residues are in binding pocket
    """
    # Compute pairwise distances (residues x ligand atoms)
    distances = np.linalg.norm(
        protein_coords[:, None, :] - ligand_coords[None, :, :],
        axis=2
    )
    
    # Residue is in pocket if any ligand atom is within cutoff
    in_pocket = np.any(distances <= cutoff, axis=1)
    
    return in_pocket


def featurize_complex(
    protein_pdb: str,
    ligand_sdf: str,
    binding_pocket_only: bool = True,
    pocket_cutoff: float = 10.0
) -> Dict:
    """
    Featurize a complete protein-ligand complex.
    
    Args:
        protein_pdb: Path to protein PDB file
        ligand_sdf: Path to ligand SDF file
        binding_pocket_only: If True, only use binding pocket residues
        pocket_cutoff: Distance cutoff for binding pocket (Angstroms)
        
    Returns:
        Dictionary with all featurized data
    """
    # Load and featurize ligand
    supplier = Chem.SDMolSupplier(ligand_sdf)
    ligand_mol = next(supplier)
    if ligand_mol is None:
        raise ValueError(f"Could not read ligand from {ligand_sdf}")
    
    ligand_atom_features, ligand_coords, ligand_bonds, ligand_bond_features = \
        featurize_ligand(ligand_mol)
    
    # Load and featurize protein (full structure first)
    protein_features_full, protein_coords_full, protein_metadata = \
        featurize_protein(protein_pdb)
    
    # Identify binding pocket
    if binding_pocket_only:
        in_pocket = identify_binding_pocket(
            protein_coords_full,
            ligand_coords,
            cutoff=pocket_cutoff
        )
        protein_features = protein_features_full[in_pocket]
        protein_coords = protein_coords_full[in_pocket]
    else:
        protein_features = protein_features_full
        protein_coords = protein_coords_full
    
    return {
        'ligand': {
            'atom_features': ligand_atom_features,
            'coords': ligand_coords,
            'bonds': ligand_bonds,
            'bond_features': ligand_bond_features,
            'num_atoms': len(ligand_atom_features),
        },
        'protein': {
            'residue_features': protein_features,
            'coords': protein_coords,
            'num_residues': len(protein_features),
            'metadata': protein_metadata,
        },
        'complex': {
            'binding_pocket_only': binding_pocket_only,
            'pocket_cutoff': pocket_cutoff,
        }
    }


if __name__ == "__main__":
    # Example usage
    print("Featurization module loaded.")
    print("Example atom features shape:", get_atom_features(Chem.MolFromSmiles('C').GetAtomWithIdx(0)).shape)
    print("Example bond features shape:", get_bond_features(Chem.MolFromSmiles('C=C').GetBondWithIdx(0)).shape)