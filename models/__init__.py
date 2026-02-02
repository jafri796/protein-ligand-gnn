"""
Models module for protein-ligand binding affinity prediction.

Provides PaiNN-based equivariant model, baseline models, and ensemble methods.
"""

from .painn_affinity import PaiNNAffinityPredictor
from .baselines import GraphDTA, RFBaseline, LinearBaseline, GraphDTAFeatureExtractor
from .ensembling import (
    SoftVotingEnsemble,
    WeightedVotingEnsemble,
    StackingEnsemble,
    create_ensemble_from_config,
)

__all__ = [
    # Main model
    'PaiNNAffinityPredictor',
    # Baselines
    'GraphDTA',
    'RFBaseline',
    'LinearBaseline',
    'GraphDTAFeatureExtractor',
    # Ensembling
    'SoftVotingEnsemble',
    'WeightedVotingEnsemble',
    'StackingEnsemble',
    'create_ensemble_from_config',
]
