"""
Equivariant layers for SE(3)-equivariant message passing.

Implements PaiNN-inspired layers following Schütt et al. (2021).
"""

from .equivariant_layers import (
    RBFExpansion,
    PaiNNMessage,
    PaiNNUpdate,
    PaiNNLayer,
    InteractionLayer,
)

__all__ = [
    'RBFExpansion',
    'PaiNNMessage',
    'PaiNNUpdate',
    'PaiNNLayer',
    'InteractionLayer',
]
