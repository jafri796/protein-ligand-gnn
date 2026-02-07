"""
Configuration and utilities for reproducibility and validation.
"""

from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Single authoritative implementations — re-exported from submodules
from .reproducibility import set_seed
from .config import validate_config


def create_directories(paths: list) -> None:
    """Create necessary directories."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


class ConfigLoader:
    """Load and validate YAML configuration files."""
    
    @staticmethod
    def load(config_path: str) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config YAML file
            
        Returns:
            Parsed configuration dictionary
        """
        from .config import load_config as _load
        return _load(config_path)
    
    @staticmethod
    def save(config: Dict, output_path: str) -> None:
        """Save configuration to YAML file."""
        from .config import save_config as _save
        _save(config, output_path)


from .config import load_config, save_config

# Re-export schemas from config.py (single source of truth)
from .config import MODEL_SCHEMA, TRAINING_SCHEMA

__all__ = [
    'set_seed', 'validate_config', 'create_directories', 'ConfigLoader',
    'load_config', 'save_config', 'MODEL_SCHEMA', 'TRAINING_SCHEMA'
]
