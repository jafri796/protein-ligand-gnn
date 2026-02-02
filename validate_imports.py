#!/usr/bin/env python3
"""Validate all imports work correctly after refactoring."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    errors = []
    
    # Test data module imports
    try:
        from data import ProteinLigandDataset, construct_complex_graph
        print("✓ data module imports")
    except Exception as e:
        errors.append(f"data module: {e}")
        print(f"✗ data module: {e}")
    
    # Test models module imports
    try:
        from models import PaiNNAffinityPredictor
        print("✓ models module imports")
    except Exception as e:
        errors.append(f"models module: {e}")
        print(f"✗ models module: {e}")
    
    # Test models.layers imports
    try:
        from models.layers import PaiNNLayer, InteractionLayer, RBFExpansion
        print("✓ models.layers module imports")
    except Exception as e:
        errors.append(f"models.layers module: {e}")
        print(f"✗ models.layers module: {e}")
    
    # Test utils imports
    try:
        from utils import ConfigLoader, set_seed, validate_config
        print("✓ utils module imports")
    except Exception as e:
        errors.append(f"utils module: {e}")
        print(f"✗ utils module: {e}")
    
    # Test experiments imports
    try:
        import experiments.train_painn
        print("✓ experiments.train_painn imports")
    except Exception as e:
        errors.append(f"experiments.train_painn: {e}")
        print(f"✗ experiments.train_painn: {e}")
    
    # Summary
    print("\n" + "="*50)
    if errors:
        print(f"FAILED: {len(errors)} import error(s)")
        for err in errors:
            print(f"  - {err}")
        return 1
    else:
        print("SUCCESS: All imports validated")
        return 0

if __name__ == "__main__":
    sys.exit(main())
