"""
Model Interpretability Module for Protein-Ligand Binding Affinity Prediction.

Provides tools to understand model predictions:
1. Attention weight extraction from InteractionLayer
2. Feature importance via SHAP values
3. Gradient-based analysis (Integrated Gradients)
4. Visualization utilities for binding site importance

Usage:
    python experiments/interpretability.py --config config/painn_config.yaml --checkpoint best_model.pt --pdb 1abc
"""

import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import ProteinLigandDataset
from models.painn_affinity import PaiNNAffinityPredictor
from utils import ConfigLoader, set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AttentionExtractor:
    """
    Extract attention weights from InteractionLayer during forward pass.
    
    Scientific Rationale:
    Attention weights reveal which protein-ligand atom pairs the model
    considers most important for binding affinity prediction.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_weights = {}
        self._hooks = []
        
    def _register_hooks(self):
        """Register forward hooks on InteractionLayer to capture attention."""
        def make_hook(name):
            def hook(module, input, output):
                # Store attention weights if available
                if hasattr(module, '_last_attention_weights'):
                    self.attention_weights[name] = module._last_attention_weights.detach().cpu()
            return hook
        
        for name, module in self.model.named_modules():
            if 'interaction' in name.lower() or 'InteractionLayer' in type(module).__name__:
                hook = module.register_forward_hook(make_hook(name))
                self._hooks.append(hook)
                
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
        
    def extract(self, batch) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for a batch.
        
        Args:
            batch: PyG Data or HeteroData batch
            
        Returns:
            Dictionary mapping layer names to attention weight tensors
        """
        self.attention_weights = {}
        self._register_hooks()
        
        try:
            self.model.eval()
            with torch.no_grad():
                _ = self.model(batch)
        finally:
            self._remove_hooks()
            
        return self.attention_weights


class IntegratedGradients:
    """
    Compute feature importance via Integrated Gradients.
    
    Reference: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
    
    Scientific Rationale:
    IG provides theoretically grounded attributions that satisfy key axioms
    (sensitivity, implementation invariance) for understanding which input
    features contribute most to predictions.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def compute_attributions(
        self,
        batch,
        target_idx: int = 0,
        n_steps: int = 50,
        baseline_type: str = 'zero'
    ) -> Dict[str, np.ndarray]:
        """
        Compute integrated gradients for input features.
        
        Args:
            batch: PyG Data batch
            target_idx: Index of target output (0 for single regression)
            n_steps: Number of interpolation steps
            baseline_type: 'zero' or 'random' baseline
            
        Returns:
            Dictionary with attributions for each feature type
        """
        self.model.eval()
        self.model.to(self.device)
        
        # Get input features
        x = batch.x.clone().to(self.device)
        x.requires_grad_(True)
        
        # Create baseline
        if baseline_type == 'zero':
            baseline = torch.zeros_like(x)
        else:
            baseline = torch.randn_like(x) * 0.01
            
        # Compute integrated gradients
        scaled_inputs = []
        for alpha in np.linspace(0, 1, n_steps):
            scaled_input = baseline + alpha * (x - baseline)
            scaled_inputs.append(scaled_input)
            
        # Stack and compute gradients
        all_gradients = []
        for scaled_input in scaled_inputs:
            scaled_input = scaled_input.clone().detach().requires_grad_(True)
            
            # Create batch copy with interpolated features
            batch_copy = batch.clone()
            batch_copy.x = scaled_input
            batch_copy = batch_copy.to(self.device)
            
            output = self.model(batch_copy)
            
            # Compute gradient
            if output.dim() > 1:
                output = output[:, target_idx]
            output = output.sum()
            
            grad = torch.autograd.grad(output, scaled_input, create_graph=False)[0]
            all_gradients.append(grad.detach().cpu())
            
        # Average gradients and multiply by (input - baseline)
        avg_gradients = torch.stack(all_gradients).mean(dim=0)
        attributions = (x.detach().cpu() - baseline.cpu()) * avg_gradients
        
        # Aggregate attributions per atom (sum over feature dimensions)
        atom_importance = attributions.sum(dim=-1).numpy()
        
        return {
            'atom_importance': atom_importance,
            'feature_attributions': attributions.numpy(),
            'n_steps': n_steps,
            'baseline_type': baseline_type
        }


class SHAPAnalyzer:
    """
    SHAP-based feature importance analysis.
    
    Uses KernelSHAP for model-agnostic explanations.
    Requires: pip install shap
    
    Scientific Rationale:
    SHAP values provide theoretically grounded, additive feature attributions
    based on game-theoretic Shapley values.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._shap_available = self._check_shap()
        
    def _check_shap(self) -> bool:
        """Check if SHAP is available."""
        try:
            import shap
            return True
        except ImportError:
            logger.warning("SHAP not installed. Install with: pip install shap")
            return False
            
    def compute_shap_values(
        self,
        dataset,
        n_background: int = 50,
        n_samples: int = 10
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Compute SHAP values for feature importance.
        
        Args:
            dataset: ProteinLigandDataset
            n_background: Number of background samples for SHAP
            n_samples: Number of samples to explain
            
        Returns:
            Dictionary with SHAP values and feature names
        """
        if not self._shap_available:
            logger.error("SHAP not available. Install with: pip install shap")
            return None
            
        import shap
        
        self.model.eval()
        self.model.to(self.device)
        
        # Create prediction function for aggregated features
        def predict_fn(X: np.ndarray) -> np.ndarray:
            """Wrapper for SHAP - expects 2D array input."""
            with torch.no_grad():
                # X is aggregated features, need to reconstruct batch
                # This is a simplified version using graph-level features
                X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                # Note: This requires a feature-based model wrapper
                return X_tensor.mean(dim=-1).cpu().numpy()
        
        # Extract graph-level features for SHAP
        logger.info(f"Extracting features from {min(n_background + n_samples, len(dataset))} samples...")
        
        features = []
        for i in range(min(n_background + n_samples, len(dataset))):
            data = dataset[i]
            if hasattr(data, 'x') and data.x is not None:
                # Aggregate node features to graph level
                feat = data.x.mean(dim=0).numpy()
                features.append(feat)
                
        if len(features) < n_background + 1:
            logger.error(f"Not enough samples: {len(features)}")
            return None
            
        features = np.array(features)
        background = features[:n_background]
        samples_to_explain = features[n_background:n_background + n_samples]
        
        # Create SHAP explainer
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(samples_to_explain)
        
        return {
            'shap_values': shap_values,
            'background_size': n_background,
            'n_samples': len(samples_to_explain)
        }


class ModelInterpreter:
    """
    Unified interface for model interpretation.
    
    Combines attention extraction, integrated gradients, and SHAP
    for comprehensive model understanding.
    """
    
    def __init__(self, model: nn.Module, config: Dict = None):
        self.model = model
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.attention_extractor = AttentionExtractor(model)
        self.ig_analyzer = IntegratedGradients(model, self.device)
        self.shap_analyzer = SHAPAnalyzer(model, self.device)
        
    def analyze_prediction(
        self,
        batch,
        methods: List[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of a single prediction.
        
        Args:
            batch: PyG Data batch
            methods: List of methods to use ('attention', 'ig', 'shap')
            
        Returns:
            Dictionary with analysis results from each method
        """
        methods = methods or ['attention', 'ig']
        results = {}
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            batch = batch.to(self.device)
            prediction = self.model(batch)
            results['prediction'] = prediction.cpu().numpy()
            
        if 'attention' in methods:
            logger.info("Extracting attention weights...")
            results['attention'] = self.attention_extractor.extract(batch)
            
        if 'ig' in methods:
            logger.info("Computing Integrated Gradients...")
            results['integrated_gradients'] = self.ig_analyzer.compute_attributions(batch)
            
        return results
        
    def get_important_atoms(
        self,
        batch,
        top_k: int = 10,
        method: str = 'ig'
    ) -> Dict[str, Any]:
        """
        Identify most important atoms for prediction.
        
        Args:
            batch: PyG Data batch
            top_k: Number of top atoms to return
            method: 'ig' for integrated gradients, 'attention' for attention weights
            
        Returns:
            Dictionary with atom indices and importance scores
        """
        if method == 'ig':
            ig_results = self.ig_analyzer.compute_attributions(batch)
            importance = np.abs(ig_results['atom_importance'])
        else:
            attn = self.attention_extractor.extract(batch)
            # Aggregate attention over all layers
            importance = np.zeros(batch.num_nodes)
            for layer_attn in attn.values():
                if layer_attn is not None:
                    # Sum attention received by each node
                    importance += layer_attn.sum(dim=-1).numpy()
                    
        # Get top-k atoms
        top_indices = np.argsort(importance)[-top_k:][::-1]
        top_scores = importance[top_indices]
        
        # Identify atom types if available
        atom_types = []
        if hasattr(batch, 'node_type'):
            for idx in top_indices:
                node_type = batch.node_type[idx].item() if hasattr(batch.node_type[idx], 'item') else batch.node_type[idx]
                atom_types.append('ligand' if node_type == 0 else 'protein')
        else:
            atom_types = ['unknown'] * len(top_indices)
            
        return {
            'top_atom_indices': top_indices.tolist(),
            'importance_scores': top_scores.tolist(),
            'atom_types': atom_types,
            'method': method
        }
        
    def save_analysis(self, results: Dict, output_path: str):
        """Save analysis results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            return obj
            
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(convert(results), f, indent=2)
            
        logger.info(f"Saved analysis to {output_path}")


def visualize_importance(
    pdb_file: str,
    importance_scores: np.ndarray,
    output_file: str = None,
    colormap: str = 'coolwarm'
):
    """
    Visualize atom importance scores on 3D structure.
    
    Requires: pip install py3Dmol (for Jupyter) or pymol
    
    Args:
        pdb_file: Path to PDB file
        importance_scores: Array of importance scores per atom
        output_file: Output file for visualization
        colormap: Matplotlib colormap name
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm
        
        # Normalize scores
        norm = Normalize(vmin=importance_scores.min(), vmax=importance_scores.max())
        cmap = cm.get_cmap(colormap)
        
        # Generate B-factor-style output for PyMOL visualization
        output_file = output_file or pdb_file.replace('.pdb', '_importance.pdb')
        
        with open(pdb_file, 'r') as f_in, open(output_file, 'w') as f_out:
            atom_idx = 0
            for line in f_in:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    if atom_idx < len(importance_scores):
                        # Replace B-factor column with importance score
                        score = importance_scores[atom_idx] * 100  # Scale for visibility
                        new_line = line[:60] + f'{score:6.2f}' + line[66:]
                        f_out.write(new_line)
                        atom_idx += 1
                    else:
                        f_out.write(line)
                else:
                    f_out.write(line)
                    
        logger.info(f"Wrote importance-annotated PDB to {output_file}")
        logger.info("Visualize in PyMOL with: spectrum b, blue_white_red")
        
        return output_file
        
    except ImportError:
        logger.warning("matplotlib not available for visualization")
        return None


def main():
    """Main function for interpretability analysis."""
    parser = argparse.ArgumentParser(description='Model Interpretability Analysis')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data-dir', type=str, help='Data directory')
    parser.add_argument('--pdb-id', type=str, help='Specific PDB ID to analyze')
    parser.add_argument('--output-dir', type=str, default='experiments/interpretability_results')
    parser.add_argument('--methods', nargs='+', default=['attention', 'ig'], 
                        choices=['attention', 'ig', 'shap'])
    parser.add_argument('--top-k', type=int, default=10, help='Top-k atoms to report')
    args = parser.parse_args()
    
    # Load config
    config = ConfigLoader.load(args.config)
    set_seed(config.get('reproducibility', {}).get('seed', 42))
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load model
    model_config = config.get('model', {})
    model = PaiNNAffinityPredictor(
        node_dim=model_config.get('node_dim', 128),
        edge_dim=model_config.get('edge_dim', 64),
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_message_passing_layers', 4),
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {args.checkpoint}")
    
    # Create interpreter
    interpreter = ModelInterpreter(model, config)
    
    # Load data
    data_dir = args.data_dir or config.get('data', {}).get('data_dir')
    if not data_dir:
        logger.error("Data directory not specified")
        return
        
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # If specific PDB ID provided, analyze that
    if args.pdb_id:
        logger.info(f"Analyzing PDB: {args.pdb_id}")
        # Load specific complex - implementation depends on dataset structure
        # For now, log placeholder
        logger.info("Single PDB analysis requires dataset loading - use with full dataset")
    else:
        # Analyze a sample from the test set
        test_split = config.get('data', {}).get('test_split', 'data/splits/test.txt')
        if Path(test_split).exists():
            dataset = ProteinLigandDataset(
                data_dir=data_dir,
                index_file=test_split,
                cache_dir=str(output_dir / 'cache')
            )
            
            if len(dataset) > 0:
                # Analyze first sample
                batch = dataset[0]
                results = interpreter.analyze_prediction(batch, methods=args.methods)
                
                # Get important atoms
                important = interpreter.get_important_atoms(batch, top_k=args.top_k)
                results['important_atoms'] = important
                
                # Save results
                interpreter.save_analysis(
                    results, 
                    str(output_dir / 'analysis_results.json')
                )
                
                logger.info(f"\n=== Top {args.top_k} Important Atoms ===")
                for i, (idx, score, atype) in enumerate(zip(
                    important['top_atom_indices'],
                    important['importance_scores'],
                    important['atom_types']
                )):
                    logger.info(f"  {i+1}. Atom {idx} ({atype}): importance = {score:.4f}")
        else:
            logger.warning(f"Test split not found: {test_split}")
            
    logger.info("Interpretability analysis complete")


if __name__ == "__main__":
    main()
