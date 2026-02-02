"""
External Test Set Evaluation
==============================
Comprehensive evaluation on held-out test sets with multiple metrics.

Features:
- Evaluation on multiple test splits (standard test, external, scaffold-based)
- Detailed per-complex metrics and analysis
- Statistical significance testing
- Error analysis and visualization
- Multi-model comparison support
- Uncertainty quantification

Usage:
    python experiments/evaluate.py \
        --model outputs/checkpoints/best_model.pt \
        --config config/painn_config.yaml \
        --test_split test \
        --batch_size 32 \
        --device cuda
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import linregress, spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import yaml

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import ProteinLigandDataset
from models.painn_affinity import PaiNNAffinityPredictor

# Alias for convenience
PaiNNAffinity = PaiNNAffinityPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    try:
        pearson, _ = pearsonr(predictions, targets)
    except:
        pearson = np.nan
    
    try:
        spearman, _ = spearmanr(predictions, targets)
    except:
        spearman = np.nan
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson,
        'spearman': spearman,
    }


class Evaluator:
    """Comprehensive model evaluation on test sets."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        batch_size: int = 32,
    ):
        """Initialize evaluator.

        Args:
            model: Trained model
            device: Device to use (cuda/cpu)
            batch_size: Evaluation batch size
        """
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.model.to(device).eval()

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        return_predictions: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate model on a dataset.

        Args:
            dataloader: PyTorch DataLoader with test data
            return_predictions: Whether to return predictions and targets

        Returns:
            Dictionary with metrics and optionally predictions
        """
        all_predictions = []
        all_targets = []
        all_ids = []
        total_loss = 0.0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(
                dataloader,
                desc="Evaluating",
                disable=len(dataloader) < 10,
            )):
                # Move batch to device
                batch = batch.to(self.device)

                # Forward pass
                predictions = self.model(batch)

                # Ensure correct shapes (use batch.y which is set by PyG DataLoader from graph.y)
                targets = batch.y.view(-1, 1) if hasattr(batch, 'y') and batch.y is not None else batch.affinity.view(-1, 1)
                loss = criterion(predictions, targets)

                # Store results
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                
                # Extract complex IDs (handle both single and batched scenarios)
                if hasattr(batch, 'pdb_id'):
                    if isinstance(batch.pdb_id, list):
                        all_ids.extend(batch.pdb_id)
                    else:
                        # Tensor or single value
                        all_ids.extend([str(x) for x in batch.pdb_id if isinstance(batch.pdb_id, (list, tuple))] or [batch.pdb_id])
                else:
                    # Fallback: use indices
                    all_ids.extend([f"complex_{batch_idx}_{i}" for i in range(len(targets))])
                
                total_loss += loss.item()

        # Convert to numpy
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        # Compute metrics
        metrics = compute_metrics(predictions, targets)
        metrics["total_loss"] = total_loss / len(dataloader)
        metrics["num_samples"] = len(targets)

        result = {
            "metrics": metrics,
            "predictions": predictions,
            "targets": targets,
            "complex_ids": all_ids,
        }

        if return_predictions:
            return result

        return metrics

    def evaluate_with_confidence(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_mcdropout: bool = False,
        mc_samples: int = 20,
    ) -> Dict[str, Any]:
        """Evaluate with uncertainty quantification (optional MC Dropout).

        Args:
            dataloader: Test DataLoader
            use_mcdropout: Whether to use MC Dropout for uncertainty
            mc_samples: Number of MC Dropout samples

        Returns:
            Dictionary with predictions, targets, and uncertainty estimates
        """
        logger.info("Running evaluation with uncertainty quantification...")

        if use_mcdropout:
            # Enable dropout during inference
            self._set_dropout_train(self.model, True)

        all_predictions = []
        all_uncertainties = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="MC Dropout Evaluation"):
                batch = batch.to(self.device)
                targets = batch.y.view(-1, 1) if hasattr(batch, 'y') and batch.y is not None else batch.affinity.view(-1, 1)

                if use_mcdropout:
                    # Multiple forward passes
                    samples = []
                    for _ in range(mc_samples):
                        output = self.model(batch)
                        samples.append(output.cpu().numpy())

                    # Compute mean and std
                    samples = np.array(samples)
                    predictions = samples.mean(axis=0).flatten()
                    uncertainties = samples.std(axis=0).flatten()
                else:
                    predictions = self.model(batch).cpu().numpy().flatten()
                    uncertainties = np.zeros_like(predictions)

                all_predictions.extend(predictions)
                all_uncertainties.extend(uncertainties)
                all_targets.extend(targets.cpu().numpy().flatten())

        if use_mcdropout:
            self._set_dropout_train(self.model, False)

        return {
            "predictions": np.array(all_predictions),
            "uncertainties": np.array(all_uncertainties),
            "targets": np.array(all_targets),
        }

    @staticmethod
    def _set_dropout_train(model: nn.Module, train: bool):
        """Recursively set dropout layers to train/eval mode."""
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train(train)

    def error_analysis(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        complex_ids: List[str],
        percentiles: List[float] = [10, 25, 50, 75, 90],
    ) -> Dict[str, Any]:
        """Analyze prediction errors by percentile.

        Args:
            predictions: Predicted affinities
            targets: True affinities
            complex_ids: Complex IDs for tracking
            percentiles: Error percentiles to compute

        Returns:
            Error analysis statistics
        """
        errors = np.abs(predictions - targets)
        error_pcts = np.percentile(errors, percentiles)

        # Identify worst predictions
        worst_idx = np.argsort(errors)[-10:]  # Top 10 worst

        analysis = {
            "mean_error": float(errors.mean()),
            "median_error": float(np.median(errors)),
            "std_error": float(errors.std()),
            "percentiles": {
                f"p{int(pct)}": float(val)
                for pct, val in zip(percentiles, error_pcts)
            },
            "worst_predictions": [
                {
                    "complex_id": complex_ids[i],
                    "target": float(targets[i]),
                    "prediction": float(predictions[i]),
                    "error": float(errors[i]),
                }
                for i in worst_idx[::-1]
            ],
        }

        return analysis


def main(args: argparse.Namespace):
    """Main evaluation pipeline.

    Args:
        args: Command-line arguments
    """
    # Setup
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    set_seed(42)
    logger.info(f"Using device: {device}")

    # Load configuration
    config = load_config(args.config)
    logger.info(f"Config: {config}")

    # Load model
    logger.info(f"Loading model from {args.model}...")
    model = PaiNNAffinity(config['model'])
    
    checkpoint = torch.load(args.model, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    logger.info("Model loaded successfully")

    # Load test dataset
    logger.info(f"Loading {args.test_split} dataset...")
    
    # Determine split file path
    split_files = {
        'test': config['data'].get('test_split', 'splits/test.txt'),
        'external': 'splits/external.txt',
        'scaffold': 'splits/scaffold.txt',
    }
    split_file = split_files.get(args.test_split, split_files['test'])
    
    test_dataset = ProteinLigandDataset(
        data_dir=args.data_root,
        index_file=split_file,
        cache_dir=config['data'].get('cache_dir', 'data/processed'),
        binding_pocket_only=config['data'].get('binding_pocket_only', True),
        pocket_cutoff=config['data'].get('pocket_cutoff', 10.0),
        interaction_cutoff=config['data'].get('interaction_cutoff', 5.0),
    )
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Create dataloader (use PyG DataLoader)
    from torch_geometric.loader import DataLoader as PyGDataLoader
    test_dataloader = PyGDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Initialize evaluator
    evaluator = Evaluator(model, device=device, batch_size=args.batch_size)

    # Run evaluation
    logger.info("=" * 60)
    logger.info("RUNNING COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)

    eval_result = evaluator.evaluate(test_dataloader, return_predictions=True)
    metrics = eval_result["metrics"]
    predictions = eval_result["predictions"]
    targets = eval_result["targets"]
    complex_ids = eval_result["complex_ids"]

    # Print metrics
    logger.info("\nTest Set Performance:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE:  {metrics['mae']:.4f}")
    logger.info(f"  R²:   {metrics['r2']:.4f}")
    logger.info(f"  Pearson r: {metrics['pearson']:.4f}")
    logger.info(f"  Spearman ρ: {metrics['spearman']:.4f}")

    # Error analysis
    error_analysis = evaluator.error_analysis(predictions, targets, complex_ids)
    logger.info("\nError Analysis:")
    logger.info(f"  Mean Error: {error_analysis['mean_error']:.4f}")
    logger.info(f"  Median Error: {error_analysis['median_error']:.4f}")
    logger.info(f"  Std Error: {error_analysis['std_error']:.4f}")
    logger.info("  Percentiles:")
    for pct_str, val in error_analysis["percentiles"].items():
        logger.info(f"    {pct_str}: {val:.4f}")

    # Optional: MC Dropout uncertainty
    if args.mc_dropout:
        logger.info("\nRunning MC Dropout uncertainty quantification...")
        mc_result = evaluator.evaluate_with_confidence(
            test_dataloader,
            use_mcdropout=True,
            mc_samples=args.mc_samples,
        )
        logger.info(f"  Mean uncertainty (std): {mc_result['uncertainties'].mean():.4f}")

    # Save results
    if args.save_results:
        output_dir = Path(args.output_dir) / f"evaluation_{args.test_split}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = output_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(
                {
                    **metrics,
                    "error_analysis": error_analysis,
                },
                f,
                indent=2,
            )
        logger.info(f"Metrics saved to {metrics_file}")

        # Save detailed predictions
        pred_df = pd.DataFrame({
            "complex_id": complex_ids,
            "target": targets,
            "prediction": predictions,
            "error": np.abs(predictions - targets),
        })
        pred_file = output_dir / "predictions.csv"
        pred_df.to_csv(pred_file, index=False)
        logger.info(f"Predictions saved to {pred_file}")

        # Save worst cases
        worst_file = output_dir / "worst_predictions.json"
        with open(worst_file, "w") as f:
            json.dump(error_analysis["worst_predictions"], f, indent=2)
        logger.info(f"Worst predictions saved to {worst_file}")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate model on test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard test set evaluation
  python experiments/evaluate.py --model outputs/checkpoints/best_model.pt

  # External test set with uncertainty
  python experiments/evaluate.py \\
      --model outputs/checkpoints/best_model.pt \\
      --test_split external \\
      --mc_dropout \\
      --mc_samples 30

  # Detailed error analysis
  python experiments/evaluate.py \\
      --model outputs/checkpoints/best_model.pt \\
      --save_results \\
      --output_dir outputs/evaluation
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="outputs/checkpoints/best_model.pt",
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/painn_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/pdbbind",
        help="Root directory of data",
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        choices=["test", "external", "scaffold"],
        help="Test split to evaluate on",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--mc_dropout",
        action="store_true",
        help="Use MC Dropout for uncertainty quantification",
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=20,
        help="Number of MC Dropout samples",
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save detailed evaluation results to disk",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results",
    )

    args = parser.parse_args()
    main(args)
