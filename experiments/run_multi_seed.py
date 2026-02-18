"""
Multi-Seed Training Wrapper

Runs training across multiple random seeds and reports aggregate statistics
(mean ± std) for all metrics. Essential for publication-quality results.

Usage:
    python experiments/run_multi_seed.py --config config/painn_config.yaml --seeds 42 123 456 789 1024
    python experiments/run_multi_seed.py --config config/painn_config.yaml --num-seeds 5
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_metrics_from_log(log_file: str) -> dict:
    """Parse final validation metrics from a training log file."""
    metrics = {}
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        # Search backwards for last validation metrics
        for line in reversed(lines):
            if 'Val Loss:' in line or 'val' in line.lower():
                # Try to extract metrics from log line
                for metric_name in ['rmse', 'mae', 'pearson', 'spearman', 'r2', 'loss']:
                    import re
                    pattern = rf'{metric_name}[:\s]+([0-9.eE+-]+)'
                    match = re.search(pattern, line, re.IGNORECASE)
                    if match:
                        try:
                            metrics[f'val_{metric_name}'] = float(match.group(1))
                        except ValueError:
                            pass
                if metrics:
                    break
    except Exception as e:
        logger.warning(f"Could not parse log file {log_file}: {e}")
    
    return metrics


def parse_metrics_from_checkpoint(checkpoint_path: str) -> dict:
    """Parse metrics from a saved checkpoint file."""
    import torch
    
    metrics = {}
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
    except Exception as e:
        logger.warning(f"Could not parse checkpoint {checkpoint_path}: {e}")
    
    return metrics


def run_single_seed(config_path: str, seed: int, gpu: int, split_type: str,
                    gradient_accumulation: int, output_base: str) -> dict:
    """Run a single training run with a specific seed."""
    seed_output_dir = Path(output_base) / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = seed_output_dir / "training.log"
    
    cmd = [
        sys.executable, "-m", "experiments.train_painn",
        "--config", config_path,
        "--seed", str(seed),
        "--gpu", str(gpu),
        "--split-type", split_type,
        "--gradient-accumulation", str(gradient_accumulation),
    ]
    
    logger.info(f"Running seed {seed}: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=86400,  # 24h timeout
        )
        
        # Save stdout/stderr
        (seed_output_dir / "stdout.txt").write_text(result.stdout)
        (seed_output_dir / "stderr.txt").write_text(result.stderr)
        
        if result.returncode != 0:
            logger.error(f"Seed {seed} failed with return code {result.returncode}")
            logger.error(f"stderr: {result.stderr[-500:]}")
            return {'seed': seed, 'status': 'failed', 'error': result.stderr[-200:]}
        
        # Try to parse metrics from checkpoint first, then log
        checkpoint_path = seed_output_dir / "checkpoints" / "best_model.pt"
        if not checkpoint_path.exists():
            # Fall back to default checkpoint location from config
            from utils.config import load_config
            config = load_config(config_path)
            checkpoint_path = Path(config['logging']['checkpoint_dir']) / "best_model.pt"
        
        metrics = {}
        if checkpoint_path.exists():
            metrics = parse_metrics_from_checkpoint(str(checkpoint_path))
        
        if not metrics and log_file.exists():
            metrics = parse_metrics_from_log(str(log_file))
        
        metrics['seed'] = seed
        metrics['status'] = 'success'
        
        logger.info(f"Seed {seed} completed: {metrics}")
        return metrics
        
    except subprocess.TimeoutExpired:
        logger.error(f"Seed {seed} timed out")
        return {'seed': seed, 'status': 'timeout'}
    except Exception as e:
        logger.error(f"Seed {seed} error: {e}")
        return {'seed': seed, 'status': 'error', 'error': str(e)}


def aggregate_results(all_results: list) -> dict:
    """Compute mean ± std across all successful seed runs."""
    successful = [r for r in all_results if r.get('status') == 'success']
    
    if not successful:
        logger.error("No successful runs to aggregate")
        return {}
    
    # Collect all metric keys (excluding non-numeric fields)
    exclude_keys = {'seed', 'status', 'error'}
    metric_keys = set()
    for r in successful:
        metric_keys.update(k for k in r.keys() if k not in exclude_keys)
    
    aggregated = {
        'num_seeds': len(successful),
        'num_failed': len(all_results) - len(successful),
        'seeds': [r['seed'] for r in successful],
    }
    
    for key in sorted(metric_keys):
        values = [r[key] for r in successful if key in r and isinstance(r[key], (int, float)) and np.isfinite(r[key])]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values,
            }
    
    return aggregated


def format_results_table(aggregated: dict) -> str:
    """Format aggregated results as a readable table."""
    lines = []
    lines.append("=" * 70)
    lines.append("MULTI-SEED RESULTS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Successful runs: {aggregated.get('num_seeds', 0)}")
    lines.append(f"Failed runs:     {aggregated.get('num_failed', 0)}")
    lines.append(f"Seeds:           {aggregated.get('seeds', [])}")
    lines.append("-" * 70)
    lines.append(f"{'Metric':<25} {'Mean':>10} {'± Std':>10} {'Min':>10} {'Max':>10}")
    lines.append("-" * 70)
    
    exclude = {'num_seeds', 'num_failed', 'seeds'}
    for key, val in sorted(aggregated.items()):
        if key in exclude or not isinstance(val, dict):
            continue
        lines.append(
            f"{key:<25} {val['mean']:>10.4f} {val['std']:>10.4f} "
            f"{val['min']:>10.4f} {val['max']:>10.4f}"
        )
    
    lines.append("=" * 70)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Multi-seed training wrapper")
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='Explicit list of seeds (e.g., --seeds 42 123 456)')
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of seeds if --seeds not provided (default: 5)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--split-type', type=str, default='random',
                       choices=['random', 'scaffold', 'lp-pdbbind'])
    parser.add_argument('--gradient-accumulation', type=int, default=1)
    parser.add_argument('--output-dir', type=str, default='outputs/multi_seed',
                       help='Base output directory')
    args = parser.parse_args()
    
    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    else:
        rng = np.random.RandomState(42)
        seeds = rng.randint(0, 10000, size=args.num_seeds).tolist()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Multi-seed training with seeds: {seeds}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run all seeds
    all_results = []
    for i, seed in enumerate(seeds):
        logger.info(f"\n{'='*70}")
        logger.info(f"RUN {i+1}/{len(seeds)} — Seed {seed}")
        logger.info(f"{'='*70}")
        
        result = run_single_seed(
            config_path=args.config,
            seed=seed,
            gpu=args.gpu,
            split_type=args.split_type,
            gradient_accumulation=args.gradient_accumulation,
            output_base=str(output_dir),
        )
        all_results.append(result)
        
        # Save intermediate results after each run
        with open(output_dir / "all_results.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # Aggregate results
    aggregated = aggregate_results(all_results)
    
    # Format and display
    table = format_results_table(aggregated)
    logger.info(f"\n{table}")
    
    # Save final results
    with open(output_dir / "aggregated_results.json", 'w') as f:
        json.dump(aggregated, f, indent=2, default=str)
    
    with open(output_dir / "results_table.txt", 'w') as f:
        f.write(table)
    
    # Save per-seed results
    with open(output_dir / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info(f"  - aggregated_results.json: Mean ± std for all metrics")
    logger.info(f"  - results_table.txt: Human-readable summary")
    logger.info(f"  - all_results.json: Per-seed raw results")


if __name__ == '__main__':
    main()
