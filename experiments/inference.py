"""Standalone inference script for PaiNN binding affinity model.

Usage:
    python experiments/inference.py \
        --protein data/pdbbind/1abc/1abc_protein.pdb \
        --ligand data/pdbbind/1abc/1abc_ligand.sdf \
        --checkpoint outputs/checkpoints/best_model.pt \
        --config config/painn_config.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
from torch_geometric.data import Batch

from data.featurization import featurize_complex
from data.graph_construction import (
    construct_complex_graph,
    construct_ligand_graph,
    construct_protein_graph,
)
from models.painn_affinity import PaiNNAffinityPredictor
from utils.config import load_config


def _build_graph(
    protein_pdb: Path,
    ligand_sdf: Path,
    binding_pocket_only: bool,
    pocket_cutoff: float,
    interaction_cutoff: float,
    protein_knn: int,
):
    """Featurize protein-ligand pair and build PyG graph."""

    complex_data = featurize_complex(
        protein_pdb=str(protein_pdb),
        ligand_sdf=str(ligand_sdf),
        binding_pocket_only=binding_pocket_only,
        pocket_cutoff=pocket_cutoff,
    )

    ligand_graph = construct_ligand_graph(
        atom_features=complex_data["ligand"]["atom_features"],
        atom_coords=complex_data["ligand"]["coords"],
        bond_indices=complex_data["ligand"]["bonds"],
        bond_features=complex_data["ligand"]["bond_features"],
    )

    protein_graph = construct_protein_graph(
        residue_features=complex_data["protein"]["residue_features"],
        residue_coords=complex_data["protein"]["coords"],
        method="knn",
        k=protein_knn,
    )

    if protein_graph.num_nodes == 0:
        raise RuntimeError("Protein graph has zero nodes; check pocket cutoff")

    complex_graph = construct_complex_graph(
        ligand_data=ligand_graph,
        protein_data=protein_graph,
        interaction_cutoff=interaction_cutoff,
        use_heterogeneous=False,
    )

    complex_graph.pdb_id = ligand_sdf.stem
    return complex_graph


def _load_model(checkpoint: Path, config_path: Path, device: torch.device) -> PaiNNAffinityPredictor:
    config = load_config(str(config_path))
    model = PaiNNAffinityPredictor(config["model"])
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def run_inference(args: argparse.Namespace) -> Dict[str, Any]:
    protein_path = Path(args.protein).resolve()
    ligand_path = Path(args.ligand).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    config_path = Path(args.config).resolve()

    if not protein_path.exists():
        raise FileNotFoundError(f"Protein file not found: {protein_path}")
    if not ligand_path.exists():
        raise FileNotFoundError(f"Ligand file not found: {ligand_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(str(config_path))
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = _load_model(checkpoint_path, config_path, device)

    binding_pocket_only = not args.full_protein if args.full_protein is not None else data_cfg.get("binding_pocket_only", True)
    pocket_cutoff = args.pocket_cutoff or data_cfg.get("pocket_cutoff", 10.0)
    interaction_cutoff = args.interaction_cutoff or data_cfg.get("interaction_cutoff", 5.0)
    protein_knn = data_cfg.get("protein_knn", 10)

    graph = _build_graph(
        protein_pdb=protein_path,
        ligand_sdf=ligand_path,
        binding_pocket_only=binding_pocket_only,
        pocket_cutoff=pocket_cutoff,
        interaction_cutoff=interaction_cutoff,
        protein_knn=protein_knn,
    )

    batch = Batch.from_data_list([graph]).to(device)
    with torch.no_grad():
        prediction = model(batch)

    if hasattr(prediction, "item"):
        pred_value = float(prediction.item())
    else:
        pred_value = float(prediction)

    result = {
        "protein": str(protein_path),
        "ligand": str(ligand_path),
        "pKd": pred_value,
        "device": str(device),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference with PaiNN affinity model")
    parser.add_argument("--protein", required=True, help="Path to protein PDB file")
    parser.add_argument("--ligand", required=True, help="Path to ligand SDF file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument(
        "--config",
        default="config/painn_config.yaml",
        help="Path to configuration YAML (default: config/painn_config.yaml)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Preferred device (will fall back to CPU if unavailable)",
    )
    parser.add_argument(
        "--pocket-cutoff",
        type=float,
        help="Override binding pocket cutoff distance in Å",
    )
    parser.add_argument(
        "--interaction-cutoff",
        type=float,
        help="Override protein-ligand interaction cutoff in Å",
    )
    parser.add_argument(
        "--full-protein",
        action="store_true",
        help="Use entire protein instead of binding pocket",
    )
    parser.add_argument(
        "--output",
        help="Optional path to JSON file for saving the prediction",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    result = run_inference(args)
    print(f"Predicted pKd: {result['pKd']:.4f}")


if __name__ == "__main__":
    main()
