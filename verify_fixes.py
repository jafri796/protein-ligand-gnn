"""
Verification script for all audit fixes.
Run: python verify_fixes.py
Results written to verify_fixes_results.txt
"""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(__file__))

results = []

def check(name, fn):
    try:
        fn()
        results.append(f"  PASS: {name}")
        return True
    except Exception as e:
        results.append(f"  FAIL: {name} — {e}")
        results.append(f"        {traceback.format_exc().splitlines()[-2]}")
        return False

results.append("=" * 70)
results.append("AUDIT FIX VERIFICATION")
results.append("=" * 70)

# ─── Issue #2: update_V removed from PaiNNUpdate ────────────────────────
results.append("\n[Issue #2] PaiNNUpdate: unused update_V removed")
def check_issue2():
    from models.layers.equivariant_layers import PaiNNUpdate
    u = PaiNNUpdate(64)
    assert not hasattr(u, 'update_V'), "update_V should have been removed"
    assert hasattr(u, 'update_U'), "update_U should still exist"
check("update_V removed, update_U present", check_issue2)

# ─── Issue #4: R² in compute_metrics ────────────────────────────────────
results.append("\n[Issue #4] compute_metrics includes R²")
def check_issue4():
    from experiments.train_painn import compute_metrics
    import numpy as np
    m = compute_metrics([1.0, 2.0, 3.0], [1.1, 2.1, 2.9])
    assert 'r2' in m, f"r2 missing from metrics keys: {list(m.keys())}"
    assert np.isfinite(m['r2']), f"r2 is not finite: {m['r2']}"
    assert m['r2'] > 0.9, f"r2 should be high for near-perfect predictions: {m['r2']}"
check("r2 key present and correct", check_issue4)

# ─── Issue #7: Docstring matches actual feature dims ────────────────────
results.append("\n[Issue #7] Feature dim docstring accuracy")
def check_issue7():
    from data.featurization import get_atom_features, LIGAND_ATOM_FEATURE_DIM
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    atom = mol.GetAtomWithIdx(0)
    feat = get_atom_features(atom)
    assert len(feat) == 49, f"Expected 49, got {len(feat)}"
    assert len(feat) == LIGAND_ATOM_FEATURE_DIM, f"Mismatch: {len(feat)} vs {LIGAND_ATOM_FEATURE_DIM}"
check("Atom features = 49 dims, matches constant", check_issue7)

# ─── Issue #11: train_enhanced scheduler uses config ─────────────────────
results.append("\n[Issue #11] train_enhanced.py scheduler uses config")
def check_issue11():
    import ast
    with open("experiments/train_enhanced.py", "r") as f:
        src = f.read()
    assert "lr_sched_cfg" in src, "Should use lr_sched_cfg variable"
    assert "lr_sched_cfg.get('patience'" in src, "Should read patience from config"
    assert "lr_sched_cfg.get('factor'" in src, "Should read factor from config"
check("Scheduler reads from config dict", check_issue11)

# ─── Issue #12: No bare except in splits.py ──────────────────────────────
results.append("\n[Issue #12] No bare except in splits.py")
def check_issue12():
    with open("data/splits.py", "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped == "except:" or stripped == "except :":
            raise AssertionError(f"Bare except found at line {i}: {stripped}")
check("No bare except clauses", check_issue12)

# ─── Issue #13: No IUPACData import in splits.py ────────────────────────
results.append("\n[Issue #13] Unused IUPACData import removed")
def check_issue13():
    with open("data/splits.py", "r") as f:
        src = f.read()
    assert "IUPACData" not in src, "IUPACData import should be removed"
check("IUPACData not in splits.py", check_issue13)

# ─── Issue #14: autocast has device_type ─────────────────────────────────
results.append("\n[Issue #14] autocast() has device_type kwarg")
def check_issue14():
    with open("experiments/train_enhanced.py", "r") as f:
        src = f.read()
    assert "autocast(device_type=" in src, "autocast should have device_type kwarg"
check("autocast(device_type='cuda') present", check_issue14)

# ─── Issue #1: target_stats comment clarified ────────────────────────────
results.append("\n[Issue #1] Target stats documentation clarified")
def check_issue1():
    with open("data/dataset.py", "r") as f:
        src = f.read()
    assert "NOT applied to graph.y" in src, "Should document that stats are not applied"
check("Clarifying comment present in dataset.py", check_issue1)

# ─── Issue #3: InteractionLayer configurable weights ─────────────────────
results.append("\n[Issue #3] InteractionLayer attention weights configurable")
def check_issue3():
    from models.layers.equivariant_layers import InteractionLayer
    import inspect
    sig = inspect.signature(InteractionLayer.__init__)
    params = list(sig.parameters.keys())
    assert 'vector_weight' in params, f"vector_weight not in params: {params}"
    assert 'edge_weight' in params, f"edge_weight not in params: {params}"
    # Test with custom values
    layer = InteractionLayer(64, num_heads=4, vector_weight=0.5, edge_weight=0.2)
    assert layer.vector_weight == 0.5, f"vector_weight not set: {layer.vector_weight}"
    assert layer.edge_weight == 0.2, f"edge_weight not set: {layer.edge_weight}"
    # Test default backwards compat
    layer2 = InteractionLayer(64, num_heads=4)
    assert layer2.vector_weight == 0.3, f"Default vector_weight wrong: {layer2.vector_weight}"
check("vector_weight/edge_weight params work", check_issue3)

# ─── Issue #5: Fixed test_validation dataset test ────────────────────────
results.append("\n[Issue #5] test_validation dataset test rewritten")
def check_issue5():
    with open("tests/test_validation.py", "r") as f:
        src = f.read()
    assert "test_dataset_validation_filters_missing_files" in src, \
        "Should have renamed test method"
    assert "MOCK PDB FILE" not in src, "Old broken mock content should be gone"
    assert "pdb_id,affinity" not in src, "Old CSV header format should be gone"
check("Test rewritten with correct assertions", check_issue5)

# ─── Issue #6: DSSP warning on failure ───────────────────────────────────
results.append("\n[Issue #6] DSSP failure produces warning")
def check_issue6():
    with open("data/featurization.py", "r") as f:
        src = f.read()
    assert "DSSP unavailable" in src, "Should have DSSP unavailable warning"
    assert "Install mkdssp" in src, "Should suggest installing mkdssp"
check("DSSP warning message present", check_issue6)

# ─── Issue #10: Multi-seed script exists ─────────────────────────────────
results.append("\n[Issue #10] Multi-seed training script")
def check_issue10():
    assert os.path.exists("experiments/run_multi_seed.py"), "Script should exist"
    with open("experiments/run_multi_seed.py", "r") as f:
        src = f.read()
    assert "aggregate_results" in src, "Should have aggregation function"
    assert "mean" in src and "std" in src, "Should compute mean/std"
    assert "format_results_table" in src, "Should format results table"
check("run_multi_seed.py exists with required functions", check_issue10)

# ─── Issue #8: CI/CD pipeline ────────────────────────────────────────────
results.append("\n[Issue #8] GitHub Actions CI/CD")
def check_issue8():
    ci_path = ".github/workflows/ci.yml"
    assert os.path.exists(ci_path), f"CI file should exist at {ci_path}"
    with open(ci_path, "r") as f:
        src = f.read()
    assert "pytest" in src, "Should run pytest"
    assert "flake8" in src, "Should run linting"
check("CI workflow exists with pytest + linting", check_issue8)

# ─── Functional: PaiNN layers still work ─────────────────────────────────
results.append("\n[Functional] PaiNN layers forward pass")
def check_painn_forward():
    import torch
    from models.layers.equivariant_layers import PaiNNLayer, RBFExpansion
    hidden_dim = 64
    num_rbf = 20
    n = 10
    s = torch.randn(n, hidden_dim)
    v = torch.randn(n, 3, hidden_dim)
    edge_index = torch.randint(0, n, (2, 20))
    rbf = RBFExpansion(num_rbf=num_rbf, cutoff=10.0)
    dists = torch.rand(20) * 10.0
    edge_rbf = rbf(dists)
    edge_vec = torch.randn(20, 3)
    edge_vec = edge_vec / (edge_vec.norm(dim=1, keepdim=True) + 1e-8)
    layer = PaiNNLayer(hidden_dim, num_rbf)
    s_out, v_out = layer(s, v, edge_index, edge_rbf, edge_vec)
    assert s_out.shape == (n, hidden_dim), f"s shape wrong: {s_out.shape}"
    assert v_out.shape == (n, 3, hidden_dim), f"v shape wrong: {v_out.shape}"
    assert torch.isfinite(s_out).all(), "s contains NaN/Inf"
    assert torch.isfinite(v_out).all(), "v contains NaN/Inf"
check("PaiNNLayer forward pass", check_painn_forward)

# ─── Functional: Rotation equivariance ───────────────────────────────────
results.append("\n[Functional] SE(3) rotation equivariance")
def check_equivariance():
    import torch
    import numpy as np
    from models.layers.equivariant_layers import PaiNNMessage
    hidden_dim = 32
    num_rbf = 10
    n = 10
    s = torch.randn(n, hidden_dim)
    v = torch.randn(n, 3, hidden_dim)
    edge_index = torch.tensor([[0,1,2,3,4],[1,2,3,4,0]])
    edge_rbf = torch.randn(5, num_rbf)
    edge_vec = torch.randn(5, 3)
    edge_vec = edge_vec / (edge_vec.norm(dim=1, keepdim=True) + 1e-8)
    # Random rotation
    angle = torch.rand(1) * 2 * np.pi
    axis = torch.randn(3); axis = axis / axis.norm()
    K = torch.zeros(3,3)
    K[0,1]=-axis[2]; K[0,2]=axis[1]; K[1,0]=axis[2]
    K[1,2]=-axis[0]; K[2,0]=-axis[1]; K[2,1]=axis[0]
    R = torch.eye(3) + torch.sin(angle)*K + (1-torch.cos(angle))*(K@K)
    v_rot = torch.einsum('ij,bjh->bih', R, v)
    ev_rot = torch.einsum('ij,ej->ei', R, edge_vec)
    layer = PaiNNMessage(hidden_dim, num_rbf)
    layer.eval()
    ds1, dv1 = layer(s, v, edge_index, edge_rbf, edge_vec)
    ds2, dv2 = layer(s, v_rot, edge_index, edge_rbf, ev_rot)
    assert torch.allclose(ds1, ds2, atol=1e-5), "Scalars not rotation invariant"
    dv1_rot = torch.einsum('ij,bjh->bih', R, dv1)
    assert torch.allclose(dv1_rot, dv2, atol=1e-4), "Vectors not rotation equivariant"
check("PaiNNMessage rotation equivariance", check_equivariance)

# ─── Functional: Full model forward pass ─────────────────────────────────
results.append("\n[Functional] Full PaiNNAffinityPredictor forward pass")
def check_full_model():
    import torch
    from torch_geometric.data import Data, Batch
    from models.painn_affinity import PaiNNAffinityPredictor
    from data.featurization import LIGAND_ATOM_FEATURE_DIM, LIGAND_BOND_FEATURE_DIM, PROTEIN_RESIDUE_FEATURE_DIM
    config = {
        'hidden_dim': 64, 'num_message_passing_layers': 2,
        'num_protein_layers': 2, 'num_rbf': 10, 'cutoff': 10.0, 'dropout': 0.1
    }
    model = PaiNNAffinityPredictor(config)
    model.eval()
    padded_dim = max(LIGAND_ATOM_FEATURE_DIM, PROTEIN_RESIDUE_FEATURE_DIM)
    edge_attr_dim = LIGAND_BOND_FEATURE_DIM + 1 + 3
    data_list = []
    for _ in range(2):
        nl, np_ = 8, 15
        nn_ = nl + np_
        x = torch.randn(nn_, padded_dim)
        pos = torch.randn(nn_, 3)
        ei = torch.randint(0, nn_, (2, 30))
        ea = torch.randn(30, edge_attr_dim)
        nt = torch.cat([torch.zeros(nl, dtype=torch.long), torch.ones(np_, dtype=torch.long)])
        data_list.append(Data(x=x, pos=pos, edge_index=ei, edge_attr=ea, node_type=nt, y=torch.tensor([7.5])))
    batch = Batch.from_data_list(data_list)
    with torch.no_grad():
        out = model(batch)
    assert out.shape == (2,), f"Expected (2,), got {out.shape}"
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"
check("Full model batch forward pass", check_full_model)

# ─── Summary ─────────────────────────────────────────────────────────────
results.append("\n" + "=" * 70)
passes = sum(1 for r in results if r.startswith("  PASS"))
fails = sum(1 for r in results if r.startswith("  FAIL"))
results.append(f"TOTAL: {passes} passed, {fails} failed out of {passes+fails} checks")
if fails == 0:
    results.append("ALL CHECKS PASSED")
else:
    results.append(f"WARNING: {fails} check(s) FAILED — review above")
results.append("=" * 70)

output = "\n".join(results)
print(output)

with open("verify_fixes_results.txt", "w") as f:
    f.write(output)

sys.exit(1 if fails > 0 else 0)
