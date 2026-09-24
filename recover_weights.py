"""
recover_pce_weights.py — Re-fit Stage 1 PCE regression and patch W into
the existing metadata bundle, without retraining the neural network.

Run once from the mps_emu directory:
    python recover_pce_weights.py

All parameters below must match exactly what was passed to train_v2.py.
Check your job log for the Stage 1 block:

    [COLA_NPCE_Keras] Stage 1 — PCE elastic-net regression
      N_train   = 37500
      N_terms   = 897
      N_t       = 46
      alpha     = 1.00e-04
      l1_ratio  = 0.9
      max_iter  = 1000
"""

import numpy as np
import train_utils_pk_emulator_v3 as utils
from npce_v3 import _build_multi_indices, _evaluate_pce_basis, _fit_elastic_net_pce

# ── Match exactly what was passed to train_v2.py ───────────────────────────
COSMO_TYPE         = "w0wacdm"
PRIOR_TYPE         = "expanded"
NL_TYPE            = "halofit"
N_BATCHES          = 50
START_BATCH        = 0
W0_MIN             = -2.0
W0WA_MAX           = -0.4
NUM_PCS            = 20        # num_pcs passed to prepare() — inferred from bundle: 20
NUM_PCS_Z          = 46        # num_pcs_z — from the "46 tPCA components" log line
PCE_MAX_DEGREE     = 8         # --pce_max_degree
PCE_NORM_Q         = 0.75      # --pce_norm_q
PCE_NORM_THRESHOLD = 1.0       # --pce_norm_threshold
ALPHA              = 1e-4      # default in COLA_NPCE_Keras
L1_RATIO           = 0.9       # default in COLA_NPCE_Keras
PCE_MAX_ITER       = 1000      # default in COLA_NPCE_Keras

# ── Load training data ─────────────────────────────────────────────────────
print("Loading training set...")
train_set = utils.COLASet(
    target_z    = utils.z_mps,
    cosmo_type  = COSMO_TYPE,
    prior_type  = PRIOR_TYPE,
    nl_type     = NL_TYPE,
    start_batch = START_BATCH,
    n_batches   = N_BATCHES,
)

# ── Apply the same prior cuts used at training time ────────────────────────
print("Applying prior cuts...")
w0_col   = utils.params.index("w")
w0wa_col = utils.params.index("w0+wa")
mask = (
    (train_set.lhs[:, w0_col]   >= W0_MIN) &
    (train_set.lhs[:, w0wa_col] <= W0WA_MAX)
)
for attr in ("lhs", "pks_target", "frac_pks", "logfracs"):
    val = getattr(train_set, attr, None)
    if val is not None:
        setattr(train_set, attr, val[mask])
if hasattr(train_set, "mps_approxes_boost") and train_set.mps_approxes_boost is not None:
    train_set.mps_approxes_boost = train_set.mps_approxes_boost[mask]
    train_set.mps_approxes       = train_set.mps_approxes_boost
elif hasattr(train_set, "mps_approxes") and train_set.mps_approxes is not None:
    train_set.mps_approxes = train_set.mps_approxes[mask]
train_set.w0_min   = W0_MIN
train_set.w0wa_max = W0WA_MAX
print(f"  {mask.sum()} cosmologies remaining after prior cuts.")

# ── Prepare PCA — must match training exactly ──────────────────────────────
# This re-fits the same scalers and tPCA, producing the same
# t_components_norm that W was originally fitted against.
print("Preparing PCA (must match training run exactly)...")
train_set.prepare(num_pcs=NUM_PCS, num_pcs_z=NUM_PCS_Z)

# ── Re-build the PCE multi-index set ──────────────────────────────────────
print("Building PCE multi-index set...")
n_dims  = train_set.lhs_norm.shape[1]
indices = _build_multi_indices(
    n_dims         = n_dims,
    max_degree     = PCE_MAX_DEGREE,
    norm_q         = PCE_NORM_Q,
    norm_threshold = PCE_NORM_THRESHOLD,
)
print(f"  {indices.shape[0]} basis terms  (expected 897)")

# ── Evaluate PCE basis on training data ───────────────────────────────────
print("Evaluating PCE basis on training data...")
Phi_train = _evaluate_pce_basis(indices, train_set.lhs_norm)
print(f"  Phi_train shape: {Phi_train.shape}")

# ── Re-fit Stage 1 elastic-net ────────────────────────────────────────────
print("Fitting elastic-net PCE regression (Stage 1)...")
W = _fit_elastic_net_pce(
    Phi_train = Phi_train,
    T_train   = train_set.t_components_norm,
    alpha     = ALPHA,
    l1_ratio  = L1_RATIO,
    max_iter  = PCE_MAX_ITER,
)
print(f"  W shape: {W.shape}  (expected ({NUM_PCS_Z}, {indices.shape[0]}))")

# Quick sanity check: PCE train MSE should be close to what was printed
# during the original training run (check your job log).
t_pce_train = (Phi_train @ W.T).astype(np.float32)
pce_mse = float(np.mean((t_pce_train - train_set.t_components_norm) ** 2))
print(f"  PCE train MSE (normalised t-space): {pce_mse:.6f}  ← should match original log")

# ── Patch W and indices into the existing bundle ──────────────────────────
# update_metadata_bundle_npce loads the bundle, adds/overwrites the two
# NPCE fields, and resaves — no other bundle contents are changed.
print("\nPatching bundle...")
utils.update_metadata_bundle_npce(
    bundle_path  = train_set._metadata_bundle_path,
    npce_indices = indices,
    npce_W       = W,
    model_type   = "npce",
)
print("Done — no retraining needed.")
print(f"Bundle location: {train_set._metadata_bundle_path}")