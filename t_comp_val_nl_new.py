"""
t_comp_val_nl_debug.py — Debug version of t_comp_val_nl.py

Adds targeted print statements at every point where k-grid shapes are
touched, so we can trace exactly where the high-k portion of the errors
goes missing.

Run this instead of t_comp_val_nl.py and inspect the output.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import train_utils_pk_emulator_v3 as utils


# ---------------------------------------------------------------------------
# Configuration  (match your working script exactly)
# ---------------------------------------------------------------------------

N_BATCHES   = 10
START_BATCH = 0
TEST_BATCH  = 100
NUM_PCS     = 20
NUM_PCS_Z   = 90
COSMO_TYPE  = "w0wacdm"
PRIOR_TYPE  = "expanded"
NL_TYPE     = "halofit"

W0_MIN      = -2.0
W0WA_MAX    = -0.4

K_MIN       = 1e-3

Z_IDX_0     = 0
Z_IDX_3     = 33

FIG_DIR     = "mps_emu/validation_figs"
VER         = "3_KminTrunc_debug"


# ---------------------------------------------------------------------------
# Filtering (unchanged)
# ---------------------------------------------------------------------------

def apply_filter(cola_set, w0_min=W0_MIN, w0wa_max=W0WA_MAX):
    w0_col   = utils.params.index("w")
    w0wa_col = utils.params.index("w0+wa")
    mask     = np.ones(len(cola_set.lhs), dtype=bool)
    if w0_min is not None:
        mask &= cola_set.lhs[:, w0_col] >= w0_min
    if w0wa_max is not None:
        mask &= cola_set.lhs[:, w0wa_col] <= w0wa_max
    for attr in ("lhs", "pks_target", "frac_pks", "logfracs"):
        if hasattr(cola_set, attr):
            setattr(cola_set, attr, getattr(cola_set, attr)[mask])
    if hasattr(cola_set, "mps_approxes_nl") and cola_set.mps_approxes_nl is not None:
        cola_set.mps_approxes_nl = cola_set.mps_approxes_nl[mask]
    elif hasattr(cola_set, "mps_approxes") and cola_set.mps_approxes is not None:
        cola_set.mps_approxes = cola_set.mps_approxes[mask]
    return int((~mask).sum())


# ---------------------------------------------------------------------------
# TruncatedPCASet with dense debug prints
# ---------------------------------------------------------------------------

class TruncatedPCASet:

    def __init__(self, logfracs_full, ks_full, num_pcs, num_pcs_z):
        self.k_mask    = ks_full >= K_MIN
        self.n_z       = logfracs_full.shape[1]
        self.num_pcs   = num_pcs
        self.num_pcs_z = num_pcs_z

        print(f"\n[DEBUG TruncatedPCASet.__init__]")
        print(f"  ks_full shape      : {ks_full.shape}")
        print(f"  ks_full range      : {ks_full[0]:.4g} -> {ks_full[-1]:.4g}")
        print(f"  K_MIN              : {K_MIN:.0e}")
        print(f"  k_mask True count  : {self.k_mask.sum()}  (k >= K_MIN, HIGH-k modes kept for PCA)")
        print(f"  k_mask False count : {(~self.k_mask).sum()}  (k <  K_MIN, LOW-k modes dropped from PCA)")
        print(f"  logfracs_full shape: {logfracs_full.shape}")

        self._lf_trunc = logfracs_full[:, :, self.k_mask]
        print(f"  _lf_trunc shape    : {self._lf_trunc.shape}  (should be N_cosmo x N_z x {self.k_mask.sum()})")

        self.scalers = []
        self.pcas    = []
        self.tpca    = None

    def fit(self):
        n_cosmo = self._lf_trunc.shape[0]
        all_pcs = []

        print(f"\n[DEBUG TruncatedPCASet.fit]")
        print(f"  Fitting on {self.k_mask.sum()} k-modes (k >= {K_MIN:.0e})")
        print(f"  n_cosmo = {n_cosmo},  n_z = {self.n_z},  num_pcs = {self.num_pcs}")

        for iz in range(self.n_z):
            lf_iz   = self._lf_trunc[:, iz, :]
            scaler  = StandardScaler()
            lf_norm = scaler.fit_transform(lf_iz)
            pca     = PCA(n_components=self.num_pcs)
            pcs     = pca.fit_transform(lf_norm)
            self.scalers.append(scaler)
            self.pcas.append(pca)
            all_pcs.append(pcs)

        all_pcs  = np.stack(all_pcs, axis=1)
        pcs_flat = all_pcs.reshape(n_cosmo, self.n_z * self.num_pcs)

        print(f"  all_pcs shape (stacked): {all_pcs.shape}")
        print(f"  pcs_flat shape          : {pcs_flat.shape}")

        self.tpca = PCA(n_components=self.num_pcs_z)
        self.tpca.fit(pcs_flat)

        print(f"  tPCA fitted with {self.num_pcs_z} components")
        return self

    def transform_and_reconstruct(self, logfracs_full_test):
        n_cosmo  = logfracs_full_test.shape[0]
        n_k_full = logfracs_full_test.shape[2]

        print(f"\n[DEBUG transform_and_reconstruct]")
        print(f"  logfracs_full_test shape : {logfracs_full_test.shape}")
        print(f"  k_mask shape             : {self.k_mask.shape}")
        print(f"  k_mask True count        : {self.k_mask.sum()}  (HIGH-k PCA modes)")
        print(f"  k_mask False count       : {(~self.k_mask).sum()}  (LOW-k assumption modes)")
        print(f"  n_k_full                 : {n_k_full}")

        # Check k_mask is compatible with logfracs_full_test
        if self.k_mask.shape[0] != n_k_full:
            print(f"  *** MISMATCH: k_mask has {self.k_mask.shape[0]} elements "
                  f"but logfracs has {n_k_full} k-modes! ***")
            print(f"  This is the likely cause of the plot showing only low-k data.")
        else:
            print(f"  k_mask size matches n_k_full — OK")

        lf_trunc = logfracs_full_test[:, :, self.k_mask]
        print(f"  lf_trunc shape (after slicing k >= K_MIN): {lf_trunc.shape}")

        # --- PCA round-trip ---
        pca_recon_trunc = np.empty_like(lf_trunc)
        all_pcs_test    = []

        for iz in range(self.n_z):
            lf_norm = self.scalers[iz].transform(lf_trunc[:, iz, :])
            pcs     = self.pcas[iz].transform(lf_norm)
            recon   = self.scalers[iz].inverse_transform(
                          self.pcas[iz].inverse_transform(pcs))
            pca_recon_trunc[:, iz, :] = recon
            all_pcs_test.append(pcs)

        print(f"  pca_recon_trunc shape : {pca_recon_trunc.shape}  "
              f"(should be N_cosmo x N_z x {self.k_mask.sum()})")

        # Check a sample value is non-trivial
        sample = pca_recon_trunc[0, Z_IDX_0, :]
        print(f"  pca_recon_trunc[0, iz=0, :] min={sample.min():.4g}, "
              f"max={sample.max():.4g}, std={sample.std():.4g}")

        # --- tPCA round-trip ---
        all_pcs_test = np.stack(all_pcs_test, axis=1)
        pcs_flat     = all_pcs_test.reshape(n_cosmo, self.n_z * self.num_pcs)
        t_comps      = self.tpca.transform(pcs_flat)
        pcs_recon    = self.tpca.inverse_transform(t_comps)
        pcs_per_z    = pcs_recon.reshape(n_cosmo, self.n_z, self.num_pcs)

        tpca_recon_trunc = np.empty_like(lf_trunc)
        for iz in range(self.n_z):
            recon = self.scalers[iz].inverse_transform(
                        self.pcas[iz].inverse_transform(pcs_per_z[:, iz, :]))
            tpca_recon_trunc[:, iz, :] = recon

        print(f"  tpca_recon_trunc shape: {tpca_recon_trunc.shape}")

        # --- Pad to full k-grid ---
        def _pad(recon_trunc):
            boost_full = np.ones((n_cosmo, self.n_z, n_k_full),
                                 dtype=recon_trunc.dtype)
            print(f"    _pad: recon_trunc shape = {recon_trunc.shape}")
            print(f"    _pad: boost_full shape  = {boost_full.shape}")
            print(f"    _pad: k_mask True count = {self.k_mask.sum()}")
            print(f"    _pad: assigning boost_full[:, :, k_mask] = exp(recon_trunc)")
            boost_full[:, :, self.k_mask] = np.exp(recon_trunc)
            # Verify: high-k region should not all be 1.0
            hk = boost_full[0, Z_IDX_0, self.k_mask]
            lk = boost_full[0, Z_IDX_0, ~self.k_mask]
            print(f"    _pad: boost_full[0,iz=0, k>=K_MIN] min={hk.min():.4g}, "
                  f"max={hk.max():.4g}, std={hk.std():.4g}")
            print(f"    _pad: boost_full[0,iz=0, k< K_MIN] min={lk.min():.4g}, "
                  f"max={lk.max():.4g}  (should all be 1.0)")
            return boost_full

        print(f"\n  --- Padding PCA reconstruction ---")
        pca_boost_full  = _pad(pca_recon_trunc)
        print(f"  --- Padding tPCA reconstruction ---")
        tpca_boost_full = _pad(tpca_recon_trunc)

        return pca_boost_full, tpca_boost_full


# ---------------------------------------------------------------------------
# Error computation with debug
# ---------------------------------------------------------------------------

def compute_errors(boost_recon_full, frac_pks_full, iz, label, k_mask):
    recon = boost_recon_full[:, iz, :]
    true  = frac_pks_full[:, iz, :]
    errors = (recon - true) / true

    print(f"\n[DEBUG compute_errors — {label}, iz={iz}]")
    print(f"  recon shape      : {recon.shape}")
    print(f"  true shape       : {true.shape}")
    print(f"  errors shape     : {errors.shape}")
    print(f"  errors full range: min={errors.min():.4g}, max={errors.max():.4g}")
    print(f"  errors high-k (k>=K_MIN): "
          f"min={errors[:, k_mask].min():.4g}, "
          f"max={errors[:, k_mask].max():.4g}, "
          f"std={errors[:, k_mask].std():.4g}")
    print(f"  errors low-k  (k< K_MIN): "
          f"min={errors[:, ~k_mask].min():.4g}, "
          f"max={errors[:, ~k_mask].max():.4g}, "
          f"std={errors[:, ~k_mask].std():.4g}")

    # Check for NaN/Inf in each region
    nan_hk = np.isnan(errors[:, k_mask]).sum()
    nan_lk = np.isnan(errors[:, ~k_mask]).sum()
    inf_hk = np.isinf(errors[:, k_mask]).sum()
    print(f"  NaN high-k: {nan_hk},  NaN low-k: {nan_lk},  Inf high-k: {inf_hk}")

    # Spot-check: does the first cosmology have non-trivial high-k error?
    first_hk = errors[0, k_mask]
    all_zero = np.allclose(first_hk, 0)
    all_one  = np.allclose(first_hk, 1)
    print(f"  errors[0, k>=K_MIN]: all-zero={all_zero}, all-one={all_one}, "
          f"sample[0:5]={first_hk[:5]}")

    return errors


# ---------------------------------------------------------------------------
# Plotting (minimal — just enough to reproduce the symptom)
# ---------------------------------------------------------------------------

def _tag():
    filter_parts = []
    if W0_MIN   is not None: filter_parts.append(f"w0min{W0_MIN}")
    if W0WA_MAX is not None: filter_parts.append(f"w0wamax{W0WA_MAX}")
    filter_str = ("_" + "_".join(filter_parts)) if filter_parts else ""
    return (f"{COSMO_TYPE}_{NL_TYPE}_{PRIOR_TYPE}{filter_str}"
            f"_n{NUM_PCS}_z{NUM_PCS_Z}_kmin{K_MIN:.0e}_v{VER}")


def plot_errors_debug(errors, ks_full, iz_label, stage, k_mask):
    """Plot with extra annotations to make the symptom visible."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: full k-range (the plot that was broken)
    ax = axes[0]
    for error in errors:
        ax.semilogx(ks_full, error, lw=0.6, alpha=0.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.axvline(K_MIN, color="red", lw=1.5, ls=":",
               label=f"K_MIN={K_MIN:.0e}")
    ax.set_xlabel(r"$k$", fontsize=12)
    ax.set_ylabel(r"$(B_\mathrm{recon} - B_\mathrm{true}) / B_\mathrm{true}$", fontsize=12)
    ax.set_title(f"{stage} errors — full k-range (z≈{iz_label})", fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)

    # Right: zoom into high-k region only (should show PCA errors)
    ax = axes[1]
    ks_hk  = ks_full[k_mask]
    err_hk = errors[:, k_mask]
    for e in err_hk:
        ax.semilogx(ks_hk, e, lw=0.6, alpha=0.5)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k$", fontsize=12)
    ax.set_ylabel(r"$(B_\mathrm{recon} - B_\mathrm{true}) / B_\mathrm{true}$", fontsize=12)
    ax.set_title(f"{stage} errors — k >= {K_MIN:.0e} only (z≈{iz_label})", fontsize=12)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = f"{FIG_DIR}/{stage.lower()}_errors_debug_z{iz_label}_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family']      = 'STIXGeneral'
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("[INFO] DEBUG RUN — t_comp_val_nl_debug.py")
    print(f"       K_MIN = {K_MIN:.0e}")
    print("=" * 60)

    # --- Load and filter training set ---
    print("\n[INFO] Loading training set...")
    train_set = utils.COLASet(
        target_z    = utils.z_mps,
        cosmo_type  = COSMO_TYPE,
        prior_type  = PRIOR_TYPE,
        nl_type     = NL_TYPE,
        n_batches   = N_BATCHES,
        start_batch = START_BATCH,
    )
    apply_filter(train_set)

    print(f"\n[DEBUG] train_set.ks shape : {train_set.ks.shape}")
    print(f"[DEBUG] train_set.ks range : {train_set.ks[0]:.4g} -> {train_set.ks[-1]:.4g}")
    print(f"[DEBUG] train_set.logfracs shape : {train_set.logfracs.shape}")
    print(f"[DEBUG] train_set.frac_pks shape : {train_set.frac_pks.shape}")
    print(f"[DEBUG] utils.ks shape     : {utils.ks.shape}")
    print(f"[DEBUG] utils.ks range     : {utils.ks[0]:.4g} -> {utils.ks[-1]:.4g}")

    # Is train_set.ks the same as utils.ks, or is it already truncated?
    if len(train_set.ks) != len(utils.ks):
        print(f"[DEBUG] *** train_set.ks ({len(train_set.ks)} pts) != utils.ks "
              f"({len(utils.ks)} pts) — COLASet has already truncated the k-grid! ***")
        print(f"[DEBUG]     This means k_mask built from train_set.ks will have "
              f"{len(train_set.ks)} elements, not 500.")
        print(f"[DEBUG]     Modes flagged as k < K_MIN by this mask correspond to "
              f"k values {train_set.ks[~(train_set.ks >= K_MIN)][0]:.4g} -> "
              f"{train_set.ks[~(train_set.ks >= K_MIN)][-1]:.4g}")
    else:
        print(f"[DEBUG] train_set.ks matches utils.ks (500 pts) — OK")

    ks_full = train_set.ks

    # --- Fit local PCA ---
    print("\n[INFO] Fitting TruncatedPCASet...")
    pca_set = TruncatedPCASet(
        logfracs_full = train_set.logfracs,
        ks_full       = ks_full,
        num_pcs       = NUM_PCS,
        num_pcs_z     = NUM_PCS_Z,
    ).fit()

    k_mask = pca_set.k_mask
    print(f"\n[DEBUG] After fit:")
    print(f"  k_mask shape      : {k_mask.shape}")
    print(f"  k_mask True  (high-k, PCA fitted) : {k_mask.sum()}")
    print(f"  k_mask False (low-k, padded w/ 1) : {(~k_mask).sum()}")
    print(f"  ks_full[k_mask] range  : {ks_full[k_mask][0]:.4g} -> {ks_full[k_mask][-1]:.4g}")
    print(f"  ks_full[~k_mask] range : {ks_full[~k_mask][0]:.4g} -> {ks_full[~k_mask][-1]:.4g}")

    # --- Load and filter test set ---
    print("\n[INFO] Loading test set...")
    test_set = utils.COLASet(
        target_z    = utils.z_mps,
        cosmo_type  = COSMO_TYPE,
        prior_type  = PRIOR_TYPE,
        nl_type     = NL_TYPE,
        start_batch = TEST_BATCH,
    )
    apply_filter(test_set)

    print(f"\n[DEBUG] test_set.ks shape     : {test_set.ks.shape}")
    print(f"[DEBUG] test_set.logfracs shape: {test_set.logfracs.shape}")
    print(f"[DEBUG] test_set.frac_pks shape: {test_set.frac_pks.shape}")

    # Key check: does test_set.logfracs have the same N_k as ks_full / k_mask?
    if test_set.logfracs.shape[2] != len(ks_full):
        print(f"[DEBUG] *** SHAPE MISMATCH: test logfracs has {test_set.logfracs.shape[2]} "
              f"k-modes but k_mask has {len(k_mask)} elements! ***")
        print(f"[DEBUG]     This WILL cause the wrong modes to be treated as high-k.")
    else:
        print(f"[DEBUG] test_set.logfracs k-dim matches k_mask — OK")

    # --- Reconstruct ---
    print("\n[INFO] Reconstructing through PCA / tPCA...")
    pca_boost_full, tpca_boost_full = pca_set.transform_and_reconstruct(
        test_set.logfracs
    )

    print(f"\n[DEBUG] pca_boost_full shape  : {pca_boost_full.shape}")
    print(f"[DEBUG] tpca_boost_full shape : {tpca_boost_full.shape}")
    print(f"[DEBUG] test_set.frac_pks shape: {test_set.frac_pks.shape}")

    # Check that pca_boost_full k-dim matches frac_pks
    if pca_boost_full.shape[2] != test_set.frac_pks.shape[2]:
        print(f"[DEBUG] *** SHAPE MISMATCH between boost_recon "
              f"({pca_boost_full.shape[2]}) and frac_pks "
              f"({test_set.frac_pks.shape[2]}) ***")

    # --- Errors ---
    pca_err_z0  = compute_errors(pca_boost_full,  test_set.frac_pks,
                                 Z_IDX_0, "PCA",  k_mask)
    tpca_err_z0 = compute_errors(tpca_boost_full, test_set.frac_pks,
                                 Z_IDX_0, "tPCA", k_mask)

    # --- Plots (two-panel: full range + high-k zoom) ---
    print("\n[INFO] Saving debug figures...")
    plot_errors_debug(pca_err_z0,  ks_full, iz_label=0, stage="PCA",  k_mask=k_mask)
    plot_errors_debug(tpca_err_z0, ks_full, iz_label=0, stage="tPCA", k_mask=k_mask)

    print("\n[INFO] Debug run complete.  Check the output above for *** warnings.")


if __name__ == "__main__":
    main()