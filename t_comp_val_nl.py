"""
t_comp_val_nl.py — Nonlinear Boost PCA and tPCA Reconstruction Validation Script

Validates that the two-stage dimensionality reduction (per-redshift PCA of
log(P_nonlin / P_nl_syren), followed by a temporal PCA across redshifts)
introduces only small reconstruction errors.  For each stage the script
produces a spaghetti plot of fractional errors:

    exp(reconstructed_logfrac) / (P_nonlin / P_nl_syren) - 1

across the test set.

Also visualises the regime boundary in cosmological parameter space, to
identify whether any bimodal structure in the boost logfracs is cleanly
separable by the diagnostic thresholds.

Outputs (saved to FIG_DIR):
  1. pca_errors_z0_<tag>.pdf   — per-cosmology PCA reconstruction errors at z=0
  2. tpca_errors_z0_<tag>.pdf  — per-cosmology tPCA reconstruction errors at z=0
  3. pca_errors_z3_<tag>.pdf   — same as (1) but at z≈3
  4. tpca_errors_z3_<tag>.pdf  — same as (2) but at z≈3
  5. regime_boundary_<tag>.pdf — regime index distribution and 2-D projections
                                 into cosmological parameter space
  6. scree_<tag>.pdf           — explained variance scree plots for the spatial
                                 PCA (sampled redshifts) and the tPCA

Unlike the old NL/LIN boost approach, frac_pks here is
P_nonlin_camb / P_nl_syren — the residual between CAMB's nonlinear spectrum
and the syren halofit prediction.  This ratio is much closer to 1 across the
full k range, so no k-truncation is needed and the PCA can use all 500 k modes.

Usage (standalone):
    python ./mps_emu/t_comp_val_nl.py

Author: Victoria Lloyd (2026)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

import train_utils_pk_emulator_v3 as utils

from sklearn.decomposition import KernelPCA


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BATCHES   = 10
START_BATCH = 0
TEST_BATCH  = 1000
NUM_PCS     = 23       # number of spatial PCA components per redshift
NUM_PCS_Z   = 65       # number of temporal PCA components across redshifts
COSMO_TYPE  = "w0wacdm"
PRIOR_TYPE  = "expanded"
NL_TYPE     = "mead2020_Tfree_mnufree" #"mead2020_Tfree_mnufree"   # nonlinear prescription: halofit, mead2020, etc.

W0_MIN      = None
W0WA_MAX    = None
OM_MIN      = None

# Omega_b / H0 triangle cut. Must match --omegab_anchor / --h0_anchor used in training.
OMEGAB_ANCHOR = 0.05
H0_ANCHOR     = 75
OMEGAB_H0_TRIANGLE_CUT = {
    'omegab_anchor': OMEGAB_ANCHOR,
    'h0_anchor':     H0_ANCHOR,
    'omegab_max':    utils.OMEGA_B_MAX,
    'h0_max':        utils.H0_MAX,
}

# Plot-time exclusion of |w0| < this value. None = no exclusion.
PLOT_W0_ABS_MIN = None

# NOTE: K_MIN truncation has been removed.  The syren halofit baseline
# (P_nl_syren) is non-trivial at all k, so the residual frac_pks =
# P_nl_camb / P_nl_syren is informative across the full k grid and there
# is no longer any benefit to discarding low-k modes before PCA fitting.
# K_MIN  = 1e-2   # (removed)

Z_IDX_0     = 0        # redshift index for z≈0
Z_IDX_3     = 33       # redshift index for z≈3 (index into utils.z_mps)

FIG_DIR     = "mps_emu/validation_figs/smaller_grid"

VER         = "0_smaller_grid"

# Redshift indices whose spatial-PCA scree curves are highlighted individually.
# All other redshifts are drawn as thin grey lines for context.
SCREE_Z_HIGHLIGHT = [0, 8, 16, 24, 33]   # z≈0, intermediate, z≈3

# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

AXES_FS = 14
TICK_FS = 12


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_data_diagnostics(train_set):
    """
    Print shape, NaN/Inf counts, and a summary of bad cosmologies for all
    key arrays in train_set.  Handles both boost mode (mps_approxes_nl) and
    lin mode (mps_approxes) gracefully.
    """
    print("=== BASIC SHAPES ===")
    print("lhs:          ", train_set.lhs.shape)
    print("pks_target:   ", train_set.pks_target.shape)
    print("frac_pks:     ", train_set.frac_pks.shape)
    print("logfracs:     ", train_set.logfracs.shape)
    if hasattr(train_set, "mps_approxes_nl") and train_set.mps_approxes_nl is not None:
        print("mps_approxes_nl:", train_set.mps_approxes_nl.shape)
    elif hasattr(train_set, "mps_approxes") and train_set.mps_approxes is not None:
        print("mps_approxes:  ", train_set.mps_approxes.shape)

    print("\n=== NaN / Inf SUMMARY ===")
    arrays_to_check = [
        ("pks_target", train_set.pks_target),
        ("frac_pks",   train_set.frac_pks),
        ("logfracs",   train_set.logfracs),
    ]
    # Report the syren NL baseline if present, else fall back to linear
    if hasattr(train_set, "mps_approxes_nl") and train_set.mps_approxes_nl is not None:
        arrays_to_check.insert(0, ("mps_approxes_nl", train_set.mps_approxes_nl))
    elif hasattr(train_set, "mps_approxes") and train_set.mps_approxes is not None:
        arrays_to_check.insert(0, ("mps_approxes", train_set.mps_approxes))

    for name, arr in arrays_to_check:
        print(f"  {name}: "
              f"NaNs={np.isnan(arr).sum()}, "
              f"infs={np.isinf(arr).sum()}, "
              f"min={np.nanmin(arr):.4g}, "
              f"max={np.nanmax(arr):.4g}")

    bad_mask    = (np.isnan(train_set.logfracs).any(axis=(1, 2))
                 | np.isinf(train_set.logfracs).any(axis=(1, 2)))
    bad_indices = np.where(bad_mask)[0]

    print(f"\n=== BAD COSMOLOGIES ===")
    print(f"  {bad_mask.sum()} / {train_set.lhs.shape[0]} cosmologies are bad")
    if bad_indices.size > 0:
        print("  First 10 bad indices:", bad_indices[:10])
        i = bad_indices[0]
        print(f"\n  Example (index {i}):")
        print(f"    LHS params:     {train_set.lhs[i]}")
        print(f"    pks_target:     "
              f"min={np.nanmin(train_set.pks_target[i]):.4g}, "
              f"max={np.nanmax(train_set.pks_target[i]):.4g}")
        print(f"    frac_pks <= 0:  {(train_set.frac_pks[i] <= 0).sum()}")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filter(cola_set, om_min=OM_MIN, w0_min=W0_MIN, w0wa_max=W0WA_MAX,
                 omegab_h0_triangle_cut=None):
    om_col     = utils.params.index("Omega_m")
    w0_col     = utils.params.index("w")
    w0wa_col   = utils.params.index("w0+wa")
    omegab_col = utils.params.index("Omega_b")
    h_col      = utils.params.index("h")
    mask       = np.ones(len(cola_set.lhs), dtype=bool)

    if om_min is not None:
        om_mask = cola_set.lhs[:, om_col] >= om_min
        print(f"  Om cut   (Om >= {om_min}): removed {(~om_mask).sum()}")
        mask &= om_mask

    if w0_min is not None:
        w0_mask = cola_set.lhs[:, w0_col] >= w0_min
        print(f"  w0 cut   (w0 >= {w0_min}): removed {(~w0_mask & mask).sum()}")
        mask &= w0_mask

    if w0wa_max is not None:
        w0wa_mask = cola_set.lhs[:, w0wa_col] <= w0wa_max
        print(f"  w0+wa cut (w0+wa <= {w0wa_max}): removed {(~w0wa_mask & mask).sum()}")
        mask &= w0wa_mask

    if omegab_h0_triangle_cut is not None:
        ob_anchor = omegab_h0_triangle_cut['omegab_anchor']
        h0_anchor = omegab_h0_triangle_cut['h0_anchor']
        ob_max    = omegab_h0_triangle_cut['omegab_max']
        h0_max    = omegab_h0_triangle_cut['h0_max']
        omegab_vals = cola_set.lhs[:, omegab_col]
        h0_vals     = cola_set.lhs[:, h_col]
        slope       = (h0_anchor - h0_max) / (ob_max - ob_anchor)
        h0_line     = h0_max + slope * (omegab_vals - ob_anchor)
        tri_mask    = h0_vals <= h0_line
        print(f"  Omega_b/H0 triangle cut: removed {(~tri_mask & mask).sum()}")
        mask &= tri_mask

    n_removed = int((~mask).sum())
    for attr in ("lhs", "pks_target", "frac_pks", "logfracs"):
        if hasattr(cola_set, attr) and getattr(cola_set, attr) is not None:
            setattr(cola_set, attr, getattr(cola_set, attr)[mask])
    if hasattr(cola_set, "mps_approxes_boost") and cola_set.mps_approxes_boost is not None:
        cola_set.mps_approxes_boost = cola_set.mps_approxes_boost[mask]
        cola_set.mps_approxes       = cola_set.mps_approxes_boost
    elif hasattr(cola_set, "mps_approxes") and cola_set.mps_approxes is not None:
        cola_set.mps_approxes = cola_set.mps_approxes[mask]
    if hasattr(cola_set, "lhs_norm") and cola_set.lhs_norm is not None:
        cola_set.lhs_norm = cola_set.lhs_norm[mask]
    return n_removed

def _exclude_outliers(errors, test_set, w0_abs_min=None, extra_mask=None):
    """
    Build a boolean keep-mask for plotting/summary purposes only — does NOT
    modify test_set or re-fit anything. Use this right before passing errors
    into print_error_summary/_spaghetti_ax, so known-pathological cosmologies
    (e.g. w0 near 0, where the symbolic syren approximation is known to break
    down) don't dominate or distort the visualization.

    Parameters
    ----------
    errors      : (N_cosmo, N_k) ndarray — output of pca_reconstruction_errors etc.
    test_set    : COLASet — used to look up parameter values for filtering
    w0_abs_min  : float or None — exclude cosmologies with |w0| < this value
    extra_mask  : array of int or None — explicit cosmology indices to exclude

    Returns
    -------
    errors_filtered : ndarray, keep_mask : boolean ndarray (N_cosmo,)
    """
    keep_mask = np.ones(len(test_set.lhs), dtype=bool)

    if w0_abs_min is not None:
        w0_col = utils.params.index("w")
        w0_vals = test_set.lhs[:, w0_col]
        near_zero_mask = np.abs(w0_vals) < w0_abs_min
        print(f"  Excluding {near_zero_mask.sum()} cosmologies with |w0| < {w0_abs_min} from plots")
        keep_mask &= ~near_zero_mask

    if extra_mask is not None:
        explicit_exclude = np.zeros(len(test_set.lhs), dtype=bool)
        explicit_exclude[extra_mask] = True
        print(f"  Excluding {explicit_exclude.sum()} explicitly-indexed cosmologies from plots")
        keep_mask &= ~explicit_exclude

    return errors[keep_mask], keep_mask

# ---------------------------------------------------------------------------
# PCA reconstruction
# ---------------------------------------------------------------------------

def pca_reconstruction_errors(train_set, test_set, iz):
    """
    Compute per-cosmology PCA reconstruction fractional errors at redshift
    index iz across the full k grid.

        exp(PCA_round_trip(logfracs[:, iz, :])) / frac_pks[:, iz, :] - 1

    where frac_pks = P_nonlin_camb / P_nl_syren.  A perfect reconstruction
    gives 0 everywhere; deviations show where the PCA truncation loses
    information.

    Parameters
    ----------
    train_set : COLASet — fitted scalers and PCAs live here (after prepare())
    test_set  : COLASet — cosmologies to evaluate
    iz        : int     — redshift index

    Returns
    -------
    errors : (N_cosmo, N_k) ndarray
    """
    scaler = train_set.frac_pks_scalers[iz]
    pca    = train_set.pcas[iz]

    normed        = scaler.transform(test_set.logfracs[:, iz, :])
    pcs           = pca.transform(normed)
    reconstructed = scaler.inverse_transform(pca.inverse_transform(pcs))
    return np.exp(reconstructed) / test_set.frac_pks[:, iz, :] - 1


# ---------------------------------------------------------------------------
# tPCA reconstruction
# ---------------------------------------------------------------------------

def tpca_stacks(train_set, test_set):
    """
    Encode all test cosmologies with the per-z PCAs, compress with tPCA,
    then reconstruct back to log-fraction space.

    Returns
    -------
    stacks : (N_cosmo, 1, N_z, N_k) ndarray of reconstructed log-fractions
    """
    n_cosmo = len(test_set.lhs)
    n_z     = len(utils.z_mps)

    all_pcs = []
    
    for i in range(n_cosmo):
        cosmos_pcs = []
        for iz in range(n_z):
            normed = train_set.frac_pks_scalers[iz].transform(
                [test_set.logfracs[i, iz, :]]
            )
            pcs = train_set.pcas[iz].transform(normed)
            cosmos_pcs.append([pcs[0]])
        all_pcs.append(cosmos_pcs)

    all_pcs  = np.transpose(np.array(all_pcs), (0, 2, 1, 3))  # (N, 1, N_z, NUM_PCS)
    pcs_flat = all_pcs.reshape(n_cosmo, n_z * NUM_PCS)

    t_comps   = train_set.tpca.transform(pcs_flat)
    pcs_recon = train_set.tpca.inverse_transform(t_comps)
    pcs_per_z = pcs_recon.reshape(n_cosmo, n_z, NUM_PCS)

    stacks = []
    for pcs_z_stack in pcs_per_z:
        reconstructed_fracs = [
            train_set.frac_pks_scalers[iz].inverse_transform(
                train_set.pcas[iz].inverse_transform(pcs_z.reshape(1, -1))
            )[0]
            for iz, pcs_z in enumerate(pcs_z_stack)
        ]
        stacks.append([np.stack(reconstructed_fracs)])

    return np.array(stacks)   # (N_cosmo, 1, N_z, N_k)

# def umap_stacks(train_set, test_set):
#     """
#     Encode all test cosmologies with the per-z PCAs, compress with UMAP,
#     then reconstruct back to log-fraction space.

#     Returns
#     -------
#     stacks : (N_cosmo, 1, N_z, N_k) ndarray of reconstructed log-fractions
#     """
#     n_cosmo = len(test_set.lhs)
#     n_z     = len(utils.z_mps)

#     all_pcs = []
#     for i in range(n_cosmo):
#         cosmos_pcs = []
#         for iz in range(n_z):
#             normed = train_set.frac_pks_scalers[iz].transform(
#                 [test_set.logfracs[i, iz, :]]
#             )
#             pcs = train_set.pcas[iz].transform(normed)
#             cosmos_pcs.append([pcs[0]])
#         all_pcs.append(cosmos_pcs)

#     all_pcs  = np.transpose(np.array(all_pcs), (0, 2, 1, 3))  # (N, 1, N_z, NUM_PCS)
#     pcs_flat = all_pcs.reshape(n_cosmo, n_z * NUM_PCS)


#     mapped       = train_set.mapper.transform(pcs_flat)
#     mapper_recon = train_set.mapper.inverse_transform(mapped)
#     mapped_pcs_per_z = mapper_recon.reshape(n_cosmo, n_z, NUM_PCS)


#     stacks = []
#     for pcs_z_stack in mapped_pcs_per_z:
#         reconstructed_fracs = [
#             train_set.frac_pks_scalers[iz].inverse_transform(
#                 train_set.pcas[iz].inverse_transform(pcs_z.reshape(1, -1))
#             )[0]
#             for iz, pcs_z in enumerate(pcs_z_stack)
#         ]
#         stacks.append([np.stack(reconstructed_fracs)])

#     return np.array(stacks)   # (N_cosmo, 1, N_z, N_k)


def tpca_reconstruction_errors(stacks, test_set, iz):
    """
    Compute tPCA fractional errors at redshift index iz across the full k grid.

        exp(reconstructed_logfrac[:, iz, :]) / frac_pks[:, iz, :] - 1

    where frac_pks = P_nonlin_camb / P_nl_syren.

    Parameters
    ----------
    stacks   : (N_cosmo, 1, N_z, N_k) — output of tpca_stacks()
    test_set : COLASet
    iz       : int

    Returns
    -------
    errors : (N_cosmo, N_k) ndarray
    """
    return np.exp(stacks[:, 0, iz, :]) / test_set.frac_pks[:, iz, :] - 1

# def umap_reconstruction_errors(stacks, test_set, iz):
#     """
#     Compute UMAP fractional errors at redshift index iz across the full k grid.

#         exp(reconstructed_logfrac[:, iz, :]) / frac_pks[:, iz, :] - 1

#     where frac_pks = P_nonlin_camb / P_nl_syren.

#     Parameters
#     ----------
#     stacks   : (N_cosmo, 1, N_z, N_k) — output of umap_stacks()
#     test_set : COLASet
#     iz       : int

#     Returns
#     -------
#     errors : (N_cosmo, N_k) ndarray
#     """
#     return np.exp(stacks[:, 0, iz, :]) / test_set.frac_pks[:, iz, :] - 1


# ---------------------------------------------------------------------------
# Regime boundary visualisation
# ---------------------------------------------------------------------------

def compute_regime_index(cola_set, iz=0):
    """
    Compute a scalar shape index for each cosmology that characterises the
    amplitude and curvature of the residual logfrac at redshift iz.

    With the syren halofit baseline the quantity is
    log(P_nl_camb / P_nl_syren), which is centred near 0 rather than being
    monotonically positive like the old NL/LIN boost.  The v_score is
    therefore more symmetric and characterises deviations from the syren
    halofit prediction, driven by cosmological parameters that syren does
    not capture perfectly (e.g. massive neutrinos, extreme w0/wa).

      - shape_index   : std of logfrac across k — large = syren misfit is k-dependent
      - low_k_excess  : mean logfrac at the 10 lowest k-modes
      - high_k_excess : mean logfrac at the 10 highest k-modes
      - v_score       : high-k excess minus mid-range (captures residual small-scale
                        power that syren halofit over- or under-predicts)

    Returns
    -------
    shape_index, low_k_excess, high_k_excess, v_score : each (N_cosmo,)
    """
    logf = cola_set.logfracs[:, iz, :]   # (N_cosmo, N_k)

    shape_index   = logf.std(axis=1)
    low_k_excess  = logf[:, :10].mean(axis=1)
    high_k_excess = logf[:, -10:].mean(axis=1)

    n_k    = logf.shape[1]
    q1, q3 = n_k // 4, 3 * n_k // 4
    centre = logf[:, q1:q3].mean(axis=1)

    v_score = high_k_excess - centre

    return shape_index, low_k_excess, high_k_excess, v_score


def _scatter_with_density_contour(ax, x, y, c, cmap, norm, xlabel, ylabel, title,
                                   nbins=40):
    sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=6, alpha=0.6,
                    linewidths=0, rasterized=True)
    H, xe, ye = np.histogram2d(x, y, bins=nbins)
    H_smooth  = gaussian_filter(H, sigma=1.2)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    levels = np.percentile(H_smooth[H_smooth > 0], [50, 80, 95])
    if len(np.unique(levels)) > 1:
        ax.contour(xc, yc, H_smooth.T, levels=levels,
                   colors="white", linewidths=0.5, alpha=0.4)
    ax.set_xlabel(xlabel, fontsize=AXES_FS)
    ax.set_ylabel(ylabel, fontsize=AXES_FS)
    ax.set_title(title,   fontsize=AXES_FS)
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    return sc


def visualize_regime_boundary(train_set, iz=0, threshold_pct=75):
    """
    Produce a multi-panel figure revealing where the high-residual /
    low-residual regime boundary sits in cosmological parameter space.

    The quantity plotted is log(P_nl_camb / P_nl_syren): where syren halofit
    is a good approximation this is near zero; deviations indicate
    cosmologies where the residual network has the most work to do.

    Panel layout
    ------------
    Row 0  : histogram of the v_score with threshold line, plus example
             log(P_nl_camb / P_nl_syren) curves for the lowest and highest
             v_score cosmologies.
    Row 1-2: 2-D scatter plots coloured by v_score for key parameter pairs.

    Parameters
    ----------
    train_set    : COLASet
    iz           : int   — redshift index (default 0)
    threshold_pct: float — percentile separating low from high residual

    Saves
    -----
    FIG_DIR/regime_boundary_<tag>.pdf
    """
    shape_index, low_k_excess, high_k_excess, v_score = compute_regime_index(
        train_set, iz)
    threshold  = np.percentile(v_score, threshold_pct)
    v_mask     = v_score >= threshold
    flat_mask  = ~v_mask

    print(f"\n=== REGIME BOUNDARY SUMMARY (z_idx={iz}, threshold={threshold_pct}th pct) ===")
    print(f"  v_score range:       {v_score.min():.4f} -> {v_score.max():.4f}")
    print(f"  Threshold:           {threshold:.4f}")
    print(f"  Low-residual regime: {flat_mask.sum()} cosmologies "
          f"({100*flat_mask.mean():.1f}%)")
    print(f"  High-residual regime:{v_mask.sum()} cosmologies "
          f"({100*v_mask.mean():.1f}%)")
    print()
    for pi, name in enumerate(utils.params):
        low_range  = (train_set.lhs[flat_mask, pi].min(),
                      train_set.lhs[flat_mask, pi].max())
        high_range = (train_set.lhs[v_mask,    pi].min(),
                      train_set.lhs[v_mask,    pi].max())
        print(f"  {name:8s}:  low-res [{low_range[0]:.3g}, {low_range[1]:.3g}]"
              f"   high-res [{high_range[0]:.3g}, {high_range[1]:.3g}]")

    proj_pairs = [
        (5, 6, "w_0",     "w_0+w_a"),
        (5, 2, "w_0",     r"\Omega_m"),
        (6, 2, "w_0+w_a", r"\Omega_m"),
        (4, 0, "n_s",     "h"),
        (3, 2, r"\Omega_b", "H_0"),
        # (8, 7, r"m_\nu",  r"\log T_{\rm AGN}"),
    ]

    n_rows = 2 + len(proj_pairs) // 2
    fig    = plt.figure(figsize=(14, 4 * n_rows))
    gs     = fig.add_gridspec(n_rows, 2, hspace=0.45, wspace=0.35)

    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=np.percentile(v_score, 2),
                     vmax=np.percentile(v_score, 98))

    # -- Row 0, left: v_score histogram -----------------------------------
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(v_score[flat_mask], bins=60, color="#4c78a8", alpha=0.7,
                 label=f"low-residual ({flat_mask.sum()})")
    ax_hist.hist(v_score[v_mask],    bins=60, color="#e45756", alpha=0.7,
                 label=f"high-residual ({v_mask.sum()})")
    ax_hist.axvline(threshold, color="k", lw=1.5, ls="--",
                    label=f"threshold ({threshold_pct}th pct)")
    ax_hist.set_xlabel(
        r"high-k excess $-$ mid-range residual  [residual score]",
        fontsize=AXES_FS - 1)
    ax_hist.set_ylabel("Count", fontsize=AXES_FS)
    ax_hist.set_title(
        f"Regime distribution ({NL_TYPE} / syren residual score)",
        fontsize=AXES_FS)
    ax_hist.legend(fontsize=TICK_FS)
    ax_hist.tick_params(labelsize=TICK_FS)

    # -- Row 0, right: example logfrac curves -----------------------------
    ax_ex = fig.add_subplot(gs[0, 1])
    low_top5  = np.argsort(v_score)[:5]
    high_top5 = np.argsort(v_score)[-5:]

    for idx in low_top5:
        ax_ex.semilogx(train_set.ks,
                        train_set.logfracs[idx, iz, :],
                        color="#4c78a8", lw=0.9, alpha=0.8)
    for idx in high_top5:
        ax_ex.semilogx(train_set.ks,
                        train_set.logfracs[idx, iz, :],
                        color="#e45756", lw=0.9, alpha=0.8)

    ax_ex.plot([], [], color="#4c78a8", lw=1.5, label="5 lowest residual")
    ax_ex.plot([], [], color="#e45756", lw=1.5, label="5 highest residual")
    ax_ex.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    ax_ex.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax_ex.set_ylabel(
        fr"$\log\!\left(P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{nl,syren}}\right)$",
        fontsize=AXES_FS)
    ax_ex.set_title(
        f"Example residual logfrac curves at z={train_set.z[iz]:.2f}",
        fontsize=AXES_FS)
    ax_ex.legend(fontsize=TICK_FS)
    ax_ex.tick_params(labelsize=TICK_FS)
    ax_ex.grid(alpha=0.3)

    # -- Rows 1+: 2-D projections -----------------------------------------
    lhs = train_set.lhs
    for k, (pi, pj, xi_label, xj_label) in enumerate(proj_pairs):
        row = 1 + k // 2
        col = k  % 2
        ax  = fig.add_subplot(gs[row, col])
        _scatter_with_density_contour(
            ax,
            x=lhs[:, pi], y=lhs[:, pj],
            c=v_score,
            cmap=cmap, norm=norm,
            xlabel=f"${xi_label}$",
            ylabel=f"${xj_label}$",
            title=f"residual score: ${xi_label}$ vs ${xj_label}$",
        )

    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.55])
    sm      = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(
        r"high-k residual score  (higher = larger CAMB$-$syren discrepancy)",
        fontsize=TICK_FS)
    cb.ax.tick_params(labelsize=TICK_FS)

    fig.suptitle(
        f"Regime boundary -- {COSMO_TYPE} / {NL_TYPE} vs syren halofit / {PRIOR_TYPE}\n"
        f"({N_BATCHES} batches, z_idx={iz}, threshold={threshold_pct}th percentile)",
        fontsize=AXES_FS + 1, y=1.01
    )
    plt.tight_layout(rect=[0, 0, 0.91, 1])

    fname = f"{FIG_DIR}/regime_boundary_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {fname}")
    return v_score, threshold


# ---------------------------------------------------------------------------
# Scree plots
# ---------------------------------------------------------------------------

def plot_scree(train_set):
    """
    Save a two-panel scree plot showing explained variance for the spatial
    PCA (left) and the tPCA (right).

    Must be called after train_set.prepare() so that train_set.pcas and
    train_set.tpca are available.

    Left panel — spatial PCA
    ~~~~~~~~~~~~~~~~~~~~~~~~
    Each redshift has its own PCA fitted on log(P_nl_camb / P_nl_syren).
    All redshifts are drawn as thin grey lines; the indices in
    SCREE_Z_HIGHLIGHT are overlaid as coloured lines with z-value labels.
    The cumulative explained variance (averaged across all redshifts) is
    shown on the right y-axis.

    Right panel — tPCA
    ~~~~~~~~~~~~~~~~~~
    The tPCA compresses the stacked spatial-PC vectors across redshifts.
    Individual and cumulative explained variance are plotted, with the
    NUM_PCS_Z cut-off marked by a vertical dashed line.

    Both panels mark the 95% and 99% cumulative variance thresholds as
    horizontal dotted lines for quick reference.

    Parameters
    ----------
    train_set : COLASet — must have .pcas and .tpca set by prepare()

    Saves
    -----
    FIG_DIR/scree_<tag>.pdf
    """
    n_z = len(utils.z_mps)

    # -- Collect explained variance ratios --------------------------------
    spatial_evr   = np.array([train_set.pcas[iz].explained_variance_ratio_
                               for iz in range(n_z)])
    spatial_cumev = np.cumsum(spatial_evr, axis=1)
    mean_cumev    = spatial_cumev.mean(axis=0)

    tpca_evr   = train_set.tpca.explained_variance_ratio_
    tpca_cumev = np.cumsum(tpca_evr)

    n_spatial  = spatial_evr.shape[1]
    n_tpca     = len(tpca_evr)
    xs_spatial = np.arange(1, n_spatial + 1)
    xs_tpca    = np.arange(1, n_tpca    + 1)

    highlight_colors = plt.get_cmap("plasma")(
        np.linspace(0.1, 0.9, len(SCREE_Z_HIGHLIGHT))
    )
    highlight_set = set(SCREE_Z_HIGHLIGHT)

    fig, (ax_sp, ax_tp) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left: spatial PCA ------------------------------------------------
    ax_sp2 = ax_sp.twinx()

    for iz in range(n_z):
        if iz not in highlight_set:
            ax_sp.semilogy(xs_spatial, spatial_evr[iz],
                       color="0.75", lw=0.5, alpha=0.6)

    for col, iz in zip(highlight_colors, SCREE_Z_HIGHLIGHT):
        iz = min(iz, n_z - 1)
        z_val = utils.z_mps[iz]
        ax_sp.semilogy(xs_spatial, spatial_evr[iz],
                   color=col, lw=1.8,
                   label=fr"$z={z_val:.2f}$ (iz={iz})")

    ax_sp2.semilogy(xs_spatial, mean_cumev,
                color="black", lw=1.5, ls="-.", alpha=0.8,
                label="mean cumulative")
    for thresh, ls in [(0.95, ":"), (0.99, "--")]:
        ax_sp2.axhline(thresh, color="black", lw=0.8, ls=ls, alpha=0.5)
        ax_sp2.text(n_spatial * 0.98, thresh + 0.005,
                    f"{int(thresh*100)}%",
                    ha="right", va="bottom", fontsize=TICK_FS - 1, color="black")

    ax_sp.axvline(NUM_PCS, color="red", lw=1.2, ls="--", alpha=0.8,
                  label=f"NUM_PCS = {NUM_PCS}")

    ax_sp.set_xlabel("Principal component index", fontsize=AXES_FS)
    ax_sp.set_ylabel("Individual explained variance ratio", fontsize=AXES_FS)
    ax_sp2.set_ylabel("Cumulative explained variance (mean over z)",
                      fontsize=AXES_FS - 1)
    ax_sp.set_title(
        fr"Spatial PCA scree -- $\log(P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{nl,syren}})$",
        fontsize=AXES_FS)
    ax_sp.set_xlim(1, n_spatial)
    ax_sp2.set_ylim(0, 1.05)
    ax_sp.tick_params(labelsize=TICK_FS)
    ax_sp2.tick_params(labelsize=TICK_FS)

    handles1, labels1 = ax_sp.get_legend_handles_labels()
    handles2, labels2 = ax_sp2.get_legend_handles_labels()
    ax_sp.legend(handles1 + handles2, labels1 + labels2,
                 fontsize=TICK_FS - 1, loc="upper right")
    ax_sp.grid(alpha=0.3)

    # -- Right: tPCA ------------------------------------------------------
    ax_tp2 = ax_tp.twinx()

    ax_tp.bar(xs_tpca, tpca_evr,
              color="#4c78a8", alpha=0.7, width=0.8, label="individual")
    ax_tp.set_yscale('log')
    ax_tp2.plot(xs_tpca, tpca_cumev,
                color="black", lw=1.5, ls="-.", alpha=0.8, label="cumulative")

    for thresh, ls in [(0.95, ":"), (0.99, "--")]:
        ax_tp2.axhline(thresh, color="black", lw=0.8, ls=ls, alpha=0.5)
        ax_tp2.text(n_tpca * 0.98, thresh + 0.005,
                    f"{int(thresh*100)}%",
                    ha="right", va="bottom", fontsize=TICK_FS - 1, color="black")

    ax_tp.axvline(NUM_PCS_Z, color="red", lw=1.2, ls="--", alpha=0.8,
                  label=f"NUM_PCS_Z = {NUM_PCS_Z}")

    ax_tp.set_xlabel("Temporal component index", fontsize=AXES_FS)
    ax_tp.set_ylabel("Individual explained variance ratio", fontsize=AXES_FS)
    ax_tp2.set_ylabel("Cumulative explained variance", fontsize=AXES_FS - 1)
    ax_tp.set_title("tPCA scree -- stacked spatial PCs across redshifts",
                    fontsize=AXES_FS)
    ax_tp.set_xlim(0.5, n_tpca + 0.5)
    ax_tp2.set_ylim(0, 1.05)
    ax_tp2.set_yscale('log')
    ax_tp.tick_params(labelsize=TICK_FS)
    ax_tp2.tick_params(labelsize=TICK_FS)

    handles1, labels1 = ax_tp.get_legend_handles_labels()
    handles2, labels2 = ax_tp2.get_legend_handles_labels()
    ax_tp.legend(handles1 + handles2, labels1 + labels2,
                 fontsize=TICK_FS - 1, loc="center right")
    ax_tp.grid(alpha=0.3)

    fig.suptitle(
        f"Scree plots -- {COSMO_TYPE} / {NL_TYPE} vs syren halofit / {PRIOR_TYPE}  "
        f"(full k grid, 500 modes)",
        fontsize=AXES_FS + 1
    )
    plt.tight_layout()

    fname = f"{FIG_DIR}/scree_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _tag():
    """Shared filename tag built from the run configuration."""
    filter_parts = []
    if W0_MIN   is not None: filter_parts.append(f"w0min{W0_MIN}")
    if W0WA_MAX is not None: filter_parts.append(f"w0wamax{W0WA_MAX}")
    if OM_MIN   is not None: filter_parts.append(f"ommin{OM_MIN}")
    if OMEGAB_H0_TRIANGLE_CUT is not None:
        filter_parts.append(f"obh0tri{OMEGAB_ANCHOR}_{H0_ANCHOR}")
    filter_str = ("_" + "_".join(filter_parts)) if filter_parts else ""
    return (f"{COSMO_TYPE}_{NL_TYPE}_{PRIOR_TYPE}{filter_str}"
            f"_n{NUM_PCS}_z{NUM_PCS_Z}_v{VER}")


def _spaghetti_ax(ax, ks, errors, ylabel, title):
    """Draw one spaghetti error curve per cosmology on ax."""
    for error in errors:
        ax.semilogx(ks, error, lw=0.6, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(ylabel, fontsize=AXES_FS)
    ax.set_title(title, fontsize=AXES_FS)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.grid(alpha=0.3)


def plot_pca_errors(errors, ks, iz_label):
    """Save PCA reconstruction error spaghetti plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _spaghetti_ax(
        ax, ks, errors,
        ylabel=(fr"$(P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{nl,syren}}) - 1$"),
        title=(fr"PCA Reconstruction Errors -- "
               fr"$\log(P_{{{NL_TYPE}}}^\mathrm{{CAMB}}/P_\mathrm{{nl,syren}})$"
               + fr" at $z\approx{iz_label}$, $N_\mathrm{{PC}}={NUM_PCS}$"),
    )
    plt.tight_layout()
    fname = f"{FIG_DIR}/pca_errors_z{iz_label}_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_tpca_errors(errors, ks, iz_label):
    """Save tPCA reconstruction error spaghetti plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _spaghetti_ax(
        ax, ks, errors,
        ylabel=(fr"$(P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{nl,syren}}) - 1$"),
        title=(fr"tPCA Reconstruction Errors -- "
               fr"$\log(P_{{{NL_TYPE}}}^\mathrm{{CAMB}}/P_\mathrm{{nl,syren}})$"
               + fr" at $z\approx{iz_label}$,"
               + fr" $N_\mathrm{{PC}}={NUM_PCS}$, $N_\mathrm{{tcomp}}={NUM_PCS_Z}$"),
    )
    plt.tight_layout()
    fname = f"{FIG_DIR}/tpca_errors_z{iz_label}_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")

# def plot_umap_errors(errors, ks, iz_label):
#     """Save tPCA reconstruction error spaghetti plot."""
#     fig, ax = plt.subplots(figsize=(8, 5))
#     _spaghetti_ax(
#         ax, ks, errors,
#         ylabel=(fr"$(P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{nl,syren}}) - 1$"),
#         title=(fr"tPCA Reconstruction Errors -- "
#                fr"$\log(P_{{{NL_TYPE}}}^\mathrm{{CAMB}}/P_\mathrm{{nl,syren}})$"
#                + fr" at $z\approx{iz_label}$,"
#                + fr" $N_\mathrm{{PC}}={NUM_PCS}$, $N_\mathrm{{tcomp}}={NUM_PCS_Z}$"),
#     )
#     plt.tight_layout()
#     fname = f"{FIG_DIR}/umap_errors_z{iz_label}_{_tag()}.pdf"
#     plt.savefig(fname, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {fname}")

from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator

# def fit_mean_logfrac_envelope(cola_set, iz, n_tagn_bins=20):
#     """
#     Fit a smooth T_AGN-dependent mean logfrac envelope across k at redshift iz.

#     The idea: the baryonic feedback signal (exponential rise at high k driven
#     by T_AGN) is shared across cosmologies in a structured way. Subtracting
#     a smooth fit to the mean logfrac as a function of (k, T_AGN) before PCA
#     leaves residuals that are smaller and more compact — the PCA can then
#     use its components for genuine cosmology-to-cosmology variation rather
#     than representing the shared feedback envelope.

#     Parameters
#     ----------
#     cola_set : COLASet — must have lhs[:, 7] = T_AGN (9-column dataset)
#     iz       : int     — redshift index
#     n_tagn_bins : int  — number of T_AGN bins for the envelope grid

#     Returns
#     -------
#     envelope_fn   : callable(tagn, k) → logfrac_envelope, shape matching inputs
#     tagn_grid     : (n_tagn_bins,) T_AGN values used for the grid
#     mean_logfracs : (n_tagn_bins, N_k) mean logfrac in each T_AGN bin
#     """
#     if cola_set.lhs.shape[1] <= 7:
#         raise ValueError("fit_mean_logfrac_envelope requires 9-column lhs (T_AGN at col 7)")

#     tagn_col   = 7   # T_AGN column index in lhs (after w0+wa overwrite)
#     tagn_vals  = cola_set.lhs[:, tagn_col]
#     logfracs_z = cola_set.logfracs[:, iz, :]   # (N_cosmo, N_k)
#     ks         = cola_set.ks

#     # Bin by T_AGN and compute mean logfrac in each bin
#     tagn_edges = np.linspace(tagn_vals.min(), tagn_vals.max(), n_tagn_bins + 1)
#     tagn_grid  = 0.5 * (tagn_edges[:-1] + tagn_edges[1:])
#     mean_logfracs = np.zeros((n_tagn_bins, len(ks)))

#     for b in range(n_tagn_bins):
#         in_bin = (tagn_vals >= tagn_edges[b]) & (tagn_vals < tagn_edges[b + 1])
#         if in_bin.sum() == 0:
#             # Empty bin — interpolate later
#             mean_logfracs[b] = np.nan
#         else:
#             # Use median rather than mean to be robust to outliers
#             mean_logfracs[b] = np.median(logfracs_z[in_bin], axis=0)

#     # Fill any empty bins by linear interpolation across T_AGN
#     for k_idx in range(len(ks)):
#         col = mean_logfracs[:, k_idx]
#         nan_mask = np.isnan(col)
#         if nan_mask.any() and (~nan_mask).sum() >= 2:
#             mean_logfracs[nan_mask, k_idx] = np.interp(
#                 tagn_grid[nan_mask], tagn_grid[~nan_mask], col[~nan_mask]
#             )

#     # Build a 2D interpolator: envelope_fn(tagn_scalar, k_array) → logfrac_array
#     # Use RegularGridInterpolator for fast vectorized evaluation at inference
#     envelope_fn = RegularGridInterpolator(
#         (tagn_grid, ks),
#         mean_logfracs,
#         method='linear',
#         bounds_error=False,
#         fill_value=None,   # extrapolate at edges
#     )

#     return envelope_fn, tagn_grid, mean_logfracs


# def plot_logfrac_envelope_diagnostic(cola_set, ks, iz, iz_label,
#                                       envelope_fn, tagn_grid, mean_logfracs):
#     """
#     Diagnostic: show the fitted T_AGN envelope and the residuals after subtracting it.
#     Two rows:
#       Top:    mean logfrac per T_AGN bin (the envelope being subtracted)
#       Bottom: logfrac residuals after envelope subtraction for a random subset
#     """
#     if cola_set.lhs.shape[1] <= 7:
#         print("  [SKIP] plot_logfrac_envelope_diagnostic requires 9-column lhs")
#         return

#     tagn_col   = 7
#     tagn_vals  = cola_set.lhs[:, tagn_col]
#     logfracs_z = cola_set.logfracs[:, iz, :]

#     # Compute residuals for all cosmologies
#     query_pts = np.column_stack([tagn_vals,
#                                   np.zeros(len(tagn_vals))])   # placeholder k
#     residuals = np.empty_like(logfracs_z)
#     for i, (tagn_i, logfrac_i) in enumerate(zip(tagn_vals, logfracs_z)):
#         pts = np.column_stack([np.full(len(ks), tagn_i), ks])
#         envelope_i = envelope_fn(pts)
#         residuals[i] = logfrac_i - envelope_i

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     # Left: the envelope itself, coloured by T_AGN
#     ax = axes[0]
#     cmap_e = plt.get_cmap("plasma")
#     norm_e = Normalize(vmin=tagn_grid.min(), vmax=tagn_grid.max())
#     for b, tagn_b in enumerate(tagn_grid):
#         color = cmap_e(norm_e(tagn_b))
#         ax.semilogx(ks, mean_logfracs[b], color=color, lw=1.2)
#     ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
#     sm_e = plt.cm.ScalarMappable(cmap=cmap_e, norm=norm_e)
#     sm_e.set_array([])
#     plt.colorbar(sm_e, ax=ax, label=r"$\log T_{\rm AGN}$")
#     ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax.set_ylabel(r"Mean $\log(P_{\rm CAMB}/P_{\rm syren})$", fontsize=AXES_FS)
#     ax.set_title(fr"Fitted $T_{{AGN}}$ envelope at $z\approx{iz_label}$", fontsize=AXES_FS)
#     ax.grid(alpha=0.3)
#     ax.tick_params(labelsize=TICK_FS)

#     # Right: residuals after subtracting the envelope, random subset
#     ax2 = axes[1]
#     rng = np.random.default_rng(seed=42)
#     sub_idx = rng.choice(len(cola_set.lhs), size=min(100, len(cola_set.lhs)), replace=False)
#     w0wa_col = utils.params.index("w0+wa")
#     w0wa_vals = cola_set.lhs[sub_idx, w0wa_col]
#     cmap_r = plt.get_cmap("coolwarm")
#     norm_r = Normalize(vmin=np.percentile(w0wa_vals, 2),
#                        vmax=np.percentile(w0wa_vals, 98))
#     for i, (idx_i, wi) in enumerate(zip(sub_idx, w0wa_vals)):
#         color = cmap_r(norm_r(wi))
#         ax2.semilogx(ks, residuals[idx_i], color=color, lw=0.5, alpha=0.5, rasterized=True)
#     ax2.axhline(0, color="k", lw=0.8, ls="--")

#     # Print residual stats vs original logfrac stats
#     orig_std  = np.abs(logfracs_z).max(axis=1).mean()
#     resid_std = np.abs(residuals).max(axis=1).mean()
#     print(f"\n  Envelope subtraction at z~{iz_label}:")
#     print(f"    Mean max |logfrac| before: {orig_std:.4f}")
#     print(f"    Mean max |residual| after: {resid_std:.4f}  "
#           f"({100*(1 - resid_std/orig_std):.1f}% reduction)")

#     sm_r = plt.cm.ScalarMappable(cmap=cmap_r, norm=norm_r)
#     sm_r.set_array([])
#     plt.colorbar(sm_r, ax=ax2, label=r"$w_0 + w_a$")
#     ax2.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax2.set_ylabel(r"Residual logfrac after envelope subtraction", fontsize=AXES_FS)
#     ax2.set_title(fr"Residuals at $z\approx{iz_label}$ (coloured by $w_0+w_a$)",
#                   fontsize=AXES_FS)
#     ax2.grid(alpha=0.3)
#     ax2.tick_params(labelsize=TICK_FS)

#     plt.suptitle(
#         f"T_AGN envelope correction — {COSMO_TYPE} / {NL_TYPE} / {PRIOR_TYPE}\n"
#         f"({N_BATCHES} batches, z_idx={iz})",
#         fontsize=AXES_FS, y=1.01
#     )
#     plt.tight_layout()
#     fname = (f"{FIG_DIR}/envelope_diagnostic_z{iz_label}_"
#              f"{COSMO_TYPE}_{NL_TYPE}_{PRIOR_TYPE}_v{VER}.pdf")
#     plt.savefig(fname, bbox_inches="tight", dpi=150)
#     plt.close()
#     print(f"  Saved: {fname}")

def plot_syren_vs_camb(cola_set, ks, iz, iz_label, n_cosmo=100, label="precut"):
    """
    For a random subset of cosmologies, plot the syren baseline accuracy:
      - Left:  CAMB / syren ratio (what the network multiplies against)
      - Right: log(CAMB / syren) = logfracs (what the network directly learns)

    Coloured by w0+wa so problem regions of parameter space are immediately
    visible. Run on the raw loaded set (before cuts) to see full prior coverage.
    Printed stats tell you whether errors are a syren-quality problem or a
    network problem.
    """
    
    w0wa_col = utils.params.index("w0+wa")

    n_total = len(cola_set.lhs)
    rng = np.random.default_rng(seed=42)
    idx = rng.choice(n_total, size=min(n_cosmo, n_total), replace=False)

    frac_z    = cola_set.frac_pks[idx, iz, :]    # (n_cosmo, N_k)  CAMB/syren
    logfrac_z = cola_set.logfracs[idx, iz, :]    # (n_cosmo, N_k)  log version
    w0wa_vals = cola_set.lhs[idx, w0wa_col]

    # Only plot/colour by finite values — don't let one blowup dominate the colourbar
    finite_per_cosmo = np.isfinite(logfrac_z).all(axis=1)
    n_nonfinite = (~finite_per_cosmo).sum()
    if n_nonfinite > 0:
        print(f"  [WARN] {n_nonfinite}/{len(idx)} sampled cosmologies have non-finite "
              f"logfracs at iz={iz} — excluded from plot but counted in stats.")

    cmap = plt.get_cmap("coolwarm")
    w0wa_finite = w0wa_vals[finite_per_cosmo]
    norm = Normalize(vmin=np.percentile(w0wa_finite, 2),
                     vmax=np.percentile(w0wa_finite, 98))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: CAMB / syren ratio ----------------------------------------
    ax = axes[0]
    for i, (fi, wi) in enumerate(zip(finite_per_cosmo, w0wa_vals)):
        if not fi:
            continue
        color = cmap(norm(wi))
        ax.semilogx(ks, frac_z[i], color=color, lw=0.5, alpha=0.5, rasterized=True)
    ax.axhline(1.0, color="black", lw=1.2, ls="--", label="Perfect syren")

    # Reference bands
    for level, ref_label, ls in [(1.01, "±1%", ":"), (1.05, "±5%", "--"),
                                  (0.99, "", ":"),     (0.95, "", "--")]:
        ax.axhline(level, color="gray", lw=0.8, ls=ls, alpha=0.5)
    ax.text(ks[1], 1.015, "1%",  fontsize=9, color="gray")
    ax.text(ks[1], 1.055, "5%",  fontsize=9, color="gray")
    ax.text(ks[1], 0.975, "-1%", fontsize=9, color="gray")
    ax.text(ks[1], 0.935, "-5%", fontsize=9, color="gray")

    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(fr"$P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{syren}}$",
                  fontsize=AXES_FS)
    ax.set_title(fr"Syren baseline ratio at $z\approx{iz_label}$", fontsize=AXES_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(alpha=0.3)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb0 = plt.colorbar(sm, ax=ax)
    cb0.set_label(r"$w_0 + w_a$", fontsize=TICK_FS)

    # --- Right: log(CAMB / syren) = logfracs ----------------------------
    ax2 = axes[1]
    for i, (fi, wi) in enumerate(zip(finite_per_cosmo, w0wa_vals)):
        if not fi:
            continue
        color = cmap(norm(wi))
        ax2.semilogx(ks, logfrac_z[i], color=color, lw=0.5, alpha=0.5, rasterized=True)
    ax2.axhline(0.0, color="black", lw=1.2, ls="--", label="Perfect syren")

    for level, ref_label, ls in [(np.log(1.01), "1%",  ":"),
                                  (np.log(1.05), "5%",  "--"),
                                  (np.log(0.99), "-1%", ":"),
                                  (np.log(0.95), "-5%", "--")]:
        ax2.axhline(level, color="gray", lw=0.8, ls=ls, alpha=0.5)
        ax2.text(ks[1], level * 1.05, ref_label, fontsize=9, color="gray")

    ax2.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax2.set_ylabel(
        fr"$\log\!\left(P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{syren}}\right)$",
        fontsize=AXES_FS)
    ax2.set_title(fr"Network learning target at $z\approx{iz_label}$", fontsize=AXES_FS)
    ax2.tick_params(labelsize=TICK_FS)
    ax2.grid(alpha=0.3)

    cb2 = plt.colorbar(sm, ax=ax2)
    cb2.set_label(r"$w_0 + w_a$", fontsize=TICK_FS)

    # --- Terminal stats --------------------------------------------------
    max_logfrac = np.abs(logfrac_z[finite_per_cosmo]).max(axis=1)
    lhs_finite  = cola_set.lhs[idx][finite_per_cosmo]
    w0wa_finite = w0wa_vals[finite_per_cosmo]

    print(f"\n  Syren baseline accuracy at z~{iz_label} "
          f"(|log(CAMB/syren)|, {finite_per_cosmo.sum()} finite cosmologies):")
    print(f"    Median max |logfrac|:   {np.median(max_logfrac):.4f}  "
          f"≈ {100*(np.exp(np.median(max_logfrac))-1):.2f}% deviation")
    print(f"    95th pct max |logfrac|: {np.percentile(max_logfrac, 95):.4f}  "
          f"≈ {100*(np.exp(np.percentile(max_logfrac, 95))-1):.2f}% deviation")
    print(f"    99th pct max |logfrac|: {np.percentile(max_logfrac, 99):.4f}  "
          f"≈ {100*(np.exp(np.percentile(max_logfrac, 99))-1):.2f}% deviation")
    print(f"    Worst |logfrac|:        {max_logfrac.max():.4f}  "
          f"≈ {100*(np.exp(max_logfrac.max())-1):.2f}% deviation")

    # --- Top 20 worst cosmologies ----------------------------------------
    n_worst = min(20, len(max_logfrac))
    worst_order = np.argsort(max_logfrac)[::-1][:n_worst]
    param_names = (utils.params_tfree_mnufree
                   if cola_set.lhs.shape[1] > 7
                   else utils.params)

    print(f"\n  Top {n_worst} worst cosmologies by max |log(CAMB/syren)| at z~{iz_label}:")
    header = f"  {'rank':>4}  {'max|logfrac|':>13}  {'~% dev':>8}  " + \
             "  ".join(f"{n:>10}" for n in param_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rank, wi in enumerate(worst_order, start=1):
        err_val  = max_logfrac[wi]
        pct_dev  = 100 * (np.exp(err_val) - 1)
        params   = lhs_finite[wi]
        row = f"  {rank:>4}  {err_val:>13.4f}  {pct_dev:>7.1f}%  " + \
              "  ".join(f"{v:>10.4g}" for v in params)
        print(row)
    
    cut_desc = "pre-cut (full prior)" if label == "precut" else f"post-cut ({label})"
    plt.suptitle(
        f"Syren baseline accuracy ({cut_desc}) — {COSMO_TYPE} / {NL_TYPE} / {PRIOR_TYPE}\n"
        f"({N_BATCHES} batches, z_idx={iz}, n={len(idx)} sampled cosmologies, "
        f"coloured by $w_0+w_a$)",
        fontsize=AXES_FS, y=1.01
    )
    plt.tight_layout()

    fname = (f"{FIG_DIR}/syren_accuracy_{label}_z{iz_label}_"
             f"{COSMO_TYPE}_{NL_TYPE}_{PRIOR_TYPE}"
             f"_n{N_BATCHES}batches_v{VER}.pdf")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_error_summary(errors, stage, iz_label):
    max_abs = np.max(np.abs(errors), axis=1)
    print(f"  {stage} (z~{iz_label}): "
          f"mean max |err| = {max_abs.mean():.5f}, "
          f"median = {np.median(max_abs):.5f}, "
          f"95th pct = {np.percentile(max_abs, 95):.5f}, "
          f"99th pct = {np.percentile(max_abs, 99):.5f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family']      = 'STIXGeneral'

    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 60)
    print("[INFO] Nonlinear Boost PCA / tPCA Reconstruction Validation")
    print(f"       cosmo_type  = {COSMO_TYPE}")
    print(f"       prior_type  = {PRIOR_TYPE}")
    print(f"       nl_type     = {NL_TYPE}")
    print(f"       baseline    = syren halofit (P_nl_camb / P_nl_syren)")
    print(f"       n_batches   = {N_BATCHES}  (starting at {START_BATCH})")
    print(f"       test_batch  = {TEST_BATCH}")
    print(f"       num_pcs     = {NUM_PCS}")
    print(f"       num_pcs_z   = {NUM_PCS_Z}")
    print(f"       Omega_b/H0 cut = anchor ({OMEGAB_ANCHOR}, {H0_ANCHOR}), "
          f"max ({utils.OMEGA_B_MAX}, {utils.H0_MAX})")
    print(f"       k grid      = {len(utils.ks)} modes")
    print("=" * 60)

    # --- Load training set ---
    print("\n[INFO] Loading training set...")
    train_set = utils.COLASet(
        target_z    = utils.z_mps,
        cosmo_type  = COSMO_TYPE,
        prior_type  = PRIOR_TYPE,
        nl_type     = NL_TYPE,
        n_batches   = N_BATCHES,
        start_batch = START_BATCH,
    )

    print_data_diagnostics(train_set)

    print("\n[INFO] Plotting syren baseline accuracy...")
    plot_syren_vs_camb(train_set, ks=train_set.ks, iz=Z_IDX_0, iz_label=0)
    plot_syren_vs_camb(train_set, ks=train_set.ks, iz=Z_IDX_3, iz_label=3)

    # --- Prior cuts (must happen before prepare() fits the PCA) ---
    n_removed = apply_filter(train_set, om_min=None, w0_min=None, w0wa_max=None,
                            omegab_h0_triangle_cut=OMEGAB_H0_TRIANGLE_CUT)
    print(f"\n[INFO] Prior cuts: removed {n_removed} cosmologies "
          f"({len(train_set.lhs)} remaining).")

    print("\n[INFO] Plotting syren baseline accuracy (post-cut)...")
    plot_syren_vs_camb(train_set, ks=train_set.ks, iz=Z_IDX_0, iz_label=0, label=_tag())
    plot_syren_vs_camb(train_set, ks=train_set.ks, iz=Z_IDX_3, iz_label=3, label=_tag())

    # if train_set.lhs.shape[1] > 7:
    #     print("\n[INFO] Fitting T_AGN logfrac envelope...")
    #     envelope_fn_z0, tagn_grid, mean_lf_z0 = fit_mean_logfrac_envelope(
    #         train_set, iz=Z_IDX_0)
    #     envelope_fn_z3, _, mean_lf_z3 = fit_mean_logfrac_envelope(
    #         train_set, iz=Z_IDX_3)

    #     plot_logfrac_envelope_diagnostic(
    #         train_set, train_set.ks, Z_IDX_0, iz_label=0,
    #         envelope_fn=envelope_fn_z0, tagn_grid=tagn_grid,
    #         mean_logfracs=mean_lf_z0)
    #     plot_logfrac_envelope_diagnostic(
    #         train_set, train_set.ks, Z_IDX_3, iz_label=3,
    #         envelope_fn=envelope_fn_z3, tagn_grid=tagn_grid,
    #         mean_logfracs=mean_lf_z3)

    # --- Regime boundary (uses full k-grid logfracs) ---
    print("\n[INFO] Visualising residual regime boundary...")
    v_score, threshold = visualize_regime_boundary(train_set, iz=Z_IDX_0)

    # --- Fit PCA + tPCA (full k-grid) ---
    print("\n[INFO] Preparing training set (PCA + tPCA fit)...")
    train_set.prepare(num_pcs=NUM_PCS, num_pcs_z=NUM_PCS_Z)

    # --- Scree plots ---
    print("\n[INFO] Saving scree plots...")
    plot_scree(train_set)

    # --- Load test set ---
    print("\n[INFO] Loading test set...")
    test_set = utils.COLASet(
        target_z    = utils.z_mps,
        cosmo_type  = COSMO_TYPE,
        prior_type  = PRIOR_TYPE,
        nl_type     = NL_TYPE,
        start_batch = TEST_BATCH,
    )
    n_removed_test = apply_filter(test_set, om_min=None, w0_min=None, w0wa_max=None,
                                  omegab_h0_triangle_cut=OMEGAB_H0_TRIANGLE_CUT)
    print(f"[INFO] Prior cuts: removed {n_removed_test} cosmologies "
          f"from test set ({len(test_set.lhs)} remaining).")

    ks = train_set.ks   # full k-grid, same for both train and test

    # --- PCA reconstruction errors ---
    print("\n[INFO] Computing PCA reconstruction errors...")
    pca_err_z0 = pca_reconstruction_errors(train_set, test_set, Z_IDX_0)
    pca_err_z3 = pca_reconstruction_errors(train_set, test_set, Z_IDX_3)

    max_abs_err = np.max(np.abs(pca_err_z0), axis=1)
    outlier_idx = np.argmax(max_abs_err)

    print(f"Outlier cosmology index: {outlier_idx}")
    print(f"Max abs error: {max_abs_err[outlier_idx]:.4f}")
    print(f"LHS params: {test_set.lhs[outlier_idx]}")
    for name, val in zip(utils.params_tfree_mnufree if test_set.lhs.shape[1] > 7 else utils.params,
                        test_set.lhs[outlier_idx]):
        print(f"  {name}: {val:.5g}")

    i = outlier_idx
    print("logfracs range:", test_set.logfracs[i, Z_IDX_0, :].min(), test_set.logfracs[i, Z_IDX_0, :].max())
    print("Any NaN/Inf in this cosmology's full (z,k) grid:", 
        ~np.isfinite(test_set.logfracs[i]).all())
    print("frac_pks range:", test_set.frac_pks[i, Z_IDX_0, :].min(), test_set.frac_pks[i, Z_IDX_0, :].max())

    pca_err_z0_plot, keep_mask = _exclude_outliers(pca_err_z0, test_set, w0_abs_min=PLOT_W0_ABS_MIN)
    pca_err_z3_plot, _         = _exclude_outliers(pca_err_z3, test_set, w0_abs_min=PLOT_W0_ABS_MIN)


    print("\n  PCA error summary:")
    print_error_summary(pca_err_z0_plot, "PCA", iz_label=0)
    print_error_summary(pca_err_z3_plot, "PCA", iz_label=3)

    # --- tPCA reconstruction errors ---
    print("\n[INFO] Computing tPCA reconstruction errors (this may take a moment)...")
    stacks      = tpca_stacks(train_set, test_set)
    tpca_err_z0 = tpca_reconstruction_errors(stacks, test_set, Z_IDX_0)
    tpca_err_z3 = tpca_reconstruction_errors(stacks, test_set, Z_IDX_3)

    tpca_err_z0_plot, keep_mask = _exclude_outliers(tpca_err_z0, test_set, w0_abs_min=0.15)
    tpca_err_z3_plot, _         = _exclude_outliers(tpca_err_z3, test_set, w0_abs_min=0.15)


    print("\n  tPCA error summary:")
    print_error_summary(tpca_err_z0_plot, "tPCA", iz_label=0)
    print_error_summary(tpca_err_z3_plot, "tPCA", iz_label=3)

    # # --- UMAP reconstruction errors ---
    # print("\n[INFO] Computing UMAP reconstruction errors (this may take a moment)...")
    # stacks      = umap_stacks(train_set, test_set)
    # umap_err_z0 = umap_reconstruction_errors(stacks, test_set, Z_IDX_0)
    # umap_err_z3 = umap_reconstruction_errors(stacks, test_set, Z_IDX_3)

    # print("\n  UMAP error summary:")
    # print_error_summary(umap_err_z0, "UMAP", iz_label=0)
    # print_error_summary(umap_err_z3, "UMAP", iz_label=3)

    # --- Figures ---
    print("\n[INFO] Saving figures...")
    plot_pca_errors(pca_err_z0_plot,  ks, iz_label=0)
    plot_tpca_errors(tpca_err_z0_plot, ks, iz_label=0)
    # plot_umap_errors(umap_err_z0, ks, iz_label=0)
    plot_pca_errors(pca_err_z3_plot,  ks, iz_label=3)
    plot_tpca_errors(tpca_err_z3_plot, ks, iz_label=3)
    # plot_umap_errors(umap_err_z3, ks, iz_label=3)


    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()