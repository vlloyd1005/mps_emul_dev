"""
evaluate_one_nl.py — Nonlinear Emulator Evaluation Script

Evaluates a single trained nonlinear boost emulator against CAMB ground-truth
power spectra on the test set and produces:

  1. Per-cosmology boost error lines (filtered subset)
  2. Two-panel comparison plots for the emulator error and the raw boost shape
  3. Histograms of the maximum absolute error across cosmologies
  4. log(P_nl_camb / P_nl_syren) diagnostic, coloured by w0
  5. Datagenerator comparison: emulator / datagenerator ratio at the fiducial
     cosmology (one panel per redshift)

The emulator predicts log(P_nl_camb / P_nl_syren_halofit).  At inference
get_pks returns:
    P_nonlin ≈ exp(log_frac) * P_nl_syren
where P_nl_syren is the syren halofit approximation used as the boost base.
The ground-truth comparison uses pks_target (the CAMB nonlinear output) from
the COLASet directly — no padding or reconstruction needed.

Prior cuts (W0_MIN_FILTER, W0WA_MAX_FILTER) must match values used during
training.

Usage (standalone):
    python ./mps_emu/evaluate_one_nl.py

Author: Victoria Lloyd (2025)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d

import emulmps_w0wa as pk_emu
import train_utils_pk_emulator_v3 as utils
from train_utils_pk_emulator_v3 import VER

# import sys; sys.path.insert(0, "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); import symbolic_pofk.linear_VL as linear
# import sys; sys.path.insert(0, "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); from symbolic_pofk.linear_VL import plin_emulated, get_approximate_D, growth_correction_R, get_eisensteinhu_nw
# import sys; sys.path.insert(0, "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); from symbolic_pofk.syrenhalofit import run_halofit_vec
# from symbolic_pofk.linear_VL import As_to_sigma8


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

START_BATCH    = 1000        # batch index used as the test/validation set
N_TRAIN        = 100         # number of training batches for the model being evaluated
COSMO_TYPE     = "w0wacdm"
NL_TYPE        = "mead2020_Tfree_mnufree"  # must be a nonlinear nl_type: halofit, mead2020, etc.
PRIOR_TYPE     = "expanded"
MODEL_TYPE     = "npce"     # "mlp" or "npce" — must match training
# VER            = "v9"
METADATA_DIR = "mps_emu/metadata"

# Target redshifts — nearest available z in utils.z_mps will be used for each
TARGET_REDSHIFTS = [0, 0.5, 1, 2, 5, 10, 30, 50]

# Prior cuts — must match values used during training
W0_MIN_FILTER   = None #-2.0      # None to disable
W0WA_MAX_FILTER = None #-0.4      # None to disable

OMEGAB_H0_TRIANGLE_CUT = {
    'omegab_anchor': 0.05,
    'h0_anchor':     75,
    'omegab_max':    0.072,
    'h0_max':        90,
}

# Diagnostic thresholds (used only for the filtered subplot / stats split)
W0_THRESHOLD    = -1.8
W0WA_THRESHOLD  = -0.75
W0_COL          = utils.params.index("w")
W0WA_COL        = utils.params.index("w0+wa")

FIG_DIR = "mps_emu/validation_figs/smaller_grid"

K_WINDOW_LO, K_WINDOW_HI = 20.0, 30.0
K_WINDOW_MASK = (utils.ks >= K_WINDOW_LO) & (utils.ks <= K_WINDOW_HI)
TAGN_COL = utils.params_tfree_mnufree.index("T_AGN")   # column 7 in lhs

# Path to single-cosmology datagenerator nonlinear reference, relative to this file.
# Should be the CAMB nonlinear output for the fiducial cosmology, shape (N_z, N_k).
# DG_PK_PATH = os.path.join(
#     os.path.dirname(os.path.abspath(__file__)),
#     "..", "mps", "output", f"one_eval_wcdm_500ks_{NL_TYPE}.npy"
# )
DG_PK_PATH = None
DG_KS = np.logspace(-5.1, 2, 500)

# Fiducial cosmology for the datagenerator comparison.
# Order: [10^9 A_s, ns, H0, Omega_b, Omega_m, w0, w0+wa]
FIDUCIAL_PARAMS = [2.1, 0.96605, 67.32, 0.04, 0.3, -0.9, -0.9, 7.3, 0.06]


# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

COLOR_50    = "#473C8A"
COLOR_90    = "#C45858"
COLOR_100   = "lightgray"
COLOR_GOOD  = "#2166ac"
COLOR_BAD   = "#d6604d"
COLOR_EMU   = "#1565c0"   # emulator bands / lines
COLOR_BOOST = "#2ca02c"   # raw CAMB boost reference
COLOR_DG    = "black"     # datagenerator reference line

AXES_FS   = 20
TICK_FS   = 17
LEGEND_FS = 17

HANDLE_95 = mlines.Line2D([], [], color=COLOR_100, label=r"$95\%$")
HANDLE_90 = mpatches.Patch(facecolor=COLOR_90, label=r"$90\%$")
HANDLE_50 = mpatches.Patch(facecolor=COLOR_50, hatch="\\", edgecolor="lightgray", label=r"$50\%$")

def _resolve_nl_metadata_path(n_params):
    """
    Resolves the NL metadata bundle path using COLASet's own tag-naming
    logic (imported from train_utils_pk_emulator_v3), so this always
    matches whatever convention COLASet.prepare() actually used to save
    it -- no hardcoded/duplicated path string to drift out of sync.
    """
    dummy = utils.COLASet.__new__(utils.COLASet)
    dummy.cosmo_type = COSMO_TYPE
    dummy.prior_type = PRIOR_TYPE
    dummy.nl_type    = NL_TYPE
    dummy.n_batches  = N_TRAIN
    dummy.w0_min     = W0_MIN_FILTER
    dummy.w0wa_max   = W0WA_MAX_FILTER
    dummy.use_boost  = True   # NL_TYPE here is always a boost type
    dummy.lhs        = np.empty((1, n_params))   # only .shape[1] is read
    tag = dummy._metadata_tag()
    path = os.path.join(METADATA_DIR, f"metadata_{tag}", "metadata.joblib")
    print(f"[INFO] Resolved NL metadata path: {path}")
    return path


def _load_nl_pca_bundle(n_params):
    import joblib
    path = _resolve_nl_metadata_path(n_params)
    if not os.path.exists(path):
        print(f"  [WARN] NL metadata bundle not found at {path} — "
              f"skipping PC-level diagnostics.")
        return None, None
    bundle = joblib.load(path)
    return bundle["pcas"], bundle["scalers"]

def plot_comparison_bands_zoomed(errors, ks, iz, z_val, k_lo=10.0, k_hi=50.0):
    """Same percentile-band plot as plot_comparison_bands, zoomed on the
    suspected transition region so it isn't washed out by the full k range."""
    mask = (ks >= k_lo) & (ks <= k_hi)
    p_vals = percentiles(errors[:, mask])
    fig, ax = plt.subplots(figsize=(8, 5))
    _fill_ax(ax, ks[mask], *p_vals,
             ylabel=fr"$P_{{{NL_TYPE}}}^\mathrm{{{MODEL_TYPE.upper()}}} / P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1$",
             label_text=f'$z={z_val:.4g}$  (k zoom {k_lo:g}-{k_hi:g})')
    ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="purple", alpha=0.08, label="k=20-30 window")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95], fontsize=LEGEND_FS, loc="best")
    plt.tight_layout()
    fname = (f"{FIG_DIR}/{MODEL_TYPE}_boost_errors_zoom_z{z_val:.4g}_{COSMO_TYPE}"
             f"_{PRIOR_TYPE}_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_max_error_vs_tagn(errors, lhs_clean, iz, z_val):
    """Scatter: max |error| within k=20-30 vs T_AGN, across all test
    cosmologies. Tests whether the localized failure tracks T_AGN,
    especially near the prior's upper edge (7.8-8.0).

    Log y-scale: max|error| spans orders of magnitude across 2000
    cosmologies (most fit well, a tail fits badly), so a linear axis
    compresses everything near the bulk into a flat-looking line at the
    bottom and hides exactly the outliers this plot exists to show.
    """
    if lhs_clean.shape[1] <= TAGN_COL:
        print("  [WARN] T_AGN column not present — skipping plot_max_error_vs_tagn.")
        return

    tagn_vals      = lhs_clean[:, TAGN_COL]
    max_err_window = np.max(np.abs(errors[:, K_WINDOW_MASK]), axis=1)
    max_err_full   = np.max(np.abs(errors), axis=1)

    # log scale can't show exact zeros -- clip to a small floor so points
    # with (near-)perfect fits still show up at the bottom of the axis
    # instead of being silently dropped by matplotlib.
    floor = max(max_err_window[max_err_window > 0].min() * 0.5, 1e-8) \
        if np.any(max_err_window > 0) else 1e-8
    max_err_window_plot = np.clip(max_err_window, floor, None)

    color_floor = max(max_err_full[max_err_full > 0].min() * 0.5, 1e-8) \
        if np.any(max_err_full > 0) else 1e-8
    max_err_full_plot = np.clip(max_err_full, color_floor, None)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(
        tagn_vals, max_err_window_plot, c=max_err_full_plot,
        cmap="YlOrRd", norm=mpl.colors.LogNorm(
            vmin=max_err_full_plot.min(), vmax=max_err_full_plot.max()),
        s=12, alpha=0.7, rasterized=True,
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"$T_\mathrm{AGN}$", fontsize=AXES_FS)
    ax.set_ylabel(fr"max $|$error$|$ in $k \in [{K_WINDOW_LO:g},{K_WINDOW_HI:g}]$", fontsize=AXES_FS - 4)
    ax.set_title(f"z={z_val:.4g}  (colour = max |error| over full k range)", fontsize=AXES_FS - 5)
    ax.grid(alpha=0.3, which="both")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("max |error| (full k)", fontsize=TICK_FS - 3)
    plt.tight_layout()
    fname = (f"{FIG_DIR}/max_error_vs_tagn_z{z_val:.4g}_{COSMO_TYPE}"
             f"_{PRIOR_TYPE}_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_worst_offenders_in_window(errors, ks, iz, z_val, n_worst=30):
    """Per-cosmology error curves for the N cosmologies with the largest
    error specifically inside the k=20-30 window -- a different selection
    criterion than your existing w0/w0+wa diagnostic split."""
    max_err_window = np.max(np.abs(errors[:, K_WINDOW_MASK]), axis=1)
    worst_idx = np.argsort(max_err_window)[::-1][:n_worst]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in worst_idx:
        ax.semilogx(ks, errors[idx], alpha=0.6, lw=1)
    ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="purple", alpha=0.08)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(fr"$P_{{{NL_TYPE}}}^\mathrm{{{MODEL_TYPE.upper()}}} / P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1$",
                  fontsize=AXES_FS - 4)
    ax.set_title(f"Worst {n_worst} cosmologies by max|error| in k=20-30 (z={z_val:.4g})",
                 fontsize=AXES_FS - 6)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = (f"{FIG_DIR}/worst_offenders_window_z{z_val:.4g}_{COSMO_TYPE}"
             f"_{PRIOR_TYPE}_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_error_heatmap_vs_tagn(errors, ks, lhs_clean, iz, z_val, n_tagn_bins=10):
    """2D heatmap: median |error| as a function of (k, T_AGN bin), aggregated
    over the full test set. Directly tests the 'transition sweep' hypothesis
    with real statistical power, instead of 4-7 discrete fiducial points."""
    if lhs_clean.shape[1] <= TAGN_COL:
        print("  [WARN] T_AGN column not present — skipping plot_error_heatmap_vs_tagn.")
        return

    tagn_vals = lhs_clean[:, TAGN_COL]
    bin_edges = np.linspace(tagn_vals.min(), tagn_vals.max(), n_tagn_bins + 1)
    bin_idx   = np.clip(np.digitize(tagn_vals, bin_edges) - 1, 0, n_tagn_bins - 1)

    heatmap = np.full((n_tagn_bins, len(ks)), np.nan)
    counts  = np.zeros(n_tagn_bins, dtype=int)
    for b in range(n_tagn_bins):
        in_bin = bin_idx == b
        counts[b] = in_bin.sum()
        if counts[b] > 0:
            heatmap[b] = np.median(np.abs(errors[in_bin]), axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    finite_vals = heatmap[np.isfinite(heatmap) & (heatmap > 0)]
    if finite_vals.size == 0:
        print(f"  [WARN] No finite positive values for heatmap at z={z_val:.4g} — skipping.")
        plt.close(fig)
        return
    im = ax.pcolormesh(ks, bin_centres, heatmap, shading="auto", cmap="YlOrRd",
                        norm=mpl.colors.LogNorm(vmin=finite_vals.min(), vmax=finite_vals.max()))
    ax.set_xscale("log")
    ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="cyan", alpha=0.15)
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(r"$T_\mathrm{AGN}$", fontsize=AXES_FS)
    ax.set_title(f"Median |error| vs (k, T_AGN)  (z={z_val:.4g})", fontsize=AXES_FS - 5)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("median |error|", fontsize=TICK_FS - 3)
    for b, c in enumerate(counts):
        ax.text(ks[-1] * 1.02, bin_centres[b], f"n={c}", fontsize=7, va="center", clip_on=False)
    plt.tight_layout()
    fname = (f"{FIG_DIR}/error_heatmap_vs_tagn_z{z_val:.4g}_{COSMO_TYPE}"
             f"_{PRIOR_TYPE}_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")

# ---------------------------------------------------------------------------
# Redshift index helpers
# ---------------------------------------------------------------------------

def find_redshift_index(target_z: float) -> tuple[int, float]:
    z_arr = np.asarray(utils.z_mps)
    iz = int(np.argmin(np.abs(z_arr - target_z)))
    return iz, float(z_arr[iz])


def resolve_redshift_indices() -> list[tuple[int, float]]:
    seen_indices = set()
    result = []
    for target in TARGET_REDSHIFTS:
        iz, z_val = find_redshift_index(target)
        if iz in seen_indices:
            print(f"  [WARN] target z={target} maps to iz={iz} (z={z_val:.4f}), "
                  "already included — skipping duplicate.")
            continue
        seen_indices.add(iz)
        result.append((iz, z_val))
    return result


# ---------------------------------------------------------------------------
# File-tag helper
# ---------------------------------------------------------------------------

def _filter_tag() -> str:
    parts = []
    if W0_MIN_FILTER   is not None:
        parts.append(f"w0min{W0_MIN_FILTER}")
    if W0WA_MAX_FILTER is not None:
        parts.append(f"w0wamax{W0WA_MAX_FILTER}")
    parts.append(VER)
    return ("_" + "_".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_set():
    """
    Load the nonlinear test set and apply prior cuts.

    pks_target on the returned COLASet is the raw CAMB nonlinear output on the
    full 500-point k-grid (no reconstruction or padding needed — the syren
    halofit baseline is non-trivial at all k so the full grid was used during
    training).
    """
    test_set = utils.COLASet(
        target_z    = utils.z_mps,
        cosmo_type  = COSMO_TYPE,
        prior_type  = PRIOR_TYPE,
        nl_type     = NL_TYPE,
        start_batch = START_BATCH,
    )

    mask = np.ones(len(test_set.lhs), dtype=bool)

    if W0_MIN_FILTER is not None:
        w0_mask = test_set.lhs[:, W0_COL] >= W0_MIN_FILTER
        print(f"[INFO] w0 cut   (w0 >= {W0_MIN_FILTER}):     "
              f"removing {(~w0_mask & mask).sum()} cosmologies.")
        mask &= w0_mask

    if W0WA_MAX_FILTER is not None:
        w0wa_mask = test_set.lhs[:, W0WA_COL] <= W0WA_MAX_FILTER
        print(f"[INFO] w0+wa cut (w0+wa <= {W0WA_MAX_FILTER}): "
                f"removing {(~w0wa_mask & mask).sum()} additional cosmologies.")
        mask &= w0wa_mask

    if OMEGAB_H0_TRIANGLE_CUT is not None:
        omegab_col = utils.params.index("Omega_b")
        h_col      = utils.params.index("h")
        ob_anchor  = OMEGAB_H0_TRIANGLE_CUT['omegab_anchor']
        h0_anchor  = OMEGAB_H0_TRIANGLE_CUT['h0_anchor']
        ob_max     = OMEGAB_H0_TRIANGLE_CUT['omegab_max']
        h0_max     = OMEGAB_H0_TRIANGLE_CUT['h0_max']

        omegab_vals = test_set.lhs[:, omegab_col]
        h0_vals     = test_set.lhs[:, h_col]

        slope = (h0_anchor - h0_max) / (ob_max - ob_anchor)
        h0_line = h0_max + slope * (omegab_vals - ob_anchor)
        triangle_mask = h0_vals <= h0_line
        print(f"[INFO] Omega_b/H0 triangle cut: "
              f"removing {(~triangle_mask & mask).sum()} additional cosmologies.")
        mask &= triangle_mask

    for attr in ("lhs", "pks_target", "frac_pks", "logfracs", "mps_approxes_boost", "mps_approxes"):
        if hasattr(test_set, attr) and getattr(test_set, attr) is not None:
            setattr(test_set, attr, getattr(test_set, attr)[mask])
    # pks_lin is None in boost mode; guard it
    # if hasattr(test_set, "pks_lin") and test_set.pks_lin is not None:
    #     test_set.pks_lin = test_set.pks_lin[mask]

    print(f"[INFO] Test set after prior cuts: {mask.sum()} / {len(mask)} cosmologies kept.")
    return test_set


def compute_predictions(test_set):
    """
    Run the emulator at every cosmology in the test set.

    Returns
    -------
    pred_pks   : (N_valid, N_z, N_k)  emulator P_nonlin on full k-grid
    true_pks   : (N_valid, N_z, N_k)  CAMB P_nonlin (ground truth)
    true_frac  : (N_valid, N_z, N_k)  CAMB frac_pks = P_nl_camb / P_nl_syren
    lhs_clean  : (N_valid, N_params)
    """
    true_pks_all  = test_set.pks_target    # (N, N_z, N_k) — full k-grid, no padding needed
    true_frac_all = test_set.frac_pks      # (N, N_z, N_k) — CAMB / syren_halofit residual
    approx_all    = test_set.mps_approxes_boost
    pks_lin_all   = true_pks_all / (true_frac_all * approx_all)
    pred_list     = []

    n_params = test_set.lhs.shape[1]

    # for row in test_set.lhs:
    for row, plin in zip(test_set.lhs, pks_lin_all):
        # lhs stores [As, ns, H0, Ob, Om, w0, w0+wa, (mnu, T_AGN)]
        # get_pks in emulmps_w0wa.py expects wa in col 6, not w0+wa
        params_for_emu = row.copy()
        # params_for_emu[6] = row[6] - row[5] 

        _, _, pk_full = pk_emu.get_pks(
            params_for_emu,
            cosmo_type             = COSMO_TYPE,
            prior_type             = PRIOR_TYPE,
            nl_type                = NL_TYPE,
            model_type             = MODEL_TYPE,
            num_batches            = N_TRAIN,
            w0_min                 = W0_MIN_FILTER,
            w0wa_max               = W0WA_MAX_FILTER,
            use_approximation_only = False,
            pk_lin=plin,
        )
        pred_list.append(pk_full)

    pred_pks = np.asarray(pred_list)

    nan_mask = np.isnan(pred_pks).any(axis=(1, 2))
    if nan_mask.any():
        print(f"  WARNING: dropping {nan_mask.sum()} cosmologies with NaN output.")

    valid = ~nan_mask
    pred_frac = pred_pks[valid] / (approx_all[valid] * pks_lin_all[valid]) #pred_pks[valid] / approx_all[valid]
    return (
        pred_pks[valid],
        true_pks_all[valid],
        true_frac_all[valid],
        pred_frac,
        test_set.lhs[valid],
    )

def _pc_responsibility(pcas, scalers, z_val, k_lo=K_WINDOW_LO, k_hi=K_WINDOW_HI, ks=None):
    ks = utils.ks if ks is None else ks
    k_mask = (ks >= k_lo) & (ks <= k_hi)
    z_key = float(f"{z_val:.3f}")
    pca, scaler = pcas[z_key], scalers[z_key]
    scale = scaler.scale_ if hasattr(scaler, "scale_") else scaler.std
    mat = pca.components_ * scale[None, :]
    window_energy = np.sum(mat[:, k_mask] ** 2, axis=1)
    total_energy  = np.sum(mat ** 2, axis=1)
    return window_energy / total_energy


def _frac_to_pcs(frac_z, z_val, pcas, scalers):
    z_key  = float(f"{z_val:.3f}")
    scaler = scalers[z_key]
    pca    = pcas[z_key]
    logfrac_norm = scaler.transform(np.log(frac_z))
    return pca.transform(logfrac_norm)


def plot_pc_true_vs_predicted(true_frac, pred_frac, pcas, scalers, lhs_clean,
                               iz, z_val, top_n=3):
    pcs_true = _frac_to_pcs(true_frac[:, iz, :], z_val, pcas, scalers)
    pcs_pred = _frac_to_pcs(pred_frac[:, iz, :], z_val, pcas, scalers)
    resp = _pc_responsibility(pcas, scalers, z_val)
    top_pcs = np.argsort(resp)[::-1][:top_n]

    has_tagn = lhs_clean.shape[1] > TAGN_COL
    color_vals = lhs_clean[:, TAGN_COL] if has_tagn else None

    fig, axes = plt.subplots(1, top_n, figsize=(5.5 * top_n, 5), squeeze=False)
    axes = axes[0]
    for ax, p in zip(axes, top_pcs):
        x, y = pcs_true[:, p], pcs_pred[:, p]
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6, zorder=1)
        if has_tagn:
            sc = ax.scatter(x, y, c=color_vals, cmap="viridis", s=10,
                             alpha=0.6, rasterized=True)
            fig.colorbar(sc, ax=ax, label=r"$T_\mathrm{AGN}$")
        else:
            ax.scatter(x, y, s=10, alpha=0.5, rasterized=True, color="C0")
        rmse = np.sqrt(np.mean((x - y) ** 2))
        corr = np.corrcoef(x, y)[0, 1]
        ax.set_xlabel("true PC coefficient")
        ax.set_ylabel("predicted PC coefficient")
        ax.set_title(f"z={z_val:.2f}, PC {p}\nresp={resp[p]:.2f}, RMSE={rmse:.3g}, r={corr:.4f}",
                     fontsize=AXES_FS - 6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = (f"{FIG_DIR}/pc_true_vs_pred_z{z_val:.4g}_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_pc_residual_vs_tagn(true_frac, pred_frac, pcas, scalers, lhs_clean,
                              iz, z_val, top_n=3):
    if lhs_clean.shape[1] <= TAGN_COL:
        print("  [WARN] T_AGN column not present — skipping plot_pc_residual_vs_tagn.")
        return
    pcs_true = _frac_to_pcs(true_frac[:, iz, :], z_val, pcas, scalers)
    pcs_pred = _frac_to_pcs(pred_frac[:, iz, :], z_val, pcas, scalers)
    resp = _pc_responsibility(pcas, scalers, z_val)
    top_pcs = np.argsort(resp)[::-1][:top_n]
    tagn_vals = lhs_clean[:, TAGN_COL]

    fig, axes = plt.subplots(1, top_n, figsize=(5.5 * top_n, 4), squeeze=False)
    axes = axes[0]
    for ax, p in zip(axes, top_pcs):
        resid = pcs_pred[:, p] - pcs_true[:, p]
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.scatter(tagn_vals, resid, s=8, alpha=0.4, rasterized=True, color="C2")
        ax.set_xlabel(r"$T_\mathrm{AGN}$")
        ax.set_ylabel("predicted - true")
        ax.set_title(f"z={z_val:.2f}, PC {p} (resp={resp[p]:.2f})", fontsize=AXES_FS - 6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = (f"{FIG_DIR}/pc_residual_vs_tagn_z{z_val:.4g}_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")

def apply_diagnostic_filter(errors, true_frac, lhs_clean):
    """
    Split into 'good' and 'bad' cosmologies using the diagnostic thresholds.

    Parameters
    ----------
    errors    : (N, N_k)  pred / true - 1
    true_frac : (N, N_k)  CAMB frac_pks (P_nl / P_nl_syren) at this z
    lhs_clean : (N, N_params)
    """
    w0_mask   = lhs_clean[:, W0_COL]   > W0_THRESHOLD
    w0wa_mask = lhs_clean[:, W0WA_COL] < W0WA_THRESHOLD
    good_mask = w0_mask & w0wa_mask
    return errors[good_mask], true_frac[good_mask], good_mask


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentiles(err):
    ps = np.percentile(err, [5, 10, 25, 75, 90, 95], axis=0)
    return ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]


def print_statistics(errors, errors_filt, iz, z_val):
    subsets = [
        ("Full dataset",                                     errors),
        (f"w0>{W0_THRESHOLD} & w0+wa<{W0WA_THRESHOLD}",    errors_filt),
    ]
    print(f"\n=== ERROR STATISTICS  (iz={iz}, z={z_val:.4f}) ===")
    for label, e in subsets:
        max_e = np.max(np.abs(e), axis=1)
        print(f"\n  {label}:")
        print(f"    Mean max |error|:     {np.mean(max_e):.6f}")
        print(f"    Median max |error|:   {np.median(max_e):.6f}")
        print(f"    95th pct max |error|: {np.percentile(max_e, 95):.6f}")
        print(f"    99th pct max |error|: {np.percentile(max_e, 99):.6f}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _to_grid(src_ks, src_Pk, tgt_ks):
    """Interpolate src_Pk onto tgt_ks; NaN outside source range."""
    mask   = (tgt_ks >= src_ks[0]) & (tgt_ks <= src_ks[-1])
    result = np.full_like(tgt_ks, np.nan, dtype=float)
    result[mask] = interp1d(src_ks, src_Pk)(tgt_ks[mask])
    return result


def _fill_ax(ax, ks, p5, p10, p25, p75, p90, p95, ylabel, label_text):
    ax.semilogx(ks, p5,  color=COLOR_100)
    ax.semilogx(ks, p95, color=COLOR_100)
    ax.fill_between(ks, p10, p90, color=COLOR_90, alpha=0.7)
    ax.fill_between(ks, p25, p75, color=COLOR_50, hatch="\\",
                    edgecolor="lightgray", alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=AXES_FS)
    ax.set_xscale("log")
    ax.grid(alpha=0.4)
    ax.set_xlim([ks[0], ks[-1]])
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.text(0.05, 0.95, label_text, transform=ax.transAxes,
            ha='left', va='top', fontsize=AXES_FS)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_boost_error_lines_filtered(errors_filt, ks, good_mask, iz, z_val):
    """Per-cosmology emulator error lines for the diagnostically-filtered subset."""
    n_plot = min(good_mask.sum(), 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_plot):
        ax.semilogx(ks, errors_filt[i, :])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(
        fr"$P_{{{NL_TYPE}}}^\mathrm{{emu}} / P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1$",
        fontsize=AXES_FS)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.text(0.05, 0.95,
            fr'$z={z_val:.4g}$,  $w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$  (N={good_mask.sum()})',
            transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FS)
    fname = (f"{FIG_DIR}/val_boost_errors_{COSMO_TYPE}_{NL_TYPE}_z{z_val:.4g}"
             f"_w0gt{W0_THRESHOLD}_w0walt{W0WA_THRESHOLD}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# def plot_comparison_bands(errors, errors_filt, ks, iz, z_val):
#     """Two panels: emulator error bands over full dataset and filtered subset."""
#     model_label = MODEL_TYPE.upper()
#     ylabel = (
#         fr"$P_{{{NL_TYPE}}}^\mathrm{{{model_label}}} / "
#         fr"P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1$"
#     )

#     panel_specs = [
#         (errors,      ""),
#         (errors_filt, fr"($w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$)"),
#     ]
#     suffix_tags = ["all", "filtered"]

#     for (e, label_text), stag in zip(panel_specs, suffix_tags):
#         p_vals = percentiles(e)
#         fig, ax = plt.subplots(figsize=(8, 5))
#         _fill_ax(ax, ks, *p_vals, ylabel=ylabel,
#                  label_text=f'$z={z_val:.4g}$  {label_text}')
#         ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#         ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95],
#                   fontsize=LEGEND_FS, loc="lower right")
#         ax.set_ylim(-0.07, 0.07)
#         plt.tight_layout()
#         fname = (f"{FIG_DIR}/{MODEL_TYPE}_boost_errors_z{z_val:.4g}_{COSMO_TYPE}"
#                  f"_{PRIOR_TYPE}_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}_{stag}.pdf")
#         plt.savefig(fname, bbox_inches="tight")
#         plt.close()
#         print(f"  Saved: {fname}")

def plot_comparison_bands(errors, ks, iz, z_val):
    """Single percentile band plot for this redshift — no filtered/unfiltered split."""
    model_label = MODEL_TYPE.upper()
    ylabel = (
        fr"$P_{{{NL_TYPE}}}^\mathrm{{{model_label}}} / "
        fr"P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1$"
    )
    p_vals = percentiles(errors)
    fig, ax = plt.subplots(figsize=(8, 5))
    _fill_ax(ax, ks, *p_vals, ylabel=ylabel, label_text=f'$z={z_val:.4g}$')
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95],
              fontsize=LEGEND_FS, loc="lower right")
    plt.tight_layout()
    fname = (f"{FIG_DIR}/{MODEL_TYPE}_boost_errors_z{z_val:.4g}_{COSMO_TYPE}"
             f"_{PRIOR_TYPE}_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# def plot_max_error_histograms(errors, errors_filt, iz, z_val):
#     """2×1 histogram grid: full dataset and filtered subset."""
#     filter_label = fr"$w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$"
#     model_label  = MODEL_TYPE.upper()
#     xlabel = (
#         fr"Max $|P_{{{NL_TYPE}}}^\mathrm{{{model_label}}} / "
#         fr"P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1|$"
#     )

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))
#     hist_specs = [
#         (axes[0], np.max(np.abs(errors),      axis=1), '#1565c0', ''),
#         (axes[1], np.max(np.abs(errors_filt), axis=1), '#C45858', filter_label),
#     ]
#     for ax, data, color, title in hist_specs:
#         ax.hist(data, bins=50, color=color, alpha=0.7, edgecolor='black')
#         ax.set_xlabel(xlabel, fontsize=16)
#         ax.set_ylabel('Count', fontsize=16)
#         ax.set_title(f'{model_label} — {title}   ($z={z_val:.4g}$)', fontsize=18)
#         ax.grid(alpha=0.3)
#         ax.tick_params(labelsize=14)

#     plt.tight_layout()
#     fname = (f"{FIG_DIR}/max_boost_error_histogram_{COSMO_TYPE}_{PRIOR_TYPE}"
#              f"_{NL_TYPE}_{MODEL_TYPE}_z{z_val:.4g}"
#              f"_w0gt{W0_THRESHOLD}_w0walt{W0WA_THRESHOLD}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
#     plt.savefig(fname, bbox_inches="tight")
#     plt.close()
#     print(f"  Saved: {fname}")

def plot_max_error_histogram(errors, iz, z_val):
    """Single histogram of max absolute error across cosmologies."""
    model_label = MODEL_TYPE.upper()
    xlabel = (
        fr"Max $|P_{{{NL_TYPE}}}^\mathrm{{{model_label}}} / "
        fr"P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1|$"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.max(np.abs(errors), axis=1), bins=50,
            color='#1565c0', alpha=0.7, edgecolor='black')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title(f'{model_label} — {COSMO_TYPE}, {NL_TYPE}   ($z={z_val:.4g}$)', fontsize=18)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    fname = (f"{FIG_DIR}/max_boost_error_histogram_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_{MODEL_TYPE}_z{z_val:.4g}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_residual_logfracs(test_set, iz, z_val):
    """
    Per-cosmology log(P_nl_camb / P_nl_syren) lines, coloured by w0.

    This is the quantity the emulator is trained to predict.  The residual
    is centred near 0 (unlike the old NL/LIN boost which was always positive),
    so we draw a horizontal reference line at 0.
    """
    w0        = test_set.lhs[:, W0_COL]
    w0wa      = test_set.lhs[:, W0WA_COL]
    mask_good = (w0 > W0_THRESHOLD) & (w0wa < W0WA_THRESHOLD)
    mask_bad  = ~mask_good
    logfracs_z = test_set.logfracs[:, iz, :]
    ks         = utils.ks

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx in np.where(mask_good)[0]:
        ax.semilogx(ks, logfracs_z[idx], color=COLOR_GOOD, alpha=0.08, lw=0.5, rasterized=True)
    for idx in np.where(mask_bad)[0]:
        ax.semilogx(ks, logfracs_z[idx], color=COLOR_BAD,  alpha=0.08, lw=0.5, rasterized=True)
    if mask_good.sum() > 0:
        ax.semilogx(ks, np.median(logfracs_z[mask_good], axis=0), color=COLOR_GOOD, lw=2.2,
                    label=fr"$w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$ (n={mask_good.sum()})")
    if mask_bad.sum() > 0:
        ax.semilogx(ks, np.median(logfracs_z[mask_bad],  axis=0), color=COLOR_BAD,  lw=2.2,
                    label=fr"fails cut (n={mask_bad.sum()})")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(
        fr"$\log\!\left(P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{nl,syren}}\right)$",
        fontsize=AXES_FS)
    ax.set_title(
        fr"$z={z_val:.4g}$   |   {COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, {MODEL_TYPE.upper()}",
        fontsize=15)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = (f"{FIG_DIR}/residual_logfracs_z{z_val:.4g}_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_{MODEL_TYPE}_w0gt{W0_THRESHOLD}_w0walt{W0WA_THRESHOLD}"
             f"{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_frac_shape_bands(true_frac, ks, iz, z_val):
    """
    Percentile bands of the CAMB residual frac_pks = P_nl_camb / P_nl_syren.

    Replaces the old 'raw CAMB boost B = P_nonlin / P_lin' plot.  The residual
    is centred near 1 so we plot as frac - 1 for visibility; a flat zero line
    means syren halofit is a perfect approximation for that cosmology.
    """
    frac_z = true_frac[:, iz, :]
    p_vals  = percentiles(frac_z - 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    _fill_ax(
        ax, ks, *p_vals,
        ylabel=fr"$P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{nl,syren}} - 1$",
        label_text=f'$z={z_val:.4g}$  (CAMB residual)',
    )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95],
              fontsize=LEGEND_FS, loc="upper left")
    plt.tight_layout()
    fname = (f"{FIG_DIR}/camb_residual_frac_z{z_val:.4g}_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Triangle plot
# ---------------------------------------------------------------------------

_PARAM_LABELS = [
    r'$10^9 A_s$', r'$n_s$', r'$H_0$', r'$\Omega_b$',
    r'$\Omega_m$', r"$w_0$", r"$w_0+w_a$",
]

_PARAM_LABELS_7 = [
    r'$10^9 A_s$', r'$n_s$', r'$H_0$', r'$\Omega_b$',
    r'$\Omega_m$', r"$w_0$", r"$w_0+w_a$",
]
_PARAM_LABELS_9 = _PARAM_LABELS_7 + [r'$m_\nu$', r'$\log T_{\rm AGN}$']

def plot_error_triangle(errors, lhs_clean, iz, z_val):
    import matplotlib.colors as mcolors
    # import matplotlib.colormaps as mcm

    n_params = min(lhs_clean.shape[1], 9)
    labels   = (_PARAM_LABELS_9 if n_params > 7 else _PARAM_LABELS_7)[:n_params]

    ref_cosmo = np.array(FIDUCIAL_PARAMS[:n_params])
    mean_abs_err = np.mean(np.abs(errors), axis=1)
    # n_params = min(lhs_clean.shape[1], 7)   # plot only the 7 base params
    # labels   = _PARAM_LABELS[:n_params]

    # vmin, vmax = mean_abs_err.min(), mean_abs_err.max()
    finite_mask = np.isfinite(mean_abs_err)
    n_nonfinite = (~finite_mask).sum()
    if n_nonfinite > 0:
        print(f"  [WARN] {n_nonfinite} cosmologies have non-finite mean error "
              f"at z={z_val:.4g} — clamping to finite max for triangle plot.")
    finite_max = mean_abs_err[finite_mask].max() if finite_mask.any() else 1.0
    mean_abs_err_plot = np.where(finite_mask, mean_abs_err, finite_max)

    vmin = mean_abs_err_plot.min()
    vmax = mean_abs_err_plot.max()
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
        # Nothing useful to plot
        print(f"  [WARN] Cannot build colormap norm at z={z_val:.4g} "
              f"(vmin={vmin}, vmax={vmax}) — skipping triangle plot.")
        return

    use_log = (vmax / max(vmin, 1e-10)) > 10
    norm = (mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax) if use_log
            else mcolors.Normalize(vmin=vmin, vmax=vmax))
    cmap = mpl.colormaps["YlOrRd"]

    fig_size = 2.2 * n_params
    fig, axes = plt.subplots(n_params, n_params,
                             figsize=(fig_size + 1.5, fig_size), squeeze=False)

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue
            if i == j:
                n_bins = 30
                counts, edges = np.histogram(lhs_clean[:, i], bins=n_bins)
                bin_centres = 0.5 * (edges[:-1] + edges[1:])
                for b in range(n_bins):
                    in_bin = (lhs_clean[:, i] >= edges[b]) & (lhs_clean[:, i] < edges[b + 1])
                    if in_bin.sum() == 0:
                        continue
                    ax.bar(bin_centres[b], counts[b], width=(edges[1] - edges[0]),
                           color=cmap(norm(np.median(mean_abs_err_plot[in_bin]))), linewidth=0)
                ax.set_xlim(edges[0], edges[-1])
                ax.yaxis.set_visible(False)
                if i < len(ref_cosmo):
                    ax.axvline(ref_cosmo[i], color="black", lw=1.4, ls="--", zorder=5)
            else:
                ax.scatter(lhs_clean[:, j], lhs_clean[:, i],
                           c=mean_abs_err_plot, cmap=cmap, norm=norm,
                           s=15, linewidths=0, rasterized=True, alpha=0.8)
                if j < len(ref_cosmo) and i < len(ref_cosmo):
                    ax.plot(ref_cosmo[j], ref_cosmo[i], marker="x", color="black",
                            markersize=9, markeredgewidth=2.0, zorder=6, linestyle="none")
            if i == n_params - 1:
                ax.set_xlabel(labels[j], fontsize=TICK_FS)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=TICK_FS)
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(axis="both", which="major", labelsize=9)

    cbar_ax = fig.add_axes([0.72, 0.55, 0.02, 0.35])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label(
        fr"Mean $|\,P_{{{NL_TYPE}}}^\mathrm{{pred}}/P_{{{NL_TYPE}}}^\mathrm{{true}} - 1\,|$",
        fontsize=TICK_FS - 1)
    cb.ax.tick_params(labelsize=9)
    ref_handle = mlines.Line2D([], [], color="black", marker="x", linestyle="none",
                               markersize=8, markeredgewidth=2.0, label="Reference cosmology")
    fig.legend(handles=[ref_handle], loc="upper right",
               bbox_to_anchor=(0.995, 0.995), fontsize=TICK_FS - 1, framealpha=0.85)
    fig.suptitle(
        (f"Emulation error vs. parameter pairs — {MODEL_TYPE.upper()}\n"
         f"{COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, z={z_val:.4g}, {N_TRAIN} batches"),
        fontsize=AXES_FS - 3, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.70, 1])
    fname = (f"{FIG_DIR}/triangle_boost_error_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_{MODEL_TYPE}_z{z_val:.4g}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Datagenerator comparison plot
# ---------------------------------------------------------------------------

def plot_datagen_comparison(ks, redshift_indices):
    """
    Multi-panel figure: one row per target redshift, three lines each:

      1. Black dashed flat line at 1.0 — datagenerator / datagenerator (reference).
      2. Green line  — CAMB residual frac_pks = P_nl_camb / P_nl_syren at the
                       fiducial cosmology (shows how much the emulator must
                       correct beyond the syren baseline).
      3. Blue line   — emulator P_nl(k,z) / datagenerator P_nl(k,z).

    This is a single-cosmology diagnostic (no bands).
    """
    if DG_PK_PATH is None:
        print("  [INFO] DG_PK_PATH is None — skipping datagenerator comparison.")
        return
    if not os.path.exists(DG_PK_PATH):
        print(f"  [WARN] Datagenerator file not found: {DG_PK_PATH}")
        print("         Skipping plot_datagen_comparison.")
        return

    # Datagenerator reference nonlinear P(k,z): shape assumed (N_z, N_k)
    dg_pk_all = np.load(DG_PK_PATH)[0]

    print("[INFO] Running emulator at fiducial cosmology for datagen comparison...")
    _, _, pk_emu_all = pk_emu.get_pks(
        FIDUCIAL_PARAMS,
        cosmo_type             = COSMO_TYPE,
        prior_type             = PRIOR_TYPE,
        nl_type                = NL_TYPE,
        model_type             = MODEL_TYPE,
        num_batches            = N_TRAIN,
        w0_min                 = W0_MIN_FILTER,
        w0wa_max               = W0WA_MAX_FILTER,
        use_approximation_only = False,
    )   # (N_z, N_k)

    # Syren halofit baseline at fiducial — this is what the emulator multiplies
    # against.  use_approximation_only=True returns P_nl_syren in boost mode.
    _, _, pk_syren_nl = pk_emu.get_pks(
        FIDUCIAL_PARAMS,
        cosmo_type             = COSMO_TYPE,
        prior_type             = PRIOR_TYPE,
        nl_type                = NL_TYPE,
        model_type             = MODEL_TYPE,
        num_batches            = N_TRAIN,
        w0_min                 = W0_MIN_FILTER,
        w0wa_max               = W0WA_MAX_FILTER,
        use_approximation_only = True,
    )   # (N_z, N_k)  — P_nl_syren

    n_rows = len(redshift_indices)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(10, 3.2 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, (iz, z_val) in zip(axes, redshift_indices):
        dg_z = _to_grid(DG_KS, dg_pk_all[iz], ks)

        # Emulator ratio to datagenerator
        ratio_emu = _to_grid(ks, pk_emu_all[iz], ks) / dg_z

        # CAMB residual at fiducial: dg_nonlin / P_nl_syren
        # Shows how much the emulator network must contribute beyond syren
        camb_residual = dg_z / _to_grid(ks, pk_syren_nl[iz], ks)

        ax.axhline(1.0, color=COLOR_DG, linestyle="--", lw=1.5,
                   label="Datagen (reference)")
        ax.semilogx(ks, camb_residual, color=COLOR_BOOST, lw=1.8,
                    label=r"CAMB $P_\mathrm{nl}$ / $P_\mathrm{nl,syren}$ (target residual)")
        ax.semilogx(ks, ratio_emu,     color=COLOR_EMU,   lw=1.8,
                    label=f"{MODEL_TYPE.upper()} / Datagen")

        ax.set_ylabel("P / P_datagen", fontsize=TICK_FS - 2)
        ax.set_xlim(ks[0], ks[-1])
        ax.set_ylim(0.88, 1.12)
        ax.grid(alpha=0.25)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FS - 3)
        ax.set_title(f"z = {z_val:.4g}", fontsize=TICK_FS, loc="right")

    axes[0].legend(fontsize=LEGEND_FS - 3, loc="upper right")
    axes[-1].set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    fig.suptitle(
        f"Emulator vs datagenerator (fiducial cosmology)\n"
        f"{MODEL_TYPE.upper()} | {COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, "
        f"{N_TRAIN} batches",
        fontsize=AXES_FS - 2, y=1.005
    )
    plt.tight_layout()
    fname = (f"{FIG_DIR}/datagen_comparison_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_{MODEL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}_allz.pdf")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family']      = 'STIXGeneral'

    os.makedirs(FIG_DIR, exist_ok=True)

    redshift_indices = resolve_redshift_indices()

    print("=" * 60)
    print("[INFO] Nonlinear Emulator Evaluation")
    print(f"       cosmo_type      = {COSMO_TYPE}")
    print(f"       prior_type      = {PRIOR_TYPE}")
    print(f"       nl_type         = {NL_TYPE}")
    print(f"       model_type      = {MODEL_TYPE}")
    print(f"       n_train         = {N_TRAIN}")
    print(f"       test_batch      = {START_BATCH}")
    print(f"       W0_MIN_FILTER   = {W0_MIN_FILTER}")
    print(f"       W0WA_MAX_FILTER = {W0WA_MAX_FILTER}")
    print(f"       W0_THRESHOLD    = {W0_THRESHOLD}  (diagnostic split only)")
    print(f"       W0WA_THRESHOLD  = {W0WA_THRESHOLD}  (diagnostic split only)")
    print(f"       DG_PK_PATH      = {DG_PK_PATH}")
    print(f"       Redshifts       = {[(iz, f'z={z:.4g}') for iz, z in redshift_indices]}")
    print("=" * 60)

    print("\n[INFO] Loading test set...")
    test_set = load_test_set()
    n_params = test_set.lhs.shape[1]
    print(f"[INFO] Parameter space: {n_params} columns "
          f"({'tfree_mnufree' if n_params > 7 else 'legacy 7-param'})")

    print("[INFO] Computing predictions (all redshifts)...")
    pred_pks, true_pks, true_frac, pred_frac, lhs_clean = compute_predictions(test_set)

    ks_full = utils.ks   # full 500-point k-grid

    nl_pcas, nl_scalers = _load_nl_pca_bundle(n_params)

    # -----------------------------------------------------------------------
    # Per-redshift evaluation loop
    # -----------------------------------------------------------------------
    for iz, z_val in redshift_indices:
        print(f"\n{'=' * 60}")
        print(f"[INFO] Evaluating iz={iz}  →  z={z_val:.4g}")
        print(f"{'=' * 60}")

        errors = pred_pks[:, iz, :] / true_pks[:, iz, :] - 1

        true_frac_z = true_frac[:, iz, :]
        errors_filt, true_frac_filt, good_mask = apply_diagnostic_filter(
            errors, true_frac_z, lhs_clean)

        # w0 = lhs_clean[:, W0_COL]
        # print(f"  Total cosmologies : {len(lhs_clean)}")
        # print(f"  Both cuts pass    : {good_mask.sum()}")
        # print(f"  w0 range (full)   : [{w0.min():.3f}, {w0.max():.3f}]")
        # if good_mask.sum() > 0:
        #     print(f"  w0 range (filt.)  : [{w0[good_mask].min():.3f}, {w0[good_mask].max():.3f}]")

        print_statistics(errors, errors_filt, iz, z_val)

        print(f"\n[INFO] Saving figures for z={z_val:.4g}...")
        plot_error_triangle(errors, lhs_clean, iz, z_val)
        plot_residual_logfracs(test_set, iz, z_val)
        plot_boost_error_lines_filtered(errors_filt, ks_full, good_mask, iz, z_val)
        plot_comparison_bands(errors, ks_full, iz, z_val)
        plot_max_error_histogram(errors, iz, z_val)
        # plot_residual_logfracs(test_set, iz, z_val)
        plot_frac_shape_bands(true_frac, ks_full, iz, z_val)
        # plot_error_triangle(errors, lhs_clean, iz, z_val)
        plot_frac_shape_bands(true_frac, ks_full, iz, z_val)
        plot_comparison_bands_zoomed(errors, ks_full, iz, z_val)
        plot_max_error_vs_tagn(errors, lhs_clean, iz, z_val)
        plot_worst_offenders_in_window(errors, ks_full, iz, z_val)
        plot_error_heatmap_vs_tagn(errors, ks_full, lhs_clean, iz, z_val)
        if nl_pcas is not None:                                                            # <-- ADD
            plot_pc_true_vs_predicted(true_frac, pred_frac, nl_pcas, nl_scalers, lhs_clean, iz, z_val)
            plot_pc_residual_vs_tagn(true_frac, pred_frac, nl_pcas, nl_scalers, lhs_clean, iz, z_val)

    # -----------------------------------------------------------------------
    # Datagenerator comparison (all z in one multi-panel figure)
    # -----------------------------------------------------------------------
    print("\n[INFO] Saving datagenerator comparison figure (all redshifts)...")
    plot_datagen_comparison(ks_full, redshift_indices)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()