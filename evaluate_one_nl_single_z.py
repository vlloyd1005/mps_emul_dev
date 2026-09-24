"""
evaluate_one_nl_single_z.py — Single-redshift Emulator Evaluation Script

Evaluates one trained single-z emulator (from train_single_z.py) against the
CAMB ground truth on a test batch and produces:

  1. Per-cosmology error lines (diagnostic subset)
  2. Percentile error bands (full k range + zoomed window)
  3. Histogram of max |error| across cosmologies
  4. log(frac_pks) diagnostic, coloured by the w0 / w0+wa split
  5. Error triangle plot, T_AGN diagnostics, PC-level diagnostics
  6. (optional) Datagenerator comparison at the fiducial cosmology

The emulator is loaded directly (Keras model + metadata bundle) — no
dependency on emulmps_w0wa.get_pks.

Error definition
----------------
The emulator predicts frac_pk:
    lin mode   : frac = P_lin_camb / P_lin_syren
    boost mode : frac = (P_nl_camb / P_lin_camb) / boost_syren_halofit
Every other factor in P(k) is identical between the emulated and true
spectrum, so
    P^emu / P^CAMB - 1  =  frac_pred / frac_true - 1
exactly.  In boost mode this isolates the nonlinear emulator's error (the
linear P(k) is taken as exact).

Prior cuts (W0_MIN_FILTER, W0WA_MAX_FILTER, OMEGAB_H0_TRIANGLE_CUT) must match
the values used during training.

Usage:
    python ./mps_emu/evaluate_one_nl_single_z.py

Author: Victoria Lloyd (2025)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d

from tensorflow import keras

import mps_emu_single_z as utils
from mps_emu_single_z import VER


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

START_BATCH    = 1000        # batch index used as the test/validation set
N_TRAIN        = 50          # number of training batches for the model being evaluated
COSMO_TYPE     = "w0wacdm"
NL_TYPE        = "mead2020_Tfree_mnufree"
PRIOR_TYPE     = "expanded"
MODEL_TYPE     = "npce"      # "mlp" or "npce" — must match training
TARGET_Z       = 1.0         # must match --target_z used in training

MODEL_DIR    = "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/models"
METADATA_DIR = "mps_emu/metadata"
FIG_DIR      = "mps_emu/validation_figs/single_z"

# Prior cuts — must match values used during training
W0_MIN_FILTER   = None      # None to disable
W0WA_MAX_FILTER = None      # None to disable

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

K_WINDOW_LO, K_WINDOW_HI = 20.0, 30.0
K_WINDOW_MASK = (utils.ks >= K_WINDOW_LO) & (utils.ks <= K_WINDOW_HI)
TAGN_COL = utils.params_tfree_mnufree.index("T_AGN")   # column 7 in lhs

USE_BOOST = NL_TYPE not in ("lin", "mead2020_Tfree_mnufree_lin")

# Datagenerator comparison (optional).  Files should have shape (1, N_z, N_k)
# on DG_KS, same convention as the old script.  In boost mode the CAMB
# *linear* datagen output for the same cosmology is also needed, since the
# emulator predicts P_nl relative to P_lin.
DG_PK_PATH     = None
DG_LIN_PK_PATH = None
DG_KS = np.logspace(-5.1, 2, 500)

# Fiducial cosmology (lhs convention):
# [10^9 A_s, ns, H0, Omega_b, Omega_m, w0, w0+wa, (log T_AGN, mnu)]
FIDUCIAL_PARAMS = [2.1, 0.96605, 67.32, 0.04, 0.3, -0.9, -0.9, 7.3, 0.06]


# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

COLOR_50    = "#473C8A"
COLOR_90    = "#C45858"
COLOR_100   = "lightgray"
COLOR_GOOD  = "#2166ac"
COLOR_BAD   = "#d6604d"
COLOR_EMU   = "#1565c0"
COLOR_BOOST = "#2ca02c"
COLOR_DG    = "black"

AXES_FS   = 20
TICK_FS   = 17
LEGEND_FS = 17

HANDLE_95 = mlines.Line2D([], [], color=COLOR_100, label=r"$95\%$")
HANDLE_90 = mpatches.Patch(facecolor=COLOR_90, label=r"$90\%$")
HANDLE_50 = mpatches.Patch(facecolor=COLOR_50, hatch="\\", edgecolor="lightgray", label=r"$50\%$")

ERR_LABEL = (fr"$P_{{{NL_TYPE}}}^\mathrm{{{MODEL_TYPE.upper()}}} / "
             fr"P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1$")


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
    return "_" + "_".join(parts)


# ---------------------------------------------------------------------------
# Emulator loading / prediction
# ---------------------------------------------------------------------------

def _resolve_tag(n_params, z_val):
    """Reproduce COLASet._metadata_tag() exactly as used at training time."""
    dummy = utils.COLASet.__new__(utils.COLASet)
    dummy.cosmo_type = COSMO_TYPE
    dummy.prior_type = PRIOR_TYPE
    dummy.nl_type    = NL_TYPE
    dummy.n_batches  = N_TRAIN
    dummy.w0_min     = W0_MIN_FILTER
    dummy.w0wa_max   = W0WA_MAX_FILTER
    dummy.use_boost  = USE_BOOST
    dummy.lhs        = np.empty((1, n_params))   # only .shape[1] is read
    dummy.z          = np.array([z_val])
    return dummy._metadata_tag()


def load_emulator(n_params, z_val):
    tag = _resolve_tag(n_params, z_val)
    bundle_path = os.path.join(METADATA_DIR, f"metadata_{tag}", "metadata.joblib")
    model_path  = os.path.join(MODEL_DIR, f"emulator_{MODEL_TYPE}_{tag}.keras")
    print(f"[INFO] Metadata bundle: {bundle_path}")
    print(f"[INFO] Keras model    : {model_path}")
    for p in (bundle_path, model_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    bundle = utils.load_metadata_bundle(bundle_path)
    model  = keras.models.load_model(
        model_path,
        custom_objects={"CustomActivationLayer": utils.CustomActivationLayer},
        compile=False,   # custom loss isn't needed for inference
    )

    if MODEL_TYPE == "npce" and bundle.get("npce_indices") is None:
        raise ValueError("MODEL_TYPE='npce' but the metadata bundle has no npce_indices.")

    z_key = sorted(bundle["pcas"].keys())[0]
    return {
        "model":        model,
        "param_scaler": bundle["param_scaler"],
        "t_scaler":     bundle["t_comp_scaler"],
        "pca":          bundle["pcas"][z_key],
        "scaler":       bundle["scalers"][z_key],
        "pcas":         bundle["pcas"],
        "scalers":      bundle["scalers"],
        "npce_indices": bundle.get("npce_indices"),
    }


def predict_frac(emu, lhs):
    """Emulated frac_pk, shape (N, N_k).  lhs in COLASet convention (w0+wa in col 6)."""
    x = emu["param_scaler"].transform(np.atleast_2d(lhs))
    if MODEL_TYPE == "npce":
        from cola_npce_single_z import _evaluate_pce_basis
        x = _evaluate_pce_basis(emu["npce_indices"], x)
    t_norm  = emu["model"].predict(x.astype(np.float32), batch_size=4096, verbose=0)
    pcs     = emu["t_scaler"].inverse_transform(t_norm)
    logfrac = emu["scaler"].inverse_transform(emu["pca"].inverse_transform(pcs))
    return np.exp(logfrac)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_set():
    """Load the single-z test set and apply the training prior cuts."""
    test_set = utils.COLASet(
        target_z    = TARGET_Z,
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

        slope   = (h0_anchor - h0_max) / (ob_max - ob_anchor)
        h0_line = h0_max + slope * (test_set.lhs[:, omegab_col] - ob_anchor)
        triangle_mask = test_set.lhs[:, h_col] <= h0_line
        print(f"[INFO] Omega_b/H0 triangle cut: "
              f"removing {(~triangle_mask & mask).sum()} additional cosmologies.")
        mask &= triangle_mask

    for attr in ("lhs", "pks_target", "frac_pks", "logfracs",
                 "mps_approxes_boost", "mps_approxes"):
        if getattr(test_set, attr, None) is not None:
            setattr(test_set, attr, getattr(test_set, attr)[mask])
    if not test_set.use_boost:
        test_set.pks_lin = test_set.pks_target

    print(f"[INFO] Test set after prior cuts: {mask.sum()} / {len(mask)} cosmologies kept.")
    return test_set


def compute_predictions(test_set, emu):
    """
    Returns
    -------
    pred_frac : (N_valid, 1, N_k)  emulated frac_pk
    true_frac : (N_valid, 1, N_k)  CAMB frac_pk
    lhs_clean : (N_valid, N_params)
    """
    pred_frac = predict_frac(emu, test_set.lhs)[:, None, :]
    true_frac = test_set.frac_pks

    nan_mask = ~np.isfinite(pred_frac).all(axis=(1, 2))
    if nan_mask.any():
        print(f"  WARNING: dropping {nan_mask.sum()} cosmologies with non-finite output.")
    valid = ~nan_mask
    return pred_frac[valid], true_frac[valid], test_set.lhs[valid]


def apply_diagnostic_filter(errors, true_frac, lhs_clean):
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


def print_statistics(errors, errors_filt, z_val):
    subsets = [
        ("Full dataset",                                  errors),
        (f"w0>{W0_THRESHOLD} & w0+wa<{W0WA_THRESHOLD}",  errors_filt),
    ]
    print(f"\n=== ERROR STATISTICS  (z={z_val:.4f}) ===")
    for label, e in subsets:
        if len(e) == 0:
            print(f"\n  {label}: (empty)")
            continue
        max_e = np.max(np.abs(e), axis=1)
        print(f"\n  {label}  (N={len(e)}):")
        print(f"    Mean max |error|:     {np.mean(max_e):.6f}")
        print(f"    Median max |error|:   {np.median(max_e):.6f}")
        print(f"    95th pct max |error|: {np.percentile(max_e, 95):.6f}")
        print(f"    99th pct max |error|: {np.percentile(max_e, 99):.6f}")
        print(f"    Frac. with max |error| < 0.5%: {np.mean(max_e < 0.005):.4f}")
        print(f"    Frac. with max |error| < 1%:   {np.mean(max_e < 0.01):.4f}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _to_grid(src_ks, src_Pk, tgt_ks):
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


def _fname(stem, z_val):
    return (f"{FIG_DIR}/{stem}_z{z_val:.4g}_{COSMO_TYPE}_{PRIOR_TYPE}"
            f"_{NL_TYPE}_{MODEL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def plot_boost_error_lines_filtered(errors_filt, ks, good_mask, z_val):
    n_plot = min(good_mask.sum(), 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_plot):
        ax.semilogx(ks, errors_filt[i, :])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(ERR_LABEL, fontsize=AXES_FS)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.text(0.05, 0.95,
            fr'$z={z_val:.4g}$,  $w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$  (N={good_mask.sum()})',
            transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FS)
    fname = _fname(f"val_errors_w0gt{W0_THRESHOLD}_w0walt{W0WA_THRESHOLD}", z_val)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_comparison_bands(errors, ks, z_val):
    p_vals = percentiles(errors)
    fig, ax = plt.subplots(figsize=(8, 5))
    _fill_ax(ax, ks, *p_vals, ylabel=ERR_LABEL, label_text=f'$z={z_val:.4g}$')
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95],
              fontsize=LEGEND_FS, loc="lower right")
    plt.tight_layout()
    fname = _fname("error_bands", z_val)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_comparison_bands_zoomed(errors, ks, z_val, k_lo=10.0, k_hi=50.0):
    mask = (ks >= k_lo) & (ks <= k_hi)
    p_vals = percentiles(errors[:, mask])
    fig, ax = plt.subplots(figsize=(8, 5))
    _fill_ax(ax, ks[mask], *p_vals, ylabel=ERR_LABEL,
             label_text=f'$z={z_val:.4g}$  (k zoom {k_lo:g}-{k_hi:g})')
    ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="purple", alpha=0.08)
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95], fontsize=LEGEND_FS, loc="best")
    plt.tight_layout()
    fname = _fname("error_bands_zoom", z_val)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_max_error_histogram(errors, z_val):
    xlabel = (fr"Max $|P_{{{NL_TYPE}}}^\mathrm{{{MODEL_TYPE.upper()}}} / "
              fr"P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1|$")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.max(np.abs(errors), axis=1), bins=50,
            color='#1565c0', alpha=0.7, edgecolor='black')
    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title(f'{MODEL_TYPE.upper()} — {COSMO_TYPE}, {NL_TYPE}   ($z={z_val:.4g}$)', fontsize=18)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=14)
    plt.tight_layout()
    fname = _fname("max_error_histogram", z_val)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_residual_logfracs(test_set, z_val):
    """Per-cosmology log(frac_pks) — the quantity the emulator learns."""
    w0        = test_set.lhs[:, W0_COL]
    w0wa      = test_set.lhs[:, W0WA_COL]
    mask_good = (w0 > W0_THRESHOLD) & (w0wa < W0WA_THRESHOLD)
    mask_bad  = ~mask_good
    logfracs_z = test_set.logfracs[:, 0, :]
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
    ax.set_ylabel(r"$\log(\mathrm{frac}_{P})$", fontsize=AXES_FS)
    ax.set_title(
        fr"$z={z_val:.4g}$   |   {COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, {MODEL_TYPE.upper()}",
        fontsize=15)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = _fname("residual_logfracs", z_val)
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_frac_shape_bands(true_frac, ks, z_val):
    """Percentile bands of the CAMB target frac_pks - 1."""
    p_vals = percentiles(true_frac[:, 0, :] - 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    _fill_ax(ax, ks, *p_vals,
             ylabel=r"$\mathrm{frac}_{P}^\mathrm{CAMB} - 1$",
             label_text=f'$z={z_val:.4g}$  (CAMB target)')
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95],
              fontsize=LEGEND_FS, loc="upper left")
    plt.tight_layout()
    fname = _fname("camb_target_frac", z_val)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_max_error_vs_tagn(errors, lhs_clean, z_val):
    if lhs_clean.shape[1] <= TAGN_COL:
        print("  [WARN] T_AGN column not present — skipping plot_max_error_vs_tagn.")
        return

    tagn_vals      = lhs_clean[:, TAGN_COL]
    max_err_window = np.max(np.abs(errors[:, K_WINDOW_MASK]), axis=1)
    max_err_full   = np.max(np.abs(errors), axis=1)

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
    ax.set_xlabel(r"$\log T_\mathrm{AGN}$", fontsize=AXES_FS)
    ax.set_ylabel(fr"max $|$error$|$ in $k \in [{K_WINDOW_LO:g},{K_WINDOW_HI:g}]$", fontsize=AXES_FS - 4)
    ax.set_title(f"z={z_val:.4g}  (colour = max |error| over full k range)", fontsize=AXES_FS - 5)
    ax.grid(alpha=0.3, which="both")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("max |error| (full k)", fontsize=TICK_FS - 3)
    plt.tight_layout()
    fname = _fname("max_error_vs_tagn", z_val)
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


def plot_worst_offenders_in_window(errors, ks, z_val, n_worst=30):
    max_err_window = np.max(np.abs(errors[:, K_WINDOW_MASK]), axis=1)
    worst_idx = np.argsort(max_err_window)[::-1][:n_worst]

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in worst_idx:
        ax.semilogx(ks, errors[idx], alpha=0.6, lw=1)
    ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="purple", alpha=0.08)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(ERR_LABEL, fontsize=AXES_FS - 4)
    ax.set_title(f"Worst {n_worst} cosmologies by max|error| in k={K_WINDOW_LO:g}-{K_WINDOW_HI:g} (z={z_val:.4g})",
                 fontsize=AXES_FS - 6)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = _fname("worst_offenders_window", z_val)
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_error_heatmap_vs_tagn(errors, ks, lhs_clean, z_val, n_tagn_bins=10):
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

    finite_vals = heatmap[np.isfinite(heatmap) & (heatmap > 0)]
    if finite_vals.size == 0:
        print(f"  [WARN] No finite positive values for heatmap at z={z_val:.4g} — skipping.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    im = ax.pcolormesh(ks, bin_centres, heatmap, shading="auto", cmap="YlOrRd",
                       norm=mpl.colors.LogNorm(vmin=finite_vals.min(), vmax=finite_vals.max()))
    ax.set_xscale("log")
    ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="cyan", alpha=0.15)
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(r"$\log T_\mathrm{AGN}$", fontsize=AXES_FS)
    ax.set_title(f"Median |error| vs (k, T_AGN)  (z={z_val:.4g})", fontsize=AXES_FS - 5)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("median |error|", fontsize=TICK_FS - 3)
    for b, c in enumerate(counts):
        ax.text(ks[-1] * 1.02, bin_centres[b], f"n={c}", fontsize=7, va="center", clip_on=False)
    plt.tight_layout()
    fname = _fname("error_heatmap_vs_tagn", z_val)
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# PC-level diagnostics
# ---------------------------------------------------------------------------

def _pc_responsibility(pca, scaler, ks=None, k_lo=K_WINDOW_LO, k_hi=K_WINDOW_HI):
    """Fraction of each PC's (de-standardised) energy inside the k window."""
    ks = utils.ks if ks is None else ks
    k_mask = (ks >= k_lo) & (ks <= k_hi)
    mat = pca.components_ * scaler.std[None, :]
    return np.sum(mat[:, k_mask] ** 2, axis=1) / np.sum(mat ** 2, axis=1)


def _frac_to_pcs(frac_z, pca, scaler):
    return pca.transform(scaler.transform(np.log(frac_z)))


def plot_pc_true_vs_predicted(true_frac, pred_frac, emu, lhs_clean, z_val, top_n=3):
    pca, scaler = emu["pca"], emu["scaler"]
    pcs_true = _frac_to_pcs(true_frac[:, 0, :], pca, scaler)
    pcs_pred = _frac_to_pcs(pred_frac[:, 0, :], pca, scaler)
    resp     = _pc_responsibility(pca, scaler)
    top_pcs  = np.argsort(resp)[::-1][:top_n]

    has_tagn = lhs_clean.shape[1] > TAGN_COL

    fig, axes = plt.subplots(1, top_n, figsize=(5.5 * top_n, 5), squeeze=False)
    for ax, p in zip(axes[0], top_pcs):
        x, y = pcs_true[:, p], pcs_pred[:, p]
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6, zorder=1)
        if has_tagn:
            sc = ax.scatter(x, y, c=lhs_clean[:, TAGN_COL], cmap="viridis", s=10,
                            alpha=0.6, rasterized=True)
            fig.colorbar(sc, ax=ax, label=r"$\log T_\mathrm{AGN}$")
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
    fname = _fname("pc_true_vs_pred", z_val)
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def plot_pc_residual_vs_tagn(true_frac, pred_frac, emu, lhs_clean, z_val, top_n=3):
    if lhs_clean.shape[1] <= TAGN_COL:
        print("  [WARN] T_AGN column not present — skipping plot_pc_residual_vs_tagn.")
        return
    pca, scaler = emu["pca"], emu["scaler"]
    pcs_true = _frac_to_pcs(true_frac[:, 0, :], pca, scaler)
    pcs_pred = _frac_to_pcs(pred_frac[:, 0, :], pca, scaler)
    resp     = _pc_responsibility(pca, scaler)
    top_pcs  = np.argsort(resp)[::-1][:top_n]
    tagn_vals = lhs_clean[:, TAGN_COL]

    fig, axes = plt.subplots(1, top_n, figsize=(5.5 * top_n, 4), squeeze=False)
    for ax, p in zip(axes[0], top_pcs):
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.scatter(tagn_vals, pcs_pred[:, p] - pcs_true[:, p],
                   s=8, alpha=0.4, rasterized=True, color="C2")
        ax.set_xlabel(r"$\log T_\mathrm{AGN}$")
        ax.set_ylabel("predicted - true")
        ax.set_title(f"z={z_val:.2f}, PC {p} (resp={resp[p]:.2f})", fontsize=AXES_FS - 6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fname = _fname("pc_residual_vs_tagn", z_val)
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Triangle plot
# ---------------------------------------------------------------------------

_PARAM_LABELS_7 = [
    r'$10^9 A_s$', r'$n_s$', r'$H_0$', r'$\Omega_b$',
    r'$\Omega_m$', r"$w_0$", r"$w_0+w_a$",
]
# Column order follows utils.params_tfree_mnufree: T_AGN (col 7), then mnu (col 8)
_PARAM_LABELS_9 = _PARAM_LABELS_7 + [r'$\log T_{\rm AGN}$', r'$m_\nu$']


def plot_error_triangle(errors, lhs_clean, z_val):
    import matplotlib.colors as mcolors

    n_params = min(lhs_clean.shape[1], 9)
    labels   = (_PARAM_LABELS_9 if n_params > 7 else _PARAM_LABELS_7)[:n_params]
    ref_cosmo = np.array(FIDUCIAL_PARAMS[:n_params])

    mean_abs_err = np.mean(np.abs(errors), axis=1)
    finite_mask  = np.isfinite(mean_abs_err)
    if (~finite_mask).any():
        print(f"  [WARN] {(~finite_mask).sum()} cosmologies have non-finite mean error "
              f"— clamping to finite max for triangle plot.")
    finite_max = mean_abs_err[finite_mask].max() if finite_mask.any() else 1.0
    mean_abs_err_plot = np.where(finite_mask, mean_abs_err, finite_max)

    vmin, vmax = mean_abs_err_plot.min(), mean_abs_err_plot.max()
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
        print(f"  [WARN] Cannot build colormap norm (vmin={vmin}, vmax={vmax}) — skipping triangle plot.")
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
                ax.axvline(ref_cosmo[i], color="black", lw=1.4, ls="--", zorder=5)
            else:
                ax.scatter(lhs_clean[:, j], lhs_clean[:, i],
                           c=mean_abs_err_plot, cmap=cmap, norm=norm,
                           s=15, linewidths=0, rasterized=True, alpha=0.8)
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
    fname = _fname("triangle_error", z_val)
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Datagenerator comparison (single z, fiducial cosmology)
# ---------------------------------------------------------------------------

def plot_datagen_comparison(emu, ks, z_val, n_params):
    """
    Single panel at the fiducial cosmology:
      - black dashed : datagen / datagen = 1
      - green        : CAMB target frac (how much the emulator must correct)
      - blue         : emulator P / datagen P

    Emulated P is rebuilt as frac_pred * baseline, where
      lin mode   : baseline = P_lin_syren
      boost mode : baseline = boost_syren_halofit * P_lin_datagen
    """
    if DG_PK_PATH is None:
        print("  [INFO] DG_PK_PATH is None — skipping datagenerator comparison.")
        return
    if not os.path.exists(DG_PK_PATH):
        print(f"  [WARN] Datagenerator file not found: {DG_PK_PATH} — skipping.")
        return
    if USE_BOOST and (DG_LIN_PK_PATH is None or not os.path.exists(DG_LIN_PK_PATH)):
        print("  [WARN] Boost mode needs DG_LIN_PK_PATH (CAMB linear at the fiducial) — skipping.")
        return

    iz_grid = int(np.argmin(np.abs(np.asarray(utils.z_mps) - z_val)))
    dg_z    = _to_grid(DG_KS, np.load(DG_PK_PATH)[0][iz_grid], ks)

    fid = np.asarray(FIDUCIAL_PARAMS[:n_params], dtype=float)
    frac_emu = predict_frac(emu, fid[None, :])[0]

    if USE_BOOST:
        plin_dg = _to_grid(DG_KS, np.load(DG_LIN_PK_PATH)[0][iz_grid], ks)
        syren_boost = utils._compute_mps_nl_approximation_parallel(
            ks, np.array([z_val]), fid[None, :], n_jobs=1)[0, 0]
        baseline = syren_boost * plin_dg
    else:
        baseline = utils._compute_mps_approximation(ks, np.array([z_val]), fid)[0]

    pk_emu = frac_emu * baseline

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(1.0, color=COLOR_DG, linestyle="--", lw=1.5, label="Datagen (reference)")
    ax.semilogx(ks, dg_z / baseline, color=COLOR_BOOST, lw=1.8,
                label="CAMB target frac")
    ax.semilogx(ks, pk_emu / dg_z,  color=COLOR_EMU,   lw=1.8,
                label=f"{MODEL_TYPE.upper()} / Datagen")
    ax.set_ylabel("P / P_datagen", fontsize=TICK_FS - 2)
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_xlim(ks[0], ks[-1])
    ax.set_ylim(0.88, 1.12)
    ax.grid(alpha=0.25)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS - 3)
    ax.set_title(f"Fiducial cosmology, z = {z_val:.4g}  |  {MODEL_TYPE.upper()}, {NL_TYPE}",
                 fontsize=TICK_FS - 2)
    ax.legend(fontsize=LEGEND_FS - 4, loc="upper left")
    plt.tight_layout()
    fname = _fname("datagen_comparison", z_val)
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

    print("=" * 60)
    print("[INFO] Single-z Emulator Evaluation")
    print(f"       cosmo_type      = {COSMO_TYPE}")
    print(f"       prior_type      = {PRIOR_TYPE}")
    print(f"       nl_type         = {NL_TYPE}  (boost mode: {USE_BOOST})")
    print(f"       model_type      = {MODEL_TYPE}")
    print(f"       target_z        = {TARGET_Z}")
    print(f"       n_train         = {N_TRAIN}")
    print(f"       test_batch      = {START_BATCH}")
    print(f"       W0_MIN_FILTER   = {W0_MIN_FILTER}")
    print(f"       W0WA_MAX_FILTER = {W0WA_MAX_FILTER}")
    print("=" * 60)

    print("\n[INFO] Loading test set...")
    test_set = load_test_set()
    n_params = test_set.lhs.shape[1]
    z_val    = float(test_set.z[0])
    print(f"[INFO] Parameter space: {n_params} columns; evaluating at z={z_val:.4g}")

    print("\n[INFO] Loading emulator...")
    emu = load_emulator(n_params, z_val)

    print("[INFO] Computing predictions...")
    pred_frac, true_frac, lhs_clean = compute_predictions(test_set, emu)

    ks = utils.ks
    errors = pred_frac[:, 0, :] / true_frac[:, 0, :] - 1

    errors_filt, _, good_mask = apply_diagnostic_filter(errors, true_frac[:, 0, :], lhs_clean)
    print_statistics(errors, errors_filt, z_val)

    print(f"\n[INFO] Saving figures for z={z_val:.4g}...")
    plot_error_triangle(errors, lhs_clean, z_val)
    plot_residual_logfracs(test_set, z_val)
    plot_boost_error_lines_filtered(errors_filt, ks, good_mask, z_val)
    plot_comparison_bands(errors, ks, z_val)
    plot_comparison_bands_zoomed(errors, ks, z_val)
    plot_max_error_histogram(errors, z_val)
    plot_frac_shape_bands(true_frac, ks, z_val)
    plot_max_error_vs_tagn(errors, lhs_clean, z_val)
    plot_worst_offenders_in_window(errors, ks, z_val)
    plot_error_heatmap_vs_tagn(errors, ks, lhs_clean, z_val)
    plot_pc_true_vs_predicted(true_frac, pred_frac, emu, lhs_clean, z_val)
    plot_pc_residual_vs_tagn(true_frac, pred_frac, emu, lhs_clean, z_val)

    print("\n[INFO] Datagenerator comparison...")
    plot_datagen_comparison(emu, ks, z_val, n_params)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()