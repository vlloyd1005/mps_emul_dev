"""
evaluate_one.py — Single-Model Emulator Evaluation Script

Evaluates a single trained emulator model against CAMB ground-truth power
spectra on the test set and produces:

  1. Per-cosmology Syren (EH approximation) error lines (filtered subset)
  2. Two-panel comparison plots (full vs. filtered) for the emulator and Syren
  3. Histograms of the maximum absolute error across cosmologies
  4. EH log-fraction diagnostic, coloured by w0
  5. Datagenerator comparison: emulator/syren error bands vs fiducial
     CAMB from ../mps/output/one_eval_wcdm_500ks_pklin.npy, one panel per z

Prior cuts: the Omega_b/H0 triangle cut (OMEGAB_H0_TRIANGLE_CUT) is applied to
the test set and must match the --omegab_anchor / --h0_anchor used during
training.  The w0 / w0+wa cuts are disabled (None) and are passed as None to
the emulator, so the model tag has no _w0min / _w0wamax suffix.

Usage (standalone):
    python ./mps_emu/evaluate_one.py

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
import train_utils_pk_emulator_v3 as utils      # same module emulmps_w0wa imports
VER = utils.VER


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------

START_BATCH    = 1000        # batch index used as the test/validation set
N_TRAIN        = 100        # number of training batches for the model being evaluated
COSMO_TYPE     = "w0wacdm"
NL_TYPE        = "mead2020_Tfree_mnufree_lin"
PRIOR_TYPE     = "expanded"
MODEL_TYPE     = "npce"      # "mlp" or "npce" — must match training

# Target redshifts — nearest available z in utils.z_mps will be used for each
TARGET_REDSHIFTS = [0, 0.5, 1, 2, 5, 10, 30, 50]

# Prior cuts — must match values used during training
W0_MIN_FILTER   = None       # disabled
W0WA_MAX_FILTER = None       # disabled

# Omega_b / H0 triangle cut: removes cosmologies above the line from
# (Omega_b = omegab_anchor, H0 = h0_max) to (Omega_b = omegab_max, H0 = h0_anchor).
# Must match --omegab_anchor / --h0_anchor used in training.  None to disable.
OMEGAB_H0_TRIANGLE_CUT = {
    'omegab_anchor': 0.05,
    'h0_anchor':     75,
    'omegab_max':    0.072,
    'h0_max':        90,
}

# Diagnostic thresholds (used only for the filtered subplot / stats split,
# not for removing cosmologies)
W0_THRESHOLD    = -1.8
W0WA_THRESHOLD  = -0.75
W0_COL          = utils.params.index("w")
W0WA_COL        = utils.params.index("w0+wa")

FIG_DIR         = "mps_emu/validation_figs/smaller_grid"

# Path to single-cosmology datagenerator reference, relative to this file
DG_PK_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "mps", "output", "one_eval_wcdm_500ks_pklin.npy"
)
# k grid used by both the datagenerator and the emulator
DG_KS = np.logspace(-5.1, 2, 500)

# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

COLOR_50    = "#473C8A"
COLOR_90    = "#C45858"
COLOR_100   = "lightgray"
COLOR_GOOD  = "#2166ac"
COLOR_BAD   = "#d6604d"
COLOR_NPCE  = "#1565c0"   # blue  — emulator bands
COLOR_SYREN = "#d6604d"   # red   — syren bands
COLOR_DG    = "black"     # datagenerator reference line

AXES_FS   = 20
TICK_FS   = 17
LEGEND_FS = 17

HANDLE_95 = mlines.Line2D([], [], color=COLOR_100, label=r"$95\%$")
HANDLE_90 = mpatches.Patch(facecolor=COLOR_90, label=r"$90\%$")
HANDLE_50 = mpatches.Patch(facecolor=COLOR_50, hatch="\\", edgecolor="lightgray", label=r"$50\%$")


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
    if W0_MIN_FILTER  is not None:
        parts.append(f"w0min{W0_MIN_FILTER}")
    if W0WA_MAX_FILTER is not None:
        parts.append(f"w0wamax{W0WA_MAX_FILTER}")
    if OMEGAB_H0_TRIANGLE_CUT is not None:
        parts.append(f"obh0cut{OMEGAB_H0_TRIANGLE_CUT['omegab_anchor']}"
                     f"-{OMEGAB_H0_TRIANGLE_CUT['h0_anchor']}")
    parts.append(VER)
    return ("_" + "_".join(parts)) if parts else ""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_test_set():
    test_set = utils.COLASet(
        target_z   = utils.z_mps,
        cosmo_type = COSMO_TYPE,
        prior_type = PRIOR_TYPE,
        nl_type    = NL_TYPE,
        start_batch= START_BATCH,
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

        # Line from (ob_anchor, h0_max) to (ob_max, h0_anchor); keep on/below it
        slope         = (h0_anchor - h0_max) / (ob_max - ob_anchor)
        h0_line       = h0_max + slope * (test_set.lhs[:, omegab_col] - ob_anchor)
        triangle_mask = test_set.lhs[:, h_col] <= h0_line
        print(f"[INFO] Omega_b/H0 triangle cut: "
              f"removing {(~triangle_mask & mask).sum()} additional cosmologies "
              f"(line from Omega_b={ob_anchor}, H0={h0_max} "
              f"to Omega_b={ob_max}, H0={h0_anchor}).")
        mask &= triangle_mask

    for attr in ("lhs", "pks_lin", "mps_approxes", "frac_pks", "logfracs"):
        setattr(test_set, attr, getattr(test_set, attr)[mask])

    print(f"[INFO] Test set after prior cuts: {mask.sum()} / {len(mask)} cosmologies kept.")
    return test_set


def compute_predictions(test_set):
    """
    Returns
    -------
    pred_pks   : (N_valid, N_z, N_k)
    syren_pks  : (N_valid, N_z, N_k)
    true_pks   : (N_valid, N_z, N_k)
    lhs_clean  : (N_valid, N_params)
    """
    true_pks_all = test_set.pks_lin
    pred_list, syren_list = [], []

    for params in test_set.lhs:
        _, _, pk_full = pk_emu.get_pks(
            params,
            cosmo_type             = COSMO_TYPE,
            prior_type             = PRIOR_TYPE,
            nl_type                = NL_TYPE,
            model_type             = MODEL_TYPE,
            num_batches            = N_TRAIN,
            w0_min                 = W0_MIN_FILTER,
            w0wa_max               = W0WA_MAX_FILTER,
            use_approximation_only = False,
        )
        _, _, pk_approx = pk_emu.get_pks(
            params,
            cosmo_type             = COSMO_TYPE,
            prior_type             = PRIOR_TYPE,
            nl_type                = NL_TYPE,
            model_type             = MODEL_TYPE,
            num_batches            = N_TRAIN,
            w0_min                 = W0_MIN_FILTER,
            w0wa_max               = W0WA_MAX_FILTER,
            use_approximation_only = True,
        )
        pred_list.append(pk_full)
        syren_list.append(pk_approx)

    pred_pks  = np.asarray(pred_list)
    syren_pks = np.asarray(syren_list)

    nan_mask = (
        np.isnan(syren_pks).any(axis=(1, 2)) |
        np.isnan(pred_pks).any(axis=(1, 2))
    )
    if nan_mask.any():
        print(f"  WARNING: dropping {nan_mask.sum()} cosmologies with NaN output.")

    valid = ~nan_mask
    return pred_pks[valid], syren_pks[valid], true_pks_all[valid], test_set.lhs[valid]


def apply_diagnostic_filter(errors, errors_syren, lhs_clean):
    w0_mask   = lhs_clean[:, W0_COL]   > W0_THRESHOLD
    w0wa_mask = lhs_clean[:, W0WA_COL] < W0WA_THRESHOLD
    good_mask = w0_mask & w0wa_mask
    return errors[good_mask], errors_syren[good_mask], good_mask


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentiles(err):
    ps = np.percentile(err, [5, 10, 25, 75, 90, 95], axis=0)
    return ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]


def print_statistics(errors, errors_syren, errors_filt, errors_syren_filt, iz, z_val):
    subsets = [
        ("Full dataset",                                        errors,      errors_syren),
        (f"w0>{W0_THRESHOLD} & w0+wa<{W0WA_THRESHOLD}",       errors_filt, errors_syren_filt),
    ]
    print(f"\n=== ERROR STATISTICS  (iz={iz}, z={z_val:.4f}) ===")
    for label, e_emu, e_syr in subsets:
        print(f"\n  {label}:")
        for name, e in [("EmulMPS", e_emu), ("Syren", e_syr)]:
            max_e = np.max(np.abs(e), axis=1)
            print(f"    {name} - Mean max |error|:     {np.mean(max_e):.6f}")
            print(f"    {name} - Median max |error|:   {np.median(max_e):.6f}")
            print(f"    {name} - 95th pct max |error|: {np.percentile(max_e, 95):.6f}")
            print(f"    {name} - 99th pct max |error|: {np.percentile(max_e, 99):.6f}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _to_grid(src_ks, src_Pk, tgt_ks):
    """Interpolate src_Pk onto tgt_ks, NaN outside range."""
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
# Existing plot functions
# ---------------------------------------------------------------------------

def plot_syren_lines_filtered(errors_syren_filt, ks, w0_mask, iz, z_val):
    n_plot = min(w0_mask.sum(), 200)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(n_plot):
        ax.semilogx(ks, errors_syren_filt[i, :])
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(r"$P_\mathrm{Syren}/P_\mathrm{CAMB} - 1$", fontsize=AXES_FS)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.text(0.05, 0.95,
            fr'$z={z_val:.4g}$,  $w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$  (N={w0_mask.sum()})',
            transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FS)
    fname = (f"{FIG_DIR}/val_fracs_{COSMO_TYPE}_z{z_val:.4g}"
             f"_w0gt{W0_THRESHOLD}_w0walt{W0WA_THRESHOLD}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_comparison_bands(errors, errors_filt, errors_syren, errors_syren_filt, ks, iz, z_val):
    model_label = MODEL_TYPE.upper()
    plot_specs = [
        (errors,       fr"$P(k)_\mathrm{{{model_label}}}/P(k)_\mathrm{{CAMB}} - 1$", MODEL_TYPE),
        (errors_syren, r"$P(k)_\mathrm{Syren}/P(k)_\mathrm{CAMB} - 1$",             "syren"),
    ]
    for e_full, ylabel, name in plot_specs:
        p_full = percentiles(e_full)
        fig, ax = plt.subplots(figsize=(8, 5))
        _fill_ax(ax, ks, *p_full, ylabel=ylabel,
                 label_text=f'$z={z_val:.4g}$ ')
        ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
        ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95],
                  fontsize=LEGEND_FS, loc="lower right")
        # ax.set_ylim(-0.07, 0.07)
        plt.tight_layout()
        fname = (f"{FIG_DIR}/{name}_errors_z{z_val:.4g}_{COSMO_TYPE}"
                 f"_{PRIOR_TYPE}_{NL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}_all.pdf")
        plt.savefig(fname, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {fname}")


def plot_max_error_histograms(errors, errors_filt, errors_syren, errors_syren_filt, iz, z_val):
    filter_label = fr"$w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$"
    model_label  = MODEL_TYPE.upper()
    max_err = {
        "emu_full":  np.max(np.abs(errors),            axis=1),
        "emu_filt":  np.max(np.abs(errors_filt),        axis=1),
        "syr_full":  np.max(np.abs(errors_syren),       axis=1),
        "syr_filt":  np.max(np.abs(errors_syren_filt),  axis=1),
    }
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    hist_specs = [
        (axes[0, 0], "emu_full", '#1565c0',
         fr'Max $|P_\mathrm{{{model_label}}}/P_\mathrm{{CAMB}} - 1|$',
         f'{model_label} — All cosmologies'),
        (axes[0, 1], "emu_filt", '#C45858',
         fr'Max $|P_\mathrm{{{model_label}}}/P_\mathrm{{CAMB}} - 1|$',
         fr'{model_label} — {filter_label}'),
        (axes[1, 0], "syr_full", '#1565c0',
         r'Max $|P_\mathrm{Syren}/P_\mathrm{CAMB} - 1|$',
         'Syren — All cosmologies'),
        (axes[1, 1], "syr_filt", '#C45858',
         r'Max $|P_\mathrm{Syren}/P_\mathrm{CAMB} - 1|$',
         fr'Syren — {filter_label}'),
    ]
    for ax, key, color, xlabel, title in hist_specs:
        ax.hist(max_err[key], bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel('Count', fontsize=16)
        ax.set_title(f'{title}   ($z={z_val:.4g}$)', fontsize=18)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=14)
    plt.tight_layout()
    fname = (f"{FIG_DIR}/max_error_histogram_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_{MODEL_TYPE}_z{z_val:.4g}"
             f"_w0gt{W0_THRESHOLD}_w0walt{W0WA_THRESHOLD}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_eh_logfracs(test_set, iz, z_val):
    w0        = test_set.lhs[:, W0_COL]
    w0wa      = test_set.lhs[:, W0WA_COL]
    mask_good = (w0 > W0_THRESHOLD) & (w0wa < W0WA_THRESHOLD)
    mask_bad  = ~mask_good
    logfracs_z = test_set.logfracs[:, iz, :]
    ks         = test_set.ks
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx in np.where(mask_good)[0]:
        ax.semilogx(ks, logfracs_z[idx], color=COLOR_GOOD, alpha=0.08, lw=0.5, rasterized=True)
    for idx in np.where(mask_bad)[0]:
        ax.semilogx(ks, logfracs_z[idx], color=COLOR_BAD, alpha=0.08, lw=0.5, rasterized=True)
    if mask_good.sum() > 0:
        ax.semilogx(ks, np.median(logfracs_z[mask_good], axis=0), color=COLOR_GOOD, lw=2.2,
                    label=fr"$w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$ (n={mask_good.sum()})")
    if mask_bad.sum() > 0:
        ax.semilogx(ks, np.median(logfracs_z[mask_bad], axis=0), color=COLOR_BAD, lw=2.2,
                    label=fr"fails cut (n={mask_bad.sum()})")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(r"$\log\!\left(P_\mathrm{CAMB} / P_\mathrm{EH}\right)$", fontsize=AXES_FS)
    ax.set_title(
        fr"$z={z_val:.4g}$   |   {COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, {MODEL_TYPE.upper()}",
        fontsize=15)
    ax.legend(fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = (f"{FIG_DIR}/eh_logfracs_z{z_val:.4g}_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_{MODEL_TYPE}_w0gt{W0_THRESHOLD}_w0walt{W0WA_THRESHOLD}"
             f"{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
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
    import matplotlib.cm as mcm

    n_params = min(lhs_clean.shape[1], 9)
    labels   = (_PARAM_LABELS_9 if n_params > 7 else _PARAM_LABELS_7)[:n_params]


    ref_cosmo = np.array([2.1, 0.96605, 67.32, 0.04, 0.3, -0.9, -0.9])
    mean_abs_err = np.mean(np.abs(errors), axis=1)
    # n_params = lhs_clean.shape[1]
    # labels   = _PARAM_LABELS[:n_params]

    vmin, vmax = mean_abs_err.min(), mean_abs_err.max()
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
                           color=cmap(norm(np.median(mean_abs_err[in_bin]))), linewidth=0)
                ax.set_xlim(edges[0], edges[-1])
                ax.yaxis.set_visible(False)
                if i < len(ref_cosmo):
                    ax.axvline(ref_cosmo[i], color="black", lw=1.4, ls="--", zorder=5)
            else:
                ax.scatter(lhs_clean[:, j], lhs_clean[:, i],
                           c=mean_abs_err, cmap=cmap, norm=norm,
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
    cb.set_label(r"Mean $|\,P_\mathrm{pred}/P_\mathrm{true} - 1\,|$", fontsize=TICK_FS - 1)
    cb.ax.tick_params(labelsize=9)
    ref_handle = mlines.Line2D([], [], color="black", marker="x", linestyle="none",
                               markersize=8, markeredgewidth=2.0, label="Reference cosmology")
    fig.legend(handles=[ref_handle], loc="upper right",
               bbox_to_anchor=(0.995, 0.995), fontsize=TICK_FS - 1, framealpha=0.85)
    fig.suptitle(
        (f"Emulation error vs. parameter pairs -- {MODEL_TYPE.upper()}\n"
         f"{COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, z={z_val:.4g}, {N_TRAIN} batches"),
        fontsize=AXES_FS - 3, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.70, 1])
    fname = (f"{FIG_DIR}/triangle_error_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_{MODEL_TYPE}_z{z_val:.4g}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Datagenerator comparison plot (single fiducial cosmology, three lines)
# ---------------------------------------------------------------------------

# Fiducial cosmology matching one_eval_wcdm_500ks_pklin.npy.
# Order matches utils.params: [As (10^9 As), ns, H0, Omega_b, Omega_m, w, w0+wa]
FIDUCIAL_PARAMS = [2.1, 0.96605, 67.32, 0.04, 0.3, -0.9, -0.9]


def plot_datagen_comparison(ks, redshift_indices):
    """
    Multi-panel figure with one row per target redshift, three lines each:

      1. Black dashed flat line at 1.0 — datagenerator / datagenerator (reference).
      2. Blue line — Syren P(k,z) / datagenerator P(k,z) at the fiducial cosmology.
      3. Orange line — NPCE P(k,z) / datagenerator P(k,z) at the fiducial cosmology.
    """
    if not os.path.exists(DG_PK_PATH):
        print(f"  [WARN] Datagenerator file not found: {DG_PK_PATH}")
        print("         Skipping plot_datagen_comparison.")
        return

    # Datagenerator reference: shape (52, 500), P in Mpc³
    dg_pk_all = np.load(DG_PK_PATH)[0]

    print("[INFO] Running emulator at fiducial cosmology for datagen comparison...")
    _, _, pk_npce_all = pk_emu.get_pks(
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

    _, _, pk_syren_all = pk_emu.get_pks(
        FIDUCIAL_PARAMS,
        cosmo_type             = COSMO_TYPE,
        prior_type             = PRIOR_TYPE,
        nl_type                = NL_TYPE,
        model_type             = MODEL_TYPE,
        num_batches            = N_TRAIN,
        w0_min                 = W0_MIN_FILTER,
        w0wa_max               = W0WA_MAX_FILTER,
        use_approximation_only = True,
    )   # (N_z, N_k)

    n_rows = len(redshift_indices)
    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(10, 3.2 * n_rows),
                             sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, (iz, z_val) in zip(axes, redshift_indices):
        dg_z = _to_grid(DG_KS, dg_pk_all[iz], ks)   # (N_k,)

        ratio_syren = _to_grid(ks, pk_syren_all[iz], ks) / dg_z
        ratio_npce  = _to_grid(ks, pk_npce_all[iz],  ks) / dg_z

        ax.axhline(1.0, color="black", linestyle="--", lw=1.5,
                   label="Datagen (reference)")
        ax.semilogx(ks, ratio_syren, color="#1565c0", lw=1.8,
                    label="Syren / Datagen")
        ax.semilogx(ks, ratio_npce,  color="#e87722", lw=1.8,
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

    redshift_indices = resolve_redshift_indices()

    print("=" * 60)
    print("[INFO] Single-Model Emulator Evaluation")
    print(f"       cosmo_type      = {COSMO_TYPE}")
    print(f"       prior_type      = {PRIOR_TYPE}")
    print(f"       nl_type         = {NL_TYPE}")
    print(f"       model_type      = {MODEL_TYPE}")
    print(f"       n_train         = {N_TRAIN}")
    print(f"       test_batch      = {START_BATCH}")
    print(f"       W0_MIN_FILTER   = {W0_MIN_FILTER}")
    print(f"       W0WA_MAX_FILTER = {W0WA_MAX_FILTER}")
    print(f"       Ob/H0 cut       = {OMEGAB_H0_TRIANGLE_CUT}")
    print(f"       W0_THRESHOLD    = {W0_THRESHOLD}  (diagnostic split only)")
    print(f"       W0WA_THRESHOLD  = {W0WA_THRESHOLD}  (diagnostic split only)")
    print(f"       DG_PK_PATH      = {DG_PK_PATH}")
    print(f"       Redshifts       = {[(iz, f'z={z:.4g}') for iz, z in redshift_indices]}")
    print("=" * 60)

    print("\n[INFO] Loading test set...")
    test_set = load_test_set()

    print("[INFO] Computing predictions (all redshifts)...")
    pred_pks, syren_pks, true_pks, lhs_clean = compute_predictions(test_set)

    syren_train = test_set.mps_approxes#[valid]            # baseline used in training
    base_diff = np.max(np.abs(syren_pks / syren_train - 1))
    print(f"[CHECK] max |syren(get_pks) / syren(training) - 1| = {base_diff:.2e}")   # expect ~1e-6

    pred_frac = pred_pks / syren_pks
    true_frac = test_set.frac_pks#[valid]

    # -----------------------------------------------------------------------
    # Per-redshift evaluation loop
    # -----------------------------------------------------------------------
    for iz, z_val in redshift_indices:
        print(f"\n{'=' * 60}")
        print(f"[INFO] Evaluating iz={iz}  →  z={z_val:.4g}")
        print(f"{'=' * 60}")

        errors = pred_frac[:, iz, :] / true_frac[:, iz, :] - 1
        errors_syren = syren_pks[:, iz, :] / true_pks[:, iz, :] - 1

        errors_filt, errors_syren_filt, w0_mask = apply_diagnostic_filter(
            errors, errors_syren, lhs_clean)

        w0 = lhs_clean[:, W0_COL]
        print(f"  Total cosmologies : {len(lhs_clean)}")
        print(f"  Both cuts pass    : {w0_mask.sum()}")
        print(f"  w0 range (full)   : [{w0.min():.3f}, {w0.max():.3f}]")
        if w0_mask.sum() > 0:
            print(f"  w0 range (filt.)  : [{w0[w0_mask].min():.3f}, {w0[w0_mask].max():.3f}]")

        print_statistics(errors, errors_syren, errors_filt, errors_syren_filt, iz, z_val)

        print(f"\n[INFO] Saving figures for z={z_val:.4g}...")
        ks = test_set.ks
        plot_syren_lines_filtered(errors_syren_filt, ks, w0_mask, iz, z_val)
        plot_comparison_bands(errors, errors_filt, errors_syren, errors_syren_filt, ks, iz, z_val)
        plot_max_error_histograms(errors, errors_filt, errors_syren, errors_syren_filt, iz, z_val)
        plot_eh_logfracs(test_set, iz, z_val)
        plot_error_triangle(errors, lhs_clean, iz, z_val)

    # -----------------------------------------------------------------------
    # Single multi-panel datagenerator comparison figure (all z at once)
    # -----------------------------------------------------------------------
    print("\n[INFO] Saving datagenerator comparison figure (all redshifts)...")
    plot_datagen_comparison(test_set.ks, redshift_indices)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()