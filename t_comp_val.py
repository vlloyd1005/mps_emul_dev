"""
t_comp_val.py — PCA and tPCA Reconstruction Validation Script

Validates that the two-stage dimensionality reduction (per-redshift PCA of
log-fractions, followed by a temporal PCA across redshifts) introduces only
small reconstruction errors.  For each stage the script produces a spaghetti
plot of fractional errors P_reconstructed / P_true - 1 across the test set.

Also visualises the regime boundary in cosmological parameter space, to
identify whether the bimodal logfrac structure (flat vs. V-shaped spectra)
is cleanly separable.

Outputs (saved to FIG_DIR):
  1. pca_errors_<tag>.pdf      — per-cosmology PCA reconstruction errors at z=0
  2. tpca_errors_<tag>.pdf     — per-cosmology tPCA reconstruction errors at z=0
  3. pca_errors_z3_<tag>.pdf   — same as (1) but at z=3
  4. tpca_errors_z3_<tag>.pdf  — same as (2) but at z=3
  5. regime_shape_<tag>.pdf    — regime index distribution and 2-D projections
                                 into cosmological parameter space

Cosmologies with w0 < W0_MIN and w0wa > W0WA_MAX are removed from both the training 
and test sets before any PCA fitting or evaluation, matching the filter applied 
during model training (see evaluate_one.py).

Usage (standalone):
    python ./mps_emu/t_comp_val.py

Author: Victoria Lloyd (2026)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter

import train_utils_pk_emulator as utils


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BATCHES   = 10
START_BATCH = 0
TEST_BATCH  = 100
NUM_PCS     = 25       # number of spatial PCA components per redshift
NUM_PCS_Z   = 30       # number of temporal PCA components across redshifts
COSMO_TYPE  = "w0wacdm"
PRIOR_TYPE  = "expanded"
NL_TYPE     = "lin"

W0_MIN      = -2.0
W0WA_MAX    = -0.4    # cosmologies with w0 < W0_MIN  and w0wa > W0WA_MAX are
                       # excluded from both training and test sets, matching the
                       # training-time filter


Z_IDX_0     = 0        # redshift index for z≈0
Z_IDX_3     = 33       # redshift index for z≈3

FIG_DIR     = "mps_emu/validation_figs"

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
    key arrays in train_set.  Intended as a sanity check before calling
    train_set.prepare().
    """
    print("=== BASIC SHAPES ===")
    print("lhs:          ", train_set.lhs.shape)
    print("mps_approxes: ", train_set.mps_approxes.shape)
    print("pks_lin:      ", train_set.pks_lin.shape)
    print("frac_pks:     ", train_set.frac_pks.shape)
    print("logfracs:     ", train_set.logfracs.shape)

    print("\n=== NaN / Inf SUMMARY ===")
    for name, arr in [
        ("mps_approxes", train_set.mps_approxes),
        ("pks_lin",      train_set.pks_lin),
        ("frac_pks",     train_set.frac_pks),
        ("logfracs",     train_set.logfracs),
    ]:
        print(f"  {name}: "
              f"NaNs={np.isnan(arr).sum()}, "
              f"infs={np.isinf(arr).sum()}, "
              f"min={np.nanmin(arr):.4g}, "
              f"max={np.nanmax(arr):.4g}")

    # Cosmologies with any NaN or Inf in logfracs (collapsed over z and k)
    bad_mask    = np.isnan(train_set.logfracs).any(axis=(1, 2)) \
                | np.isinf(train_set.logfracs).any(axis=(1, 2))
    bad_indices = np.where(bad_mask)[0]

    print(f"\n=== BAD COSMOLOGIES ===")
    print(f"  {bad_mask.sum()} / {train_set.lhs.shape[0]} cosmologies are bad")
    if bad_indices.size > 0:
        print("  First 10 bad indices:", bad_indices[:10])
        i = bad_indices[0]
        print(f"\n  Example (index {i}):")
        print(f"    LHS params:      {train_set.lhs[i]}")
        print(f"    mps_approxes:    "
              f"min={np.nanmin(train_set.mps_approxes[i]):.4g}, "
              f"max={np.nanmax(train_set.mps_approxes[i]):.4g}")
        print(f"    pks_lin:         "
              f"min={np.nanmin(train_set.pks_lin[i]):.4g}, "
              f"max={np.nanmax(train_set.pks_lin[i]):.4g}")
        print(f"    zeros in mps_approxes: "
              f"{(train_set.mps_approxes[i] == 0).sum()}")
        print(f"    non-positive frac_pks: "
              f"{(train_set.frac_pks[i] <= 0).sum()}")


# ---------------------------------------------------------------------------
# filtering
# ---------------------------------------------------------------------------

def apply_filter(cola_set, w0_min=W0_MIN, w0wa_max=W0WA_MAX):
    """
    Remove cosmologies with w0 < w0_min from a COLASet in-place.

    Mirrors the filter applied in evaluate_one.load_test_set() so that the
    validation PCA and error plots reflect exactly the same population the
    trained model will see at inference time.

    Parameters
    ----------
    cola_set : COLASet — modified in-place
    w0_min   : float  — lower bound on w0 (default W0_MIN = -2.0)

    Returns
    -------
    n_removed : int — number of cosmologies removed
    """
    w0_col = utils.params.index("w")
    w0wa_col = utils.params.index("w0+wa")
    mask = np.ones(len(cola_set.lhs), dtype=bool)
 
    if w0_min is not None:
        w0_mask = cola_set.lhs[:, w0_col] >= w0_min
        print(f"  w0 cut   (w0 >= {w0_min}):    removed {(~w0_mask).sum()}")
        mask &= w0_mask
 
    if w0wa_max is not None:
        w0wa_mask = cola_set.lhs[:, w0wa_col] <= w0wa_max
        print(f"  w0+wa cut (w0+wa <= {w0wa_max}): removed {(~w0wa_mask & mask).sum()} "
              f"additional cosmologies")
        mask &= w0wa_mask
    n_removed = int((~mask).sum())

    for attr in ("lhs", "pks_lin", "mps_approxes", "frac_pks", "logfracs"):
        if hasattr(cola_set, attr):
            setattr(cola_set, attr, getattr(cola_set, attr)[mask])

    # lhs_norm is only present after prepare() — guard accordingly
    if hasattr(cola_set, "lhs_norm") and cola_set.lhs_norm is not None:
        cola_set.lhs_norm = cola_set.lhs_norm[mask]

    return int(n_removed)


# ---------------------------------------------------------------------------
# PCA reconstruction
# ---------------------------------------------------------------------------

def pca_reconstruction_errors(train_set, test_set, iz):
    """
    Compute per-cosmology PCA reconstruction fractional errors at redshift
    index iz:  exp(inverse_transform(PCA(logfracs))) / frac_pks - 1.

    Parameters
    ----------
    train_set : COLASet — fitted scalers and PCAs live here
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

    # --- encode: (N_cosmo, N_z, 1, NUM_PCS) ---
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

    # --- tPCA round-trip ---
    t_comps   = train_set.tpca.transform(pcs_flat)
    pcs_recon = train_set.tpca.inverse_transform(t_comps)
    pcs_per_z = pcs_recon.reshape(n_cosmo, n_z, NUM_PCS)

    # --- decode back to log-fraction space ---
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


def tpca_reconstruction_errors(stacks, test_set, iz):
    """
    Compute tPCA fractional errors at redshift index iz.

    Parameters
    ----------
    stacks   : (N_cosmo, 1, N_z, N_k) ndarray — output of tpca_stacks()
    test_set : COLASet
    iz       : int — redshift index

    Returns
    -------
    errors : (N_cosmo, N_k) ndarray
    """
    return np.exp(stacks[:, 0, iz, :]) / test_set.frac_pks[:, iz, :] - 1


# ---------------------------------------------------------------------------
# Regime boundary visualisation
# ---------------------------------------------------------------------------

def compute_regime_index(cola_set, iz=0):
    """
    Compute a scalar 'shape index' for each cosmology that discriminates
    between flat and V-shaped logfrac curves at redshift index iz.

    The index is the standard deviation of logfrac(k) across k-modes.  Flat
    spectra (centre of parameter space) produce small std; V-shaped spectra
    (parameter-space edges, where EH significantly mis-estimates the BAO peak
    position or the broadband slope) produce large std.

    We also compute two secondary diagnostics:
      - low_k_excess : mean logfrac at the 10 lowest k-modes, capturing the
                       overall amplitude offset at large scales
      - high_k_excess: mean logfrac at the 10 highest k-modes, capturing the
                       small-scale upturn present in V-shaped spectra

    Parameters
    ----------
    cola_set : COLASet
    iz       : int — redshift index to evaluate (default 0, i.e. z≈0)

    Returns
    -------
    shape_index   : (N_cosmo,) — logfrac std across k
    low_k_excess  : (N_cosmo,) — mean logfrac at low k
    high_k_excess : (N_cosmo,) — mean logfrac at high k
    v_score       : (N_cosmo,) — composite V-shape score:
                                 0.5*(low_k_excess + high_k_excess) - centre,
                                 positive = V-shaped, negative/zero = flat
    """
    logf = cola_set.logfracs[:, iz, :]   # (N_cosmo, N_k)

    shape_index   = logf.std(axis=1)
    low_k_excess  = logf[:, :10].mean(axis=1)
    high_k_excess = logf[:, -10:].mean(axis=1)

    # Centre of k range: avoid the BAO bump region by using mid-quartile mean
    n_k    = logf.shape[1]
    q1, q3 = n_k // 4, 3 * n_k // 4
    centre = logf[:, q1:q3].mean(axis=1)

    v_score = 0.5 * (low_k_excess + high_k_excess) - centre

    return shape_index, low_k_excess, high_k_excess, v_score


def _scatter_with_density_contour(ax, x, y, c, cmap, norm, xlabel, ylabel, title,
                                   nbins=40):
    """
    Scatter plot coloured by c, with a smoothed density contour overlay
    so the regime boundary emerges even where points are dense.
    """
    sc = ax.scatter(x, y, c=c, cmap=cmap, norm=norm, s=6, alpha=0.6,
                    linewidths=0, rasterized=True)

    # Overlay a 2-D histogram as faint contours to show where most
    # cosmologies live, independent of the colour axis.
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
    Produce a multi-panel figure that reveals where the V-shaped / flat
    regime boundary sits in cosmological parameter space.

    Panel layout
    ------------
    Row 0  : histogram of the V-score (shape index), with a dashed threshold
             line separating the two regimes, plus example logfrac curves for
             the flattest and most V-shaped cosmologies.
    Row 1-2: 2-D scatter plots of every pair of the three cosmological
             parameters most likely to drive the boundary: w0, w0+wa, Om.
             Points are coloured by V-score so the boundary is visible.

    Additionally prints summary statistics of the regime split, including
    which parameter ranges dominate each regime.

    Parameters
    ----------
    train_set    : COLASet (must have logfracs and lhs populated)
    iz           : int   — redshift index for the shape analysis (default 0)
    threshold_pct: float — percentile of V-score used to define 'V-shaped'
                           (default 75: top quartile = V-shaped regime)

    Saves
    -----
    FIG_DIR/regime_boundary_<tag>.pdf
    """
    shape_index, low_k_excess, high_k_excess, v_score = compute_regime_index(train_set, iz)
    threshold  = np.percentile(v_score, threshold_pct)
    v_mask     = v_score >= threshold    # True = V-shaped regime
    flat_mask  = ~v_mask

    # ── summary statistics ────────────────────────────────────────────────
    print(f"\n=== REGIME BOUNDARY SUMMARY (z_idx={iz}, threshold={threshold_pct}th pct) ===")
    print(f"  V-score range:   {v_score.min():.4f} → {v_score.max():.4f}")
    print(f"  Threshold:       {threshold:.4f}")
    print(f"  Flat regime:     {flat_mask.sum()} cosmologies "
          f"({100*flat_mask.mean():.1f}%)")
    print(f"  V-shaped regime: {v_mask.sum()} cosmologies "
          f"({100*v_mask.mean():.1f}%)")
    print()
    param_names = utils.params   # ['h','Omega_b','Omega_m','As','ns','w','w0+wa']
    for pi, name in enumerate(param_names):
        flat_range = (train_set.lhs[flat_mask, pi].min(),
                      train_set.lhs[flat_mask, pi].max())
        v_range    = (train_set.lhs[v_mask,    pi].min(),
                      train_set.lhs[v_mask,    pi].max())
        print(f"  {name:8s}:  flat [{flat_range[0]:.3g}, {flat_range[1]:.3g}]"
              f"   V-shaped [{v_range[0]:.3g}, {v_range[1]:.3g}]")

    # ── figure ────────────────────────────────────────────────────────────
    # Parameter indices for 2-D projections: w0(5), w0+wa(6), Om(2), ns(4), h(0)
    proj_pairs = [
        (5, 6, "w_0",   "w_0+w_a"),
        (5, 2, "w_0",   r"\Omega_m"),
        (6, 2, "w_0+w_a", r"\Omega_m"),
        (4, 0, "n_s",   "h"),
    ]

    n_rows = 2 + len(proj_pairs) // 2
    fig    = plt.figure(figsize=(14, 4 * n_rows))
    gs     = fig.add_gridspec(n_rows, 2, hspace=0.45, wspace=0.35)

    cmap = plt.get_cmap("plasma")
    norm = Normalize(vmin=np.percentile(v_score, 2),
                     vmax=np.percentile(v_score, 98))

    # ── Row 0, left: V-score histogram ───────────────────────────────────
    ax_hist = fig.add_subplot(gs[0, 0])
    ax_hist.hist(v_score[flat_mask], bins=60, color="#4c78a8", alpha=0.7,
                 label=f"flat ({flat_mask.sum()})")
    ax_hist.hist(v_score[v_mask],    bins=60, color="#e45756", alpha=0.7,
                 label=f"V-shaped ({v_mask.sum()})")
    ax_hist.axvline(threshold, color="k", lw=1.5, ls="--",
                    label=f"threshold ({threshold_pct}th pct)")
    ax_hist.set_xlabel("V-score  [0.5·(low-k + high-k mean) − mid-k mean]",
                        fontsize=AXES_FS - 1)
    ax_hist.set_ylabel("Count", fontsize=AXES_FS)
    ax_hist.set_title("Regime distribution (V-score)", fontsize=AXES_FS)
    ax_hist.legend(fontsize=TICK_FS)
    ax_hist.tick_params(labelsize=TICK_FS)

    # ── Row 0, right: example logfrac curves ─────────────────────────────
    ax_ex = fig.add_subplot(gs[0, 1])

    # 5 flattest and 5 most V-shaped, chosen by absolute V-score rank
    flat_top5 = np.argsort(v_score)[:5]
    v_top5    = np.argsort(v_score)[-5:]

    for idx in flat_top5:
        ax_ex.semilogx(train_set.ks,
                        train_set.logfracs[idx, iz, :],
                        color="#4c78a8", lw=0.9, alpha=0.8)
    for idx in v_top5:
        ax_ex.semilogx(train_set.ks,
                        train_set.logfracs[idx, iz, :],
                        color="#e45756", lw=0.9, alpha=0.8)

    # Invisible proxy artists for the legend
    ax_ex.plot([], [], color="#4c78a8", lw=1.5, label="5 flattest")
    ax_ex.plot([], [], color="#e45756", lw=1.5, label="5 most V-shaped")
    ax_ex.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=AXES_FS)
    ax_ex.set_ylabel(r"$\log(P_\mathrm{CAMB}/P_\mathrm{EH})$", fontsize=AXES_FS)
    ax_ex.set_title(f"Example logfrac curves at z={train_set.z[iz]:.2f}",
                     fontsize=AXES_FS)
    ax_ex.legend(fontsize=TICK_FS)
    ax_ex.tick_params(labelsize=TICK_FS)
    ax_ex.grid(alpha=0.3)

    # ── Rows 1+: 2-D projections ──────────────────────────────────────────
    lhs = train_set.lhs
    for k, (pi, pj, xi_label, xj_label) in enumerate(proj_pairs):
        row = 1 + k // 2
        col = k  % 2
        ax  = fig.add_subplot(gs[row, col])
        sc  = _scatter_with_density_contour(
            ax,
            x=lhs[:, pi], y=lhs[:, pj],
            c=v_score,
            cmap=cmap, norm=norm,
            xlabel=f"${xi_label}$",
            ylabel=f"${xj_label}$",
            title=f"V-score: ${xi_label}$ vs ${xj_label}$",
        )

    # Shared colourbar on the right
    cbar_ax = fig.add_axes([0.92, 0.08, 0.015, 0.55])
    sm      = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("V-score (higher = more V-shaped)", fontsize=TICK_FS)
    cb.ax.tick_params(labelsize=TICK_FS)

    fig.suptitle(
        f"Regime boundary visualisation — {COSMO_TYPE} / {NL_TYPE} / {PRIOR_TYPE}\n"
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
# Plotting
# ---------------------------------------------------------------------------

def _tag():
    """Shared filename tag built from the run configuration."""
    return (f"{COSMO_TYPE}_{NL_TYPE}_{PRIOR_TYPE}"
            f"_w0min{W0_MIN}_w0wamax{W0WA_MAX}_n{NUM_PCS}_z{NUM_PCS_Z}")


def _spaghetti_ax(ax, ks, errors, ylabel, title):
    """Draw one spaghetti error curve per cosmology on ax."""
    for error in errors:
        ax.semilogx(ks, error, lw=0.6, alpha=0.7)
    ax.set_xlabel(r"$k \; (h/\mathrm{Mpc})$", fontsize=AXES_FS)
    ax.set_ylabel(ylabel, fontsize=AXES_FS)
    ax.set_title(title, fontsize=AXES_FS)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.grid(alpha=0.3)


def plot_pca_errors(errors, ks, iz_label):
    """Save PCA reconstruction error spaghetti plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _spaghetti_ax(
        ax, ks, errors,
        ylabel=r"$P_\mathrm{PCA-reconstructed}/P - 1$",
        title=(r"PCA Reconstruction Errors"
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
        ylabel=r"$P_\mathrm{tPCA-reconstructed}/P - 1$",
        title=(r"tPCA Reconstruction Errors"
               + fr" at $z\approx{iz_label}$,"
               + fr" $N_\mathrm{{PC}}={NUM_PCS}$, $N_\mathrm{{tcomp}}={NUM_PCS_Z}$"),
    )
    plt.tight_layout()
    fname = f"{FIG_DIR}/tpca_errors_z{iz_label}_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family']      = 'STIXGeneral'

    print("=" * 60)
    print("[INFO] PCA / tPCA Reconstruction Validation")
    print(f"       cosmo_type  = {COSMO_TYPE}")
    print(f"       prior_type  = {PRIOR_TYPE}")
    print(f"       nl_type     = {NL_TYPE}")
    print(f"       n_batches   = {N_BATCHES}  (starting at {START_BATCH})")
    print(f"       test_batch  = {TEST_BATCH}")
    print(f"       num_pcs     = {NUM_PCS}")
    print(f"       num_pcs_z   = {NUM_PCS_Z}")
    print("=" * 60)

    # --- Load data ---
    print("\n[INFO] Loading training set...")
    train_set = utils.COLASet(
        target_z=utils.z_mps,
        cosmo_type=COSMO_TYPE,
        prior_type=PRIOR_TYPE,
        nl_type=NL_TYPE,
        n_batches=N_BATCHES,
        start_batch=START_BATCH,
    )

    print_data_diagnostics(train_set)

    # --- w0 filter (must happen before prepare() fits the PCA) ---
    n_removed = apply_filter(train_set, w0_min=W0_MIN, w0wa_max=W0WA_MAX)
    print(f"\n[INFO] w0 filter (w0 >= {W0_MIN}) w0wa filter(w0+wa <= {W0WA_MAX})): removed {n_removed} cosmologies "
          f"from training set ({len(train_set.lhs)} remaining).")

    # --- Regime boundary (before prepare, so we see the raw logfracs) ---
    print("\n[INFO] Visualising regime boundary...")
    v_score, threshold = visualize_regime_boundary(train_set, iz=Z_IDX_0)

    print("\n[INFO] Preparing training set (PCA + tPCA fit)...")
    train_set.prepare(num_pcs=NUM_PCS, num_pcs_z=NUM_PCS_Z)

    print("\n[INFO] Loading test set...")
    test_set = utils.COLASet(
        target_z=utils.z_mps,
        cosmo_type=COSMO_TYPE,
        nl_type=NL_TYPE,
        start_batch=TEST_BATCH,
    )
    n_removed_test = apply_filter(test_set, w0_min=W0_MIN, w0wa_max=W0WA_MAX)
    print(f"[INFO] w0 filter (w0 >= {W0_MIN}): removed {n_removed_test} cosmologies "
          f"from test set ({len(test_set.lhs)} remaining).")

    ks = train_set.ks

    # --- PCA errors ---
    print("\n[INFO] Computing PCA reconstruction errors...")
    pca_err_z0 = pca_reconstruction_errors(train_set, test_set, Z_IDX_0)
    pca_err_z3 = pca_reconstruction_errors(train_set, test_set, Z_IDX_3)

    # --- tPCA errors ---
    print("[INFO] Computing tPCA reconstruction errors...")
    stacks       = tpca_stacks(train_set, test_set)
    tpca_err_z0  = tpca_reconstruction_errors(stacks, test_set, Z_IDX_0)
    tpca_err_z3  = tpca_reconstruction_errors(stacks, test_set, Z_IDX_3)

    # --- Figures ---
    print("\n[INFO] Saving figures...")
    plot_pca_errors(pca_err_z0,  ks, iz_label=0)
    plot_tpca_errors(tpca_err_z0, ks, iz_label=0)
    plot_pca_errors(pca_err_z3,  ks, iz_label=3)
    plot_tpca_errors(tpca_err_z3, ks, iz_label=3)

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()