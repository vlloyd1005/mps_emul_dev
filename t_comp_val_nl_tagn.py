"""
t_comp_val_nl_envelope.py — PCA/tPCA Reconstruction Validation with T_AGN Envelope Correction

Identical to t_comp_val_nl.py, but adds a T_AGN-dependent mean logfrac envelope
correction as a preprocessing step before PCA fitting.

The motivation: AGN baryonic feedback drives a systematic k-dependent shape in
log(P_CAMB / P_syren) that is strongly correlated with T_AGN. This shared
envelope uses up PCA components that could otherwise represent genuine
cosmology-to-cosmology variation. By subtracting the envelope before PCA and
adding it back at reconstruction, the PCA only needs to represent the residuals,
which should be smaller and more compactly representable.

Pipeline:
    1. Fit envelope: median logfrac in each T_AGN bin → 2D interpolator f(T_AGN, k)
    2. Subtract envelope from logfracs before Scaler.fit + PCA.fit
    3. Reconstruct: PCA inverse → add envelope back → compare to frac_pks
    4. tPCA runs on the envelope-corrected PCA coefficients

All other diagnostics (regime boundary, scree, syren accuracy plots) are
identical to t_comp_val_nl.py so results are directly comparable.

Usage:
    python ./mps_emu/t_comp_val_nl_envelope.py

Author: Victoria Lloyd (2026)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

import train_utils_pk_emulator_v3 as utils
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_BATCHES   = 10
START_BATCH = 0
TEST_BATCH  = 1000
NUM_PCS     = 30
NUM_PCS_Z   = 80
COSMO_TYPE  = "w0wacdm"
PRIOR_TYPE  = "expanded"
NL_TYPE     = "mead2020_Tfree_mnufree"

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

# Envelope correction settings
N_TAGN_BINS = 20    # number of T_AGN bins for the envelope grid
TAGN_COL    = 7     # column index of T_AGN in lhs (after w0+wa overwrite)

Z_IDX_0     = 0
Z_IDX_3     = 33

FIG_DIR     = "mps_emu/validation_figs/smaller_grid"
VER         = "0_smaller_grid"

SCREE_Z_HIGHLIGHT = [0, 8, 16, 24, 33]

AXES_FS = 14
TICK_FS = 12


# ---------------------------------------------------------------------------
# Filtering  (identical to t_comp_val_nl.py)
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


# ---------------------------------------------------------------------------
# Tag
# ---------------------------------------------------------------------------

def _tag():
    parts = []
    if W0_MIN   is not None: parts.append(f"w0min{W0_MIN}")
    if W0WA_MAX is not None: parts.append(f"w0wamax{W0WA_MAX}")
    if OM_MIN   is not None: parts.append(f"ommin{OM_MIN}")
    if OMEGAB_H0_TRIANGLE_CUT is not None:
        parts.append(f"obh0tri{OMEGAB_ANCHOR}_{H0_ANCHOR}")
    fstr = ("_" + "_".join(parts)) if parts else ""
    return f"{COSMO_TYPE}_{NL_TYPE}_{PRIOR_TYPE}{fstr}_n{NUM_PCS}_z{NUM_PCS_Z}_v{VER}"

# ---------------------------------------------------------------------------
# Envelope fitting
# ---------------------------------------------------------------------------

def fit_envelope_per_z(cola_set, n_tagn_bins=N_TAGN_BINS):
    """
    Fit a smooth T_AGN-dependent mean logfrac envelope at every redshift.

    For each redshift iz, compute the median logfrac across cosmologies in
    each T_AGN bin, producing a 2D surface f(T_AGN, k). Store as a
    RegularGridInterpolator so it can be evaluated quickly for any
    (T_AGN, k) pair.

    Stores on cola_set:
        .envelope_fns  : list of RegularGridInterpolator, one per redshift
        .tagn_grid     : (n_tagn_bins,) T_AGN bin centres
        .envelope_mean_logfracs : (N_z, n_tagn_bins, N_k) raw bin medians

    Returns
    -------
    None  (modifies cola_set in place)
    """
    if cola_set.lhs.shape[1] <= 7:
        raise ValueError(
            "Envelope correction requires 9-column lhs (T_AGN at col 7). "
            "Set NL_TYPE = 'mead2020_Tfree_mnufree'."
        )

    tagn_vals = cola_set.lhs[:, TAGN_COL]
    ks        = cola_set.ks
    n_z       = len(utils.z_mps)

    tagn_edges = np.linspace(tagn_vals.min(), tagn_vals.max(), n_tagn_bins + 1)
    tagn_grid  = 0.5 * (tagn_edges[:-1] + tagn_edges[1:])

    envelope_fns        = []
    all_mean_logfracs   = np.full((n_z, n_tagn_bins, len(ks)), np.nan)

    for iz in range(n_z):
        logfracs_z = cola_set.logfracs[:, iz, :]   # (N_cosmo, N_k)
        mean_lf    = np.full((n_tagn_bins, len(ks)), np.nan)

        for b in range(n_tagn_bins):
            in_bin = (tagn_vals >= tagn_edges[b]) & (tagn_vals < tagn_edges[b + 1])
            if in_bin.sum() == 0:
                continue
            # Use median — robust to the extreme outlier cosmologies
            finite_mask = np.isfinite(logfracs_z[in_bin]).all(axis=1)
            if finite_mask.sum() > 0:
                mean_lf[b] = np.median(logfracs_z[in_bin][finite_mask], axis=0)

        # Fill any empty bins by linear interpolation across T_AGN
        for k_idx in range(len(ks)):
            col      = mean_lf[:, k_idx]
            nan_mask = np.isnan(col)
            if nan_mask.any() and (~nan_mask).sum() >= 2:
                mean_lf[nan_mask, k_idx] = np.interp(
                    tagn_grid[nan_mask], tagn_grid[~nan_mask], col[~nan_mask]
                )
            elif nan_mask.all():
                mean_lf[:, k_idx] = 0.0   # fallback: no correction

        all_mean_logfracs[iz] = mean_lf

        fn = RegularGridInterpolator(
            (tagn_grid, ks),
            mean_lf,
            method='linear',
            bounds_error=False,
            fill_value=None,   # linear extrapolation beyond grid edges
        )
        envelope_fns.append(fn)

    cola_set.envelope_fns             = envelope_fns
    cola_set.tagn_grid                = tagn_grid
    cola_set.envelope_mean_logfracs   = all_mean_logfracs

    print(f"  Envelope fitted: {n_tagn_bins} T_AGN bins, "
          f"{n_z} redshifts, T_AGN range [{tagn_vals.min():.2f}, {tagn_vals.max():.2f}]")


def evaluate_envelope(envelope_fn, tagn_vals, ks):
    """
    Evaluate the envelope for a batch of cosmologies at all k modes.

    Parameters
    ----------
    envelope_fn : RegularGridInterpolator — from fit_envelope_per_z
    tagn_vals   : (N_cosmo,) T_AGN values
    ks          : (N_k,) k modes

    Returns
    -------
    envelope : (N_cosmo, N_k)
    """
    N_cosmo = len(tagn_vals)
    N_k     = len(ks)
    # Build query points: all (tagn, k) combinations
    tagn_rep = np.repeat(tagn_vals, N_k)      # (N_cosmo * N_k,)
    k_rep    = np.tile(ks, N_cosmo)            # (N_cosmo * N_k,)
    pts      = np.column_stack([tagn_rep, k_rep])
    return envelope_fn(pts).reshape(N_cosmo, N_k)


# ---------------------------------------------------------------------------
# Envelope-corrected prepare()
# ---------------------------------------------------------------------------

def prepare_with_envelope(train_set, num_pcs, num_pcs_z,
                           metadata_dir="mps_emu/metadata"):
    """
    Fit scalers + PCA + tPCA on envelope-corrected logfracs.

    Identical to COLASet.prepare() except:
      - The T_AGN envelope is subtracted from logfracs[:, iz, :] before
        Scaler.fit() and PCA.fit() at each redshift.
      - The corrected PCA coefficients and scalers are stored on train_set
        under the same attribute names as the original prepare(), so all
        downstream reconstruction functions work unchanged.
      - The envelope_fns are already on train_set from fit_envelope_per_z().

    The metadata bundle is written to disk so the emulator can reload it,
    but the envelope functions themselves are stored as Python objects on
    train_set (they are not currently serialised into the bundle — that
    would be the next integration step).
    """
    from sklearn.preprocessing import StandardScaler
    import joblib

    if not hasattr(train_set, 'envelope_fns'):
        raise RuntimeError("Call fit_envelope_per_z(train_set) before prepare_with_envelope().")

    train_set.num_pcs   = num_pcs
    tagn_vals           = train_set.lhs[:, TAGN_COL]
    ks                  = train_set.ks
    n_z                 = len(utils.z_mps)

    metadata_subdir = os.path.join(
        metadata_dir, f"metadata_{train_set._metadata_tag()}_envelope"
    )
    os.makedirs(metadata_subdir, exist_ok=True)

    # --- Param scaler (unchanged) ---
    from sklearn.preprocessing import MinMaxScaler
    train_set.param_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_set.lhs)
    train_set.lhs_norm     = train_set.param_scaler.transform(train_set.lhs)

    # --- Per-z logfrac scalers + PCA on envelope-corrected logfracs ---
    train_set.frac_pks_scalers = []
    all_pcs                    = []
    pcas_dict                  = {}
    scalers_dict               = {}

    from train_utils_pk_emulator_v3 import Scaler

    for iz, z_val in enumerate(utils.z_mps):
        z_key      = float(f"{z_val:.3f}")
        logfracs_z = train_set.logfracs[:, iz, :].copy()   # (N_cosmo, N_k)

        # Subtract T_AGN envelope
        envelope_z = evaluate_envelope(train_set.envelope_fns[iz], tagn_vals, ks)
        logfracs_z_corr = logfracs_z - envelope_z

        # Fit scaler and PCA on the corrected logfracs
        scaler = Scaler()
        scaler.fit(logfracs_z_corr)
        normed  = scaler.transform(logfracs_z_corr)

        pca = PCA(n_components=num_pcs)
        pca.fit(normed)
        pca_vals = pca.transform(normed)

        train_set.frac_pks_scalers.append(scaler)
        pcas_dict[z_key]    = pca
        scalers_dict[z_key] = scaler
        all_pcs.append(pca_vals)

    train_set.pcas    = [pcas_dict[float(f"{z:.3f}")] for z in utils.z_mps]
    train_set.all_pcs = np.transpose(np.array(all_pcs), (1, 0, 2))
    pcs_flat          = train_set.all_pcs.reshape(len(train_set.lhs), n_z * num_pcs)

    # --- tPCA ---
    pca_z = PCA(n_components=num_pcs_z)
    pca_z.fit(pcs_flat)
    train_set.tpca         = pca_z
    train_set.t_components = pca_z.transform(pcs_flat)

    # --- t-component scaler ---
    from train_utils_pk_emulator_v3 import TComponentScaler
    train_set.t_comp_scaler     = TComponentScaler().fit(train_set.t_components)
    train_set.t_components_norm = train_set.t_comp_scaler.transform(train_set.t_components)

    stds = train_set.t_components.std(axis=0)
    n_components = train_set.t_components.shape[1]
    print(f"\n  Prepared {n_components} envelope-corrected tPCA components.")
    print(f"  t-component std range (raw):  {stds.min():.3f} → {stds.max():.3f}  "
          f"(ratio = {stds.max()/stds.min():.1f}x)")
    print(f"  t-component std range (norm): "
          f"{train_set.t_components_norm.std(axis=0).min():.3f} → "
          f"{train_set.t_components_norm.std(axis=0).max():.3f}  (should be ~1.0)")

    train_set._metadata_bundle_path = os.path.join(metadata_subdir, "metadata.joblib")
    print(f"  Metadata subdir: {metadata_subdir}")


# ---------------------------------------------------------------------------
# Envelope-corrected PCA reconstruction errors
# ---------------------------------------------------------------------------

def pca_reconstruction_errors_envelope(train_set, test_set, iz):
    """
    Compute PCA reconstruction errors with envelope correction.

    Steps:
      1. Subtract envelope from test logfracs at iz
      2. Transform with scaler + PCA (fitted on envelope-corrected train logfracs)
      3. Inverse PCA + inverse scaler
      4. Add envelope back
      5. Compare exp(reconstructed) to frac_pks

    Parameters
    ----------
    train_set : COLASet — has .envelope_fns, .frac_pks_scalers, .pcas
    test_set  : COLASet — has .lhs[:, TAGN_COL] = T_AGN, .logfracs, .frac_pks
    iz        : int

    Returns
    -------
    errors : (N_cosmo, N_k)
    """
    if not hasattr(train_set, 'envelope_fns'):
        raise RuntimeError("train_set has no envelope_fns — run fit_envelope_per_z first.")

    tagn_vals  = test_set.lhs[:, TAGN_COL]
    ks         = test_set.logfracs.shape[2]   # N_k
    ks_arr     = train_set.ks

    scaler = train_set.frac_pks_scalers[iz]
    pca    = train_set.pcas[iz]

    # Subtract envelope from test logfracs
    envelope_z = evaluate_envelope(train_set.envelope_fns[iz], tagn_vals, ks_arr)
    logfracs_corr = test_set.logfracs[:, iz, :] - envelope_z

    # PCA round-trip on corrected logfracs
    normed        = scaler.transform(logfracs_corr)
    pcs           = pca.transform(normed)
    reconstructed_corr = scaler.inverse_transform(pca.inverse_transform(pcs))

    # Add envelope back before converting to ratio space
    reconstructed = reconstructed_corr + envelope_z

    return np.exp(reconstructed) / test_set.frac_pks[:, iz, :] - 1


# ---------------------------------------------------------------------------
# Envelope-corrected tPCA stacks
# ---------------------------------------------------------------------------

def tpca_stacks_envelope(train_set, test_set):
    """
    Encode test cosmologies with envelope-corrected per-z PCAs, compress
    with tPCA, reconstruct back to log-fraction space, add envelope back.

    Returns
    -------
    stacks : (N_cosmo, 1, N_z, N_k) reconstructed log-fractions (with envelope)
    """
    if not hasattr(train_set, 'envelope_fns'):
        raise RuntimeError("train_set has no envelope_fns — run fit_envelope_per_z first.")

    n_cosmo   = len(test_set.lhs)
    n_z       = len(utils.z_mps)
    tagn_vals = test_set.lhs[:, TAGN_COL]
    ks_arr    = train_set.ks

    # Precompute all envelopes for test set: (N_z, N_cosmo, N_k)
    envelopes = np.stack([
        evaluate_envelope(train_set.envelope_fns[iz], tagn_vals, ks_arr)
        for iz in range(n_z)
    ], axis=0)   # (N_z, N_cosmo, N_k)

    all_pcs = []
    for i in range(n_cosmo):
        cosmos_pcs = []
        for iz in range(n_z):
            logfrac_corr = test_set.logfracs[i, iz, :] - envelopes[iz, i, :]
            normed = train_set.frac_pks_scalers[iz].transform([logfrac_corr])
            pcs    = train_set.pcas[iz].transform(normed)
            cosmos_pcs.append([pcs[0]])
        all_pcs.append(cosmos_pcs)

    all_pcs  = np.transpose(np.array(all_pcs), (0, 2, 1, 3))  # (N, 1, N_z, NUM_PCS)
    pcs_flat = all_pcs.reshape(n_cosmo, n_z * NUM_PCS)

    t_comps   = train_set.tpca.transform(pcs_flat)
    pcs_recon = train_set.tpca.inverse_transform(t_comps)
    pcs_per_z = pcs_recon.reshape(n_cosmo, n_z, NUM_PCS)

    stacks = []
    for i, pcs_z_stack in enumerate(pcs_per_z):
        reconstructed_fracs = []
        for iz, pcs_z in enumerate(pcs_z_stack):
            # Inverse PCA + inverse scaler (gives corrected logfrac)
            logfrac_corr = train_set.frac_pks_scalers[iz].inverse_transform(
                train_set.pcas[iz].inverse_transform(pcs_z.reshape(1, -1))
            )[0]
            # Add envelope back for this cosmology
            logfrac = logfrac_corr + envelopes[iz, i, :]
            reconstructed_fracs.append(logfrac)
        stacks.append([np.stack(reconstructed_fracs)])

    return np.array(stacks)   # (N_cosmo, 1, N_z, N_k)


def tpca_reconstruction_errors(stacks, test_set, iz):
    return np.exp(stacks[:, 0, iz, :]) / test_set.frac_pks[:, iz, :] - 1


# ---------------------------------------------------------------------------
# Envelope diagnostic plot
# ---------------------------------------------------------------------------

def plot_envelope_diagnostic(train_set, iz, iz_label):
    """
    Two-panel figure:
      Left:  Median logfrac per T_AGN bin (the envelope itself)
      Right: Residuals after envelope subtraction for a random subset,
             coloured by w0+wa — shows what the PCA now needs to learn
    """
    tagn_vals  = train_set.lhs[:, TAGN_COL]
    logfracs_z = train_set.logfracs[:, iz, :]
    ks         = train_set.ks
    tagn_grid  = train_set.tagn_grid
    mean_lf    = train_set.envelope_mean_logfracs[iz]
    envelope_z = evaluate_envelope(train_set.envelope_fns[iz], tagn_vals, ks)
    residuals  = logfracs_z - envelope_z

    w0wa_col  = utils.params.index("w0+wa")
    w0wa_vals = train_set.lhs[:, w0wa_col]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: envelope coloured by T_AGN
    ax = axes[0]
    cmap_e = plt.get_cmap("plasma")
    norm_e = Normalize(vmin=tagn_grid.min(), vmax=tagn_grid.max())
    for b, tagn_b in enumerate(tagn_grid):
        ax.semilogx(ks, mean_lf[b], color=cmap_e(norm_e(tagn_b)), lw=1.2)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
    sm_e = plt.cm.ScalarMappable(cmap=cmap_e, norm=norm_e)
    sm_e.set_array([])
    plt.colorbar(sm_e, ax=ax, label=r"$\log T_{\rm AGN}$")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(r"Median $\log(P_{\rm CAMB}/P_{\rm syren})$ per $T_{\rm AGN}$ bin",
                  fontsize=AXES_FS - 1)
    ax.set_title(fr"Fitted $T_{{\rm AGN}}$ envelope at $z\approx{iz_label}$", fontsize=AXES_FS)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=TICK_FS)

    # Right: residuals coloured by w0+wa
    ax2 = axes[1]
    rng = np.random.default_rng(seed=42)
    sub = rng.choice(len(train_set.lhs), size=min(150, len(train_set.lhs)), replace=False)
    cmap_r = plt.get_cmap("coolwarm")
    norm_r = Normalize(vmin=np.percentile(w0wa_vals[sub], 2),
                       vmax=np.percentile(w0wa_vals[sub], 98))
    for idx in sub:
        ax2.semilogx(ks, residuals[idx], color=cmap_r(norm_r(w0wa_vals[idx])),
                     lw=0.5, alpha=0.4, rasterized=True)
    ax2.axhline(0, color="k", lw=0.8, ls="--")

    # Stats
    orig_max  = np.abs(logfracs_z[np.isfinite(logfracs_z).all(axis=1)]).max(axis=1).mean()
    resid_max = np.abs(residuals[np.isfinite(residuals).all(axis=1)]).max(axis=1).mean()
    reduction = 100 * (1 - resid_max / orig_max)
    print(f"\n  Envelope correction at z~{iz_label}:")
    print(f"    Mean max |logfrac| before: {orig_max:.4f}")
    print(f"    Mean max |residual| after: {resid_max:.4f}  ({reduction:.1f}% reduction)")

    sm_r = plt.cm.ScalarMappable(cmap=cmap_r, norm=norm_r)
    sm_r.set_array([])
    plt.colorbar(sm_r, ax=ax2, label=r"$w_0+w_a$")
    ax2.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax2.set_ylabel(r"Residual logfrac (after envelope subtraction)", fontsize=AXES_FS - 1)
    ax2.set_title(fr"Residuals at $z\approx{iz_label}$ (coloured by $w_0+w_a$)",
                  fontsize=AXES_FS)
    ax2.grid(alpha=0.3)
    ax2.tick_params(labelsize=TICK_FS)

    plt.suptitle(
        f"T_AGN envelope correction — {COSMO_TYPE} / {NL_TYPE} / {PRIOR_TYPE}\n"
        f"({N_BATCHES} batches, z_idx={iz}, {N_TAGN_BINS} T_AGN bins)",
        fontsize=AXES_FS, y=1.01
    )
    plt.tight_layout()
    fname = f"{FIG_DIR}/envelope_diagnostic_z{iz_label}_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Spaghetti plots  (identical interface to original)
# ---------------------------------------------------------------------------

def _exclude_outliers(errors, test_set, w0_abs_min=None):
    keep_mask = np.ones(len(test_set.lhs), dtype=bool)
    if w0_abs_min is not None:
        w0_col    = utils.params.index("w")
        w0_vals   = test_set.lhs[:, w0_col]
        near_zero = np.abs(w0_vals) < w0_abs_min
        print(f"  Excluding {near_zero.sum()} cosmologies with |w0| < {w0_abs_min} from plots")
        keep_mask &= ~near_zero
    return errors[keep_mask], keep_mask


def _spaghetti_ax(ax, ks, errors, ylabel, title):
    for error in errors:
        ax.semilogx(ks, error, lw=0.6, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(ylabel, fontsize=AXES_FS)
    ax.set_title(title, fontsize=AXES_FS)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.grid(alpha=0.3)


def plot_pca_errors(errors, ks, iz_label):
    fig, ax = plt.subplots(figsize=(8, 5))
    _spaghetti_ax(ax, ks, errors,
        ylabel=fr"$(P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{nl,syren}}) - 1$",
        title=(fr"PCA Errors (envelope-corrected) — "
               fr"$\log(P_{{{NL_TYPE}}}^\mathrm{{CAMB}}/P_\mathrm{{nl,syren}})$"
               + fr" at $z\approx{iz_label}$, $N_\mathrm{{PC}}={NUM_PCS}$"))
    plt.tight_layout()
    fname = f"{FIG_DIR}/pca_errors_z{iz_label}_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def plot_tpca_errors(errors, ks, iz_label):
    fig, ax = plt.subplots(figsize=(8, 5))
    _spaghetti_ax(ax, ks, errors,
        ylabel=fr"$(P_{{{NL_TYPE}}}^\mathrm{{CAMB}} / P_\mathrm{{nl,syren}}) - 1$",
        title=(fr"tPCA Errors (envelope-corrected) — "
               fr"$\log(P_{{{NL_TYPE}}}^\mathrm{{CAMB}}/P_\mathrm{{nl,syren}})$"
               + fr" at $z\approx{iz_label}$,"
               + fr" $N_\mathrm{{PC}}={NUM_PCS}$, $N_\mathrm{{tcomp}}={NUM_PCS_Z}$"))
    plt.tight_layout()
    fname = f"{FIG_DIR}/tpca_errors_z{iz_label}_{_tag()}.pdf"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


def print_error_summary(errors, stage, iz_label):
    max_abs = np.max(np.abs(errors), axis=1)
    finite  = np.isfinite(max_abs)
    max_abs = max_abs[finite]
    print(f"  {stage} (z~{iz_label}): "
          f"mean max |err| = {max_abs.mean():.5f}, "
          f"median = {np.median(max_abs):.5f}, "
          f"95th pct = {np.percentile(max_abs, 95):.5f}, "
          f"99th pct = {np.percentile(max_abs, 99):.5f}")


# ---------------------------------------------------------------------------
# Scree plot  (identical to original)
# ---------------------------------------------------------------------------

def plot_scree(train_set):
    n_z = len(utils.z_mps)
    spatial_evr   = np.array([train_set.pcas[iz].explained_variance_ratio_ for iz in range(n_z)])
    spatial_cumev = np.cumsum(spatial_evr, axis=1).mean(axis=0)
    tpca_evr      = train_set.tpca.explained_variance_ratio_
    tpca_cumev    = np.cumsum(tpca_evr)
    n_spatial     = spatial_evr.shape[1]
    n_tpca        = len(tpca_evr)
    xs_spatial    = np.arange(1, n_spatial + 1)
    xs_tpca       = np.arange(1, n_tpca + 1)

    highlight_colors = plt.get_cmap("plasma")(np.linspace(0.1, 0.9, len(SCREE_Z_HIGHLIGHT)))
    highlight_set    = set(SCREE_Z_HIGHLIGHT)

    fig, (ax_sp, ax_tp) = plt.subplots(1, 2, figsize=(14, 5))
    ax_sp2 = ax_sp.twinx()

    for iz in range(n_z):
        if iz not in highlight_set:
            ax_sp.semilogy(xs_spatial, spatial_evr[iz], color="0.75", lw=0.5, alpha=0.6)
    for col, iz in zip(highlight_colors, SCREE_Z_HIGHLIGHT):
        iz = min(iz, n_z - 1)
        ax_sp.semilogy(xs_spatial, spatial_evr[iz], color=col, lw=1.8,
                       label=fr"$z={utils.z_mps[iz]:.2f}$")
    ax_sp2.semilogy(xs_spatial, spatial_cumev, color="black", lw=1.5, ls="-.", alpha=0.8,
                    label="mean cumulative")
    for thresh, ls in [(0.95, ":"), (0.99, "--")]:
        ax_sp2.axhline(thresh, color="black", lw=0.8, ls=ls, alpha=0.5)
    ax_sp.axvline(NUM_PCS, color="red", lw=1.2, ls="--", label=f"NUM_PCS={NUM_PCS}")
    ax_sp.set_xlabel("PC index", fontsize=AXES_FS)
    ax_sp.set_ylabel("Individual EVR", fontsize=AXES_FS)
    ax_sp2.set_ylabel("Cumulative EVR (mean over z)", fontsize=AXES_FS - 1)
    ax_sp.set_title(fr"Spatial PCA scree (envelope-corrected)", fontsize=AXES_FS)
    ax_sp.set_xlim(1, n_spatial); ax_sp2.set_ylim(0, 1.05)
    ax_sp.tick_params(labelsize=TICK_FS); ax_sp2.tick_params(labelsize=TICK_FS)
    h1, l1 = ax_sp.get_legend_handles_labels()
    h2, l2 = ax_sp2.get_legend_handles_labels()
    ax_sp.legend(h1 + h2, l1 + l2, fontsize=TICK_FS - 1, loc="upper right")
    ax_sp.grid(alpha=0.3)

    ax_tp2 = ax_tp.twinx()
    ax_tp.bar(xs_tpca, tpca_evr, color="#4c78a8", alpha=0.7, width=0.8, label="individual")
    ax_tp.set_yscale('log')
    ax_tp2.plot(xs_tpca, tpca_cumev, color="black", lw=1.5, ls="-.", alpha=0.8, label="cumulative")
    for thresh, ls in [(0.95, ":"), (0.99, "--")]:
        ax_tp2.axhline(thresh, color="black", lw=0.8, ls=ls, alpha=0.5)
    ax_tp.axvline(NUM_PCS_Z, color="red", lw=1.2, ls="--", label=f"NUM_PCS_Z={NUM_PCS_Z}")
    ax_tp.set_xlabel("tPCA index", fontsize=AXES_FS)
    ax_tp.set_ylabel("Individual EVR", fontsize=AXES_FS)
    ax_tp2.set_ylabel("Cumulative EVR", fontsize=AXES_FS - 1)
    ax_tp.set_title("tPCA scree (envelope-corrected)", fontsize=AXES_FS)
    ax_tp.set_xlim(0.5, n_tpca + 0.5); ax_tp2.set_ylim(0, 1.05)
    ax_tp.tick_params(labelsize=TICK_FS); ax_tp2.tick_params(labelsize=TICK_FS)
    h1, l1 = ax_tp.get_legend_handles_labels()
    h2, l2 = ax_tp2.get_legend_handles_labels()
    ax_tp.legend(h1 + h2, l1 + l2, fontsize=TICK_FS - 1, loc="center right")
    ax_tp.grid(alpha=0.3)

    fig.suptitle(f"Scree plots (envelope-corrected) — {COSMO_TYPE} / {NL_TYPE} / {PRIOR_TYPE}",
                 fontsize=AXES_FS + 1)
    plt.tight_layout()
    fname = f"{FIG_DIR}/scree_{_tag()}.pdf"
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
    print("[INFO] PCA/tPCA Validation WITH T_AGN Envelope Correction")
    print(f"       nl_type    = {NL_TYPE}")
    print(f"       W0WA_MAX   = {W0WA_MAX}  |  OM_MIN = {OM_MIN}")
    print(f"       NUM_PCS    = {NUM_PCS}   |  NUM_PCS_Z = {NUM_PCS_Z}")
    print(f"       N_TAGN_BINS = {N_TAGN_BINS}")
    print("=" * 60)

    # --- Load ---
    print("\n[INFO] Loading training set...")
    train_set = utils.COLASet(
        target_z    = utils.z_mps,
        cosmo_type  = COSMO_TYPE,
        prior_type  = PRIOR_TYPE,
        nl_type     = NL_TYPE,
        n_batches   = N_BATCHES,
        start_batch = START_BATCH,
    )

    # --- Cuts ---
    n_removed = apply_filter(train_set, om_min=None, w0_min=None, w0wa_max=None,
                             omegab_h0_triangle_cut=OMEGAB_H0_TRIANGLE_CUT)
    print(f"\n[INFO] Prior cuts: removed {n_removed} ({len(train_set.lhs)} remaining).")

    # --- Fit envelope ---
    print("\n[INFO] Fitting T_AGN logfrac envelope per redshift...")
    fit_envelope_per_z(train_set, n_tagn_bins=N_TAGN_BINS)

    # --- Envelope diagnostic ---
    print("\n[INFO] Saving envelope diagnostic plots...")
    plot_envelope_diagnostic(train_set, iz=Z_IDX_0, iz_label=0)
    plot_envelope_diagnostic(train_set, iz=Z_IDX_3, iz_label=3)

    # --- Prepare with envelope correction ---
    print("\n[INFO] Preparing PCA/tPCA with envelope correction...")
    prepare_with_envelope(train_set, num_pcs=NUM_PCS, num_pcs_z=NUM_PCS_Z)

    # --- Scree ---
    print("\n[INFO] Saving scree plots...")
    plot_scree(train_set)

    # --- Test set ---
    print("\n[INFO] Loading test set...")
    test_set = utils.COLASet(
        target_z    = utils.z_mps,
        cosmo_type  = COSMO_TYPE,
        prior_type  = PRIOR_TYPE,
        nl_type     = NL_TYPE,
        start_batch = TEST_BATCH,
    )
    apply_filter(test_set, om_min=None, w0_min=None, w0wa_max=None,
                 omegab_h0_triangle_cut=OMEGAB_H0_TRIANGLE_CUT)
    print(f"[INFO] Test set: {len(test_set.lhs)} cosmologies after cuts.")

    ks = train_set.ks

    # --- PCA errors ---
    print("\n[INFO] Computing envelope-corrected PCA reconstruction errors...")
    pca_err_z0 = pca_reconstruction_errors_envelope(train_set, test_set, Z_IDX_0)
    pca_err_z3 = pca_reconstruction_errors_envelope(train_set, test_set, Z_IDX_3)

    pca_err_z0_plot, _ = _exclude_outliers(pca_err_z0, test_set, w0_abs_min=PLOT_W0_ABS_MIN)
    pca_err_z3_plot, _ = _exclude_outliers(pca_err_z3, test_set, w0_abs_min=PLOT_W0_ABS_MIN)

    print("\n  PCA error summary (envelope-corrected):")
    print_error_summary(pca_err_z0_plot, "PCA", iz_label=0)
    print_error_summary(pca_err_z3_plot, "PCA", iz_label=3)

    # --- tPCA errors ---
    print("\n[INFO] Computing envelope-corrected tPCA reconstruction errors...")
    stacks      = tpca_stacks_envelope(train_set, test_set)
    tpca_err_z0 = tpca_reconstruction_errors(stacks, test_set, Z_IDX_0)
    tpca_err_z3 = tpca_reconstruction_errors(stacks, test_set, Z_IDX_3)

    tpca_err_z0_plot, _ = _exclude_outliers(tpca_err_z0, test_set, w0_abs_min=PLOT_W0_ABS_MIN)
    tpca_err_z3_plot, _ = _exclude_outliers(tpca_err_z3, test_set, w0_abs_min=PLOT_W0_ABS_MIN)

    print("\n  tPCA error summary (envelope-corrected):")
    print_error_summary(tpca_err_z0_plot, "tPCA", iz_label=0)
    print_error_summary(tpca_err_z3_plot, "tPCA", iz_label=3)

    # --- Figures ---
    print("\n[INFO] Saving figures...")
    plot_pca_errors(pca_err_z0_plot,  ks, iz_label=0)
    plot_tpca_errors(tpca_err_z0_plot, ks, iz_label=0)
    plot_pca_errors(pca_err_z3_plot,  ks, iz_label=3)
    plot_tpca_errors(tpca_err_z3_plot, ks, iz_label=3)

    print("\n[INFO] Done.")
    print("\n[INFO] Compare these PCA/tPCA errors against t_comp_val_nl.py output")
    print("       to quantify the benefit of the envelope correction.")


if __name__ == "__main__":
    main()