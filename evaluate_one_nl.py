"""
evaluate_one_nl.py — Nonlinear Emulator Evaluation Script (end-to-end)

Evaluates a trained nonlinear boost emulator against CAMB nonlinear ground
truth, using the *linear emulator* (not CAMB linear) for P_lin:

    P_nl^emu = frac_NL^emu * B_syren * P_lin^emu
    error    = P_nl^emu / P_nl^CAMB - 1

Grids
-----
Each emulator runs on the grid it was trained on, independent of whatever
grid train_utils_pk_emulator_v3 currently defines:

  * linear emulator : full grid (FULL_KS: 500 k, FULL_ZS: 52 z)
  * boost emulator  : z taken from the PCA keys in its metadata bundle,
                      k = FULL_KS within [BOOST_KMIN, BOOST_KMAX]
                      (checked against the PCA length)

The linear output and the CAMB test set are both sliced onto the boost grid.

Error decomposition (exact):
    1 + err_total = (1 + err_NL) * (1 + err_lin)
    err_NL  = frac_pred / frac_true - 1     with frac_true = B_camb / B_syren
    err_lin = P_lin^emu / P_lin^CAMB - 1
B_syren is the same one the emulator uses (computed on the boost k grid), so
err_NL is consistent with the network's training target.

Usage (standalone):
    python ./mps_emu/evaluate_one_nl.py

Author: Victoria Lloyd (2025)
"""

import os
import joblib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.interpolate import interp1d

import emulmps_w0wa as pk_emu
import train_utils_pk_emulator_v3 as utils
from train_utils_pk_emulator_v3 import VER


# ---------------------------------------------------------------------------
# Run configuration — nonlinear (boost) emulator
# ---------------------------------------------------------------------------

START_BATCH    = 100
N_BATCHES      = 20
N_TRAIN        = 100
COSMO_TYPE     = "w0wacdm"
NL_TYPE        = "mead2020_Tfree_mnufree"
PRIOR_TYPE     = "expanded"
MODEL_TYPE     = "npce"
METADATA_DIR   = "mps_emu/metadata"

import numpy as np, train_utils_pk_emulator_v3 as utils   # full 52-z grid restored

z1_mps = np.linspace(0,3,33,endpoint=False)
z2_mps = np.linspace(3,10,7,endpoint=False)
z3_mps = np.linspace(10,50,12)
FULL_ZS = np.concatenate((z1_mps, z2_mps, z3_mps), axis=0) #, z3_mps


z_hi = np.concatenate(([9.0], FULL_ZS[FULL_ZS >= 10]))      # z=9 anchors options 3/4
ts = utils.COLASet(target_z=z_hi, cosmo_type=COSMO_TYPE, prior_type=PRIOR_TYPE,
                   nl_type=NL_TYPE, start_batch=START_BATCH, n_batches=N_BATCHES)

frac    = ts.frac_pks                     # B_camb / B_syren       (N, Nz, Nk)
B_syren = ts.mps_approxes_boost
B_camb  = frac * B_syren
f9, f, Bc = frac[:, :1], frac[:, 1:], B_camb[:, 1:]
decay = ((1 + 9.0) / (1 + z_hi[1:]))[None, :, None] ** 2

errs = {
    "1 linear only":      1 / Bc - 1,
    "2 syren only":       1 / f - 1,
    "3 freeze frac(z=9)": f9 / f - 1,
    "4 D^2 decay":        f9 ** decay / f - 1,
}
ks = np.asarray(ts.ks)
for name, e in errs.items():
    print(f"\n{name}")
    for iz, z in enumerate(z_hi[1:]):
        row = [np.percentile(np.max(np.abs(e[:, iz, (ks >= lo) & (ks < hi)]), axis=1), 95)
               for lo, hi in [(0, 1), (1, 10), (10, 50)]]
        print(f"  z={z:5.1f}  95th pct max|err|  k<1: {row[0]:.2e}  1–10: {row[1]:.2e}  10–50: {row[2]:.2e}")

# k range the boost emulator was trained on (cut of FULL_KS)
BOOST_KMIN, BOOST_KMAX = 0.005, 50.0

# ---------------------------------------------------------------------------
# Run configuration — linear emulator (full grid)
# ---------------------------------------------------------------------------

LIN_NL_TYPE       = "mead2020_Tfree_mnufree_lin"
LIN_MODEL_TYPE    = "npce"
LIN_MODEL_PATH    = "mps_emu/models/emulator_npce_w0wacdm_expanded_mead2020_Tfree_mnufree_lin_tfreemnufree_nTrain100_v15.keras"
LIN_METADATA_PATH = "mps_emu/metadata/metadata_w0wacdm_expanded_mead2020_Tfree_mnufree_lin_tfreemnufree_nTrain100_v15/metadata.joblib"

# Full training grid (linear emulator and datagenerator files)
FULL_KS = np.logspace(-5.1, 2, 500)
FULL_ZS = np.concatenate((
    np.linspace(0, 3, 33, endpoint=False),
    np.linspace(3, 10, 7, endpoint=False),
    np.linspace(10, 50, 12),
))

# Target redshifts — snapped to the nearest boost-grid redshift
TARGET_REDSHIFTS = [0, 0.5, 1, 2, 5, 10, 30, 50]

# Prior cuts — must match training
W0_MIN_FILTER   = None
W0WA_MAX_FILTER = None

OMEGAB_H0_TRIANGLE_CUT = {
    'omegab_anchor': 0.048,
    'h0_anchor':     74.5,
    'omegab_max':    0.072,
    'h0_max':        90,
}

W0_W0WA_TRIANGLE_CUT = {
    'w0_min':      -2.0,    # test-prior w0 minimum
    'w0wa_anchor': -0.95,    # w0+wa where the line meets w0 = w0_min
    'w0_anchor':   -1.03,    # w0 where the line meets w0+wa = w0wa_max
    'w0wa_max':    -0.01,   # test-prior w0+wa maximum
}

# Test-prior box (ML convention, lhs column order).  Cosmologies outside any
# of these ranges are dropped before evaluation.  None to disable.
# Note: lhs column 6 stores w0+wa, so "w0wa" bounds apply to w0+wa.
TEST_PRIORS = {
    "As_1e9": (1.0,   3.5),
    "ns":     (0.8,   1.05),
    "H0":     (55.0,  88.0),
    "Ob":     (0.03,  0.07),
    "Om":     (0.2,   0.5),
    "w0":     (-2.0,  -0.01),
    "w0wa":   (-4.0,  -0.01),
    "Tagn":   (6.5,   8.0),
    "mnu":    (0.055, 0.6),
}
# lhs column for each name: [As, ns, H0, Ob, Om, w0, w0+wa, T_AGN, mnu]
TEST_PRIOR_COLS = {"As_1e9": 0, "ns": 1, "H0": 2, "Ob": 3, "Om": 4,
                   "w0": 5, "w0wa": 6, "Tagn": 7, "mnu": 8}

# Diagnostic thresholds (stats / filtered-subset split only)
W0_THRESHOLD    = -1.8
W0WA_THRESHOLD  = -0.75
W0_COL          = utils.params.index("w")
W0WA_COL        = utils.params.index("w0+wa")

FIG_DIR = "mps_emu/validation_figs/smaller_grid_linemu_fulltest"

K_WINDOW_LO, K_WINDOW_HI = 20.0, 30.0
TAGN_COL = utils.params_tfree_mnufree.index("T_AGN")   # column 7

# Datagenerator NL reference, shape (1, 52, 500) on FULL grid. None to skip.
DG_PK_PATH = None

# Fiducial: [10^9 A_s, ns, H0, Omega_b, Omega_m, w0, w0+wa, log T_AGN, mnu]
FIDUCIAL_PARAMS = [2.1, 0.96605, 67.32, 0.04, 0.3, -0.9, -0.9, 7.3, 0.06]

# Filled by setup_grids(): boost grid + index maps into the full grid
GRID = {}


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
# Grids
# ---------------------------------------------------------------------------

def _nl_n_params():
    return 9 if "Tfree_mnufree" in NL_TYPE else 7


def _resolve_nl_metadata_path():
    dummy = utils.COLASet.__new__(utils.COLASet)
    dummy.cosmo_type = COSMO_TYPE
    dummy.prior_type = PRIOR_TYPE
    dummy.nl_type    = NL_TYPE
    dummy.n_batches  = N_TRAIN
    dummy.w0_min     = W0_MIN_FILTER
    dummy.w0wa_max   = W0WA_MAX_FILTER
    dummy.use_boost  = True
    dummy.lhs        = np.empty((1, _nl_n_params()))
    return os.path.join(METADATA_DIR, f"metadata_{dummy._metadata_tag()}", "metadata.joblib")


def setup_grids():
    """Derive the boost grid from the boost model's own metadata bundle."""
    path = _resolve_nl_metadata_path()
    print(f"[INFO] Boost metadata bundle: {path}")
    bundle = joblib.load(path)

    # z: PCA keys are float(f"{z:.3f}") of the training redshifts
    z_keys = np.array(sorted(bundle["pcas"].keys()), dtype=float)
    z_idx  = np.array([np.argmin(np.abs(FULL_ZS - zk)) for zk in z_keys])
    if np.max(np.abs(FULL_ZS[z_idx] - z_keys)) > 6e-4:
        raise ValueError("Boost-model redshifts are not a subset of FULL_ZS.")

    # k: fixed cut of the full grid; must match the PCA vector length
    k_idx  = np.where((FULL_KS >= BOOST_KMIN) & (FULL_KS <= BOOST_KMAX))[0]
    n_k_pca = next(iter(bundle["pcas"].values())).components_.shape[1]
    if n_k_pca != len(k_idx):
        raise ValueError(
            f"Boost PCA has {n_k_pca} k-points but [{BOOST_KMIN}, {BOOST_KMAX}] "
            f"selects {len(k_idx)} — adjust BOOST_KMIN / BOOST_KMAX."
        )

    GRID.update(
        ks=FULL_KS[k_idx], zs=FULL_ZS[z_idx],
        full_k_idx=k_idx, full_z_idx=z_idx,
        pcas=bundle["pcas"], scalers=bundle["scalers"],
    )
    GRID["k_window"] = (GRID["ks"] >= K_WINDOW_LO) & (GRID["ks"] <= K_WINDOW_HI)
    print(f"[INFO] Boost grid: {len(k_idx)} k in [{GRID['ks'][0]:.4g}, {GRID['ks'][-1]:.4g}], "
          f"{len(z_idx)} z in [{GRID['zs'][0]:.3g}, {GRID['zs'][-1]:.3g}]")


def _grid_emulator_class(ks, zs, name):
    """PkEmulator subclass pinned to a given grid (methods read self.K_MODES / Z_MODES)."""
    return type(name, (pk_emu.PkEmulator,),
                dict(K_MODES=ks, N_K_MODES=len(ks), Z_MODES=zs, N_ZS=len(zs)))


def load_emulators():
    LinEmu   = _grid_emulator_class(FULL_KS, FULL_ZS, "FullGridPkEmulator")
    BoostEmu = _grid_emulator_class(GRID["ks"], GRID["zs"], "BoostGridPkEmulator")

    print(f"[INFO] Loading linear emulator: {LIN_MODEL_PATH}")
    lin_emu = LinEmu.from_paths(LIN_MODEL_PATH, LIN_METADATA_PATH,
                                nl_type=LIN_NL_TYPE, model_type=LIN_MODEL_TYPE)
    if lin_emu.use_boost:
        raise ValueError(f"LIN_NL_TYPE='{LIN_NL_TYPE}' is not a linear nl_type.")

    print("[INFO] Loading boost emulator (by tag)...")
    nl_emu = BoostEmu(cosmo_type=COSMO_TYPE, prior_type=PRIOR_TYPE, nl_type=NL_TYPE,
                      model_type=MODEL_TYPE, num_batches=N_TRAIN,
                      w0_min=W0_MIN_FILTER, w0wa_max=W0WA_MAX_FILTER)
    return lin_emu, nl_emu


def emulate_plin_on_boost_grid(lin_emu, params):
    _, _, pk_full = lin_emu.get_pks(np.asarray(params))          # (52, 500)
    return pk_full[np.ix_(GRID["full_z_idx"], GRID["full_k_idx"])]


# ---------------------------------------------------------------------------
# Redshift / tag helpers
# ---------------------------------------------------------------------------

def resolve_redshift_indices():
    seen, result = set(), []
    for target in TARGET_REDSHIFTS:
        iz = int(np.argmin(np.abs(GRID["zs"] - target)))
        z_val = float(GRID["zs"][iz])
        if iz in seen:
            print(f"  [WARN] target z={target} maps to iz={iz} (z={z_val:.4f}) "
                  "already included — skipping.")
            continue
        seen.add(iz)
        result.append((iz, z_val))
    return result


def _filter_tag():
    parts = []
    if W0_MIN_FILTER   is not None:
        parts.append(f"w0min{W0_MIN_FILTER}")
    if W0WA_MAX_FILTER is not None:
        parts.append(f"w0wamax{W0WA_MAX_FILTER}")
    parts += ["linemu", VER]
    return "_" + "_".join(parts)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_test_set():
    """Load CAMB test data at the boost redshifts, apply cuts, slice k to the boost grid."""
    test_set = utils.COLASet(
        target_z    = GRID["zs"],
        cosmo_type  = COSMO_TYPE,
        prior_type  = PRIOR_TYPE,
        nl_type     = NL_TYPE,
        start_batch = START_BATCH,
        n_batches   = N_BATCHES,
    )

    mask = np.ones(len(test_set.lhs), dtype=bool)
    if W0_MIN_FILTER is not None:
        mask &= test_set.lhs[:, W0_COL] >= W0_MIN_FILTER
    if W0WA_MAX_FILTER is not None:
        mask &= test_set.lhs[:, W0WA_COL] <= W0WA_MAX_FILTER
    if OMEGAB_H0_TRIANGLE_CUT is not None:
        c = OMEGAB_H0_TRIANGLE_CUT
        ob = test_set.lhs[:, utils.params.index("Omega_b")]
        h0 = test_set.lhs[:, utils.params.index("h")]
        slope = (c['h0_anchor'] - c['h0_max']) / (c['omegab_max'] - c['omegab_anchor'])
        tri = h0 <= c['h0_max'] + slope * (ob - c['omegab_anchor'])
        print(f"[INFO] Omega_b/H0 triangle cut: removing {(~tri & mask).sum()} cosmologies.")
        mask &= tri
    if TEST_PRIORS is not None:
        print("[INFO] Test-prior box cut:")
        for name, (lo, hi) in TEST_PRIORS.items():
            col = TEST_PRIOR_COLS[name]
            if col >= test_set.lhs.shape[1]:
                print(f"    {name:<7s} column {col} not in lhs — skipped.")
                continue
            vals   = test_set.lhs[:, col]
            in_box = (vals >= lo) & (vals <= hi)
            print(f"    {name:<7s} [{lo:g}, {hi:g}]: removing {(~in_box & mask).sum()} "
                  f"additional (data range {vals.min():.4g} – {vals.max():.4g})")
            mask &= in_box
    if W0_W0WA_TRIANGLE_CUT is not None:
        c    = W0_W0WA_TRIANGLE_CUT
        w0   = test_set.lhs[:, W0_COL]
        w0wa = test_set.lhs[:, W0WA_COL]                      # column 6 holds w0+wa
        slope = (c['w0wa_max'] - c['w0wa_anchor']) / (c['w0_anchor'] - c['w0_min'])
        w0wa_line = c['w0wa_anchor'] + slope * (w0 - c['w0_min'])
        tri_w = w0wa <= w0wa_line                             # keep on/below the line
        print(f"[INFO] w0/w0+wa corner cut: removing {(~tri_w & mask).sum()} additional "
              f"cosmologies (line from w0={c['w0_min']}, w0+wa={c['w0wa_anchor']} "
              f"to w0={c['w0_anchor']}, w0+wa={c['w0wa_max']}).")
        mask &= tri_w
    print(f"[INFO] Test set after prior cuts: {mask.sum()} / {len(mask)} cosmologies kept.")

    # CAMB P_lin recovered exactly: pks_target = frac * B_syren(test) * P_lin_camb
    plin_camb = test_set.pks_target / (test_set.frac_pks * test_set.mps_approxes_boost)

    # k slice: boost k inside the test set's k grid (utils.ks)
    tks = np.asarray(test_set.ks)
    kt_idx = np.array([np.argmin(np.abs(np.log(tks) - np.log(k))) for k in GRID["ks"]])
    if not np.allclose(tks[kt_idx], GRID["ks"], rtol=1e-10, atol=0):
        raise ValueError("Boost k grid is not a subset of the test set's k grid (utils.ks).")

    return dict(
        lhs       = test_set.lhs[mask],
        pk_nl     = test_set.pks_target[mask][..., kt_idx],    # CAMB P_nl  (N, Nz_b, Nk_b)
        plin_camb = plin_camb[mask][..., kt_idx],              # CAMB P_lin (N, Nz_b, Nk_b)
    )


def compute_predictions(data, lin_emu, nl_emu):
    """
    Returns pred_pks, true_pks, true_frac, pred_frac, err_lin, lhs — all on the
    boost grid.  true_frac uses the emulator's own B_syren (boost k grid).
    """
    pred_l, frac_l, plin_l = [], [], []
    for row in data["lhs"]:
        plin_emu = emulate_plin_on_boost_grid(lin_emu, row)
        _, _, pk     = nl_emu.get_pks(row, pk_lin=plin_emu)   # frac * B_syren * P_lin^emu
        _, _, fpred  = nl_emu.get_boost(row)                   # frac (network output)
        pred_l.append(pk); frac_l.append(fpred); plin_l.append(plin_emu)

    pred_pks  = np.asarray(pred_l, dtype=np.float64)
    pred_frac = np.asarray(frac_l, dtype=np.float64)
    plin_emu  = np.asarray(plin_l, dtype=np.float64)

    b_syren   = pred_pks / (pred_frac * plin_emu)          # exactly the emulator's B_syren
    b_camb    = data["pk_nl"] / data["plin_camb"]
    true_frac = b_camb / b_syren
    err_lin   = plin_emu / data["plin_camb"] - 1

    valid = np.isfinite(pred_pks).all(axis=(1, 2)) & np.isfinite(true_frac).all(axis=(1, 2))
    if (~valid).any():
        print(f"  WARNING: dropping {(~valid).sum()} cosmologies with non-finite output.")
    return (pred_pks[valid], data["pk_nl"][valid], true_frac[valid],
            pred_frac[valid], err_lin[valid], data["lhs"][valid])


def apply_diagnostic_filter(errors, lhs):
    good = (lhs[:, W0_COL] > W0_THRESHOLD) & (lhs[:, W0WA_COL] < W0WA_THRESHOLD)
    return errors[good], good


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def percentiles(err):
    ps = np.percentile(err, [5, 10, 25, 75, 90, 95], axis=0)
    return ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]


def _max_err_summary(label, e):
    m = np.max(np.abs(e), axis=1)
    print(f"    {label:<22s} mean {np.mean(m):.6f} | median {np.median(m):.6f} | "
          f"95% {np.percentile(m, 95):.6f} | 99% {np.percentile(m, 99):.6f}")


def print_statistics(err_total, err_nl, err_lin, good, iz, z_val):
    print(f"\n=== ERROR STATISTICS  (iz={iz}, z={z_val:.4f}) — max |error| over k ===")
    for label, m in [("Full dataset", np.ones(len(err_total), bool)),
                     (f"w0>{W0_THRESHOLD} & w0+wa<{W0WA_THRESHOLD}", good)]:
        print(f"\n  {label}  (N={m.sum()}):")
        if m.sum() == 0:
            continue
        _max_err_summary("total (lin emu x NL)", err_total[m])
        _max_err_summary("NL emulator only",     err_nl[m])
        _max_err_summary("linear emulator only", err_lin[m])


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
    ax.fill_between(ks, p25, p75, color=COLOR_50, hatch="\\", edgecolor="lightgray", alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=AXES_FS)
    ax.set_xscale("log")
    ax.grid(alpha=0.4)
    ax.set_xlim([ks[0], ks[-1]])
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.text(0.05, 0.95, label_text, transform=ax.transAxes, ha='left', va='top', fontsize=AXES_FS)


def _save(fig, stem, z_val, dpi=None):
    fname = (f"{FIG_DIR}/{stem}_z{z_val:.4g}_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_{MODEL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
    fig.savefig(fname, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_error_decomposition(err_total, err_nl, err_lin, ks, z_val):
    fig, ax = plt.subplots(figsize=(9, 5))
    for e, color, label in [(err_lin, "C0", "linear emulator"),
                            (err_nl, "C2", "NL emulator (boost)"),
                            (err_total, "C3", "total")]:
        lo, med, hi = np.percentile(e, [5, 50, 95], axis=0)
        ax.fill_between(ks, lo, hi, color=color, alpha=0.2)
        ax.semilogx(ks, med, color=color, lw=1.8, label=f"{label} (median, 5–95%)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xscale("log"); ax.set_xlim(ks[0], ks[-1])
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(r"$P^\mathrm{emu}/P^\mathrm{CAMB} - 1$", fontsize=AXES_FS)
    ax.set_title(f"Error decomposition, z={z_val:.4g}", fontsize=AXES_FS - 4)
    ax.legend(fontsize=LEGEND_FS - 5); ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS - 2)
    plt.tight_layout()
    _save(fig, "error_decomposition", z_val)


def plot_boost_error_lines_filtered(errors_filt, ks, good, z_val):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i in range(min(good.sum(), 200)):
        ax.semilogx(ks, errors_filt[i])
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(ERR_LABEL, fontsize=AXES_FS)
    ax.grid(alpha=0.3); ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    ax.text(0.05, 0.95,
            fr'$z={z_val:.4g}$,  $w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$  (N={good.sum()})',
            transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FS)
    _save(fig, f"val_boost_errors_w0gt{W0_THRESHOLD}_w0walt{W0WA_THRESHOLD}", z_val)


def plot_comparison_bands(errors, ks, z_val):
    fig, ax = plt.subplots(figsize=(8, 5))
    _fill_ax(ax, ks, *percentiles(errors), ylabel=ERR_LABEL, label_text=f'$z={z_val:.4g}$')
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95], fontsize=LEGEND_FS, loc="lower right")
    plt.tight_layout()
    _save(fig, "boost_errors", z_val)


def plot_comparison_bands_zoomed(errors, ks, z_val, k_lo=10.0, k_hi=50.0):
    m = (ks >= k_lo) & (ks <= k_hi)
    fig, ax = plt.subplots(figsize=(8, 5))
    _fill_ax(ax, ks[m], *percentiles(errors[:, m]), ylabel=ERR_LABEL,
             label_text=f'$z={z_val:.4g}$  (k zoom {k_lo:g}-{k_hi:g})')
    ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="purple", alpha=0.08)
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95], fontsize=LEGEND_FS, loc="best")
    plt.tight_layout()
    _save(fig, "boost_errors_zoom", z_val)


def plot_max_error_histogram(errors, z_val):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.max(np.abs(errors), axis=1), bins=50, color='#1565c0', alpha=0.7, edgecolor='black')
    ax.set_xlabel(fr"Max $|P_{{{NL_TYPE}}}^\mathrm{{{MODEL_TYPE.upper()}}} / "
                  fr"P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1|$", fontsize=16)
    ax.set_ylabel('Count', fontsize=16)
    ax.set_title(f'{MODEL_TYPE.upper()} — {COSMO_TYPE}, {NL_TYPE}   ($z={z_val:.4g}$)', fontsize=18)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=14)
    plt.tight_layout()
    _save(fig, "max_boost_error_histogram", z_val)


def plot_residual_logfracs(true_frac, lhs, ks, iz, z_val):
    """log(B_camb / B_syren) per cosmology — the boost emulator's training target."""
    good = (lhs[:, W0_COL] > W0_THRESHOLD) & (lhs[:, W0WA_COL] < W0WA_THRESHOLD)
    lf = np.log(true_frac[:, iz, :])
    fig, ax = plt.subplots(figsize=(9, 5))
    for idx in np.where(good)[0]:
        ax.semilogx(ks, lf[idx], color=COLOR_GOOD, alpha=0.08, lw=0.5, rasterized=True)
    for idx in np.where(~good)[0]:
        ax.semilogx(ks, lf[idx], color=COLOR_BAD, alpha=0.08, lw=0.5, rasterized=True)
    if good.sum():
        ax.semilogx(ks, np.median(lf[good], axis=0), color=COLOR_GOOD, lw=2.2,
                    label=fr"$w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$ (n={good.sum()})")
    if (~good).sum():
        ax.semilogx(ks, np.median(lf[~good], axis=0), color=COLOR_BAD, lw=2.2,
                    label=fr"fails cut (n={(~good).sum()})")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(r"$\log\!\left(B^\mathrm{CAMB} / B^\mathrm{syren}\right)$", fontsize=AXES_FS)
    ax.set_title(fr"$z={z_val:.4g}$   |   {COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, {MODEL_TYPE.upper()}",
                 fontsize=15)
    ax.legend(fontsize=14); ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
    plt.tight_layout()
    _save(fig, "residual_logfracs", z_val, dpi=150)


def plot_frac_shape_bands(true_frac, ks, iz, z_val):
    fig, ax = plt.subplots(figsize=(8, 5))
    _fill_ax(ax, ks, *percentiles(true_frac[:, iz, :] - 1),
             ylabel=r"$B^\mathrm{CAMB} / B^\mathrm{syren} - 1$",
             label_text=f'$z={z_val:.4g}$  (CAMB target)')
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95], fontsize=LEGEND_FS, loc="upper left")
    plt.tight_layout()
    _save(fig, "camb_residual_frac", z_val)


def plot_max_error_vs_tagn(errors, lhs, z_val):
    if lhs.shape[1] <= TAGN_COL:
        print("  [WARN] T_AGN column not present — skipping plot_max_error_vs_tagn.")
        return
    win  = np.max(np.abs(errors[:, GRID["k_window"]]), axis=1)
    full = np.max(np.abs(errors), axis=1)
    win  = np.clip(win,  max(win[win > 0].min() * 0.5, 1e-8)  if np.any(win > 0)  else 1e-8, None)
    full = np.clip(full, max(full[full > 0].min() * 0.5, 1e-8) if np.any(full > 0) else 1e-8, None)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(lhs[:, TAGN_COL], win, c=full, cmap="YlOrRd",
                    norm=mpl.colors.LogNorm(vmin=full.min(), vmax=full.max()),
                    s=12, alpha=0.7, rasterized=True)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log T_\mathrm{AGN}$", fontsize=AXES_FS)
    ax.set_ylabel(fr"max $|$error$|$ in $k \in [{K_WINDOW_LO:g},{K_WINDOW_HI:g}]$", fontsize=AXES_FS - 4)
    ax.set_title(f"z={z_val:.4g}  (colour = max |error| over full k range)", fontsize=AXES_FS - 5)
    ax.grid(alpha=0.3, which="both")
    fig.colorbar(sc, ax=ax).set_label("max |error| (full k)", fontsize=TICK_FS - 3)
    plt.tight_layout()
    _save(fig, "max_error_vs_tagn", z_val, dpi=150)


def plot_worst_offenders_in_window(errors, ks, z_val, n_worst=30):
    worst = np.argsort(np.max(np.abs(errors[:, GRID["k_window"]]), axis=1))[::-1][:n_worst]
    fig, ax = plt.subplots(figsize=(8, 5))
    for idx in worst:
        ax.semilogx(ks, errors[idx], alpha=0.6, lw=1)
    ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="purple", alpha=0.08)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(ERR_LABEL, fontsize=AXES_FS - 4)
    ax.set_title(f"Worst {n_worst} by max|error| in k={K_WINDOW_LO:g}-{K_WINDOW_HI:g} (z={z_val:.4g})",
                 fontsize=AXES_FS - 6)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _save(fig, "worst_offenders_window", z_val)


def plot_error_heatmap_vs_tagn(errors, ks, lhs, z_val, n_bins=10):
    if lhs.shape[1] <= TAGN_COL:
        print("  [WARN] T_AGN column not present — skipping plot_error_heatmap_vs_tagn.")
        return
    t = lhs[:, TAGN_COL]
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    bidx  = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    heat  = np.full((n_bins, len(ks)), np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = bidx == b
        counts[b] = m.sum()
        if counts[b]:
            heat[b] = np.median(np.abs(errors[m]), axis=0)
    fv = heat[np.isfinite(heat) & (heat > 0)]
    if fv.size == 0:
        print(f"  [WARN] No finite positive heatmap values at z={z_val:.4g} — skipping.")
        return
    centres = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.pcolormesh(ks, centres, heat, shading="auto", cmap="YlOrRd",
                       norm=mpl.colors.LogNorm(vmin=fv.min(), vmax=fv.max()))
    ax.set_xscale("log")
    ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="cyan", alpha=0.15)
    ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    ax.set_ylabel(r"$\log T_\mathrm{AGN}$", fontsize=AXES_FS)
    ax.set_title(f"Median |error| vs (k, T_AGN)  (z={z_val:.4g})", fontsize=AXES_FS - 5)
    fig.colorbar(im, ax=ax).set_label("median |error|", fontsize=TICK_FS - 3)
    for b, c in enumerate(counts):
        ax.text(ks[-1] * 1.02, centres[b], f"n={c}", fontsize=7, va="center", clip_on=False)
    plt.tight_layout()
    _save(fig, "error_heatmap_vs_tagn", z_val, dpi=150)


# ---------------------------------------------------------------------------
# PC-level diagnostics (boost emulator only)
# ---------------------------------------------------------------------------

def _pca_scaler(z_val):
    z_key = float(f"{z_val:.3f}")
    return GRID["pcas"][z_key], GRID["scalers"][z_key]


def _pc_responsibility(z_val):
    pca, scaler = _pca_scaler(z_val)
    scale = scaler.scale_ if hasattr(scaler, "scale_") else scaler.std
    mat = pca.components_ * scale[None, :]
    return np.sum(mat[:, GRID["k_window"]] ** 2, axis=1) / np.sum(mat ** 2, axis=1)


def _frac_to_pcs(frac_z, z_val):
    pca, scaler = _pca_scaler(z_val)
    return pca.transform(scaler.transform(np.log(frac_z)))


def plot_pc_true_vs_predicted(true_frac, pred_frac, lhs, iz, z_val, top_n=3):
    pt, pp = _frac_to_pcs(true_frac[:, iz], z_val), _frac_to_pcs(pred_frac[:, iz], z_val)
    resp = _pc_responsibility(z_val)
    has_tagn = lhs.shape[1] > TAGN_COL
    fig, axes = plt.subplots(1, top_n, figsize=(5.5 * top_n, 5), squeeze=False)
    for ax, p in zip(axes[0], np.argsort(resp)[::-1][:top_n]):
        x, y = pt[:, p], pp[:, p]
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", lw=1, alpha=0.6, zorder=1)
        if has_tagn:
            sc = ax.scatter(x, y, c=lhs[:, TAGN_COL], cmap="viridis", s=10, alpha=0.6, rasterized=True)
            fig.colorbar(sc, ax=ax, label=r"$\log T_\mathrm{AGN}$")
        else:
            ax.scatter(x, y, s=10, alpha=0.5, rasterized=True, color="C0")
        ax.set_xlabel("true PC coefficient"); ax.set_ylabel("predicted PC coefficient")
        ax.set_title(f"z={z_val:.2f}, PC {p}\nresp={resp[p]:.2f}, "
                     f"RMSE={np.sqrt(np.mean((x - y) ** 2)):.3g}, r={np.corrcoef(x, y)[0, 1]:.4f}",
                     fontsize=AXES_FS - 6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "pc_true_vs_pred", z_val, dpi=150)


def plot_pc_residual_vs_tagn(true_frac, pred_frac, lhs, iz, z_val, top_n=3):
    if lhs.shape[1] <= TAGN_COL:
        print("  [WARN] T_AGN column not present — skipping plot_pc_residual_vs_tagn.")
        return
    pt, pp = _frac_to_pcs(true_frac[:, iz], z_val), _frac_to_pcs(pred_frac[:, iz], z_val)
    resp = _pc_responsibility(z_val)
    fig, axes = plt.subplots(1, top_n, figsize=(5.5 * top_n, 4), squeeze=False)
    for ax, p in zip(axes[0], np.argsort(resp)[::-1][:top_n]):
        ax.axhline(0, color="gray", ls=":", lw=1)
        ax.scatter(lhs[:, TAGN_COL], pp[:, p] - pt[:, p], s=8, alpha=0.4, rasterized=True, color="C2")
        ax.set_xlabel(r"$\log T_\mathrm{AGN}$"); ax.set_ylabel("predicted - true")
        ax.set_title(f"z={z_val:.2f}, PC {p} (resp={resp[p]:.2f})", fontsize=AXES_FS - 6)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "pc_residual_vs_tagn", z_val, dpi=150)


# ---------------------------------------------------------------------------
# Triangle plot
# ---------------------------------------------------------------------------

_PARAM_LABELS_7 = [r'$10^9 A_s$', r'$n_s$', r'$H_0$', r'$\Omega_b$',
                   r'$\Omega_m$', r"$w_0$", r"$w_0+w_a$"]
# Column order follows utils.params_tfree_mnufree: T_AGN (col 7), then mnu (col 8)
_PARAM_LABELS_9 = _PARAM_LABELS_7 + [r'$\log T_{\rm AGN}$', r'$m_\nu$']



def plot_error_triangle(errors, lhs, z_val):
    import matplotlib.colors as mcolors
    n = min(lhs.shape[1], 9)
    labels = (_PARAM_LABELS_9 if n > 7 else _PARAM_LABELS_7)[:n]
    ref = np.array(FIDUCIAL_PARAMS[:n])

    mae = np.mean(np.abs(errors), axis=1)
    fin = np.isfinite(mae)
    mae = np.where(fin, mae, mae[fin].max() if fin.any() else 1.0)
    vmin, vmax = mae.min(), mae.max()
    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
        print(f"  [WARN] Cannot build colormap norm at z={z_val:.4g} — skipping triangle plot.")
        return
    norm = (mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax) if vmax / max(vmin, 1e-10) > 10
            else mcolors.Normalize(vmin=vmin, vmax=vmax))
    cmap = mpl.colormaps["YlOrRd"]

    fs = 2.2 * n
    fig, axes = plt.subplots(n, n, figsize=(fs + 1.5, fs), squeeze=False)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False); continue
            if i == j:
                counts, edges = np.histogram(lhs[:, i], bins=30)
                centres = 0.5 * (edges[:-1] + edges[1:])
                for b in range(30):
                    m = (lhs[:, i] >= edges[b]) & (lhs[:, i] < edges[b + 1])
                    if m.any():
                        ax.bar(centres[b], counts[b], width=edges[1] - edges[0],
                               color=cmap(norm(np.median(mae[m]))), linewidth=0)
                ax.set_xlim(edges[0], edges[-1]); ax.yaxis.set_visible(False)
                ax.axvline(ref[i], color="black", lw=1.4, ls="--", zorder=5)
            else:
                ax.scatter(lhs[:, j], lhs[:, i], c=mae, cmap=cmap, norm=norm,
                           s=15, linewidths=0, rasterized=True, alpha=0.8)
                ax.plot(ref[j], ref[i], marker="x", color="black", markersize=9,
                        markeredgewidth=2.0, zorder=6, linestyle="none")
            if i == n - 1:
                ax.set_xlabel(labels[j], fontsize=TICK_FS)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i], fontsize=TICK_FS)
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(axis="both", which="major", labelsize=9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = fig.colorbar(sm, cax=fig.add_axes([0.72, 0.55, 0.02, 0.35]))
    cb.set_label(fr"Mean $|\,P_{{{NL_TYPE}}}^\mathrm{{pred}}/P_{{{NL_TYPE}}}^\mathrm{{true}} - 1\,|$",
                 fontsize=TICK_FS - 1)
    cb.ax.tick_params(labelsize=9)
    fig.legend(handles=[mlines.Line2D([], [], color="black", marker="x", linestyle="none",
                                      markersize=8, markeredgewidth=2.0, label="Reference cosmology")],
               loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=TICK_FS - 1, framealpha=0.85)
    fig.suptitle(f"Emulation error vs. parameter pairs — {MODEL_TYPE.upper()} (lin emu x NL)\n"
                 f"{COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, z={z_val:.4g}, {N_TRAIN} batches",
                 fontsize=AXES_FS - 3, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.70, 1])
    _save(fig, "triangle_boost_error", z_val, dpi=150)


# ---------------------------------------------------------------------------
# Datagenerator comparison (fiducial)
# ---------------------------------------------------------------------------

def plot_datagen_comparison(redshift_indices, lin_emu, nl_emu):
    if DG_PK_PATH is None:
        print("  [INFO] DG_PK_PATH is None — skipping datagenerator comparison.")
        return
    if not os.path.exists(DG_PK_PATH):
        print(f"  [WARN] Datagenerator file not found: {DG_PK_PATH} — skipping.")
        return

    ks = GRID["ks"]
    dg_all = np.load(DG_PK_PATH)[0]                           # (52, 500) full grid
    plin = emulate_plin_on_boost_grid(lin_emu, FIDUCIAL_PARAMS)
    _, _, pk_emu_fid = nl_emu.get_pks(FIDUCIAL_PARAMS, pk_lin=plin)
    _, _, pk_base    = nl_emu.get_pks(FIDUCIAL_PARAMS, pk_lin=plin, use_approximation_only=True)

    fig, axes = plt.subplots(len(redshift_indices), 1,
                             figsize=(10, 3.2 * len(redshift_indices)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (iz, z_val) in zip(axes, redshift_indices):
        dg_z = dg_all[GRID["full_z_idx"][iz]][GRID["full_k_idx"]]
        ax.axhline(1.0, color=COLOR_DG, linestyle="--", lw=1.5, label="Datagen (reference)")
        ax.semilogx(ks, dg_z / pk_base[iz], color=COLOR_BOOST, lw=1.8,
                    label=r"Datagen / ($B_\mathrm{syren} P_\mathrm{lin}^\mathrm{emu}$)")
        ax.semilogx(ks, pk_emu_fid[iz] / dg_z, color=COLOR_EMU, lw=1.8,
                    label=f"{MODEL_TYPE.upper()} (lin emu x NL) / Datagen")
        ax.set_ylabel("P / P_datagen", fontsize=TICK_FS - 2)
        ax.set_xlim(ks[0], ks[-1]); ax.set_ylim(0.88, 1.12); ax.grid(alpha=0.25)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FS - 3)
        ax.set_title(f"z = {z_val:.4g}", fontsize=TICK_FS, loc="right")
    axes[0].legend(fontsize=LEGEND_FS - 3, loc="upper right")
    axes[-1].set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
    plt.tight_layout()
    fname = (f"{FIG_DIR}/datagen_comparison_{COSMO_TYPE}_{PRIOR_TYPE}"
             f"_{NL_TYPE}_{MODEL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}_allz.pdf")
    fig.savefig(fname, bbox_inches="tight", dpi=150); plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family']      = 'STIXGeneral'
    os.makedirs(FIG_DIR, exist_ok=True)

    setup_grids()
    redshift_indices = resolve_redshift_indices()

    print("=" * 60)
    print("[INFO] Nonlinear Emulator Evaluation (linear EMULATOR x boost emulator)")
    print(f"       nl_type / model = {NL_TYPE} / {MODEL_TYPE}, nTrain={N_TRAIN}")
    print(f"       lin model       = {LIN_MODEL_PATH}")
    print(f"       test_batch      = {START_BATCH}")
    print(f"       Ob/H0 cut       = {OMEGAB_H0_TRIANGLE_CUT}")
    print(f"       Redshifts       = {[(iz, f'z={z:.4g}') for iz, z in redshift_indices]}")
    print("=" * 60)

    lin_emu, nl_emu = load_emulators()

    print("\n[INFO] Loading test set...")
    data = load_test_set()

    print("[INFO] Computing predictions (linear emulator -> boost emulator)...")
    pred_pks, true_pks, true_frac, pred_frac, err_lin, lhs = \
        compute_predictions(data, lin_emu, nl_emu)

    ks = GRID["ks"]
    for iz, z_val in redshift_indices:
        print(f"\n{'=' * 60}\n[INFO] Evaluating iz={iz}  →  z={z_val:.4g}\n{'=' * 60}")

        errors   = pred_pks[:, iz] / true_pks[:, iz] - 1        # total
        err_nl_z = pred_frac[:, iz] / true_frac[:, iz] - 1      # boost emulator only
        err_lin_z = err_lin[:, iz]                              # linear emulator only

        errors_filt, good = apply_diagnostic_filter(errors, lhs)
        print_statistics(errors, err_nl_z, err_lin_z, good, iz, z_val)

        print(f"\n[INFO] Saving figures for z={z_val:.4g}...")
        plot_error_decomposition(errors, err_nl_z, err_lin_z, ks, z_val)
        plot_error_triangle(errors, lhs, z_val)
        plot_residual_logfracs(true_frac, lhs, ks, iz, z_val)
        plot_boost_error_lines_filtered(errors_filt, ks, good, z_val)
        plot_comparison_bands(errors, ks, z_val)
        plot_comparison_bands_zoomed(errors, ks, z_val)
        plot_max_error_histogram(errors, z_val)
        plot_frac_shape_bands(true_frac, ks, iz, z_val)
        plot_max_error_vs_tagn(errors, lhs, z_val)
        plot_worst_offenders_in_window(errors, ks, z_val)
        plot_error_heatmap_vs_tagn(errors, ks, lhs, z_val)
        plot_pc_true_vs_predicted(true_frac, pred_frac, lhs, iz, z_val)
        plot_pc_residual_vs_tagn(true_frac, pred_frac, lhs, iz, z_val)

    print("\n[INFO] Datagenerator comparison...")
    plot_datagen_comparison(redshift_indices, lin_emu, nl_emu)
    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()



# """
# evaluate_one_nl.py — Nonlinear Emulator Evaluation Script (end-to-end)

# Evaluates a trained nonlinear boost emulator against CAMB nonlinear ground
# truth, using the *linear emulator* (not CAMB linear) for P_lin:

#     P_nl^emu = frac_NL^emu * B_syren * P_lin^emu
#     error    = P_nl^emu / P_nl^CAMB - 1

# Grids
# -----
# Each emulator runs on the grid it was trained on, independent of whatever
# grid train_utils_pk_emulator_v3 currently defines:

#   * linear emulator : full grid (FULL_KS: 500 k, FULL_ZS: 52 z)
#   * boost emulator  : z taken from the PCA keys in its metadata bundle,
#                       k = FULL_KS within [BOOST_KMIN, BOOST_KMAX]
#                       (checked against the PCA length)

# The linear output and the CAMB test set are both sliced onto the boost grid.

# Error decomposition (exact):
#     1 + err_total = (1 + err_NL) * (1 + err_lin)
#     err_NL  = frac_pred / frac_true - 1     with frac_true = B_camb / B_syren
#     err_lin = P_lin^emu / P_lin^CAMB - 1
# B_syren is the same one the emulator uses (computed on the boost k grid), so
# err_NL is consistent with the network's training target.

# Usage (standalone):
#     python ./mps_emu/evaluate_one_nl.py

# Author: Victoria Lloyd (2025)
# """

# import os
# import joblib
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches
# from scipy.interpolate import interp1d

# import emulmps_w0wa as pk_emu
# import train_utils_pk_emulator_v3 as utils
# from train_utils_pk_emulator_v3 import VER


# # ---------------------------------------------------------------------------
# # Run configuration — nonlinear (boost) emulator
# # ---------------------------------------------------------------------------

# START_BATCH    = 100
# N_BATCHES      = 10
# N_TRAIN        = 100
# COSMO_TYPE     = "w0wacdm"
# NL_TYPE        = "mead2020_Tfree_mnufree"
# PRIOR_TYPE     = "expanded"
# MODEL_TYPE     = "npce"
# METADATA_DIR   = "mps_emu/metadata"

# # k range the boost emulator was trained on (cut of FULL_KS)
# BOOST_KMIN, BOOST_KMAX = 0.005, 50.0

# # ---------------------------------------------------------------------------
# # Run configuration — linear emulator (full grid)
# # ---------------------------------------------------------------------------

# LIN_NL_TYPE       = "mead2020_Tfree_mnufree_lin"
# LIN_MODEL_TYPE    = "npce"
# LIN_MODEL_PATH    = "mps_emu/models/emulator_npce_w0wacdm_expanded_mead2020_Tfree_mnufree_lin_tfreemnufree_nTrain100_v15.keras"
# LIN_METADATA_PATH = "mps_emu/metadata/metadata_w0wacdm_expanded_mead2020_Tfree_mnufree_lin_tfreemnufree_nTrain100_v15/metadata.joblib"

# # Full training grid (linear emulator and datagenerator files)
# FULL_KS = np.logspace(-5.1, 2, 500)
# FULL_ZS = np.concatenate((
#     np.linspace(0, 3, 33, endpoint=False),
#     np.linspace(3, 10, 7, endpoint=False),
#     np.linspace(10, 50, 12),
# ))

# # Target redshifts — snapped to the nearest boost-grid redshift
# TARGET_REDSHIFTS = [0, 0.5, 1, 2, 5, 10, 30, 50]

# # Prior cuts — must match training
# W0_MIN_FILTER   = None
# W0WA_MAX_FILTER = None

# OMEGAB_H0_TRIANGLE_CUT = {
#     'omegab_anchor': 0.05,
#     'h0_anchor':     75,
#     'omegab_max':    0.072,
#     'h0_max':        90,
# }

# # Diagnostic thresholds (stats / filtered-subset split only)
# W0_THRESHOLD    = -1.8
# W0WA_THRESHOLD  = -0.75
# W0_COL          = utils.params.index("w")
# W0WA_COL        = utils.params.index("w0+wa")

# FIG_DIR = "mps_emu/validation_figs/smaller_grid_linemu_fulltest"

# K_WINDOW_LO, K_WINDOW_HI = 20.0, 30.0
# TAGN_COL = utils.params_tfree_mnufree.index("T_AGN")   # column 7

# # Datagenerator NL reference, shape (1, 52, 500) on FULL grid. None to skip.
# DG_PK_PATH = None

# # Fiducial: [10^9 A_s, ns, H0, Omega_b, Omega_m, w0, w0+wa, log T_AGN, mnu]
# FIDUCIAL_PARAMS = [2.1, 0.96605, 67.32, 0.04, 0.3, -0.9, -0.9, 7.3, 0.06]

# # Filled by setup_grids(): boost grid + index maps into the full grid
# GRID = {}


# # ---------------------------------------------------------------------------
# # Plotting constants
# # ---------------------------------------------------------------------------

# COLOR_50    = "#473C8A"
# COLOR_90    = "#C45858"
# COLOR_100   = "lightgray"
# COLOR_GOOD  = "#2166ac"
# COLOR_BAD   = "#d6604d"
# COLOR_EMU   = "#1565c0"
# COLOR_BOOST = "#2ca02c"
# COLOR_DG    = "black"

# AXES_FS   = 20
# TICK_FS   = 17
# LEGEND_FS = 17

# HANDLE_95 = mlines.Line2D([], [], color=COLOR_100, label=r"$95\%$")
# HANDLE_90 = mpatches.Patch(facecolor=COLOR_90, label=r"$90\%$")
# HANDLE_50 = mpatches.Patch(facecolor=COLOR_50, hatch="\\", edgecolor="lightgray", label=r"$50\%$")

# ERR_LABEL = (fr"$P_{{{NL_TYPE}}}^\mathrm{{{MODEL_TYPE.upper()}}} / "
#              fr"P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1$")


# # ---------------------------------------------------------------------------
# # Grids
# # ---------------------------------------------------------------------------

# def _nl_n_params():
#     return 9 if "Tfree_mnufree" in NL_TYPE else 7


# def _resolve_nl_metadata_path():
#     dummy = utils.COLASet.__new__(utils.COLASet)
#     dummy.cosmo_type = COSMO_TYPE
#     dummy.prior_type = PRIOR_TYPE
#     dummy.nl_type    = NL_TYPE
#     dummy.n_batches  = N_TRAIN
#     dummy.w0_min     = W0_MIN_FILTER
#     dummy.w0wa_max   = W0WA_MAX_FILTER
#     dummy.use_boost  = True
#     dummy.lhs        = np.empty((1, _nl_n_params()))
#     return os.path.join(METADATA_DIR, f"metadata_{dummy._metadata_tag()}", "metadata.joblib")


# def setup_grids():
#     """Derive the boost grid from the boost model's own metadata bundle."""
#     path = _resolve_nl_metadata_path()
#     print(f"[INFO] Boost metadata bundle: {path}")
#     bundle = joblib.load(path)

#     # z: PCA keys are float(f"{z:.3f}") of the training redshifts
#     z_keys = np.array(sorted(bundle["pcas"].keys()), dtype=float)
#     z_idx  = np.array([np.argmin(np.abs(FULL_ZS - zk)) for zk in z_keys])
#     if np.max(np.abs(FULL_ZS[z_idx] - z_keys)) > 6e-4:
#         raise ValueError("Boost-model redshifts are not a subset of FULL_ZS.")

#     # k: fixed cut of the full grid; must match the PCA vector length
#     k_idx  = np.where((FULL_KS >= BOOST_KMIN) & (FULL_KS <= BOOST_KMAX))[0]
#     n_k_pca = next(iter(bundle["pcas"].values())).components_.shape[1]
#     if n_k_pca != len(k_idx):
#         raise ValueError(
#             f"Boost PCA has {n_k_pca} k-points but [{BOOST_KMIN}, {BOOST_KMAX}] "
#             f"selects {len(k_idx)} — adjust BOOST_KMIN / BOOST_KMAX."
#         )

#     GRID.update(
#         ks=FULL_KS[k_idx], zs=FULL_ZS[z_idx],
#         full_k_idx=k_idx, full_z_idx=z_idx,
#         pcas=bundle["pcas"], scalers=bundle["scalers"],
#     )
#     GRID["k_window"] = (GRID["ks"] >= K_WINDOW_LO) & (GRID["ks"] <= K_WINDOW_HI)
#     print(f"[INFO] Boost grid: {len(k_idx)} k in [{GRID['ks'][0]:.4g}, {GRID['ks'][-1]:.4g}], "
#           f"{len(z_idx)} z in [{GRID['zs'][0]:.3g}, {GRID['zs'][-1]:.3g}]")


# def _grid_emulator_class(ks, zs, name):
#     """PkEmulator subclass pinned to a given grid (methods read self.K_MODES / Z_MODES)."""
#     return type(name, (pk_emu.PkEmulator,),
#                 dict(K_MODES=ks, N_K_MODES=len(ks), Z_MODES=zs, N_ZS=len(zs)))


# def load_emulators():
#     LinEmu   = _grid_emulator_class(FULL_KS, FULL_ZS, "FullGridPkEmulator")
#     BoostEmu = _grid_emulator_class(GRID["ks"], GRID["zs"], "BoostGridPkEmulator")

#     print(f"[INFO] Loading linear emulator: {LIN_MODEL_PATH}")
#     lin_emu = LinEmu.from_paths(LIN_MODEL_PATH, LIN_METADATA_PATH,
#                                 nl_type=LIN_NL_TYPE, model_type=LIN_MODEL_TYPE)
#     if lin_emu.use_boost:
#         raise ValueError(f"LIN_NL_TYPE='{LIN_NL_TYPE}' is not a linear nl_type.")

#     print("[INFO] Loading boost emulator (by tag)...")
#     nl_emu = BoostEmu(cosmo_type=COSMO_TYPE, prior_type=PRIOR_TYPE, nl_type=NL_TYPE,
#                       model_type=MODEL_TYPE, num_batches=N_TRAIN,
#                       w0_min=W0_MIN_FILTER, w0wa_max=W0WA_MAX_FILTER)
#     return lin_emu, nl_emu


# def emulate_plin_on_boost_grid(lin_emu, params):
#     _, _, pk_full = lin_emu.get_pks(np.asarray(params))          # (52, 500)
#     return pk_full[np.ix_(GRID["full_z_idx"], GRID["full_k_idx"])]


# # ---------------------------------------------------------------------------
# # Redshift / tag helpers
# # ---------------------------------------------------------------------------

# def resolve_redshift_indices():
#     seen, result = set(), []
#     for target in TARGET_REDSHIFTS:
#         iz = int(np.argmin(np.abs(GRID["zs"] - target)))
#         z_val = float(GRID["zs"][iz])
#         if iz in seen:
#             print(f"  [WARN] target z={target} maps to iz={iz} (z={z_val:.4f}) "
#                   "already included — skipping.")
#             continue
#         seen.add(iz)
#         result.append((iz, z_val))
#     return result


# def _filter_tag():
#     parts = []
#     if W0_MIN_FILTER   is not None:
#         parts.append(f"w0min{W0_MIN_FILTER}")
#     if W0WA_MAX_FILTER is not None:
#         parts.append(f"w0wamax{W0WA_MAX_FILTER}")
#     parts += ["linemu", VER]
#     return "_" + "_".join(parts)


# # ---------------------------------------------------------------------------
# # Data
# # ---------------------------------------------------------------------------

# def load_test_set():
#     """Load CAMB test data at the boost redshifts, apply cuts, slice k to the boost grid."""
#     test_set = utils.COLASet(
#         target_z    = GRID["zs"],
#         cosmo_type  = COSMO_TYPE,
#         prior_type  = PRIOR_TYPE,
#         nl_type     = NL_TYPE,
#         start_batch = START_BATCH,
#         n_batches   = N_BATCHES,
#     )

#     mask = np.ones(len(test_set.lhs), dtype=bool)
#     if W0_MIN_FILTER is not None:
#         mask &= test_set.lhs[:, W0_COL] >= W0_MIN_FILTER
#     if W0WA_MAX_FILTER is not None:
#         mask &= test_set.lhs[:, W0WA_COL] <= W0WA_MAX_FILTER
#     if OMEGAB_H0_TRIANGLE_CUT is not None:
#         c = OMEGAB_H0_TRIANGLE_CUT
#         ob = test_set.lhs[:, utils.params.index("Omega_b")]
#         h0 = test_set.lhs[:, utils.params.index("h")]
#         slope = (c['h0_anchor'] - c['h0_max']) / (c['omegab_max'] - c['omegab_anchor'])
#         tri = h0 <= c['h0_max'] + slope * (ob - c['omegab_anchor'])
#         print(f"[INFO] Omega_b/H0 triangle cut: removing {(~tri & mask).sum()} cosmologies.")
#         mask &= tri
#     print(f"[INFO] Test set after prior cuts: {mask.sum()} / {len(mask)} cosmologies kept.")

#     # CAMB P_lin recovered exactly: pks_target = frac * B_syren(test) * P_lin_camb
#     plin_camb = test_set.pks_target / (test_set.frac_pks * test_set.mps_approxes_boost)

#     # k slice: boost k inside the test set's k grid (utils.ks)
#     tks = np.asarray(test_set.ks)
#     kt_idx = np.array([np.argmin(np.abs(np.log(tks) - np.log(k))) for k in GRID["ks"]])
#     if not np.allclose(tks[kt_idx], GRID["ks"], rtol=1e-10, atol=0):
#         raise ValueError("Boost k grid is not a subset of the test set's k grid (utils.ks).")

#     return dict(
#         lhs       = test_set.lhs[mask],
#         pk_nl     = test_set.pks_target[mask][..., kt_idx],    # CAMB P_nl  (N, Nz_b, Nk_b)
#         plin_camb = plin_camb[mask][..., kt_idx],              # CAMB P_lin (N, Nz_b, Nk_b)
#     )


# def compute_predictions(data, lin_emu, nl_emu):
#     """
#     Returns pred_pks, true_pks, true_frac, pred_frac, err_lin, lhs — all on the
#     boost grid.  true_frac uses the emulator's own B_syren (boost k grid).
#     """
#     pred_l, frac_l, plin_l = [], [], []
#     for row in data["lhs"]:
#         plin_emu = emulate_plin_on_boost_grid(lin_emu, row)
#         _, _, pk     = nl_emu.get_pks(row, pk_lin=plin_emu)   # frac * B_syren * P_lin^emu
#         _, _, fpred  = nl_emu.get_boost(row)                   # frac (network output)
#         pred_l.append(pk); frac_l.append(fpred); plin_l.append(plin_emu)

#     pred_pks  = np.asarray(pred_l, dtype=np.float64)
#     pred_frac = np.asarray(frac_l, dtype=np.float64)
#     plin_emu  = np.asarray(plin_l, dtype=np.float64)

#     b_syren   = pred_pks / (pred_frac * plin_emu)          # exactly the emulator's B_syren
#     b_camb    = data["pk_nl"] / data["plin_camb"]
#     true_frac = b_camb / b_syren
#     err_lin   = plin_emu / data["plin_camb"] - 1

#     valid = np.isfinite(pred_pks).all(axis=(1, 2)) & np.isfinite(true_frac).all(axis=(1, 2))
#     if (~valid).any():
#         print(f"  WARNING: dropping {(~valid).sum()} cosmologies with non-finite output.")
#     return (pred_pks[valid], data["pk_nl"][valid], true_frac[valid],
#             pred_frac[valid], err_lin[valid], data["lhs"][valid])


# def apply_diagnostic_filter(errors, lhs):
#     good = (lhs[:, W0_COL] > W0_THRESHOLD) & (lhs[:, W0WA_COL] < W0WA_THRESHOLD)
#     return errors[good], good


# # ---------------------------------------------------------------------------
# # Statistics
# # ---------------------------------------------------------------------------

# def percentiles(err):
#     ps = np.percentile(err, [5, 10, 25, 75, 90, 95], axis=0)
#     return ps[0], ps[1], ps[2], ps[3], ps[4], ps[5]


# def _max_err_summary(label, e):
#     m = np.max(np.abs(e), axis=1)
#     print(f"    {label:<22s} mean {np.mean(m):.6f} | median {np.median(m):.6f} | "
#           f"95% {np.percentile(m, 95):.6f} | 99% {np.percentile(m, 99):.6f}")


# def print_statistics(err_total, err_nl, err_lin, good, iz, z_val):
#     print(f"\n=== ERROR STATISTICS  (iz={iz}, z={z_val:.4f}) — max |error| over k ===")
#     for label, m in [("Full dataset", np.ones(len(err_total), bool)),
#                      (f"w0>{W0_THRESHOLD} & w0+wa<{W0WA_THRESHOLD}", good)]:
#         print(f"\n  {label}  (N={m.sum()}):")
#         if m.sum() == 0:
#             continue
#         _max_err_summary("total (lin emu x NL)", err_total[m])
#         _max_err_summary("NL emulator only",     err_nl[m])
#         _max_err_summary("linear emulator only", err_lin[m])


# # ---------------------------------------------------------------------------
# # Plot helpers
# # ---------------------------------------------------------------------------

# def _to_grid(src_ks, src_Pk, tgt_ks):
#     mask   = (tgt_ks >= src_ks[0]) & (tgt_ks <= src_ks[-1])
#     result = np.full_like(tgt_ks, np.nan, dtype=float)
#     result[mask] = interp1d(src_ks, src_Pk)(tgt_ks[mask])
#     return result


# def _fill_ax(ax, ks, p5, p10, p25, p75, p90, p95, ylabel, label_text):
#     ax.semilogx(ks, p5,  color=COLOR_100)
#     ax.semilogx(ks, p95, color=COLOR_100)
#     ax.fill_between(ks, p10, p90, color=COLOR_90, alpha=0.7)
#     ax.fill_between(ks, p25, p75, color=COLOR_50, hatch="\\", edgecolor="lightgray", alpha=0.7)
#     ax.set_ylabel(ylabel, fontsize=AXES_FS)
#     ax.set_xscale("log")
#     ax.grid(alpha=0.4)
#     ax.set_xlim([ks[0], ks[-1]])
#     ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
#     ax.text(0.05, 0.95, label_text, transform=ax.transAxes, ha='left', va='top', fontsize=AXES_FS)


# def _save(fig, stem, z_val, dpi=None):
#     fname = (f"{FIG_DIR}/{stem}_z{z_val:.4g}_{COSMO_TYPE}_{PRIOR_TYPE}"
#              f"_{NL_TYPE}_{MODEL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}.pdf")
#     fig.savefig(fname, bbox_inches="tight", dpi=dpi)
#     plt.close(fig)
#     print(f"  Saved: {fname}")


# # ---------------------------------------------------------------------------
# # Plots
# # ---------------------------------------------------------------------------

# def plot_error_decomposition(err_total, err_nl, err_lin, ks, z_val):
#     fig, ax = plt.subplots(figsize=(9, 5))
#     for e, color, label in [(err_lin, "C0", "linear emulator"),
#                             (err_nl, "C2", "NL emulator (boost)"),
#                             (err_total, "C3", "total")]:
#         lo, med, hi = np.percentile(e, [5, 50, 95], axis=0)
#         ax.fill_between(ks, lo, hi, color=color, alpha=0.2)
#         ax.semilogx(ks, med, color=color, lw=1.8, label=f"{label} (median, 5–95%)")
#     ax.axhline(0, color="black", lw=0.8, ls="--")
#     ax.set_xscale("log"); ax.set_xlim(ks[0], ks[-1])
#     ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax.set_ylabel(r"$P^\mathrm{emu}/P^\mathrm{CAMB} - 1$", fontsize=AXES_FS)
#     ax.set_title(f"Error decomposition, z={z_val:.4g}", fontsize=AXES_FS - 4)
#     ax.legend(fontsize=LEGEND_FS - 5); ax.grid(alpha=0.3)
#     ax.tick_params(axis='both', which='major', labelsize=TICK_FS - 2)
#     plt.tight_layout()
#     _save(fig, "error_decomposition", z_val)


# def plot_boost_error_lines_filtered(errors_filt, ks, good, z_val):
#     fig, ax = plt.subplots(figsize=(8, 5))
#     for i in range(min(good.sum(), 200)):
#         ax.semilogx(ks, errors_filt[i])
#     ax.axhline(0, color="black", lw=0.8, ls="--")
#     ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax.set_ylabel(ERR_LABEL, fontsize=AXES_FS)
#     ax.grid(alpha=0.3); ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
#     ax.text(0.05, 0.95,
#             fr'$z={z_val:.4g}$,  $w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$  (N={good.sum()})',
#             transform=ax.transAxes, ha='left', va='top', fontsize=LEGEND_FS)
#     _save(fig, f"val_boost_errors_w0gt{W0_THRESHOLD}_w0walt{W0WA_THRESHOLD}", z_val)


# def plot_comparison_bands(errors, ks, z_val):
#     fig, ax = plt.subplots(figsize=(8, 5))
#     _fill_ax(ax, ks, *percentiles(errors), ylabel=ERR_LABEL, label_text=f'$z={z_val:.4g}$')
#     ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95], fontsize=LEGEND_FS, loc="lower right")
#     plt.tight_layout()
#     _save(fig, "boost_errors", z_val)


# def plot_comparison_bands_zoomed(errors, ks, z_val, k_lo=10.0, k_hi=50.0):
#     m = (ks >= k_lo) & (ks <= k_hi)
#     fig, ax = plt.subplots(figsize=(8, 5))
#     _fill_ax(ax, ks[m], *percentiles(errors[:, m]), ylabel=ERR_LABEL,
#              label_text=f'$z={z_val:.4g}$  (k zoom {k_lo:g}-{k_hi:g})')
#     ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="purple", alpha=0.08)
#     ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95], fontsize=LEGEND_FS, loc="best")
#     plt.tight_layout()
#     _save(fig, "boost_errors_zoom", z_val)


# def plot_max_error_histogram(errors, z_val):
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.hist(np.max(np.abs(errors), axis=1), bins=50, color='#1565c0', alpha=0.7, edgecolor='black')
#     ax.set_xlabel(fr"Max $|P_{{{NL_TYPE}}}^\mathrm{{{MODEL_TYPE.upper()}}} / "
#                   fr"P_{{{NL_TYPE}}}^\mathrm{{CAMB}} - 1|$", fontsize=16)
#     ax.set_ylabel('Count', fontsize=16)
#     ax.set_title(f'{MODEL_TYPE.upper()} — {COSMO_TYPE}, {NL_TYPE}   ($z={z_val:.4g}$)', fontsize=18)
#     ax.grid(alpha=0.3); ax.tick_params(labelsize=14)
#     plt.tight_layout()
#     _save(fig, "max_boost_error_histogram", z_val)


# def plot_residual_logfracs(true_frac, lhs, ks, iz, z_val):
#     """log(B_camb / B_syren) per cosmology — the boost emulator's training target."""
#     good = (lhs[:, W0_COL] > W0_THRESHOLD) & (lhs[:, W0WA_COL] < W0WA_THRESHOLD)
#     lf = np.log(true_frac[:, iz, :])
#     fig, ax = plt.subplots(figsize=(9, 5))
#     for idx in np.where(good)[0]:
#         ax.semilogx(ks, lf[idx], color=COLOR_GOOD, alpha=0.08, lw=0.5, rasterized=True)
#     for idx in np.where(~good)[0]:
#         ax.semilogx(ks, lf[idx], color=COLOR_BAD, alpha=0.08, lw=0.5, rasterized=True)
#     if good.sum():
#         ax.semilogx(ks, np.median(lf[good], axis=0), color=COLOR_GOOD, lw=2.2,
#                     label=fr"$w_0>{W0_THRESHOLD}$ & $w_0+w_a<{W0WA_THRESHOLD}$ (n={good.sum()})")
#     if (~good).sum():
#         ax.semilogx(ks, np.median(lf[~good], axis=0), color=COLOR_BAD, lw=2.2,
#                     label=fr"fails cut (n={(~good).sum()})")
#     ax.axhline(0, color="k", lw=0.8, ls="--")
#     ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax.set_ylabel(r"$\log\!\left(B^\mathrm{CAMB} / B^\mathrm{syren}\right)$", fontsize=AXES_FS)
#     ax.set_title(fr"$z={z_val:.4g}$   |   {COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, {MODEL_TYPE.upper()}",
#                  fontsize=15)
#     ax.legend(fontsize=14); ax.grid(alpha=0.3)
#     ax.tick_params(axis='both', which='major', labelsize=TICK_FS)
#     plt.tight_layout()
#     _save(fig, "residual_logfracs", z_val, dpi=150)


# def plot_frac_shape_bands(true_frac, ks, iz, z_val):
#     fig, ax = plt.subplots(figsize=(8, 5))
#     _fill_ax(ax, ks, *percentiles(true_frac[:, iz, :] - 1),
#              ylabel=r"$B^\mathrm{CAMB} / B^\mathrm{syren} - 1$",
#              label_text=f'$z={z_val:.4g}$  (CAMB target)')
#     ax.axhline(0, color="black", lw=0.8, ls="--")
#     ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax.legend(handles=[HANDLE_50, HANDLE_90, HANDLE_95], fontsize=LEGEND_FS, loc="upper left")
#     plt.tight_layout()
#     _save(fig, "camb_residual_frac", z_val)


# def plot_max_error_vs_tagn(errors, lhs, z_val):
#     if lhs.shape[1] <= TAGN_COL:
#         print("  [WARN] T_AGN column not present — skipping plot_max_error_vs_tagn.")
#         return
#     win  = np.max(np.abs(errors[:, GRID["k_window"]]), axis=1)
#     full = np.max(np.abs(errors), axis=1)
#     win  = np.clip(win,  max(win[win > 0].min() * 0.5, 1e-8)  if np.any(win > 0)  else 1e-8, None)
#     full = np.clip(full, max(full[full > 0].min() * 0.5, 1e-8) if np.any(full > 0) else 1e-8, None)

#     fig, ax = plt.subplots(figsize=(8, 5))
#     sc = ax.scatter(lhs[:, TAGN_COL], win, c=full, cmap="YlOrRd",
#                     norm=mpl.colors.LogNorm(vmin=full.min(), vmax=full.max()),
#                     s=12, alpha=0.7, rasterized=True)
#     ax.set_yscale("log")
#     ax.set_xlabel(r"$\log T_\mathrm{AGN}$", fontsize=AXES_FS)
#     ax.set_ylabel(fr"max $|$error$|$ in $k \in [{K_WINDOW_LO:g},{K_WINDOW_HI:g}]$", fontsize=AXES_FS - 4)
#     ax.set_title(f"z={z_val:.4g}  (colour = max |error| over full k range)", fontsize=AXES_FS - 5)
#     ax.grid(alpha=0.3, which="both")
#     fig.colorbar(sc, ax=ax).set_label("max |error| (full k)", fontsize=TICK_FS - 3)
#     plt.tight_layout()
#     _save(fig, "max_error_vs_tagn", z_val, dpi=150)


# def plot_worst_offenders_in_window(errors, ks, z_val, n_worst=30):
#     worst = np.argsort(np.max(np.abs(errors[:, GRID["k_window"]]), axis=1))[::-1][:n_worst]
#     fig, ax = plt.subplots(figsize=(8, 5))
#     for idx in worst:
#         ax.semilogx(ks, errors[idx], alpha=0.6, lw=1)
#     ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="purple", alpha=0.08)
#     ax.axhline(0, color="black", lw=0.8, ls="--")
#     ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax.set_ylabel(ERR_LABEL, fontsize=AXES_FS - 4)
#     ax.set_title(f"Worst {n_worst} by max|error| in k={K_WINDOW_LO:g}-{K_WINDOW_HI:g} (z={z_val:.4g})",
#                  fontsize=AXES_FS - 6)
#     ax.grid(alpha=0.3)
#     plt.tight_layout()
#     _save(fig, "worst_offenders_window", z_val)


# def plot_error_heatmap_vs_tagn(errors, ks, lhs, z_val, n_bins=10):
#     if lhs.shape[1] <= TAGN_COL:
#         print("  [WARN] T_AGN column not present — skipping plot_error_heatmap_vs_tagn.")
#         return
#     t = lhs[:, TAGN_COL]
#     edges = np.linspace(t.min(), t.max(), n_bins + 1)
#     bidx  = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
#     heat  = np.full((n_bins, len(ks)), np.nan)
#     counts = np.zeros(n_bins, dtype=int)
#     for b in range(n_bins):
#         m = bidx == b
#         counts[b] = m.sum()
#         if counts[b]:
#             heat[b] = np.median(np.abs(errors[m]), axis=0)
#     fv = heat[np.isfinite(heat) & (heat > 0)]
#     if fv.size == 0:
#         print(f"  [WARN] No finite positive heatmap values at z={z_val:.4g} — skipping.")
#         return
#     centres = 0.5 * (edges[:-1] + edges[1:])
#     fig, ax = plt.subplots(figsize=(9, 5))
#     im = ax.pcolormesh(ks, centres, heat, shading="auto", cmap="YlOrRd",
#                        norm=mpl.colors.LogNorm(vmin=fv.min(), vmax=fv.max()))
#     ax.set_xscale("log")
#     ax.axvspan(K_WINDOW_LO, K_WINDOW_HI, color="cyan", alpha=0.15)
#     ax.set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     ax.set_ylabel(r"$\log T_\mathrm{AGN}$", fontsize=AXES_FS)
#     ax.set_title(f"Median |error| vs (k, T_AGN)  (z={z_val:.4g})", fontsize=AXES_FS - 5)
#     fig.colorbar(im, ax=ax).set_label("median |error|", fontsize=TICK_FS - 3)
#     for b, c in enumerate(counts):
#         ax.text(ks[-1] * 1.02, centres[b], f"n={c}", fontsize=7, va="center", clip_on=False)
#     plt.tight_layout()
#     _save(fig, "error_heatmap_vs_tagn", z_val, dpi=150)


# # ---------------------------------------------------------------------------
# # PC-level diagnostics (boost emulator only)
# # ---------------------------------------------------------------------------

# def _pca_scaler(z_val):
#     z_key = float(f"{z_val:.3f}")
#     return GRID["pcas"][z_key], GRID["scalers"][z_key]


# def _pc_responsibility(z_val):
#     pca, scaler = _pca_scaler(z_val)
#     scale = scaler.scale_ if hasattr(scaler, "scale_") else scaler.std
#     mat = pca.components_ * scale[None, :]
#     return np.sum(mat[:, GRID["k_window"]] ** 2, axis=1) / np.sum(mat ** 2, axis=1)


# def _frac_to_pcs(frac_z, z_val):
#     pca, scaler = _pca_scaler(z_val)
#     return pca.transform(scaler.transform(np.log(frac_z)))


# def plot_pc_true_vs_predicted(true_frac, pred_frac, lhs, iz, z_val, top_n=3):
#     pt, pp = _frac_to_pcs(true_frac[:, iz], z_val), _frac_to_pcs(pred_frac[:, iz], z_val)
#     resp = _pc_responsibility(z_val)
#     has_tagn = lhs.shape[1] > TAGN_COL
#     fig, axes = plt.subplots(1, top_n, figsize=(5.5 * top_n, 5), squeeze=False)
#     for ax, p in zip(axes[0], np.argsort(resp)[::-1][:top_n]):
#         x, y = pt[:, p], pp[:, p]
#         lims = [min(x.min(), y.min()), max(x.max(), y.max())]
#         ax.plot(lims, lims, "k--", lw=1, alpha=0.6, zorder=1)
#         if has_tagn:
#             sc = ax.scatter(x, y, c=lhs[:, TAGN_COL], cmap="viridis", s=10, alpha=0.6, rasterized=True)
#             fig.colorbar(sc, ax=ax, label=r"$\log T_\mathrm{AGN}$")
#         else:
#             ax.scatter(x, y, s=10, alpha=0.5, rasterized=True, color="C0")
#         ax.set_xlabel("true PC coefficient"); ax.set_ylabel("predicted PC coefficient")
#         ax.set_title(f"z={z_val:.2f}, PC {p}\nresp={resp[p]:.2f}, "
#                      f"RMSE={np.sqrt(np.mean((x - y) ** 2)):.3g}, r={np.corrcoef(x, y)[0, 1]:.4f}",
#                      fontsize=AXES_FS - 6)
#         ax.grid(alpha=0.3)
#     fig.tight_layout()
#     _save(fig, "pc_true_vs_pred", z_val, dpi=150)


# def plot_pc_residual_vs_tagn(true_frac, pred_frac, lhs, iz, z_val, top_n=3):
#     if lhs.shape[1] <= TAGN_COL:
#         print("  [WARN] T_AGN column not present — skipping plot_pc_residual_vs_tagn.")
#         return
#     pt, pp = _frac_to_pcs(true_frac[:, iz], z_val), _frac_to_pcs(pred_frac[:, iz], z_val)
#     resp = _pc_responsibility(z_val)
#     fig, axes = plt.subplots(1, top_n, figsize=(5.5 * top_n, 4), squeeze=False)
#     for ax, p in zip(axes[0], np.argsort(resp)[::-1][:top_n]):
#         ax.axhline(0, color="gray", ls=":", lw=1)
#         ax.scatter(lhs[:, TAGN_COL], pp[:, p] - pt[:, p], s=8, alpha=0.4, rasterized=True, color="C2")
#         ax.set_xlabel(r"$\log T_\mathrm{AGN}$"); ax.set_ylabel("predicted - true")
#         ax.set_title(f"z={z_val:.2f}, PC {p} (resp={resp[p]:.2f})", fontsize=AXES_FS - 6)
#         ax.grid(alpha=0.3)
#     fig.tight_layout()
#     _save(fig, "pc_residual_vs_tagn", z_val, dpi=150)


# # ---------------------------------------------------------------------------
# # Triangle plot
# # ---------------------------------------------------------------------------

# _PARAM_LABELS_7 = [r'$10^9 A_s$', r'$n_s$', r'$H_0$', r'$\Omega_b$',
#                    r'$\Omega_m$', r"$w_0$", r"$w_0+w_a$"]
# # Column order follows utils.params_tfree_mnufree: T_AGN (col 7), then mnu (col 8)
# _PARAM_LABELS_9 = _PARAM_LABELS_7 + [r'$\log T_{\rm AGN}$', r'$m_\nu$']


# # ---------------------------------------------------------------------------
# # Training-prior outline (hypersphere in datagen parameters)
# # ---------------------------------------------------------------------------
# # Datagen parameter: (center, radius).  The prior is the ball
# #     sum_p ((x_p - center_p) / radius_p)^2 <= PRIOR_RADIUS_SCALE^2
# # Plotting only: nothing here touches data, normalisation or predictions.
# PRIOR_SPHERE = {
#     "lnAs":  (2.605, 0.995),
#     "ns":    (1.0,   0.3),
#     "H0":    (70.0,  15.0),
#     "ob_h2": (0.021, 0.019),
#     "oc_h2": (0.135, 0.105),
#     "w0":    (-0.85, 1.35),
#     "wa":    (-1.5,  2.5),
#     "Tagn":  (7.25,  0.75),
# }
# PRIOR_RADIUS_SCALE = 1.0      # set to e.g. 1.3 to draw an enlarged sphere
# PRIOR_MNU          = 0.06     # mnu used in the ob_h2/oc_h2 -> Om conversion
# DATAGEN_ORDER      = ["lnAs", "ns", "H0", "ob_h2", "oc_h2", "w0", "wa", "Tagn"]

# # Datagen parameters each ML lhs column depends on
# # ML lhs: [As_1e9, ns, H0, Ob, Om, w0, w0+wa, T_AGN, mnu]
# _ML_DEPS = [
#     {"lnAs"}, {"ns"}, {"H0"}, {"ob_h2", "H0"}, {"ob_h2", "oc_h2", "H0"},
#     {"w0"}, {"w0", "wa"}, {"Tagn"}, set(),          # mnu: not part of the sphere
# ]


# def _datagen_to_ml_lhs(dg):
#     """
#     Same conversion as datagen_to_ml, followed by the COLASet convention of
#     storing w0+wa in column 6.
#     dg: (N, 8) in DATAGEN_ORDER -> (N, 8) [As_1e9, ns, H0, Ob, Om, w0, w0+wa, Tagn]
#     """
#     lnAs, ns, H0, ob_h2, oc_h2, w0, wa, tagn = dg.T
#     h = H0 / 100.0
#     Ob = ob_h2 / h**2
#     mnu_contrib = (PRIOR_MNU * (3.046 / 3) ** 0.75) / 94.0708
#     Om = (oc_h2 + ob_h2 + mnu_contrib) / h**2
#     As_1e9 = np.exp(lnAs) / 10.0
#     return np.column_stack((As_1e9, ns, H0, Ob, Om, w0, w0 + wa, tagn))


# def _prior_surface_samples(cols, n=None, seed=0):
#     """
#     Points on the sphere surface restricted to the datagen params that the ML
#     columns `cols` depend on (other params at their centre; they don't affect
#     these columns), converted to ML parameters.
#     """
#     deps = sorted(set().union(*[_ML_DEPS[c] for c in cols]), key=DATAGEN_ORDER.index)
#     if n is None:   # more samples in higher dimensions so the outline stays smooth
#         n = min(50_000 * 4 ** max(len(deps) - 2, 0), 1_000_000)
#     rng = np.random.default_rng(seed)
#     u = rng.normal(size=(n, len(deps)))
#     u /= np.linalg.norm(u, axis=1, keepdims=True)
#     centre = np.array([PRIOR_SPHERE[p][0] for p in DATAGEN_ORDER])
#     dg = np.tile(centre, (n, 1))
#     for k, p in enumerate(deps):
#         dg[:, DATAGEN_ORDER.index(p)] += PRIOR_RADIUS_SCALE * PRIOR_SPHERE[p][1] * u[:, k]
#     return _datagen_to_ml_lhs(dg), _datagen_to_ml_lhs(centre[None, :])[0]


# _PRIOR_CACHE = {}


# def _prior_outline_2d(i, j, n_bins=720):
#     """
#     Outline of the prior projected onto ML columns (i -> x, j -> y): for each
#     polar angle around the projected centre, the outermost projected point.
#     Exact ellipse when both columns are linear in the datagen params; the
#     polar-max outline otherwise (projections here are star-shaped).
#     """
#     key = ("2d", i, j)
#     if key in _PRIOR_CACHE:
#         return _PRIOR_CACHE[key]
#     if not _ML_DEPS[i] or not _ML_DEPS[j]:
#         _PRIOR_CACHE[key] = None
#         return None
#     ml, c = _prior_surface_samples([i, j])
#     x, y = ml[:, i], ml[:, j]
#     dx = (x - c[i]) / np.ptp(x)
#     dy = (y - c[j]) / np.ptp(y)
#     ang = np.arctan2(dy, dx)
#     r   = np.hypot(dx, dy)
#     bins = ((ang + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
#     order = np.argsort(r)                      # later (larger r) writes win
#     best = np.full(n_bins, -1)
#     best[bins[order]] = order
#     best = best[best >= 0]
#     xo = np.append(x[best], x[best[0]])
#     yo = np.append(y[best], y[best[0]])
#     _PRIOR_CACHE[key] = (xo, yo)
#     return xo, yo


# def _prior_extent_1d(i):
#     key = ("1d", i)
#     if key not in _PRIOR_CACHE:
#         if not _ML_DEPS[i]:
#             _PRIOR_CACHE[key] = None
#         else:
#             ml, _ = _prior_surface_samples([i])
#             _PRIOR_CACHE[key] = (ml[:, i].min(), ml[:, i].max())
#     return _PRIOR_CACHE[key]


# PRIOR_LINE_KW = dict(color="0.25", ls="--", lw=0.8, zorder=4)


# def plot_error_triangle(errors, lhs, z_val):
#     import matplotlib.colors as mcolors
#     n = min(lhs.shape[1], 9)
#     labels = (_PARAM_LABELS_9 if n > 7 else _PARAM_LABELS_7)[:n]
#     ref = np.array(FIDUCIAL_PARAMS[:n])

#     mae = np.mean(np.abs(errors), axis=1)
#     fin = np.isfinite(mae)
#     mae = np.where(fin, mae, mae[fin].max() if fin.any() else 1.0)
#     vmin, vmax = mae.min(), mae.max()
#     if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmin == vmax:
#         print(f"  [WARN] Cannot build colormap norm at z={z_val:.4g} — skipping triangle plot.")
#         return
#     norm = (mcolors.LogNorm(vmin=max(vmin, 1e-6), vmax=vmax) if vmax / max(vmin, 1e-10) > 10
#             else mcolors.Normalize(vmin=vmin, vmax=vmax))
#     cmap = mpl.colormaps["YlOrRd"]

#     fs = 2.2 * n
#     fig, axes = plt.subplots(n, n, figsize=(fs + 1.5, fs), squeeze=False)
#     for i in range(n):
#         for j in range(n):
#             ax = axes[i, j]
#             if j > i:
#                 ax.set_visible(False); continue
#             if i == j:
#                 counts, edges = np.histogram(lhs[:, i], bins=30)
#                 centres = 0.5 * (edges[:-1] + edges[1:])
#                 for b in range(30):
#                     m = (lhs[:, i] >= edges[b]) & (lhs[:, i] < edges[b + 1])
#                     if m.any():
#                         ax.bar(centres[b], counts[b], width=edges[1] - edges[0],
#                                color=cmap(norm(np.median(mae[m]))), linewidth=0)
#                 ax.yaxis.set_visible(False)
#                 ax.axvline(ref[i], color="black", lw=1.4, ls="--", zorder=5)
#                 lo_hi = _prior_extent_1d(i)
#                 if lo_hi is not None:
#                     for v in lo_hi:
#                         ax.axvline(v, **PRIOR_LINE_KW)
#                     pad = 0.03 * (max(edges[-1], lo_hi[1]) - min(edges[0], lo_hi[0]))
#                     ax.set_xlim(min(edges[0], lo_hi[0]) - pad, max(edges[-1], lo_hi[1]) + pad)
#                 else:
#                     ax.set_xlim(edges[0], edges[-1])
#             else:
#                 ax.scatter(lhs[:, j], lhs[:, i], c=mae, cmap=cmap, norm=norm,
#                            s=15, linewidths=0, rasterized=True, alpha=0.8)
#                 ax.plot(ref[j], ref[i], marker="x", color="black", markersize=9,
#                         markeredgewidth=2.0, zorder=6, linestyle="none")
#                 outline = _prior_outline_2d(j, i)          # x = column j, y = column i
#                 if outline is not None:
#                     ax.plot(*outline, **PRIOR_LINE_KW)
#             if i == n - 1:
#                 ax.set_xlabel(labels[j], fontsize=TICK_FS)
#             else:
#                 ax.tick_params(labelbottom=False)
#             if j == 0 and i != 0:
#                 ax.set_ylabel(labels[i], fontsize=TICK_FS)
#             else:
#                 ax.tick_params(labelleft=False)
#             ax.tick_params(axis="both", which="major", labelsize=9)

#     sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
#     cb = fig.colorbar(sm, cax=fig.add_axes([0.72, 0.55, 0.02, 0.35]))
#     cb.set_label(fr"Mean $|\,P_{{{NL_TYPE}}}^\mathrm{{pred}}/P_{{{NL_TYPE}}}^\mathrm{{true}} - 1\,|$",
#                  fontsize=TICK_FS - 1)
#     cb.ax.tick_params(labelsize=9)
#     fig.legend(handles=[mlines.Line2D([], [], color="black", marker="x", linestyle="none",
#                                       markersize=8, markeredgewidth=2.0, label="Reference cosmology"),
#                         mlines.Line2D([], [], label=f"Training prior (R={PRIOR_RADIUS_SCALE:g})",
#                                       **{k: v for k, v in PRIOR_LINE_KW.items() if k != "zorder"})],
#                loc="upper right", bbox_to_anchor=(0.995, 0.995), fontsize=TICK_FS - 1, framealpha=0.85)
#     fig.suptitle(f"Emulation error vs. parameter pairs — {MODEL_TYPE.upper()} (lin emu x NL)\n"
#                  f"{COSMO_TYPE}, {PRIOR_TYPE}, {NL_TYPE}, z={z_val:.4g}, {N_TRAIN} batches",
#                  fontsize=AXES_FS - 3, y=1.01)
#     plt.tight_layout(rect=[0, 0, 0.70, 1])
#     _save(fig, "triangle_boost_error", z_val, dpi=150)


# # ---------------------------------------------------------------------------
# # Datagenerator comparison (fiducial)
# # ---------------------------------------------------------------------------

# def plot_datagen_comparison(redshift_indices, lin_emu, nl_emu):
#     if DG_PK_PATH is None:
#         print("  [INFO] DG_PK_PATH is None — skipping datagenerator comparison.")
#         return
#     if not os.path.exists(DG_PK_PATH):
#         print(f"  [WARN] Datagenerator file not found: {DG_PK_PATH} — skipping.")
#         return

#     ks = GRID["ks"]
#     dg_all = np.load(DG_PK_PATH)[0]                           # (52, 500) full grid
#     plin = emulate_plin_on_boost_grid(lin_emu, FIDUCIAL_PARAMS)
#     _, _, pk_emu_fid = nl_emu.get_pks(FIDUCIAL_PARAMS, pk_lin=plin)
#     _, _, pk_base    = nl_emu.get_pks(FIDUCIAL_PARAMS, pk_lin=plin, use_approximation_only=True)

#     fig, axes = plt.subplots(len(redshift_indices), 1,
#                              figsize=(10, 3.2 * len(redshift_indices)), sharex=True)
#     axes = np.atleast_1d(axes)
#     for ax, (iz, z_val) in zip(axes, redshift_indices):
#         dg_z = dg_all[GRID["full_z_idx"][iz]][GRID["full_k_idx"]]
#         ax.axhline(1.0, color=COLOR_DG, linestyle="--", lw=1.5, label="Datagen (reference)")
#         ax.semilogx(ks, dg_z / pk_base[iz], color=COLOR_BOOST, lw=1.8,
#                     label=r"Datagen / ($B_\mathrm{syren} P_\mathrm{lin}^\mathrm{emu}$)")
#         ax.semilogx(ks, pk_emu_fid[iz] / dg_z, color=COLOR_EMU, lw=1.8,
#                     label=f"{MODEL_TYPE.upper()} (lin emu x NL) / Datagen")
#         ax.set_ylabel("P / P_datagen", fontsize=TICK_FS - 2)
#         ax.set_xlim(ks[0], ks[-1]); ax.set_ylim(0.88, 1.12); ax.grid(alpha=0.25)
#         ax.tick_params(axis='both', which='major', labelsize=TICK_FS - 3)
#         ax.set_title(f"z = {z_val:.4g}", fontsize=TICK_FS, loc="right")
#     axes[0].legend(fontsize=LEGEND_FS - 3, loc="upper right")
#     axes[-1].set_xlabel(r"$k \; [1/\mathrm{Mpc}]$", fontsize=AXES_FS)
#     plt.tight_layout()
#     fname = (f"{FIG_DIR}/datagen_comparison_{COSMO_TYPE}_{PRIOR_TYPE}"
#              f"_{NL_TYPE}_{MODEL_TYPE}{_filter_tag()}_nTrain{N_TRAIN}_allz.pdf")
#     fig.savefig(fname, bbox_inches="tight", dpi=150); plt.close(fig)
#     print(f"  Saved: {fname}")


# # ---------------------------------------------------------------------------
# # Main
# # ---------------------------------------------------------------------------

# def main():
#     mpl.rcParams['mathtext.fontset'] = 'stix'
#     mpl.rcParams['font.family']      = 'STIXGeneral'
#     os.makedirs(FIG_DIR, exist_ok=True)

#     setup_grids()
#     redshift_indices = resolve_redshift_indices()

#     print("=" * 60)
#     print("[INFO] Nonlinear Emulator Evaluation (linear EMULATOR x boost emulator)")
#     print(f"       nl_type / model = {NL_TYPE} / {MODEL_TYPE}, nTrain={N_TRAIN}")
#     print(f"       lin model       = {LIN_MODEL_PATH}")
#     print(f"       test_batch      = {START_BATCH}")
#     print(f"       Ob/H0 cut       = {OMEGAB_H0_TRIANGLE_CUT}")
#     print(f"       Redshifts       = {[(iz, f'z={z:.4g}') for iz, z in redshift_indices]}")
#     print("=" * 60)

#     lin_emu, nl_emu = load_emulators()

#     print("\n[INFO] Loading test set...")
#     data = load_test_set()

#     print("[INFO] Computing predictions (linear emulator -> boost emulator)...")
#     pred_pks, true_pks, true_frac, pred_frac, err_lin, lhs = \
#         compute_predictions(data, lin_emu, nl_emu)

#     ks = GRID["ks"]
#     for iz, z_val in redshift_indices:
#         print(f"\n{'=' * 60}\n[INFO] Evaluating iz={iz}  →  z={z_val:.4g}\n{'=' * 60}")

#         errors   = pred_pks[:, iz] / true_pks[:, iz] - 1        # total
#         err_nl_z = pred_frac[:, iz] / true_frac[:, iz] - 1      # boost emulator only
#         err_lin_z = err_lin[:, iz]                              # linear emulator only

#         errors_filt, good = apply_diagnostic_filter(errors, lhs)
#         print_statistics(errors, err_nl_z, err_lin_z, good, iz, z_val)

#         print(f"\n[INFO] Saving figures for z={z_val:.4g}...")
#         plot_error_decomposition(errors, err_nl_z, err_lin_z, ks, z_val)
#         plot_error_triangle(errors, lhs, z_val)
#         plot_residual_logfracs(true_frac, lhs, ks, iz, z_val)
#         plot_boost_error_lines_filtered(errors_filt, ks, good, z_val)
#         plot_comparison_bands(errors, ks, z_val)
#         plot_comparison_bands_zoomed(errors, ks, z_val)
#         plot_max_error_histogram(errors, z_val)
#         plot_frac_shape_bands(true_frac, ks, iz, z_val)
#         plot_max_error_vs_tagn(errors, lhs, z_val)
#         plot_worst_offenders_in_window(errors, ks, z_val)
#         plot_error_heatmap_vs_tagn(errors, ks, lhs, z_val)
#         plot_pc_true_vs_predicted(true_frac, pred_frac, lhs, iz, z_val)
#         plot_pc_residual_vs_tagn(true_frac, pred_frac, lhs, iz, z_val)

#     print("\n[INFO] Datagenerator comparison...")
#     plot_datagen_comparison(redshift_indices, lin_emu, nl_emu)
#     print("\n[INFO] Done.")


# if __name__ == "__main__":
#     main()