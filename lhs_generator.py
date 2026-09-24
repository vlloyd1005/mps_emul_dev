"""
lhs_generator.py — Latin Hypercube Sampling for Cosmological Parameter Batches

Generates batches of cosmological parameter vectors used to train and test the
MPS emulator.  Each batch is saved in two conventions:

  - ML convention    [As_1e9, ns, H0, Ob, Om, w0, wa (, Tagn)]
    The format expected by the emulator and train_utils_pk_emulator.py.

  - Datagen convention  [lnAs, ns, H0, ob_h2, oc_h2, w0, wa (, Tagn)]
    The format expected by the CAMB data-generation pipeline.

The w0/wa space is sampled by drawing w0 and w0+wa independently from uniform
distributions, then deriving wa = (w0+wa) - w0.  This avoids the triangular
sampling artefact that arises when w0 and wa are drawn independently.

An optional Tagn parameter controls the AGN feedback temperature in the
HMcode-2020 baryonic feedback model.  Include it by setting include_tagn=True.

Usage (standalone):
    python ./mps_emu/lhs_generator.py

Author: Victoria Lloyd (2025)
"""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Parameter index conventions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamIndexML:
    """
    Column indices for arrays stored in ML convention.

    ML convention: [As_1e9, ns, H0, Ob, Om, w0, wa (, Tagn)]
    wa is derived after sampling as (w0+wa) - w0.
    """
    As_1e9: int = 0
    ns:     int = 1
    H0:     int = 2
    Ob:     int = 3
    Om:     int = 4
    w0:     int = 5
    wa:     int = 6
    Tagn:   int = 7


@dataclass(frozen=True)
class ParamIndexDatagen:
    """
    Column indices for arrays stored in datagen convention.

    Datagen convention: [lnAs, ns, H0, ob_h2, oc_h2, w0, wa (, Tagn)]
    This is the format passed directly to the CAMB data-generation pipeline.
    """
    lnAs:  int = 0
    ns:    int = 1
    H0:    int = 2
    ob_h2: int = 3
    oc_h2: int = 4
    w0:    int = 5
    wa:    int = 6
    Tagn:  int = 7


# Convenience singletons for column access (e.g. params[:, P_ML.w0])
P_ML = ParamIndexML()
P_DG = ParamIndexDatagen()

# Human-readable parameter name tuples (used for logging / array annotation)
PARAM_KEYS_ML       = ("As_1e9", "ns", "H0", "Ob", "Om", "w0", "wa")
PARAM_KEYS_ML_TFREE = ("As_1e9", "ns", "H0", "Ob", "Om", "w0", "wa", "Tagn")

# Internal sampling keys: wa is replaced by w0wa (the sum w0+wa) during LHS
# sampling, then converted back to wa after the draw.
_SAMPLE_KEYS       = ("As_1e9", "ns", "H0", "Ob", "Om", "w0", "w0wa")
_SAMPLE_KEYS_TFREE = ("As_1e9", "ns", "H0", "Ob", "Om", "w0", "w0wa", "Tagn")


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

# Test-set priors in ML convention.
# Note: w0wa is the sum w0+wa, not wa itself.
test_priors_ml = {
    "As_1e9": (1.0,  3.5),
    "ns":     (0.8,  1.05),
    "H0":     (55,   88),
    "Ob":     (0.03, 0.07),
    "Om":     (0.2,  0.5),
    "w0":     (-3.5, -0.01),
    "w0wa":   (-4.0, -0.01),   # w0 + wa sampled directly; wa = w0wa - w0
    "Tagn":   (6.5,  8.0),
}

# Bound on w(z) at high redshift: w(z=10) = w0 + wa*(1 - 1/11)
# Applied optionally in enforce_w_constraints().
W_HIGHZ_BOUNDS = (-3.0, -0.6)

# Training priors are slightly wider than test priors to avoid edge effects.
PRIOR_EXPANSION_FACTOR = 0.05


def expand_priors(test_priors: dict, expansion_factor: float = PRIOR_EXPANSION_FACTOR) -> dict:
    """
    Expand each prior bound outward by expansion_factor * width on each side.

    Parameters
    ----------
    test_priors      : dict mapping param name -> (min, max)
    expansion_factor : fractional expansion per side (default 0.05 = 5%)

    Returns
    -------
    dict with the same keys and expanded bounds
    """
    train_priors = {}
    for key, (pmin, pmax) in test_priors.items():
        expansion = (pmax - pmin) * expansion_factor
        train_priors[key] = (pmin - expansion, pmax + expansion)
    return train_priors


train_priors_ml = expand_priors(test_priors_ml)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

OUTDIR = Path("./params")
OUTDIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def lhs_sample(
    n_samples: int,
    priors: dict,
    seed: int = None,
    include_tagn: bool = True,
) -> np.ndarray:
    """
    Draw n_samples cosmologies via Latin Hypercube Sampling within prior bounds.

    w0 and w0+wa are sampled independently; wa is then derived as
    wa = w0wa - w0.  This produces a uniform joint distribution over the
    (w0, wa) plane without the triangular cut-off that arises from sampling
    w0 and wa independently.

    Parameters
    ----------
    n_samples    : int  — number of samples to draw
    priors       : dict — maps param name to (min, max); must contain 'w0wa', not 'wa'
    seed         : int or None — RNG seed for reproducibility
    include_tagn : bool — if True, also sample Tagn

    Returns
    -------
    (n_samples, N_params) ndarray in ML convention [As_1e9, ns, H0, Ob, Om, w0, wa (, Tagn)]
    """
    sample_keys = _SAMPLE_KEYS_TFREE if include_tagn else _SAMPLE_KEYS
    ndim = len(sample_keys)

    sampler      = qmc.LatinHypercube(d=ndim, seed=seed)
    unit_samples = sampler.random(n=n_samples)   # uniform in [0, 1]^ndim

    # Scale to actual prior ranges
    bounds      = np.array([priors[k] for k in sample_keys])
    raw_samples = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * unit_samples

    # Replace the w0wa column with wa = w0wa - w0
    w0_col   = sample_keys.index("w0")
    w0wa_col = sample_keys.index("w0wa")
    samples  = raw_samples.copy()
    samples[:, w0wa_col] = raw_samples[:, w0wa_col] - raw_samples[:, w0_col]

    # Column order is now [..., w0, wa, ...] — matches ML convention
    return samples


def enforce_w_constraints(params: np.ndarray) -> np.ndarray:
    """
    Retain only cosmologies where w(z=10) lies within W_HIGHZ_BOUNDS.

    w(z=10) = w0 + wa * (1 - a),  a = 1 / (1 + z) = 1/11

    Parameters
    ----------
    params : (N, N_params) ndarray in ML convention

    Returns
    -------
    Filtered (M, N_params) ndarray, M <= N
    """
    w0 = params[:, P_ML.w0]
    wa = params[:, P_ML.wa]

    a_highz = 1.0 / 11.0
    w_highz = w0 + wa * (1.0 - a_highz)
    mask    = (w_highz >= W_HIGHZ_BOUNDS[0]) & (w_highz <= W_HIGHZ_BOUNDS[1])
    return params[mask]


def enforce_lcdm(params: np.ndarray) -> np.ndarray:
    """
    Override w0=-1 and wa=0 for all rows (ΛCDM limit).

    Parameters
    ----------
    params : (N, N_params) ndarray in ML convention

    Returns
    -------
    Copy of params with w0 and wa columns fixed
    """
    params = params.copy()
    params[:, P_ML.w0] = -1.0
    params[:, P_ML.wa] =  0.0
    return params


def collect_samples(
    priors: dict,
    n_samples: int,
    lcdm: bool,
    seed: int = None,
    include_tagn: bool = True,
    apply_w_constraints: bool = True,
) -> np.ndarray:
    """
    Draw batches of LHS samples until exactly n_samples pass all constraints.

    Parameters
    ----------
    priors              : dict — parameter priors (must include 'w0wa', not 'wa')
    n_samples           : int  — target number of accepted samples
    lcdm                : bool — if True, fix w0=-1, wa=0 (ΛCDM)
    seed                : int or None — base RNG seed; incremented per batch
    include_tagn        : bool — if True, also sample Tagn
    apply_w_constraints : bool — if True, apply high-z w(z) check (ignored for ΛCDM)

    Returns
    -------
    (n_samples, N_params) ndarray in ML convention
    """
    collected = []
    total     = 0
    batch_num = 0

    # Less oversampling needed when w0+wa is drawn uniformly (no triangular cut)
    oversample_factor = 1.5 if (apply_w_constraints and not lcdm) else 1.2

    while total < n_samples:
        batch_seed = (seed + batch_num) if seed is not None else None
        n_raw      = int(oversample_factor * (n_samples - total))

        raw = lhs_sample(n_raw, priors, seed=batch_seed, include_tagn=include_tagn)

        if lcdm:
            raw = enforce_lcdm(raw)
        elif apply_w_constraints:
            raw = enforce_w_constraints(raw)

        if len(raw) > 0:
            collected.append(raw)
            total += len(raw)

        batch_num += 1

        if batch_num > 100:
            print(f"  Warning: {batch_num} batches drawn, only {total}/{n_samples} "
                  f"samples accepted. Consider relaxing constraints or widening priors.")
            break

    result          = np.vstack(collected)[:n_samples]
    acceptance_rate = n_samples / (batch_num * n_raw / oversample_factor)
    print(f"  Acceptance rate: {acceptance_rate:.2%}  "
          f"({n_samples} samples from {batch_num} batch(es))")
    return result


# ---------------------------------------------------------------------------
# Parameter convention conversions
# ---------------------------------------------------------------------------

def _unpack(params: np.ndarray):
    """Transpose params so columns can be unpacked as named scalars."""
    return params.T


def ml_to_datagen(params_ml: np.ndarray, mnu: float = 0.06, include_tagn: bool = None) -> np.ndarray:
    """
    Convert from ML convention to datagen convention.

    ML:      [As_1e9, ns, H0, Ob, Om,    w0, wa (, Tagn)]
    Datagen: [lnAs,   ns, H0, ob_h2, oc_h2, w0, wa (, Tagn)]

    Derived quantities:
      lnAs  = ln(10^10 * As) = ln(10 * As_1e9)
      ob_h2 = Ob * (H0/100)^2
      oc_h2 = Om * h^2 - ob_h2 - neutrino_contribution

    Parameters
    ----------
    params_ml    : (N, 7 or 8) ndarray
    mnu          : float — total neutrino mass in eV (default 0.06)
    include_tagn : bool or None — inferred from shape if None

    Returns
    -------
    (N, 7 or 8) ndarray in datagen convention
    """
    if include_tagn is None:
        include_tagn = (params_ml.shape[1] == 8)

    if include_tagn:
        As_1e9, ns, H0, Ob, Om, w0, wa, Tagn = _unpack(params_ml)
    else:
        As_1e9, ns, H0, Ob, Om, w0, wa = _unpack(params_ml)

    h     = H0 / 100.0
    ob_h2 = Ob * h**2
    lnAs  = np.log(10.0 * As_1e9)

    # Neutrino density contribution: mnu in eV, using standard N_eff=3.046
    mnu_contrib = (mnu * (3.046 / 3) ** 0.75) / 94.0708
    oc_h2       = Om * h**2 - ob_h2 - mnu_contrib

    if include_tagn:
        return np.column_stack((lnAs, ns, H0, ob_h2, oc_h2, w0, wa, Tagn))
    else:
        return np.column_stack((lnAs, ns, H0, ob_h2, oc_h2, w0, wa))


def datagen_to_ml(params_datagen: np.ndarray, mnu: float = 0.06, include_tagn: bool = None) -> np.ndarray:
    """
    Convert from datagen convention to ML convention (inverse of ml_to_datagen).

    Datagen: [lnAs, ns, H0, ob_h2, oc_h2, w0, wa (, Tagn)]
    ML:      [As_1e9, ns, H0, Ob, Om, w0, wa (, Tagn)]

    Parameters
    ----------
    params_datagen : (N, 7 or 8) ndarray
    mnu            : float — total neutrino mass in eV (default 0.06)
    include_tagn   : bool or None — inferred from shape if None

    Returns
    -------
    (N, 7 or 8) ndarray in ML convention
    """
    if include_tagn is None:
        include_tagn = (params_datagen.shape[1] == 8)

    if include_tagn:
        lnAs, ns, H0, ob_h2, oc_h2, w0, wa, Tagn = _unpack(params_datagen)
    else:
        lnAs, ns, H0, ob_h2, oc_h2, w0, wa = _unpack(params_datagen)

    h    = H0 / 100.0
    Ob   = ob_h2 / h**2

    mnu_contrib = (mnu * (3.046 / 3) ** 0.75) / 94.0708
    Om          = (oc_h2 + ob_h2 + mnu_contrib) / h**2
    As_1e9      = np.exp(lnAs) / 10.0

    if include_tagn:
        return np.column_stack((As_1e9, ns, H0, Ob, Om, w0, wa, Tagn))
    else:
        return np.column_stack((As_1e9, ns, H0, Ob, Om, w0, wa))


def convert_datagen_file_to_ml(datagen_filepath: str, ml_filepath: str = None, mnu: float = 0.06) -> np.ndarray:
    """
    Load a saved datagen .npy file, convert to ML convention, and save.

    Parameters
    ----------
    datagen_filepath : str — path to the datagen .npy file
    ml_filepath      : str or None — output path; defaults to replacing
                       '_datagen_' with '_ml_' in the input filename
    mnu              : float — total neutrino mass in eV

    Returns
    -------
    (N, N_params) ndarray in ML convention
    """
    datagen_params = np.load(datagen_filepath)
    ml_params      = datagen_to_ml(datagen_params, mnu=mnu)

    if ml_filepath is None:
        ml_filepath = str(datagen_filepath).replace("_datagen_", "_ml_")

    np.save(ml_filepath, ml_params)
    print(f"Converted {datagen_filepath} -> {ml_filepath}")
    return ml_params


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------

def generate_batches(
    priors: dict,
    batch_indices,
    n_per_batch: int = 2000,
    lcdm: bool = False,
    include_tagn: bool = True,
    apply_w_constraints: bool = True,
) -> None:
    """
    Generate and save cosmological parameter batches in both ML and datagen
    conventions.

    Each batch is saved as two .npy files in OUTDIR:
      train_<model>_mps_ml_<idx>_expanded2.npy      — ML convention
      train_<model>_mps_datagen_<idx>_expanded2.npy — Datagen convention

    Parameters
    ----------
    priors              : dict — parameter priors in ML convention (with 'w0wa' key)
    batch_indices       : iterable of int — batch index numbers for file naming
    n_per_batch         : int  — number of cosmologies per batch (default 2000)
    lcdm                : bool — if True, fix w0=-1, wa=0 for all samples
    include_tagn        : bool — if True, include Tagn as a sampled parameter
    apply_w_constraints : bool — if True, filter on w(z=10) within W_HIGHZ_BOUNDS
    """
    model_tag = "lcdm" if lcdm else "w0wacdm"

    for batch_idx in batch_indices:
        print(f"[Batch {batch_idx}] Generating {n_per_batch} samples ({model_tag})...")
        params_ml      = collect_samples(priors, n_per_batch, lcdm,
                                         seed=batch_idx, include_tagn=include_tagn,
                                         apply_w_constraints=apply_w_constraints)
        params_datagen = ml_to_datagen(params_ml, include_tagn=include_tagn)

        ml_file      = OUTDIR / f"train_{model_tag}_mps_ml_{batch_idx}_expanded2.npy"
        datagen_file = OUTDIR / f"train_{model_tag}_mps_datagen_{batch_idx}_expanded2.npy"

        np.save(ml_file,      params_ml)
        np.save(datagen_file, params_datagen)
        print(f"  Saved {ml_file.name} and {datagen_file.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("[INFO] LHS Generator")
    print("=" * 60)

    print("\nTest priors (ML convention):")
    for k, v in test_priors_ml.items():
        print(f"  {k}: ({v[0]:.6f}, {v[1]:.6f})")

    print("\nTraining priors (ML convention, expanded by "
          f"{PRIOR_EXPANSION_FACTOR:.0%} per side):")
    for k, v in train_priors_ml.items():
        print(f"  {k}: ({v[0]:.6f}, {v[1]:.6f})")

    print(f"\nHigh-z w(z) safety bounds: w(z=10) in {W_HIGHZ_BOUNDS}")
    print(f"Output directory: {OUTDIR.resolve()}")

    # --- Training batches ---
    print("\n[INFO] Generating training batches (indices 20–49)...")
    generate_batches(
        train_priors_ml,
        batch_indices=range(20, 50),
        n_per_batch=2000,
        lcdm=False,
        include_tagn=False,
        apply_w_constraints=False,
    )

    # --- Test batch ---
    print("\n[INFO] Generating test batch (index 100)...")
    generate_batches(
        test_priors_ml,
        batch_indices=[100],
        n_per_batch=2000,
        lcdm=False,
        include_tagn=False,
        apply_w_constraints=False,
    )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()