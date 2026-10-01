# Auxiliary functions for Power Spectrum Emulation
# Author: João Victor Silva Rebouças (2022), updated by Victoria Lloyd (2025)
import math
import pickle
from itertools import product
import os
import joblib
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from keras import models, layers, optimizers
from keras.regularizers import l1_l2
from tqdm import tqdm
import time
from joblib import Parallel, delayed


import sys; sys.path.insert(0, "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); import symbolic_pofk.linear_VL as linear
import sys; sys.path.insert(0, "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); from symbolic_pofk.linear_VL import plin_emulated, get_approximate_D, growth_correction_R, get_eisensteinhu_nw
import sys; sys.path.insert(0, "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); from symbolic_pofk.syrenhalofit import run_halofit_vec
from symbolic_pofk.linear_VL import As_to_sigma8


keras.mixed_precision.set_global_policy('mixed_bfloat16')
tf.config.optimizer.set_jit(True)


# import subprocess
# import sys

# def install(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# # Example: Install the 'requests' library
# install("umap-learn==0.5.3")

from sklearn.decomposition import KernelPCA

# ----------------------------------------------------------------------------------------------------
# Parameter space
params = ['As', 'ns', 'h', 'Omega_b', 'Omega_m', 'w', 'w0+wa']
params_latex = [r'$10^9 A_s$', r'$n_s$', r'$H_0$', r'$\Omega_b$', r'$\Omega_m$', r'$w_0$', r'$w_0+w_a$']

# Extended parameter set for mnu/T_AGN-free datasets (tfree_mnufree).
# Datasets using the base 7-parameter set are unaffected; these are only
# appended when COLASet detects a 9-column `lhs` array (see __init__).
params_tfree_mnufree = params + ['T_AGN', 'mnu']
params_latex_tfree_mnufree = params_latex + [r'$\log T_{\rm AGN}$', r'$m_\nu$']

DEFAULT_MNU = 0.06

#k=1e-5 to 1e2 (with 500 steps in log space)
ks = np.logspace(-5.1, 2, 500)
# ks_lin = np.logspace(-5.1, 2, 500)          # original grid (must match the saved files)

# kmin, kmax = 0.005, 50.0
# k_mask = (ks_lin >= kmin) & (ks_lin <= kmax)

# ks = ks_lin[k_mask]    
# Boost truncation: the NL/LIN boost is ~1 (log~0) for k < 1e-2, so training
# on that region wastes PCA modes and network capacity.  In boost mode we
# truncate to k >= BOOST_K_MIN and pad with 1.0 at inference time.
# NOTE: No longer needed now that the boost denominator is the syren halofit
# prediction (which is non-trivial at all k), so truncation is commented out.
# BOOST_K_MIN  = 1e-2
# BOOST_K_MASK = ks >= BOOST_K_MIN   # shape (500,), True for the nonlinear k range
# ks_boost     = ks[BOOST_K_MASK]

# Kept as pass-through aliases so any code that references them still works,
# but no truncation is actually applied.
BOOST_K_MIN  = ks[0]              # effectively no truncation
BOOST_K_MASK = np.ones(len(ks), dtype=bool)
ks_boost     = ks

z1_mps = np.linspace(0,3,33,endpoint=False)
z2_mps = np.linspace(3,10,7,endpoint=False)
z3_mps = np.linspace(10,50,12)
z_mps = np.concatenate((z1_mps, z2_mps, z3_mps), axis=0) #, z3_mps

H0_MAX = 90
OMEGA_B_MAX = 0.072

BASE_PATH = "/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps/"

# ----------------------------------------------------------------------------------------------------
# nl_type registry
NL_TYPE_REGISTRY = {
    "lin":                      ("pklin",                           "Linear P(k)"),
    "halofit":                  ("pknonlin",                        "Non-linear P(k) via HaloFit"),
    "mead2020":                 ("mead2020_pknonlin",               "Non-linear P(k) via HMcode Mead2020"),
    "mead2020_feedback":        ("mead2020_feedback_pknonlin",      "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (fixed T_AGN)"),
    "mead2020_feedback_Tfree":  ("mead2020_feedback_Tfree_pknonlin","Non-linear P(k) via HMcode Mead2020 + baryonic feedback (free T_AGN)"),
    "mead2020_Tfree_mnufree_lin":  ("mead2020_Tfree_mnufree_pklin",   "Linear P(k) via HMcode Mead2020 + baryonic feedback + mnu free on expanded priors(free T_AGN)"),
    "mead2020_Tfree_mnufree":      ("mead2020_Tfree_mnufree_pknonlin",   "Non-linear P(k) via HMcode Mead2020 + baryonic feedback + mnu free on expanded priors(free T_AGN)"),
}

VALID_NL_TYPES = {
    "expanded":    {"lin", "halofit", "mead2020_Tfree_mnufree_lin", "mead2020_Tfree_mnufree"},
    "constrained": set(NL_TYPE_REGISTRY.keys()),
}

VER = "_v15"

def _validate_prior_and_nl_type(prior_type: str, nl_type: str):
    if prior_type not in VALID_NL_TYPES:
        raise ValueError(
            f"Unknown prior_type '{prior_type}'. "
            f"Must be one of: {sorted(VALID_NL_TYPES.keys())}"
        )
    if nl_type not in NL_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown nl_type '{nl_type}'. "
            f"Must be one of: {sorted(NL_TYPE_REGISTRY.keys())}"
        )
    if nl_type not in VALID_NL_TYPES[prior_type]:
        raise ValueError(
            f"nl_type '{nl_type}' is not available for prior_type '{prior_type}'. "
            f"Valid nl_types for '{prior_type}': {sorted(VALID_NL_TYPES[prior_type])}"
        )


def _make_file_paths(base_path: str, cosmo_type: str, prior_type: str, nl_type: str, batch: int):
    pk_suffix = NL_TYPE_REGISTRY[nl_type][0]
    if prior_type == "expanded":
        input_name  = f"train_{cosmo_type}_mps_ml_{batch}_expanded_uniform.npy"
        output_name = f"train_{cosmo_type}_{batch}_expanded_uniform_{pk_suffix}.npy"
        if "mead2020_Tfree_mnufree" in nl_type:
            input_name  = f"train_{cosmo_type}_mps_ml_{batch}_tfree_mnufree.npy"
            output_name = f"train_{cosmo_type}_expanded_{batch}_{pk_suffix}.npy"
    else:
        input_name  = f"old_prior/train_{cosmo_type}_mps_ml_{batch}_new.npy"
        output_name = f"train_{cosmo_type}_condensed_{batch}_new_{pk_suffix}.npy"
    input_path  = os.path.join(base_path, "input/input_hs",  input_name)
    output_path = os.path.join(base_path, "output/output_hs", output_name)
    return input_path, output_path


# ----------------------------------------------------------------------------------------------------
def _compute_mps_approximation(ks, zs, params: np.ndarray, use_eh=False) -> np.ndarray:
    """
    Computes the analytical P_lin(k, z) approximation in physical units (Mpc³).

    Args:
        params: 1D array [10^9 A_s, ns, H0, Ob, Om, w0, wa].

    Returns:
        np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³.
    """
    As, ns, H0_in, Ob, Om, w0, w0wa = params[:7]
    wa = w0wa - w0 
    h = H0_in / 100.0
    mnu = params[8] if len(params) > 7 else DEFAULT_MNU
    k_for_plin = ks / h
    if use_eh:
        pk_fid_hmpc = get_eisensteinhu_nw(k_for_plin, As, Om, Ob, h, ns, mnu=mnu, w0=w0, wa=wa)
    else:
        pk_fid_hmpc = plin_emulated(k_for_plin, Om, Ob, h, ns, As=As, mnu=mnu, w0=w0, wa=wa)
    a_array = 1.0 / (zs + 1)
    D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
    Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)
    R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
    Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)
    growth_factors = (Dz/D0) * (Dz/D0) * (Rz/R0)
    result = pk_fid_hmpc[None, :] * growth_factors[:, None]
    result = result / h**3
    return result.astype(np.float32)


def _compute_mps_nl_approximation(ks, zs, params: np.ndarray) -> np.ndarray:
    """
    Computes the syren halofit nonlinear P(k, z) approximation in physical
    units (Mpc³).

    This is used as the denominator of the boost that the emulator learns
    in nonlinear mode:

        frac_pks = P_nl_camb / P_nl_syren

    so the network only has to learn the residual between CAMB's nonlinear
    spectrum and the syren halofit prediction, which is much smaller and
    smoother than the raw NL/LIN boost.

    Args:
        ks:     k-modes in 1/Mpc (Mpc⁻¹), shape (N_K_MODES,).
        zs:     redshift array, shape (N_ZS,).
        params: 1D array [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa].

    Returns:
        np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³.
    """
    As, ns, H0_in, Ob, Om, w0, w0wa = params[:7]
    wa = w0wa - w0
    h = H0_in / 100.0
    mnu = params[8] if len(params) > 7 else DEFAULT_MNU

    # k in h/Mpc for syren calls
    k_hmpc = ks / h

    # Linear Pk in (Mpc/h)^3 from syren
    pk_lin_hmpc = plin_emulated(k_hmpc, Om, Ob, h, ns, As=As, mnu=mnu, w0=w0, wa=wa)  # shape (N_K,)

    a_array = 1.0 / (np.asarray(zs) + 1.0)

    # Apply growth factor to get P_lin(k, z) in (Mpc/h)^3
    D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
    Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)
    R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
    Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)
    growth_factors = (Dz / D0) ** 2 * (Rz / R0)          # shape (N_ZS,)

    pk_lin_hmpc_z = pk_lin_hmpc[None, :] * growth_factors[:, None]  # (N_ZS, N_K)

    # sigma8 at z=0 for halofit (halofit uses sigma8 at z=0 + a(z) internally)
    sigma8_z0 = As_to_sigma8(As, Om, Ob, h, ns, mnu=mnu, w0=w0, wa=wa)

    # run_halofit_vec expects P_lin in (Mpc/h)^3 and k in h/Mpc;
    # return_boost=False → returns P_nl in (Mpc/h)^3
    pk_nl_hmpc_z = run_halofit_vec(
        k_hmpc,
        sigma8_z0,
        Om, Ob, h, ns,
        a_array,
        return_boost=False,
        Plin_in=pk_lin_hmpc_z,
    )  # shape (N_ZS, N_K)

    boost = pk_nl_hmpc_z / pk_lin_hmpc_z

    # Convert (Mpc/h)^3 → Mpc³
    # pk_nl_mpc_z = pk_nl_hmpc_z / h**3
    return boost.astype(np.float32)

def _compute_mps_nl_approximation_parallel(ks, zs, params_batch, n_jobs=-1, chunk_size=20000):
    As, ns, H0_in, Ob, Om, w0, w0wa = [params_batch[:, i] for i in range(7)]
    wa = w0wa - w0
    h  = H0_in / 100.0

    N_cosmo = params_batch.shape[0]
    N_z, N_k = len(zs), len(ks)

    if params_batch.shape[1] > 7:
        mnu = params_batch[:, 8]
    else:
        mnu = np.full(N_cosmo, DEFAULT_MNU, dtype=params_batch.dtype)

    a_array = 1.0 / (np.asarray(zs) + 1.0)
    boost = np.empty((N_cosmo, N_z, N_k), dtype=np.float32)

    def _halofit_one(j, k_hmpc_j, sigma8_z0_j, Om_j, Ob_j, h_j, ns_j, Plin_j):
        return run_halofit_vec(
            k_hmpc_j, sigma8_z0_j, Om_j, Ob_j, h_j, ns_j,
            a_array, return_boost=True, Plin_in=Plin_j,
        )

    for start in tqdm(range(0, N_cosmo, chunk_size), desc="syren halofit (chunked)", unit="chunk"):
        end = min(start + chunk_size, N_cosmo)
        sl = slice(start, end)

        As_b, Om_b, Ob_b, h_b, ns_b, w0_b, wa_b, mnu_b = [
            p[sl, None] for p in (As, Om, Ob, h, ns, w0, wa, mnu)
        ]
        k_hmpc = ks[None, :] / h_b   # (chunk, N_k)

        pk_lin_hmpc = plin_emulated(k_hmpc, Om_b, Ob_b, h_b, ns_b, As=As_b, mnu=mnu_b, w0=w0_b, wa=wa_b)

        a_b = a_array[None, :]
        D0 = get_approximate_D(k=1e-4, As=As_b, Om=Om_b, Ob=Ob_b, h=h_b, ns=ns_b, mnu=mnu_b, w0=w0_b, wa=wa_b, a=1)
        Dz = get_approximate_D(k=1e-4, As=As_b, Om=Om_b, Ob=Ob_b, h=h_b, ns=ns_b, mnu=mnu_b, w0=w0_b, wa=wa_b, a=a_b)
        R0 = growth_correction_R(As=As_b, Om=Om_b, Ob=Ob_b, h=h_b, ns=ns_b, mnu=mnu_b, w0=w0_b, wa=wa_b, a=1)
        Rz = growth_correction_R(As=As_b, Om=Om_b, Ob=Ob_b, h=h_b, ns=ns_b, mnu=mnu_b, w0=w0_b, wa=wa_b, a=a_b)
        growth_factors = (Dz / D0) ** 2 * (Rz / R0)
        pk_lin_hmpc_z = pk_lin_hmpc[:, None, :] * growth_factors[:, :, None]   # (chunk, N_z, N_k)

        sigma8_z0 = As_to_sigma8(As_b, Om_b, Ob_b, h_b, ns_b, mnu=mnu_b, w0=w0_b, wa=wa_b)

        results = Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(_halofit_one)(
                j, ks / h_b[j - start, 0], float(sigma8_z0[j - start, 0]),
                float(Om_b[j - start, 0]), float(Ob_b[j - start, 0]),
                float(h_b[j - start, 0]), float(ns_b[j - start, 0]),
                pk_lin_hmpc_z[j - start],
            )
            for j in range(start, end)
        )
        for offset, r in enumerate(results):
            boost[start + offset] = r

        del pk_lin_hmpc, pk_lin_hmpc_z, growth_factors, results
        import gc; gc.collect()

    return boost

def make_trimmed_huber_loss(trim_top_frac: float, huber_delta: float = 1.0):
    """
    Same as before, but uses tf.math.top_k instead of tf.sort — O(n) to find
    the k worst elements instead of O(n log n) to sort everything just to
    slice off the top fraction.
    """
    @tf.function
    def trimmed_huber(y_true, y_pred):
        per_sample = tf.reduce_mean(
            tf.keras.losses.huber(y_true, y_pred, delta=huber_delta),
            axis=-1
        )
        if trim_top_frac <= 0.0:
            return tf.reduce_mean(per_sample)

        batch_size = tf.shape(per_sample)[0]
        n_trim = tf.cast(
            tf.math.floor(tf.cast(batch_size, tf.float32) * trim_top_frac),
            tf.int32
        )
        n_trim = tf.maximum(n_trim, 0)

        # top_k finds the LARGEST n_trim values -> these are exactly the ones
        # we want to EXCLUDE (the worst-fit samples), so we mask them out
        # rather than slicing a sorted array.
        _, worst_indices = tf.math.top_k(per_sample, k=n_trim, sorted=False)
        # sorted=False is fine here -- we don't care about order, just which
        # indices to exclude, so skip the extra cost of sorting the top-k too.

        mask = tf.ones_like(per_sample, dtype=tf.bool)
        mask = tf.tensor_scatter_nd_update(
            mask,
            tf.expand_dims(worst_indices, axis=1),
            tf.zeros_like(worst_indices, dtype=tf.bool),
        )

        kept = tf.boolean_mask(per_sample, mask)
        return tf.reduce_mean(kept)
    return trimmed_huber

# ----------------------------------------------------------------------------------------------------
def load_set(
    n_batches=1,
    start_batch=0,
    z_indices=None,
    cosmo_type="w0wacdm",
    prior_type="constrained",
    nl_type="lin",
    base_path=BASE_PATH,
    check_unphysical=True,
    verbose=True
):
    """Load cosmological input/output data batches."""
    _validate_prior_and_nl_type(prior_type, nl_type)
    inputs_all, outputs_all = [], []

    for b in tqdm(range(start_batch, start_batch + n_batches), desc="Loading batches", unit="batch"):
        input_path, output_path = _make_file_paths(base_path, cosmo_type, prior_type, nl_type, b)
        # if "lin" in nl_type: output_path = "/lustre/nvwulf/home/vlloyd/pklins_large_hypersphere_R1.3.npy"
        # else: output_path = "/lustre/nvwulf/home/vlloyd/pknls_large_hypersphere_R1.3.npy"
        # input_path = "/lustre/nvwulf/home/vlloyd/cosmos_large_hypersphere_R1.3.npy"
        if not (os.path.exists(input_path) and os.path.exists(output_path)):
            if verbose:
                print(f"⚠️  Skipping missing batch {b}  (prior='{prior_type}', nl_type='{nl_type}')")
                print(f"    expected input : {input_path}")
                print(f"    expected output: {output_path}")
            continue

        x = np.load(input_path,  mmap_mode="r")
        y = np.load(output_path, mmap_mode="r").astype(np.float32, copy=False)#[..., k_mask]
        if z_indices is not None:
            y = y[:, z_indices, :]
        if check_unphysical:
            bad = np.all((y == 0) | (y == 1), axis=(1, 2))
            if np.any(bad) and verbose:
                print(f"🧹 Removed {np.sum(bad)} unphysical cosmologies from batch {b}")
            x, y = x[~bad], y[~bad]
        inputs_all.append(x)
        outputs_all.append(y)

    if inputs_all:
        inputs_all  = np.concatenate(inputs_all,  axis=0)
        outputs_all = np.concatenate(outputs_all, axis=0)
    else:
        inputs_all  = np.empty((0, 5))
        outputs_all = np.empty((0, 122, 2000))
    return inputs_all, outputs_all

import time

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_batch_begin(self, batch, logs=None):
        self._batch_start = time.perf_counter()

    def on_train_batch_end(self, batch, logs=None):
        dur = time.perf_counter() - self._batch_start
        if batch < 20:   # only log the first few, avoid flooding the output file
            print(f"[TIMING] batch {batch}: {dur*1000:.2f} ms", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        print(f"[TIMING] epoch {epoch} loss={logs.get('loss'):.6f}", flush=True)

# ----------------------------------------------------------------------------------------------------
class TComponentScaler:
    """
    Per-column standardisation for t-components.

    Stores mean and std computed on the training set so that inference can
    invert the transform consistently.  Saved to / loaded from disk via joblib
    alongside the other metadata.

    Why this matters
    ----------------
    The raw tPCA output has a 288x std ratio across the 15 columns (t[0] std ≈ 143,
    t[14] std ≈ 0.5).  Under MSE loss this means gradients for t[14] are ~83 000×
    smaller than for t[0], so the network essentially never learns to predict the
    smaller components accurately.  Standardising to unit variance before training
    gives every component equal weight in the loss.
    """
    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, T: np.ndarray) -> "TComponentScaler":
        self.mean_ = T.mean(axis=0)
        self.std_  = T.std(axis=0)
        self.std_  = np.where(self.std_ == 0, 1.0, self.std_)  # guard zero-std columns
        return self

    def transform(self, T: np.ndarray) -> np.ndarray:
        return (T - self.mean_) / self.std_

    def inverse_transform(self, T_norm: np.ndarray) -> np.ndarray:
        return T_norm * self.std_ + self.mean_


# ----------------------------------------------------------------------------------------------------
# Metadata bundle helpers
# ----------------------------------------------------------------------------------------------------

def _build_metadata_bundle(
    param_scaler,
    t_comp_pca,
    t_comp_scaler,
    pcas_dict: dict,
    scalers_dict: dict,
    npce_indices=None,
    model_type: str = "mlp",
) -> dict:
    return {
        "param_scaler":  param_scaler,
        "t_comp_pca":    t_comp_pca,
        "t_comp_scaler": t_comp_scaler,
        "pcas":          pcas_dict,
        "scalers":       scalers_dict,
        "npce_indices":  npce_indices,
        "model_type":    model_type,
    }


def save_metadata_bundle(bundle: dict, path: str, compress: int = 3) -> None:
    """Save a metadata bundle to a single joblib file."""
    joblib.dump(bundle, path, compress=compress)
    logging.info(
        f"Metadata bundle saved to {path}  "
        f"(model_type={bundle['model_type']}, "
        f"n_z={len(bundle['pcas'])}, "
        f"t_comp_scaler={'present' if bundle['t_comp_scaler'] is not None else 'None'}, "
        f"npce_indices={'present' if bundle['npce_indices'] is not None else 'None'})"
    )


def load_metadata_bundle(path: str) -> dict:
    """Load a metadata bundle from disk and return it as a plain dict."""
    bundle = joblib.load(path)
    logging.info(
        f"Metadata bundle loaded from {path}  "
        f"(model_type={bundle.get('model_type', 'unknown')}, "
        f"n_z={len(bundle.get('pcas', {}))})"
    )
    return bundle


def update_metadata_bundle_npce(bundle_path: str, npce_indices: np.ndarray,
                                  model_type: str = "npce") -> None:
    bundle = load_metadata_bundle(bundle_path)
    bundle["npce_indices"] = npce_indices
    bundle["model_type"]   = model_type       # caller controls this
    save_metadata_bundle(bundle, bundle_path)


# ----------------------------------------------------------------------------------------------------
class COLASet:
    def __init__(
        self,
        path=BASE_PATH,
        target_z=None,
        cosmo_type="w0wacdm",
        prior_type="constrained",
        nl_type="lin",
        n_batches=1,
        start_batch=0,
        verbose=True
    ):
        _validate_prior_and_nl_type(prior_type, nl_type)
        self.n_batches  = n_batches
        self.cosmo_type = cosmo_type
        self.prior_type = prior_type
        self.nl_type    = nl_type

        # True for any nonlinear nl_type: emulate NL/syren_halofit boost instead
        # of LIN/syren_linear.
        self.use_boost = nl_type not in ("lin", "mead2020_Tfree_mnufree_lin")

        if target_z is not None:
            target_z_array = np.atleast_1d(target_z).astype(float)
            diffs     = np.abs(target_z_array[:, None] - z_mps[None, :])
            z_indices = np.argmin(diffs, axis=1).tolist()
            self.z    = target_z_array
        else:
            z_indices = None
            self.z    = np.array(z_mps)

        # ------------------------------------------------------------------
        # Load primary (target) dataset
        # ------------------------------------------------------------------
        self.lhs, self.pks_target = load_set(
            base_path   = path,
            z_indices   = z_indices,
            cosmo_type  = cosmo_type,
            prior_type  = prior_type,
            nl_type     = nl_type,
            n_batches   = n_batches,
            start_batch = start_batch,
            verbose     = verbose,
        )

        self.lhs = self.lhs.copy()
        self.lhs[:, 6] = self.lhs[:, 5] + self.lhs[:, 6]   # store w0+wa in col 6

        print(f"pks_target shape = {self.pks_target.shape}")
        self.ks = ks

        if isinstance(self.z, (list, np.ndarray)) and len(np.shape(self.z)) > 1:
            self.z = np.ravel(self.z).tolist()

        # ------------------------------------------------------------------
        # Compute frac_pks  (the quantity the emulator actually learns)
        #
        #   use_boost = False  (nl_type == "lin"):
        #       frac_pks = P_lin_camb / P_lin_syren
        #       — ratio of CAMB linear to the symbolic linear approximation.
        #
        #   use_boost = True   (nl_type != "lin"):
        #       frac_pks = P_nl_camb / P_nl_syren_halofit
        #       — residual boost relative to the syren halofit prediction.
        #       This is much closer to 1 than the old NL/LIN boost, so the
        #       network has an easier target, and no k-truncation is needed.
        #
        # NOTE: The old NL/LIN boost path (loading a companion linear dataset
        # and dividing) has been replaced.  We now compute the syren halofit
        # approximation analytically for every cosmology and use that as the
        # denominator, mirroring exactly what the linear emulator does with
        # the syren linear approximation.
        # ------------------------------------------------------------------
        if self.use_boost:

            lin_type = "mead2020_Tfree_mnufree_lin" if nl_type == "mead2020_Tfree_mnufree" else "lin" 
            _, self.pks_lin = load_set(
                base_path   = path,
                z_indices   = z_indices,
                cosmo_type  = cosmo_type,
                prior_type  = prior_type,
                nl_type     = lin_type,
                n_batches   = n_batches,
                start_batch = start_batch,
                verbose     = verbose,
            )   # not used in NL mode; syren provides the baseline

            print(
                f"Computing NL/syren_halofit boost "
                f"(frac_pks = Boost_{nl_type}_camb / boost_nl_syren) "
                f"over full k grid ({len(self.ks)} points)"
            )

            self.boost = self.pks_target / self.pks_lin
            del self.pks_lin
            import gc; gc.collect()
            # start_time0 = time.perf_counter()
            # self.mps_approxes_boost0 = np.array([
            #     _compute_mps_nl_approximation(self.ks, self.z, p)[0]
            #     if len(self.z) == 1 else
            #     _compute_mps_nl_approximation(self.ks, self.z, p)
            #     for p in tqdm(self.lhs, desc="syren halofit approx", unit="cosmo")
            # ])
            # end_time0 = time.perf_counter()
            # print(f"Time taken: {end_time0 - start_time0:.6f} seconds")
            # print(f"mps_approxes_boost shape = {self.mps_approxes_boost0.shape}")
            # start_time = time.perf_counter()
            self.mps_approxes_boost = _compute_mps_nl_approximation_parallel(
                self.ks, self.z, self.lhs, n_jobs=16#-1
            )
            self.mps_approxes = self.mps_approxes_boost
            # end_time = time.perf_counter()
            # print(f"Time taken: {end_time - start_time:.6f} seconds")
            print(f"mps_approxes_boost_batch shape = {self.mps_approxes_boost.shape}")
            # print(np.allclose(self.mps_approxes_boost, self.mps_approxes_boost0, rtol=1e-5))


            with np.errstate(divide="ignore", invalid="ignore"):
                self.frac_pks = self.boost / self.mps_approxes_boost
            del self.boost
            gc.collect()

            # Keep mps_approxes as an alias pointing to the NL approximation
            # so downstream code that references either name keeps working.
            # self.mps_approxes = self.mps_approxes_boost

        else:
            # Original path: emulate P_lin_camb / P_lin_syren
            self.pks_lin = self.pks_target   # alias for back-compat

            self.mps_approxes = np.array([
                _compute_mps_approximation(self.ks, self.z, p)[0]
                if len(self.z) == 1 else
                _compute_mps_approximation(self.ks, self.z, p)
                for p in self.lhs
            ])
            print(f"mps_approxes shape = {self.mps_approxes.shape}")
            print("Computing LIN/syren ratio (frac_pks = P_lin / P_syren)")

            with np.errstate(divide="ignore", invalid="ignore"):
                self.frac_pks = self.pks_lin / self.mps_approxes

            self.mps_approxes_boost = None   # not used in linear mode

        self.logfracs = np.log(self.frac_pks)

        # ------------------------------------------------------------------
        # Remove unphysical / non-finite cosmologies
        # ------------------------------------------------------------------
        all_zero_mask   = (self.pks_target == 0).all(axis=(1, 2))
        non_finite_mask = ~np.isfinite(self.logfracs).all(axis=(1, 2))
        bad_cosmo_mask  = all_zero_mask | non_finite_mask

        print(f"⚠️  Found {bad_cosmo_mask.sum()} / {len(bad_cosmo_mask)} cosmologies with issues:")
        print(f"   - {all_zero_mask.sum()} with all-zero P (unphysical)")
        print(f"   - {non_finite_mask.sum()} with non-finite log(frac_pks)")

        if bad_cosmo_mask.sum() > 0:
            bad_indices = np.where(bad_cosmo_mask)[0]
            print("First 10 bad cosmology indices:", bad_indices[:10])
            print("Example bad LHS:", self.lhs[bad_indices[0]])

        good_mask         = ~bad_cosmo_mask
        self.lhs          = self.lhs[good_mask]
        self.pks_target   = self.pks_target[good_mask]
        self.frac_pks     = self.frac_pks[good_mask]
        self.logfracs     = self.logfracs[good_mask]
        if self.use_boost:
            self.mps_approxes_boost = self.mps_approxes_boost[good_mask]
            self.mps_approxes    = self.mps_approxes_boost   # keep alias in sync
        else:
            self.mps_approxes = self.mps_approxes[good_mask]
            self.pks_lin      = self.pks_lin[good_mask]   # alias stays in sync

        self.w0_min   = None
        self.w0wa_max = None
        self._metadata_bundle_path = None

    # -------------------------------------------------------
    def _metadata_tag(self):
        w0_tag    = f"_w0min{self.w0_min}"     if self.w0_min   is not None else ""
        w0wa_tag  = f"_w0wamax{self.w0wa_max}" if self.w0wa_max is not None else ""
        mode_tag  = "_boost" if self.use_boost else ""
        param_tag = "_tfreemnufree" if self.lhs.shape[1] > 7 else ""
        # kmin_tag removed: k-truncation no longer applied
        return (
            f"{self.cosmo_type}_{self.prior_type}{w0_tag}{w0wa_tag}"
            f"_{self.nl_type}{mode_tag}{param_tag}_nTrain{self.n_batches}{VER}"
        )

    # -------------------------------------------------------
    def change_ks(self, ks):
        new_frac_pks = np.array([
            [CubicSpline(self.ks, self.frac_pks[i, j, :])(ks)
             for j in range(len(self.z))]
            for i in range(len(self.lhs))
        ])
        self.frac_pks = new_frac_pks
        self.logfracs = np.log(self.frac_pks)
        self.ks = ks
        if hasattr(self, "num_pcs") and self.num_pcs is not None:
            self.prepare(self.num_pcs)

    # -------------------------------------------------------
    def update(self, cosmos, frac_pks):
        frac_pks = np.atleast_3d(frac_pks)
        self.lhs      = np.vstack([self.lhs, cosmos])
        self.frac_pks = np.vstack([self.frac_pks, frac_pks])
        self.logfracs = np.log(self.frac_pks)
        if hasattr(self, "num_pcs") and self.num_pcs is not None:
            self.prepare(self.num_pcs)

    # -------------------------------------------------------
    def prepare(self, num_pcs, num_pcs_z, metadata_dir="mps_emu/metadata", multihead=False):
        """
        Fit scalers and PCA for each redshift, then run tPCA and normalise
        the resulting t-components to zero mean / unit variance per column.
        """

        if not multihead and num_pcs_z is None:
            raise ValueError("num_pcs_z must be provided when multihead=False.")
        
        self.num_pcs = num_pcs
        metadata_subdir = os.path.join(metadata_dir, f"metadata_{self._metadata_tag()}")
        os.makedirs(metadata_subdir, exist_ok=True)

        # --- Cosmological parameter scaler ---
        self.param_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.lhs)
        self.lhs_norm     = self.param_scaler.transform(self.lhs)

        # --- Per-z logfrac scalers + PCA ---
        self.frac_pks_scalers = []
        self.logfracs_norm    = np.empty_like(self.logfracs)
        all_pcs               = []
        pcas_dict             = {}
        scalers_dict          = {}

        for iz, z_val in enumerate(tqdm(self.z, desc="Preparing PCA/scalers", unit="z")):
            z_key = float(f"{z_val:.3f}")

            scaler = Scaler()
            scaler.fit(self.logfracs[:, iz, :])
            logfracs_norm_z = scaler.transform(self.logfracs[:, iz, :])

            pca = PCA(n_components=num_pcs)
            pca.fit(logfracs_norm_z)
            pca_vals = pca.transform(logfracs_norm_z)

            self.frac_pks_scalers.append(scaler)
            pcas_dict[z_key]    = pca
            scalers_dict[z_key] = scaler
            all_pcs.append(pca_vals)
            self.logfracs_norm[:, iz, :] = logfracs_norm_z

        self.pcas = [pcas_dict[float(f"{z:.3f}")] for z in self.z]

        self.all_pcs = np.transpose(np.array(all_pcs), (1, 0, 2))
        pcs_flat     = self.all_pcs.reshape(len(self.lhs), len(self.z) * num_pcs)

        # # --- tPCA ---
        # pca_z = PCA(n_components=num_pcs_z)
        # pca_z.fit(pcs_flat)
        # self.tpca         = pca_z
        # self.t_components = pca_z.transform(pcs_flat)

        # --- Multi-head targets: shape (N_cosmo, N_z * num_pcs) ---
        # Store all PCA coefficients as a flat array, one row per cosmology.
        # No tPCA compression — the shared backbone handles the redshift correlation.
        self.all_pcs_flat = pcs_flat   # (N_cosmo, N_z * num_pcs) — already computed above

        if multihead:
            # No tPCA: targets are the raw stacked PCA coefficients
            self.tpca         = None
            self.t_components = pcs_flat   # (N_cosmo, N_z * num_pcs)
        else:
            pca_z = PCA(n_components=num_pcs_z)
            pca_z.fit(pcs_flat)
            self.tpca         = pca_z
            self.t_components = pca_z.transform(pcs_flat)

        # Normalise per-column so all heads have equal gradient scale
        self.t_comp_scaler     = TComponentScaler().fit(self.t_components)
        self.t_components_norm = self.t_comp_scaler.transform(self.t_components)

        # Keep tpca = None so COLAModel.predict_pcs_from_t knows which path to take
        # self.tpca         = None
        # self.t_components = self.all_pcs_flat

        # # --- UMAP ---
        # kpca = KernelPCA(
        #     n_components=num_pcs_z,
        #     kernel="rbf",          # or "poly"
        #     gamma=0.01,            # tune this
        #     fit_inverse_transform=True,   # trains the inverse automatically
        #     random_state=42,
        # )
        # kpca.fit(pcs_flat)
        # self.mapper       = kpca
        # self.map_decomp   = kpca.transform(pcs_flat)

        # --- Normalise t-components to zero mean / unit variance per column ---
        # self.t_comp_scaler     = TComponentScaler().fit(self.t_components)
        # self.t_components_norm = self.t_comp_scaler.transform(self.t_components)

        stds = self.t_components.std(axis=0)
        n_components = self.t_components.shape[1]
        label        = "raw PCA coefficients (multihead)" if multihead else "tPCA components"
        print(f"\n✅ Prepared all cosmologies with {n_components} {label} each.")
        print(f"   t-component std range (raw):  {stds.min():.3f} → {stds.max():.3f}  "
              f"(ratio = {stds.max()/stds.min():.1f}x)")
        print(f"   t-component std range (norm): "
              f"{self.t_components_norm.std(axis=0).min():.3f} → "
              f"{self.t_components_norm.std(axis=0).max():.3f}  (should be ~1.0)")

        bundle = _build_metadata_bundle(
            param_scaler  = self.param_scaler,
            t_comp_pca    = self.tpca,   # None for multihead
            t_comp_scaler = self.t_comp_scaler,
            pcas_dict     = pcas_dict,
            scalers_dict  = scalers_dict,
            npce_indices  = None,
            model_type    = "multihead" if multihead else "mlp",
        )
        output_path = os.path.join(metadata_subdir, "metadata.joblib")
        save_metadata_bundle(bundle, output_path)
        self._metadata_bundle_path = output_path
        print(f"   Metadata saved to: {output_path}")

    # -------------------------------------------------------
    @classmethod
    def from_metadata_bundle(cls, bundle_path: str) -> "COLASet":
        """Reconstruct a minimal COLASet from a saved metadata bundle."""
        bundle = load_metadata_bundle(bundle_path)

        obj = cls.__new__(cls)

        obj.param_scaler   = bundle["param_scaler"]
        obj.tpca           = bundle["t_comp_pca"]
        obj.t_comp_scaler  = bundle["t_comp_scaler"]

        z_keys = sorted(bundle["pcas"].keys())
        obj.z                = np.array(z_keys)
        obj.pcas             = [bundle["pcas"][z]    for z in z_keys]
        obj.num_pcs          = obj.pcas[0].n_components_
        obj.frac_pks_scalers = [bundle["scalers"][z] for z in z_keys]

        obj._pcas_dict    = bundle["pcas"]
        obj._scalers_dict = bundle["scalers"]

        obj.npce_indices = bundle.get("npce_indices", None)

        obj.lhs_norm           = None
        obj.t_components_norm  = None
        obj._metadata_bundle_path = bundle_path

        return obj


# ----------------------------------------------------------------------------------------------------
class COLAModel:
    def __init__(self, trainSet):
        self.param_scaler      = trainSet.param_scaler
        self.frac_pks_scalers  = trainSet.frac_pks_scalers
        self.pcas              = trainSet.pcas
        self.z_vals            = trainSet.z
        self.t_comp_scaler     = trainSet.t_comp_scaler
        self.tpca              = trainSet.tpca
        self.use_boost         = trainSet.use_boost
        self.models            = {}

    def fit(self, trainSet, num_epochs):
        raise NotImplementedError("ERROR: COLAModel must override `fit` method")

    def predict_t_components(self, x):
        raise NotImplementedError("ERROR: COLAModel must override `predict_t_components` method")

    def predict_pcs_from_t(self, t_components_raw, z_idx):
        if self.tpca is None:
            raise NotImplementedError(
                "predict_pcs_from_t called on base COLAModel with tpca=None. "
                "Multihead subclass should have overridden this method."
            )
        pcs_flat = self.tpca.inverse_transform(t_components_raw)
        num_pcs  = self.pcas[z_idx].n_components_
        n_z      = len(self.z_vals)
        pcs_all  = pcs_flat.reshape(-1, n_z, num_pcs)
        return pcs_all[:, z_idx, :]

    def predict_logfrac(self, x, z_idx):
        t_raw        = self.predict_t_components(x)
        pcs_z        = self.predict_pcs_from_t(t_raw, z_idx)
        logfrac_norm = self.pcas[z_idx].inverse_transform(pcs_z)
        return self.frac_pks_scalers[z_idx].inverse_transform(logfrac_norm)

    def predict(self, x, z_idx):
        """
        Return frac_pk for the given z index on the full ks grid.

        In boost mode the emulator predicts frac_pk = P_nl_camb / P_nl_syren.
        The caller is responsible for multiplying by the syren halofit baseline
        to recover the full nonlinear P(k).  The output is always the same
        length as ks (no truncation is applied).
        """
        frac_truncated = np.exp(self.predict_logfrac(x, z_idx))
        # No k-padding needed: the full k grid is used in both linear and boost modes.
        return frac_truncated

    def plot_errors(self, testSet, z_idx):
        preds   = self.predict(testSet.lhs, z_idx)
        targets = np.exp(testSet.logfracs[:, z_idx, :])
        fig, ax = plt.subplots()
        for pred, target in zip(preds, targets):
            ax.semilogx(ks, pred / target - 1)
        ax.fill_between(ks, -0.0025, 0.0025, color="gray", alpha=0.75)
        ax.fill_between(ks, -0.005,  0.005,  color="gray", alpha=0.5)
        ax.set_xlabel("k")
        ax.set_ylabel(f"Emulation Error (z={testSet.z[z_idx]:.3f})")
        return fig, ax

    def get_outliers(self, testSet, z_idx, log=False):
        preds         = self.predict(testSet.lhs, z_idx)
        frac_pks_test = np.exp(testSet.logfracs[:, z_idx, :])
        cosmos, boosts = [], []
        for pred, target, cosmo in zip(preds, frac_pks_test, testSet.lhs):
            error = np.abs(pred / target - 1)
            if np.any(error > 0.005):
                if log:
                    h, Ob, Om, As, ns, w0, w0wa = cosmo
                    print(f"Outlier (z={testSet.z[z_idx]:.3f}): "
                        f"h={h:.3f}, Ob={Ob:.3f}, Om={Om:.3f}, As={As:.2e}, ns={ns:.3f}, "
                        f"w0={w0:.3f}, w0+wa={w0wa:.3f} (max error={np.max(error):.4f})")
                cosmos.append(cosmo)
                boosts.append(target)
        return cosmos, boosts

    def save(self, path):
        raise DeprecationWarning("Pickling models is not advised.")

# ----------------------------------------------------------------------------------------------------
class COLA_MultiHead_Keras(COLAModel):
    """
    Shared-backbone + per-redshift-head emulator.

    Architecture:
        7 params → shared MLP backbone (width=1024, depth=num_layers)
                 → N_z heads, each outputting NUM_PCS PCA coefficients

    The backbone learns the cosmology-dependent latent representation once;
    each head learns how to decode that into PCA coefficients at its redshift.
    This avoids the tPCA bottleneck entirely — the network predicts all
    N_z * NUM_PCS coefficients but shares parameters across redshifts.
    """
    def __init__(self, trainSet, num_layers=3, num_neurons=1024):
        super().__init__(trainSet)
        self.num_layers  = num_layers
        self.num_neurons = num_neurons
        self.trainSet    = trainSet
        self.models      = {}
        self.n_z         = len(trainSet.z)
        self.num_pcs     = trainSet.num_pcs

    def build_model(self):
        """
        Shared backbone with N_z output heads.
        Each head predicts NUM_PCS normalised PCA coefficients for one redshift.
        All heads are trained jointly under a single MSE loss on the flattened
        (N_z * NUM_PCS) output vector — identical interface to the tPCA model.
        """
        input_shape  = self.trainSet.lhs_norm.shape[1]     # 7
        output_shape = self.n_z * self.num_pcs              # e.g. 34 * 20 = 680

        # --- Shared backbone ---
        inputs = layers.Input(shape=(input_shape,))
        x = layers.Dense(self.num_neurons)(inputs)
        x = CustomActivationLayer(self.num_neurons)(x)

        for _ in range(self.num_layers - 1):
            x = layers.Dense(self.num_neurons)(x)
            x = CustomActivationLayer(self.num_neurons)(x)

        backbone_out = x   # shared representation, shape (batch, num_neurons)

        # --- Per-redshift heads ---
        # Each head is a single linear layer on top of the shared backbone.
        # You can make heads deeper if needed by adding more layers here.
        head_outputs = []
        for iz in range(self.n_z):
            head = layers.Dense(
                self.num_pcs,
                name=f"head_z{iz}",
                dtype='float32'
            )(backbone_out)
            head_outputs.append(head)

        # Concatenate all heads → (batch, N_z * NUM_PCS)
        if self.n_z > 1:
            outputs = layers.Concatenate(axis=-1)(head_outputs)
        else:
            outputs = head_outputs[0]

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model

    def fit_multihead(
        self,
        trainSet,
        num_epochs,
        batch_size=512,
        initial_lr=1e-3,
        final_lr=1e-5,
        huber_delta=1.0,
        trim_top_frac=0.0,      # <-- add this
    ):
        mlp = self.build_model()

        n_train         = len(trainSet.lhs_norm)
        steps_per_epoch = max(1, n_train // batch_size)
        total_steps     = num_epochs * steps_per_epoch

        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            alpha=final_lr / initial_lr,
        )

        loss_fn = make_trimmed_huber_loss(trim_top_frac, huber_delta)

        print(
            f"\n[fit_multihead] Training config:\n"
            f"  input           = {trainSet.lhs_norm.shape[1]} params\n"
            f"  output          = {self.n_z} heads × {self.num_pcs} PCs "
            f"= {self.n_z * self.num_pcs} total\n"
            f"  batch_size      = {batch_size} "
            f"({steps_per_epoch} steps/epoch, {total_steps} total steps)\n"
            f"  LR schedule     = CosineDecay {initial_lr:.2e} → {final_lr:.2e}\n"
            f"  loss            = {'trimmed Huber' if trim_top_frac > 0 else 'Huber'}"
            f"(delta={huber_delta}"
            f"{f', trim={trim_top_frac}' if trim_top_frac > 0 else ''})\n"
            f"  epochs          = {num_epochs}"
        )

        mlp.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=loss_fn,
        )

        nn_model_train_keras(
            mlp,
            epochs=num_epochs,
            input_data=trainSet.lhs_norm,
            truths=trainSet.t_components_norm,
            batch_size=batch_size,
        )
        self.models["t-component"] = mlp
        return mlp

    def predict_t_components(self, x):
        """
        Run the network and invert the per-column normalisation.
        Returns raw (un-normalised) PCA coefficients, shape (N_cosmo, N_z * NUM_PCS).
        """
        mlp = self.models["t-component"]
        if x.max() > 1.5 or x.min() < -1.5:
            x = self.param_scaler.transform(x)
        t_norm = mlp.predict(x, verbose=0)
        return self.t_comp_scaler.inverse_transform(t_norm)

    def predict_pcs_from_t(self, t_components_raw, z_idx):
        """
        Extract PCA coefficients for a single redshift from the flat output.
        Overrides COLAModel.predict_pcs_from_t — no tPCA inverse needed.
        """
        pcs_all = t_components_raw.reshape(-1, self.n_z, self.num_pcs)
        return pcs_all[:, z_idx, :]

# ----------------------------------------------------------------------------------------------------
class COLA_MultiHead_NPCE_Keras(COLA_MultiHead_Keras):
    """
    Multi-headed emulator with a Neural PCE shared backbone.

    The PCE layer expands the 7 cosmological parameters into a polynomial
    basis before passing them to the shared dense backbone, giving the
    network explicit polynomial interactions between parameters without
    needing to learn them from scratch.  The per-redshift heads are
    identical to COLA_MultiHead_Keras.
    """
    def __init__(
        self,
        trainSet,
        max_degree=6,
        norm_q=0.75,
        norm_threshold=1.0,
        num_layers=3,
        num_neurons=1024,
    ):
        super().__init__(trainSet, num_layers=num_layers, num_neurons=num_neurons)
        self.max_degree     = max_degree
        self.norm_q         = norm_q
        self.norm_threshold = norm_threshold

        # Build the PCE multi-index set (same logic as COLA_NPCE_Keras)
        n_params = trainSet.lhs_norm.shape[1]   # 7
        self._indices = self._build_pce_indices(n_params)
        print(
            f"[NPCE backbone] {len(self._indices)} PCE basis terms "
            f"(max_degree={max_degree}, q={norm_q}, threshold={norm_threshold})"
        )

    def _build_pce_indices(self, n_params):
        """
        Build the hyperbolic-cross truncated multi-index set.
        Mirrors COLA_NPCE_Keras._build_indices — extract to a shared
        utility function if you want to avoid duplication.
        """
        from itertools import product as iproduct
        indices = []
        for idx in iproduct(range(self.max_degree + 1), repeat=n_params):
            lq_norm = sum(i ** self.norm_q for i in idx)
            if lq_norm <= self.norm_threshold:
                indices.append(idx)
        return np.array(indices, dtype=np.int32)

    def build_model(self):
        """
        PCE feature expansion → shared backbone → per-redshift heads.
        """
        input_shape  = self.trainSet.lhs_norm.shape[1]   # 7
        n_pce_terms  = len(self._indices)
        output_shape = self.n_z * self.num_pcs

        # --- Input ---
        inputs = layers.Input(shape=(input_shape,), name="cosmo_params")

        # --- PCE feature expansion (same as COLA_NPCE_Keras) ---
        # For each multi-index α, compute ∏_i x_i^α_i
        pce_features = self._pce_layer(inputs)   # (batch, n_pce_terms)

        # --- Shared backbone ---
        x = layers.Dense(self.num_neurons, name="backbone_dense_0")(pce_features)
        x = CustomActivationLayer(self.num_neurons, name="backbone_act_0")(x)

        for i in range(self.num_layers - 1):
            x = layers.Dense(self.num_neurons, name=f"backbone_dense_{i+1}")(x)
            x = CustomActivationLayer(self.num_neurons, name=f"backbone_act_{i+1}")(x)

        # --- Per-redshift heads ---
        head_outputs = []
        for iz in range(self.n_z):
            head = layers.Dense(self.num_pcs, name=f"head_z{iz}")(x)
            head_outputs.append(head)

        outputs = layers.Concatenate(axis=-1)(head_outputs) if self.n_z > 1 else head_outputs[0]

        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model

    def _pce_layer(self, inputs):
        """
        Compute polynomial chaos basis features from normalised inputs.
        Each basis term is a monomial ∏_i x_i^α_i for multi-index α.
        Uses TensorFlow ops so it sits inside the Keras graph.
        """
        terms = []
        for idx in self._indices:
            term = tf.ones_like(inputs[:, 0:1])   # (batch, 1)
            for dim, power in enumerate(idx):
                if power > 0:
                    term = term * tf.pow(inputs[:, dim:dim+1], float(power))
            terms.append(term)
        return tf.concat(terms, axis=1)   # (batch, n_pce_terms)

# ----------------------------------------------------------------------------------------------------
class COLA_NN_Keras(COLAModel):
    def __init__(self, trainSet, num_layers=3, num_neurons=1024):
        super().__init__(trainSet)
        self.num_layers  = num_layers
        self.num_neurons = num_neurons
        self.trainSet    = trainSet
        self.models      = {}

    def build_model_for_t_comps(self):
        input_shape  = self.trainSet.lhs_norm.shape[1]
        output_shape = self.trainSet.t_components_norm.shape[1]
        return generate_mlp(
            input_shape=input_shape,
            output_shape=output_shape,
            num_layers=self.num_layers,
            num_neurons=self.num_neurons,
            activation="custom",
            alpha=0,
            l1_ratio=0
        )

    def fit_t_componets(
        self,
        trainSet,
        num_epochs,
        batch_size=512,
        initial_lr=1e-3,
        final_lr=1e-5,
        huber_delta=1.0,
        trim_top_frac=0.0,
        decayevery=None,
        decayrate=None,
    ):
        mlp = self.build_model_for_t_comps()

        n_train     = len(trainSet.lhs_norm)
        steps_per_epoch = max(1, n_train // batch_size)
        total_steps     = num_epochs * steps_per_epoch

        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=total_steps,
            alpha=final_lr / initial_lr,
        )

        print(
            f"\n[fit_t_componets] Training config:\n"
            f"  batch_size      = {batch_size}  "
            f"({steps_per_epoch} steps/epoch, {total_steps} total steps)\n"
            f"  LR schedule     = CosineDecay  "
            f"{initial_lr:.2e} → {final_lr:.2e}\n"
            f"  loss            = {'trimmed MSE' if trim_top_frac > 0 else 'Huber'}\n"
            f"  epochs          = {num_epochs}"
        )

        mlp.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
            loss=keras.losses.Huber(delta=huber_delta),
        )

        nn_model_train_keras(
            mlp,
            epochs=num_epochs,
            input_data=trainSet.lhs_norm,
            truths=trainSet.t_components_norm,
            batch_size=batch_size,
        )
        self.models["t-component"] = mlp
        return mlp

    def predict_t_components(self, x):
        mlp = self.models["t-component"]
        if x.max() > 1.5 or x.min() < -1.5:
            x = self.param_scaler.transform(x)
        t_norm = mlp.predict(x, verbose=0)
        return self.t_comp_scaler.inverse_transform(t_norm)


# ----------------------------------------------------------------------------------------------------
class Scaler:
    """Standard-score normaliser (mean/std) for logfrac slices."""
    def __init__(self):
        self.mean = None
        self.std  = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std  = np.std(X, axis=0)
        self.std =  np.where(self.std == 0, 1.0, self.std)  # guard zero-std columns

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return (X * self.std) + self.mean


# ----------------------------------------------------------------------------------------------------
class CustomActivationLayer(layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomActivationLayer, self).__init__(**kwargs)
        self.units = units
        self.input_spec = layers.InputSpec(min_ndim=2)

    def build(self, input_shape):
        self.beta  = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="beta")
        self.gamma = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="gamma")
        super(CustomActivationLayer, self).build(input_shape)

    def call(self, x):
        # Cast weights to match the input's compute dtype (handles mixed precision).
        # self.beta/self.gamma are stored as float32 variables but must be cast
        # to whatever dtype `x` arrives in (bfloat16 under mixed_bfloat16 policy).
        beta  = tf.cast(self.beta, x.dtype)
        gamma = tf.cast(self.gamma, x.dtype)
        func = tf.add(
            gamma,
            tf.multiply(tf.sigmoid(tf.multiply(beta, x)), tf.subtract(tf.cast(1.0, x.dtype), gamma))
        )
        return tf.multiply(func, x)

    def get_config(self):
        config = super(CustomActivationLayer, self).get_config()
        config.update({'units': self.units})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


def generate_mlp(input_shape, output_shape, num_layers, num_neurons,
                 activation="custom", alpha=0.01, l1_ratio=0.01,
                 learning_rate=1e-3, optimizer='adam', loss='mse'):
    """Generates an MLP model."""
    reg = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio)) if alpha != 0 else None

    def apply_activation(x):
        if activation == "custom":
            return CustomActivationLayer(num_neurons)(x)
        elif activation == "relu":
            return keras.activations.relu(x)
        elif activation == "sigmoid":
            return keras.activations.sigmoid(x)
        else:
            raise ValueError(f"Unexpected activation '{activation}'")

    inputs = layers.Input(shape=(input_shape,))
    x = layers.Dense(num_neurons, kernel_regularizer=reg)(inputs)
    x = apply_activation(x)

    for _ in range(num_layers - 1):
        x = layers.Dense(num_neurons, kernel_regularizer=reg)(x)
        x = apply_activation(x)

    outputs = layers.Dense(output_shape, dtype='float32')(x)

    if optimizer.lower() == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer.lower() == 'sgd':
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.99, nesterov=True)
    else:
        raise ValueError(f"Unhandled optimizer: {optimizer}")

    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    model.compile(optimizer=opt, loss=loss)
    return model


# ----------------------------------------------------------------------------------------------------
def generate_resnet(input_shape, output_shape, num_res_blocks=1, num_of_neurons=512,
                    activation="relu", alpha=1e-5, l1_ratio=0.1, dropout=0.1):
    """Generates a ResNet model with `num_res_blocks` residual blocks."""
    reg = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio))
    input_layer = layers.Input(shape=input_shape)

    hid1 = layers.Dense(units=num_of_neurons, kernel_regularizer=reg, bias_regularizer=reg)(input_layer)
    act1 = CustomActivationLayer(num_of_neurons)(hid1)
    hid2 = layers.Dense(units=num_of_neurons, kernel_regularizer=reg, bias_regularizer=reg)(act1)
    act2 = CustomActivationLayer(num_of_neurons)(hid2)
    residual = layers.Add()([act1, act2])

    for _ in range(num_res_blocks - 1):
        hid1 = layers.Dense(units=num_of_neurons, kernel_regularizer=reg, bias_regularizer=reg)(residual)
        act1 = CustomActivationLayer(num_of_neurons)(hid1)
        hid2 = layers.Dense(units=num_of_neurons, kernel_regularizer=reg, bias_regularizer=reg)(act1)
        act2 = CustomActivationLayer(num_of_neurons)(hid2)
        residual = layers.Add()([act1, act2])

    output_layer = layers.Dense(units=output_shape)(residual)
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanAbsoluteError())
    return model


# ----------------------------------------------------------------------------------------------------
def nn_model_train_keras(
    model,
    epochs,
    input_data,
    truths,
    batch_size=512,
    validation_features=None,
    validation_truths=None,
    decayevery=None,
    decayrate=None,
):
    # Build a tf.data pipeline that prefetches to GPU
    dataset = (
        tf.data.Dataset.from_tensor_slices((
            input_data.astype(np.float32),
            truths.astype(np.float32),
        ))
        .shuffle(buffer_size=min(len(input_data), 50_000), reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=True)   # drop_remainder avoids retracing
        .prefetch(tf.data.AUTOTUNE)                # overlap CPU prep with GPU compute
    )

    callbacks = []
    # callbacks.append(TimingCallback())
    if decayevery is not None and decayrate is not None:
        def scheduler(epoch, lr):
            return lr / decayrate if (epoch != 0 and epoch % decayevery == 0) else lr
        callbacks.append(keras.callbacks.LearningRateScheduler(scheduler))

    # Verbose=2 gives one line per epoch without the slow progress bar
    history = model.fit(dataset, epochs=epochs, callbacks=callbacks, verbose=2)
    return history.history['loss'][-1]
