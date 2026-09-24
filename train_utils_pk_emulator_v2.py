# # Auxiliary functions for Power Spectrum Emulation
# # Author: João Victor Silva Rebouças (2022), updated by Victoria Lloyd (2025)
# import math
# import pickle
# from itertools import product
# import os
# import joblib
# import logging

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import CubicSpline
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error

# import tensorflow as tf
# from tensorflow import keras
# from keras import models, layers, optimizers
# from keras.regularizers import l1_l2
# from tqdm import tqdm

# import sys; sys.path.insert(0, "/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); import symbolic_pofk.linear_VL as linear
# import sys; sys.path.insert(0, "/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); from symbolic_pofk.linear_VL import plin_emulated, get_approximate_D, growth_correction_R, get_eisensteinhu_nw

# # ----------------------------------------------------------------------------------------------------
# # Parameter space
# params = ['As', 'ns', 'h', 'Omega_b', 'Omega_m', 'w', 'w0+wa']
# params_latex = [r'$10^9 A_s$', r'$n_s$', r'$H_0$', r'$\Omega_b$', r'$\Omega_m$', r'$w_0$', r'$w_0+w_a$']

# #k=1e-5 to 1e2 (with 2400 steps in log space)
# ks = np.logspace(-5.1, 2, 500)
# z1_mps = np.linspace(0,3,33,endpoint=False)
# z2_mps = np.linspace(3,10,7,endpoint=False)
# z3_mps = np.linspace(10,50,12)
# z_mps = np.concatenate((z1_mps, z2_mps, z3_mps), axis=0)

# BASE_PATH = "/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps/"

# # ----------------------------------------------------------------------------------------------------
# # nl_type registry
# NL_TYPE_REGISTRY = {
#     "lin":                      ("pklin",                           "Linear P(k)"),
#     "halofit":                  ("pknonlin",                        "Non-linear P(k) via HaloFit"),
#     "mead2020":                 ("mead2020_pknonlin",               "Non-linear P(k) via HMcode Mead2020"),
#     "mead2020_feedback":        ("mead2020_feedback_pknonlin",      "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (fixed T_AGN)"),
#     "mead2020_feedback_Tfree":  ("mead2020_feedback_Tfree_pknonlin","Non-linear P(k) via HMcode Mead2020 + baryonic feedback (free T_AGN)"),
# }

# VALID_NL_TYPES = {
#     "expanded":    {"lin", "halofit"},
#     "constrained": set(NL_TYPE_REGISTRY.keys()),
# }

# VER = "_v5"

# def _validate_prior_and_nl_type(prior_type: str, nl_type: str):
#     if prior_type not in VALID_NL_TYPES:
#         raise ValueError(
#             f"Unknown prior_type '{prior_type}'. "
#             f"Must be one of: {sorted(VALID_NL_TYPES.keys())}"
#         )
#     if nl_type not in NL_TYPE_REGISTRY:
#         raise ValueError(
#             f"Unknown nl_type '{nl_type}'. "
#             f"Must be one of: {sorted(NL_TYPE_REGISTRY.keys())}"
#         )
#     if nl_type not in VALID_NL_TYPES[prior_type]:
#         raise ValueError(
#             f"nl_type '{nl_type}' is not available for prior_type '{prior_type}'. "
#             f"Valid nl_types for '{prior_type}': {sorted(VALID_NL_TYPES[prior_type])}"
#         )


# def _make_file_paths(base_path: str, cosmo_type: str, prior_type: str, nl_type: str, batch: int):
#     pk_suffix = NL_TYPE_REGISTRY[nl_type][0]
#     if prior_type == "expanded":
#         input_name  = f"train_{cosmo_type}_mps_ml_{batch}_expanded_uniform.npy"
#         output_name = f"train_{cosmo_type}_{batch}_expanded_uniform_{pk_suffix}.npy"
#     else:
#         input_name  = f"old_prior/train_{cosmo_type}_mps_ml_{batch}_new.npy"
#         output_name = f"train_{cosmo_type}_condensed_{batch}_new_{pk_suffix}.npy"
#     input_path  = os.path.join(base_path, "input",  input_name)
#     output_path = os.path.join(base_path, "output", output_name)
#     return input_path, output_path


# # ----------------------------------------------------------------------------------------------------
# def _compute_mps_approximation(ks, zs, params: np.ndarray, use_eh=False) -> np.ndarray:
#     """
#     Computes the analytical P_lin(k, z) approximation in physical units (Mpc³).

#     Args:
#         params: 1D array [10^9 A_s, ns, H0, Ob, Om, w0, wa].

#     Returns:
#         np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³.
#     """
#     As, ns, H0_in, Ob, Om, w0, w0wa = params
#     wa = w0wa - w0 
#     h = H0_in / 100.0
#     k_for_plin = ks / h
#     if use_eh:
#         pk_fid_hmpc = get_eisensteinhu_nw(k_for_plin, As, Om, Ob, h, ns, mnu=0.06, w0=w0, wa=wa)
#     else:
#         pk_fid_hmpc = plin_emulated(k_for_plin, Om, Ob, h, ns, As=As, w0=w0, wa=wa)
#     a_array = 1.0 / (zs + 1)
#     D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
#     Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
#     R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
#     Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
#     growth_factors = (Dz/D0) * (Dz/D0) * (Rz/R0)
#     result = pk_fid_hmpc[None, :] * growth_factors[:, None]
#     result = result / h**3
#     return result.astype(np.float32)

# # ----------------------------------------------------------------------------------------------------
# def load_set(
#     n_batches=1,
#     start_batch=0,
#     z_indices=None,
#     cosmo_type="w0wacdm",
#     prior_type="constrained",
#     nl_type="lin",
#     base_path=BASE_PATH,
#     check_unphysical=True,
#     verbose=True
# ):
#     """Load cosmological input/output data batches."""
#     _validate_prior_and_nl_type(prior_type, nl_type)
#     inputs_all, outputs_all = [], []

#     for b in tqdm(range(start_batch, start_batch + n_batches), desc="Loading batches", unit="batch"):
#         input_path, output_path = _make_file_paths(base_path, cosmo_type, prior_type, nl_type, b)
#         if not (os.path.exists(input_path) and os.path.exists(output_path)):
#             if verbose:
#                 print(f"⚠️  Skipping missing batch {b}  (prior='{prior_type}', nl_type='{nl_type}')")
#                 print(f"    expected input : {input_path}")
#                 print(f"    expected output: {output_path}")
#             continue

#         x = np.load(input_path,  mmap_mode="r")
#         y = np.load(output_path, mmap_mode="r")
#         if z_indices is not None:
#             y = y[:, z_indices, :]
#         if check_unphysical:
#             bad = np.all((y == 0) | (y == 1), axis=(1, 2))
#             if np.any(bad) and verbose:
#                 print(f"🧹 Removed {np.sum(bad)} unphysical cosmologies from batch {b}")
#             x, y = x[~bad], y[~bad]
#         inputs_all.append(x)
#         outputs_all.append(y)

#     if inputs_all:
#         inputs_all  = np.concatenate(inputs_all,  axis=0)
#         outputs_all = np.concatenate(outputs_all, axis=0)
#     else:
#         inputs_all  = np.empty((0, 5))
#         outputs_all = np.empty((0, 122, 2000))
#     return inputs_all, outputs_all


# # ----------------------------------------------------------------------------------------------------
# class TComponentScaler:
#     """
#     Per-column standardisation for t-components.

#     Stores mean and std computed on the training set so that inference can
#     invert the transform consistently.  Saved to / loaded from disk via joblib
#     alongside the other metadata.

#     Why this matters
#     ----------------
#     The raw tPCA output has a 288x std ratio across the 15 columns (t[0] std ≈ 143,
#     t[14] std ≈ 0.5).  Under MSE loss this means gradients for t[14] are ~83 000×
#     smaller than for t[0], so the network essentially never learns to predict the
#     smaller components accurately.  Standardising to unit variance before training
#     gives every component equal weight in the loss.
#     """
#     def __init__(self):
#         self.mean_ = None
#         self.std_  = None

#     def fit(self, T: np.ndarray) -> "TComponentScaler":
#         self.mean_ = T.mean(axis=0)
#         self.std_  = T.std(axis=0)
#         self.std_  = np.where(self.std_ == 0, 1.0, self.std_)  # guard zero-std columns
#         return self

#     def transform(self, T: np.ndarray) -> np.ndarray:
#         return (T - self.mean_) / self.std_

#     def inverse_transform(self, T_norm: np.ndarray) -> np.ndarray:
#         return T_norm * self.std_ + self.mean_


# # ----------------------------------------------------------------------------------------------------
# # Metadata bundle helpers
# # ----------------------------------------------------------------------------------------------------

# def _build_metadata_bundle(
#     param_scaler,
#     t_comp_pca,
#     t_comp_scaler,
#     pcas_dict: dict,
#     scalers_dict: dict,
#     npce_indices=None,
#     model_type: str = "mlp",
# ) -> dict:
#     """
#     Assemble the canonical metadata bundle that is written to / read from disk.

#     Parameters
#     ----------
#     pcas_dict    : {float(z): PCA object}  — keyed by z value rounded to 3 dp
#     scalers_dict : {float(z): Scaler}      — same keys as pcas_dict
#     npce_indices : (N_terms, n_dims) int ndarray or None
#                    PCE multi-index matrix.  None for MLP models; populated
#                    after training for NPCE models via update_metadata_bundle_npce().
#     model_type   : 'mlp' or 'npce'
#     """
#     return {
#         "param_scaler":  param_scaler,
#         "t_comp_pca":    t_comp_pca,
#         "t_comp_scaler": t_comp_scaler,
#         "pcas":          pcas_dict,
#         "scalers":       scalers_dict,
#         "npce_indices":  npce_indices,
#         "model_type":    model_type,
#     }


# def save_metadata_bundle(bundle: dict, path: str, compress: int = 3) -> None:
#     """Save a metadata bundle to a single joblib file."""
#     joblib.dump(bundle, path, compress=compress)
#     logging.info(
#         f"Metadata bundle saved to {path}  "
#         f"(model_type={bundle['model_type']}, "
#         f"n_z={len(bundle['pcas'])}, "
#         f"t_comp_scaler={'present' if bundle['t_comp_scaler'] is not None else 'None'}, "
#         f"npce_indices={'present' if bundle['npce_indices'] is not None else 'None'})"
#     )


# def load_metadata_bundle(path: str) -> dict:
#     """
#     Load a metadata bundle from disk and return it as a plain dict.

#     Compatible with bundles produced both by this module and by the legacy
#     repackage_metadata.py script — the key names are identical.
#     """
#     bundle = joblib.load(path)
#     logging.info(
#         f"Metadata bundle loaded from {path}  "
#         f"(model_type={bundle.get('model_type', 'unknown')}, "
#         f"n_z={len(bundle.get('pcas', {}))})"
#     )
#     return bundle


# def update_metadata_bundle_npce(bundle_path: str, npce_indices: np.ndarray) -> None:
#     """
#     Patch an existing metadata bundle with NPCE-specific fields.

#     Called by train.py after NPCE training completes, so that the bundle
#     written during prepare() (which has npce_indices=None, model_type='mlp')
#     is updated to reflect the actual model type.

#     Parameters
#     ----------
#     bundle_path  : str        — path to the metadata.joblib to update in-place
#     npce_indices : np.ndarray — (N_terms, n_dims) PCE multi-index matrix
#                                 from COLA_NPCE_Keras._indices
#     """
#     bundle = load_metadata_bundle(bundle_path)
#     bundle["npce_indices"] = npce_indices
#     bundle["model_type"]   = "npce"
#     save_metadata_bundle(bundle, bundle_path)
#     print(
#         f"   Bundle updated: model_type='npce', "
#         f"npce_indices shape={npce_indices.shape}  →  {bundle_path}"
#     )


# # ----------------------------------------------------------------------------------------------------
# class COLASet:
#     def __init__(
#         self,
#         path=BASE_PATH,
#         target_z=None,
#         cosmo_type="w0wacdm",
#         prior_type="constrained",
#         nl_type="lin",
#         n_batches=1,
#         start_batch=0,
#         verbose=True
#     ):
#         _validate_prior_and_nl_type(prior_type, nl_type)
#         self.n_batches  = n_batches
#         self.cosmo_type = cosmo_type
#         self.prior_type = prior_type
#         self.nl_type    = nl_type

#         # True for any nonlinear nl_type: emulate NL/LIN boost instead of LIN/syren
#         self.use_boost = (nl_type != "lin")

#         if target_z is not None:
#             target_z_array = np.atleast_1d(target_z).astype(float)
#             diffs     = np.abs(target_z_array[:, None] - z_mps[None, :])
#             z_indices = np.argmin(diffs, axis=1).tolist()
#             self.z    = target_z_array
#         else:
#             z_indices = None
#             self.z    = np.array(z_mps)

#         # ------------------------------------------------------------------
#         # Load primary (target) dataset
#         # ------------------------------------------------------------------
#         self.lhs, self.pks_target = load_set(
#             base_path   = path,
#             z_indices   = z_indices,
#             cosmo_type  = cosmo_type,
#             prior_type  = prior_type,
#             nl_type     = nl_type,
#             n_batches   = n_batches,
#             start_batch = start_batch,
#             verbose     = verbose,
#         )

#         self.lhs = self.lhs.copy()
#         self.lhs[:, 6] = self.lhs[:, 5] + self.lhs[:, 6]   # store w0+wa in col 6

#         print(f"pks_target shape = {self.pks_target.shape}")
#         self.ks = ks

#         if isinstance(self.z, (list, np.ndarray)) and len(np.shape(self.z)) > 1:
#             self.z = np.ravel(self.z).tolist()

#         # ------------------------------------------------------------------
#         # Compute frac_pks  (the quantity the emulator actually learns)
#         #
#         #   use_boost = False  (nl_type == "lin"):
#         #       frac_pks = P_lin / P_syren   — ratio to the symbolic approximation
#         #
#         #   use_boost = True   (nl_type != "lin"):
#         #       frac_pks = P_nonlin / P_lin  — nonlinear boost
#         #       We load the matching linear dataset to form the denominator.
#         # ------------------------------------------------------------------
#         if self.use_boost:
#             # Load the linear counterpart (same prior/cosmo type, same batches)
#             lhs_lin, self.pks_lin = load_set(
#                 base_path   = path,
#                 z_indices   = z_indices,
#                 cosmo_type  = cosmo_type,
#                 prior_type  = prior_type,
#                 nl_type     = "lin",
#                 n_batches   = n_batches,
#                 start_batch = start_batch,
#                 verbose     = verbose,
#             )
#             print(f"pks_lin shape  = {self.pks_lin.shape}")

#             # Align rows: keep only cosmologies present in BOTH datasets.
#             # Both sets are drawn from the same LHS so should be identical in
#             # practice, but batch skips or unphysical-row removal can cause
#             # a mismatch — guard against it explicitly.
#             lin_lhs_w0wa = lhs_lin.copy()
#             lin_lhs_w0wa[:, 6] = lin_lhs_w0wa[:, 5] + lin_lhs_w0wa[:, 6]
#             nl_keys  = set(map(tuple, self.lhs.round(8)))
#             lin_keys = set(map(tuple, lin_lhs_w0wa.round(8)))
#             common   = nl_keys & lin_keys
#             if len(common) < len(self.lhs):
#                 print(
#                     f"⚠️  Row-alignment: {len(self.lhs) - len(common)} NL cosmologies "
#                     f"have no matching LIN row — dropping them."
#                 )
#             nl_mask  = np.array([tuple(r.round(8)) in common for r in self.lhs])
#             lin_mask = np.array([tuple(r.round(8)) in common for r in lin_lhs_w0wa])
#             self.lhs        = self.lhs[nl_mask]
#             self.pks_target = self.pks_target[nl_mask]
#             self.pks_lin    = self.pks_lin[lin_mask]

#             self.mps_approxes = None   # not used in boost mode

#             print(f"Computing NL/LIN boost (frac_pks = P_{nl_type} / P_lin)")
#             with np.errstate(divide="ignore", invalid="ignore"):
#                 self.frac_pks = self.pks_target / self.pks_lin

#         else:
#             # Original path: emulate P_lin / P_syren
#             self.pks_lin = self.pks_target   # alias for back-compat

#             self.mps_approxes = np.array([
#                 _compute_mps_approximation(self.ks, self.z, p)[0]
#                 if len(self.z) == 1 else
#                 _compute_mps_approximation(self.ks, self.z, p)
#                 for p in self.lhs
#             ])
#             print(f"mps_approxes shape = {self.mps_approxes.shape}")
#             print("Computing LIN/syren ratio (frac_pks = P_lin / P_syren)")

#             with np.errstate(divide="ignore", invalid="ignore"):
#                 self.frac_pks = self.pks_lin / self.mps_approxes

#         self.logfracs = np.log(self.frac_pks)

#         # ------------------------------------------------------------------
#         # Remove unphysical / non-finite cosmologies
#         # ------------------------------------------------------------------
#         all_zero_mask   = (self.pks_target == 0).all(axis=(1, 2))
#         non_finite_mask = ~np.isfinite(self.logfracs).all(axis=(1, 2))
#         bad_cosmo_mask  = all_zero_mask | non_finite_mask

#         print(f"⚠️  Found {bad_cosmo_mask.sum()} / {len(bad_cosmo_mask)} cosmologies with issues:")
#         print(f"   - {all_zero_mask.sum()} with all-zero P (unphysical)")
#         print(f"   - {non_finite_mask.sum()} with non-finite log(frac_pks)")

#         if bad_cosmo_mask.sum() > 0:
#             bad_indices = np.where(bad_cosmo_mask)[0]
#             print("First 10 bad cosmology indices:", bad_indices[:10])
#             print("Example bad LHS:", self.lhs[bad_indices[0]])

#         good_mask         = ~bad_cosmo_mask
#         self.lhs          = self.lhs[good_mask]
#         self.pks_target   = self.pks_target[good_mask]
#         self.frac_pks     = self.frac_pks[good_mask]
#         self.logfracs     = self.logfracs[good_mask]
#         if self.use_boost:
#             self.pks_lin      = self.pks_lin[good_mask]
#         else:
#             self.mps_approxes = self.mps_approxes[good_mask]
#             self.pks_lin      = self.pks_lin[good_mask]   # alias stays in sync

#         self.w0_min   = None
#         self.w0wa_max = None

#         # Populated by prepare(); used by train.py to patch the bundle after
#         # NPCE training without needing to reconstruct the path externally.
#         self._metadata_bundle_path = None

#     # -------------------------------------------------------
#     def _metadata_tag(self):
#         w0_tag   = f"_w0min{self.w0_min}"     if self.w0_min   is not None else ""
#         w0wa_tag = f"_w0wamax{self.w0wa_max}" if self.w0wa_max is not None else ""
#         mode_tag = "_boost" if self.use_boost else ""
#         return (
#             f"{self.cosmo_type}_{self.prior_type}{w0_tag}{w0wa_tag}"
#             f"_{self.nl_type}{mode_tag}_nTrain{self.n_batches}{VER}"
#         )

#     # -------------------------------------------------------
#     def change_ks(self, ks):
#         new_frac_pks = np.array([
#             [CubicSpline(self.ks, self.frac_pks[i, j, :])(ks)
#              for j in range(len(self.z))]
#             for i in range(len(self.lhs))
#         ])
#         self.frac_pks = new_frac_pks
#         self.logfracs = np.log(self.frac_pks)
#         self.ks = ks
#         if hasattr(self, "num_pcs") and self.num_pcs is not None:
#             self.prepare(self.num_pcs)

#     # -------------------------------------------------------
#     def update(self, cosmos, frac_pks):
#         frac_pks = np.atleast_3d(frac_pks)
#         self.lhs      = np.vstack([self.lhs, cosmos])
#         self.frac_pks = np.vstack([self.frac_pks, frac_pks])
#         self.logfracs = np.log(self.frac_pks)
#         if hasattr(self, "num_pcs") and self.num_pcs is not None:
#             self.prepare(self.num_pcs)

#     # -------------------------------------------------------
#     def prepare(self, num_pcs, num_pcs_z, metadata_dir="mps_emu/metadata"):
#         """
#         Fit scalers and PCA for each redshift, then run tPCA and normalise
#         the resulting t-components to zero mean / unit variance per column.

#         All metadata is saved as a single 'metadata.joblib' bundle.  For MLP
#         models this bundle is complete.  For NPCE models, train.py calls
#         update_metadata_bundle_npce(train_set._metadata_bundle_path, indices)
#         after training to patch in the PCE multi-index matrix and set
#         model_type='npce'.  The bundle path is stored on self for convenience.

#         The bundle format is identical to the one produced by the legacy
#         repackage_metadata.py script, so existing deployed bundles remain
#         fully compatible.
#         """
#         self.num_pcs = num_pcs
#         metadata_subdir = os.path.join(metadata_dir, f"metadata_{self._metadata_tag()}")
#         os.makedirs(metadata_subdir, exist_ok=True)

#         # --- Cosmological parameter scaler ---
#         self.param_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(self.lhs)
#         self.lhs_norm     = self.param_scaler.transform(self.lhs)

#         # --- Per-z logfrac scalers + PCA ---
#         self.frac_pks_scalers = []
#         self.logfracs_norm    = np.empty_like(self.logfracs)
#         all_pcs               = []
#         pcas_dict             = {}   # {float(z): PCA}   — for the bundle
#         scalers_dict          = {}   # {float(z): Scaler} — for the bundle

#         for iz, z_val in enumerate(tqdm(self.z, desc="Preparing PCA/scalers", unit="z")):
#             z_key = float(f"{z_val:.3f}")

#             scaler = Scaler()
#             scaler.fit(self.logfracs[:, iz, :])
#             logfracs_norm_z = scaler.transform(self.logfracs[:, iz, :])

#             pca = PCA(n_components=num_pcs)
#             pca.fit(logfracs_norm_z)
#             pca_vals = pca.transform(logfracs_norm_z)

#             self.frac_pks_scalers.append(scaler)
#             pcas_dict[z_key]    = pca
#             scalers_dict[z_key] = scaler
#             all_pcs.append(pca_vals)
#             self.logfracs_norm[:, iz, :] = logfracs_norm_z

#         # Keep self.pcas as an ordered list (indexed by iz) for COLAModel
#         self.pcas = [pcas_dict[float(f"{z:.3f}")] for z in self.z]

#         self.all_pcs = np.transpose(np.array(all_pcs), (1, 0, 2))
#         pcs_flat     = self.all_pcs.reshape(len(self.lhs), len(self.z) * num_pcs)

#         # --- tPCA ---
#         pca_z = PCA(n_components=num_pcs_z)
#         pca_z.fit(pcs_flat)
#         self.tpca         = pca_z
#         self.t_components = pca_z.transform(pcs_flat)

#         # --- Normalise t-components to zero mean / unit variance per column ---
#         self.t_comp_scaler     = TComponentScaler().fit(self.t_components)
#         self.t_components_norm = self.t_comp_scaler.transform(self.t_components)

#         stds = self.t_components.std(axis=0)
#         print(f"\n✅ Prepared all cosmologies with {num_pcs_z} tPCA components each.")
#         print(f"   t-component std range (raw):  {stds.min():.3f} → {stds.max():.3f}  "
#               f"(ratio = {stds.max()/stds.min():.1f}x)")
#         print(f"   t-component std range (norm): "
#               f"{self.t_components_norm.std(axis=0).min():.3f} → "
#               f"{self.t_components_norm.std(axis=0).max():.3f}  (should be ~1.0)")

#         # --- Save single bundle (npce_indices=None / model_type='mlp' for now) ---
#         # For NPCE models, train.py will call update_metadata_bundle_npce() after
#         # training to patch in the PCE indices and flip model_type to 'npce'.
#         bundle = _build_metadata_bundle(
#             param_scaler  = self.param_scaler,
#             t_comp_pca    = self.tpca,
#             t_comp_scaler = self.t_comp_scaler,
#             pcas_dict     = pcas_dict,
#             scalers_dict  = scalers_dict,
#             npce_indices  = None,
#             model_type    = "mlp",
#         )
#         output_path = os.path.join(metadata_subdir, "metadata.joblib")
#         save_metadata_bundle(bundle, output_path)
#         self._metadata_bundle_path = output_path   # stored for train.py to use
#         print(f"   Metadata saved to: {output_path}")

#     # -------------------------------------------------------
#     @classmethod
#     def from_metadata_bundle(cls, bundle_path: str) -> "COLASet":
#         """
#         Reconstruct a minimal COLASet from a saved metadata bundle.

#         The returned object has no training data (lhs, pks_lin, etc.) — it
#         carries only the fitted scalers/PCAs needed to run COLAModel at
#         inference time.  Pass it directly to COLAModel.__init__.

#         Parameters
#         ----------
#         bundle_path : str
#             Path to a metadata.joblib file produced by prepare() or by the
#             legacy repackage_metadata.py script.

#         Example
#         -------
#         >>> train_set = COLASet.from_metadata_bundle("metadata/metadata.joblib")
#         >>> model = COLA_NN_Keras(train_set)
#         >>> model.models["t-component"] = keras.models.load_model("model.keras")
#         """
#         bundle = load_metadata_bundle(bundle_path)

#         obj = cls.__new__(cls)

#         # Scalers / PCAs the model needs at inference time
#         obj.param_scaler   = bundle["param_scaler"]
#         obj.tpca           = bundle["t_comp_pca"]
#         obj.t_comp_scaler  = bundle["t_comp_scaler"]

#         # Reconstruct ordered lists from the z-keyed dicts, sorted by z
#         z_keys = sorted(bundle["pcas"].keys())
#         obj.z                = np.array(z_keys)
#         obj.pcas             = [bundle["pcas"][z]    for z in z_keys]
#         obj.frac_pks_scalers = [bundle["scalers"][z] for z in z_keys]

#         # Expose raw dicts too, in case callers want keyed access
#         obj._pcas_dict    = bundle["pcas"]
#         obj._scalers_dict = bundle["scalers"]

#         # NPCE-specific — None for MLP bundles
#         obj.npce_indices = bundle.get("npce_indices", None)

#         # Fields that don't exist for an inference-only set
#         obj.lhs_norm           = None
#         obj.t_components_norm  = None
#         obj._metadata_bundle_path = bundle_path

#         return obj


# # ----------------------------------------------------------------------------------------------------
# class COLAModel:
#     def __init__(self, trainSet):
#         self.param_scaler      = trainSet.param_scaler
#         self.frac_pks_scalers  = trainSet.frac_pks_scalers
#         self.pcas              = trainSet.pcas
#         self.z_vals            = trainSet.z
#         self.t_comp_scaler     = trainSet.t_comp_scaler
#         self.tpca              = trainSet.tpca
#         self.models            = {}

#     def fit(self, trainSet, num_epochs):
#         raise NotImplementedError("ERROR: COLAModel must override `fit` method")

#     def predict_t_components(self, x):
#         raise NotImplementedError("ERROR: COLAModel must override `predict_t_components` method")

#     def predict_pcs_from_t(self, t_components_raw, z_idx):
#         pcs_flat = self.tpca.inverse_transform(t_components_raw)
#         num_pcs  = self.pcas[z_idx].n_components_
#         n_z      = len(self.z_vals)
#         pcs_all  = pcs_flat.reshape(-1, n_z, num_pcs)
#         return pcs_all[:, z_idx, :]

#     def predict_logfrac(self, x, z_idx):
#         t_raw        = self.predict_t_components(x)
#         pcs_z        = self.predict_pcs_from_t(t_raw, z_idx)
#         logfrac_norm = self.pcas[z_idx].inverse_transform(pcs_z)
#         return self.frac_pks_scalers[z_idx].inverse_transform(logfrac_norm)

#     def predict(self, x, z_idx):
#         """Return frac_pk = exp(logfrac) for the given z index."""
#         return np.exp(self.predict_logfrac(x, z_idx))

#     def plot_errors(self, testSet, z_idx):
#         preds   = self.predict(testSet.lhs, z_idx)
#         targets = np.exp(testSet.logfracs[:, z_idx, :])
#         fig, ax = plt.subplots()
#         for pred, target in zip(preds, targets):
#             ax.semilogx(testSet.ks, pred / target - 1)
#         ax.fill_between(testSet.ks, -0.0025, 0.0025, color="gray", alpha=0.75)
#         ax.fill_between(testSet.ks, -0.005,  0.005,  color="gray", alpha=0.5)
#         ax.set_xlabel("k")
#         ax.set_ylabel(f"Emulation Error (z={testSet.z[z_idx]:.3f})")
#         return fig, ax

#     def get_outliers(self, testSet, z_idx, log=False):
#         preds         = self.predict(testSet.lhs, z_idx)
#         frac_pks_test = np.exp(testSet.logfracs[:, z_idx, :])
#         cosmos, boosts = [], []
#         for pred, target, cosmo in zip(preds, frac_pks_test, testSet.lhs):
#             error = np.abs(pred / target - 1)
#             if np.any(error > 0.005):
#                 if log:
#                     h, Ob, Om, As, ns, w0, w0wa = cosmo
#                     print(f"Outlier (z={testSet.z[z_idx]:.3f}): "
#                         f"h={h:.3f}, Ob={Ob:.3f}, Om={Om:.3f}, As={As:.2e}, ns={ns:.3f}, "
#                         f"w0={w0:.3f}, w0+wa={w0wa:.3f} (max error={np.max(error):.4f})")
#                 cosmos.append(cosmo)
#                 boosts.append(target)
#         return cosmos, boosts

#     def save(self, path):
#         raise DeprecationWarning("Pickling models is not advised.")


# # ----------------------------------------------------------------------------------------------------
# class COLA_NN_Keras(COLAModel):
#     def __init__(self, trainSet, num_layers=3, num_neurons=1024):
#         super().__init__(trainSet)
#         self.num_layers  = num_layers
#         self.num_neurons = num_neurons
#         self.trainSet    = trainSet
#         self.models      = {}

#     def build_model_for_t_comps(self):
#         input_shape  = self.trainSet.lhs_norm.shape[1]
#         output_shape = self.trainSet.t_components_norm.shape[1]
#         return generate_mlp(
#             input_shape=input_shape,
#             output_shape=output_shape,
#             num_layers=self.num_layers,
#             num_neurons=self.num_neurons,
#             activation="custom",
#             alpha=0,
#             l1_ratio=0
#         )

#     def fit_t_componets(
#         self,
#         trainSet,
#         num_epochs,
#         batch_size=512,
#         initial_lr=1e-3,
#         final_lr=1e-5,
#         huber_delta=1.0,
#         # Legacy step-decay params kept for backwards compatibility but ignored
#         # when cosine annealing is active (which is always by default).
#         decayevery=None,
#         decayrate=None,
#     ):
#         """
#         Train the t-component emulator with cosine annealing and Huber loss.

#         Parameters
#         ----------
#         trainSet : COLASet
#             The prepared training dataset.
#         num_epochs : int
#             Total number of training epochs.
#         batch_size : int, optional
#             Mini-batch size.  Default 512.  Previously hardcoded to 30, which
#             produced very noisy gradients (1000 steps/epoch on 30k samples).
#             Larger batches give smoother gradients and allow a higher initial LR.
#             Recommended range: 256–1024.
#         initial_lr : float, optional
#             Starting learning rate for cosine annealing.  Default 1e-3.
#         final_lr : float, optional
#             Floor learning rate at the end of training (alpha in CosineDecay).
#             Default 1e-5.  Set lower (e.g. 1e-6) for very long runs.
#         huber_delta : float, optional
#             Transition point of the Huber loss (in normalised t-component units,
#             where std ≈ 1).  Default 1.0 — quadratic below 1 sigma, linear above.
#             Increase to 2–3 if you find the model underfits extreme cosmologies;
#             decrease to 0.5 if outliers are still dominating training.
#         decayevery, decayrate : ignored
#             Retained for backwards compatibility with old call sites.  Cosine
#             annealing supersedes step-decay entirely.
#         """
#         mlp = self.build_model_for_t_comps()

#         n_train     = len(trainSet.lhs_norm)
#         steps_per_epoch = max(1, n_train // batch_size)
#         total_steps     = num_epochs * steps_per_epoch

#         lr_schedule = keras.optimizers.schedules.CosineDecay(
#             initial_learning_rate=initial_lr,
#             decay_steps=total_steps,
#             alpha=final_lr / initial_lr,
#         )

#         print(
#             f"\n[fit_t_componets] Training config:\n"
#             f"  batch_size      = {batch_size}  "
#             f"({steps_per_epoch} steps/epoch, {total_steps} total steps)\n"
#             f"  LR schedule     = CosineDecay  "
#             f"{initial_lr:.2e} → {final_lr:.2e}\n"
#             f"  loss            = MSE\n"
#             f"  epochs          = {num_epochs}"
#         )

#         mlp.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
#             loss=keras.losses.Huber(delta=huber_delta),
#         )

#         nn_model_train_keras(
#             mlp,
#             epochs=num_epochs,
#             input_data=trainSet.lhs_norm,
#             truths=trainSet.t_components_norm,
#             batch_size=batch_size,
#         )
#         self.models["t-component"] = mlp
#         return mlp

#     def predict_t_components(self, x):
#         """
#         Run the NN and invert the t-component normalisation.
#         """
#         mlp = self.models["t-component"]
#         if x.max() > 1.5 or x.min() < -1.5:
#             x = self.param_scaler.transform(x)
#         t_norm = mlp.predict(x, verbose=0)
#         return self.t_comp_scaler.inverse_transform(t_norm)


# # ----------------------------------------------------------------------------------------------------
# class Scaler:
#     """Standard-score normaliser (mean/std) for logfrac slices."""
#     def __init__(self):
#         self.mean = None
#         self.std  = None

#     def fit(self, X):
#         self.mean = np.mean(X, axis=0)
#         self.std  = np.std(X, axis=0)
#         self.std =  np.where(self.std == 0, 1.0, self.std)  # guard zero-std columns

#     def transform(self, X):
#         return (X - self.mean) / self.std

#     def inverse_transform(self, X):
#         return (X * self.std) + self.mean


# # ----------------------------------------------------------------------------------------------------
# class CustomActivationLayer(layers.Layer):
#     def __init__(self, units, **kwargs):
#         super(CustomActivationLayer, self).__init__(**kwargs)
#         self.units = units
#         self.input_spec = layers.InputSpec(min_ndim=2)

#     def build(self, input_shape):
#         self.beta  = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="beta")
#         self.gamma = self.add_weight(shape=(self.units,), initializer='random_normal', trainable=True, name="gamma")
#         super(CustomActivationLayer, self).build(input_shape)

#     def call(self, x):
#         func = tf.add(
#             self.gamma,
#             tf.multiply(tf.sigmoid(tf.multiply(self.beta, x)), tf.subtract(1.0, self.gamma))
#         )
#         return tf.multiply(func, x)

#     def get_config(self):
#         config = super(CustomActivationLayer, self).get_config()
#         config.update({'units': self.units})
#         return config

#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.units)


# def generate_mlp(input_shape, output_shape, num_layers, num_neurons,
#                  activation="custom", alpha=0.01, l1_ratio=0.01,
#                  learning_rate=1e-3, optimizer='adam', loss='mse'):
#     """Generates an MLP model."""
#     reg = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio)) if alpha != 0 else None

#     def apply_activation(x):
#         if activation == "custom":
#             return CustomActivationLayer(num_neurons)(x)
#         elif activation == "relu":
#             return keras.activations.relu(x)
#         elif activation == "sigmoid":
#             return keras.activations.sigmoid(x)
#         else:
#             raise ValueError(f"Unexpected activation '{activation}'")

#     inputs = layers.Input(shape=(input_shape,))
#     x = layers.Dense(num_neurons, kernel_regularizer=reg)(inputs)
#     x = apply_activation(x)

#     for _ in range(num_layers - 1):
#         x = layers.Dense(num_neurons, kernel_regularizer=reg)(x)
#         x = apply_activation(x)

#     outputs = layers.Dense(output_shape)(x)

#     if optimizer.lower() == 'adam':
#         opt = optimizers.Adam(learning_rate=learning_rate)
#     elif optimizer.lower() == 'sgd':
#         opt = optimizers.SGD(learning_rate=learning_rate, momentum=0.99, nesterov=True)
#     else:
#         raise ValueError(f"Unhandled optimizer: {optimizer}")

#     model = models.Model(inputs=inputs, outputs=outputs)
#     model.summary()
#     model.compile(optimizer=opt, loss=loss)
#     return model


# # ----------------------------------------------------------------------------------------------------
# def generate_resnet(input_shape, output_shape, num_res_blocks=1, num_of_neurons=512,
#                     activation="relu", alpha=1e-5, l1_ratio=0.1, dropout=0.1):
#     """Generates a ResNet model with `num_res_blocks` residual blocks."""
#     reg = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio))
#     input_layer = layers.Input(shape=input_shape)

#     hid1 = layers.Dense(units=num_of_neurons, kernel_regularizer=reg, bias_regularizer=reg)(input_layer)
#     act1 = CustomActivationLayer(num_of_neurons)(hid1)
#     hid2 = layers.Dense(units=num_of_neurons, kernel_regularizer=reg, bias_regularizer=reg)(act1)
#     act2 = CustomActivationLayer(num_of_neurons)(hid2)
#     residual = layers.Add()([act1, act2])

#     for _ in range(num_res_blocks - 1):
#         hid1 = layers.Dense(units=num_of_neurons, kernel_regularizer=reg, bias_regularizer=reg)(residual)
#         act1 = CustomActivationLayer(num_of_neurons)(hid1)
#         hid2 = layers.Dense(units=num_of_neurons, kernel_regularizer=reg, bias_regularizer=reg)(act1)
#         act2 = CustomActivationLayer(num_of_neurons)(hid2)
#         residual = layers.Add()([act1, act2])

#     output_layer = layers.Dense(units=output_shape)(residual)
#     model = keras.models.Model(inputs=input_layer, outputs=output_layer)
#     model.summary()
#     model.compile(optimizer=keras.optimizers.Adam(), loss=keras.losses.MeanAbsoluteError())
#     return model


# # ----------------------------------------------------------------------------------------------------
# def nn_model_train_keras(
#     model,
#     epochs,
#     input_data,
#     truths,
#     batch_size=512,
#     validation_features=None,
#     validation_truths=None,
#     decayevery=None,
#     decayrate=None,
# ):
#     """
#     Trains a neural network model.

#     The LR schedule is expected to be embedded in the optimizer (e.g. via
#     CosineDecay) when called from fit_t_componets.  The legacy decayevery /
#     decayrate step-decay is still supported for direct callers that haven't
#     migrated, but is a no-op when decayevery=None (the new default).

#     Parameters
#     ----------
#     batch_size : int
#         Mini-batch size.  Default 512.  Previously hardcoded to 30.
#     decayevery, decayrate : int / float or None
#         Legacy step-decay parameters.  If both are provided, a
#         LearningRateScheduler callback is added (for backwards compatibility).
#         When called from fit_t_componets these are None and the schedule
#         embedded in the Adam optimizer is used instead.
#     """
#     class PrintCallback(tf.keras.callbacks.Callback):
#         epoch = 0

#         def on_epoch_begin(self, epoch, logs=None):
#             self.epoch = epoch

#         def on_epoch_end(self, batch, logs=None):
#             print(f'Epoch: {self.epoch} => Loss = {logs["loss"]}', end="\r")

#     callbacks = [PrintCallback()]

#     if decayevery is not None and decayrate is not None:
#         def scheduler(epoch, lr):
#             return lr / decayrate if (epoch != 0 and epoch % decayevery == 0) else lr
#         callbacks.append(keras.callbacks.LearningRateScheduler(scheduler))

#     fit_kwargs = dict(
#         batch_size=batch_size,
#         epochs=epochs,
#         callbacks=callbacks,
#         verbose=0,
#     )
#     if validation_features is not None and validation_truths is not None:
#         fit_kwargs["validation_data"] = (validation_features, validation_truths)

#     history = model.fit(input_data, truths, **fit_kwargs)
#     return history.history['loss'][-1]

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

import sys; sys.path.insert(0, "/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); import symbolic_pofk.linear_VL as linear
import sys; sys.path.insert(0, "/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); from symbolic_pofk.linear_VL import plin_emulated, get_approximate_D, growth_correction_R, get_eisensteinhu_nw
import sys; sys.path.insert(0, "/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emulator_train/symbolic_pofk"); from symbolic_pofk.syrenhalofit import run_halofit_vec
from symbolic_pofk.linear_VL import As_to_sigma8

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

#k=1e-5 to 1e2 (with 500 steps in log space)
ks = np.logspace(-5.1, 2, 500)

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
z_mps = np.concatenate((z1_mps, z2_mps, z3_mps), axis=0)

BASE_PATH = "/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps/"

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
    "expanded":    {"lin", "halofit", "mead2020_Tfree_mnufree"},
    "constrained": set(NL_TYPE_REGISTRY.keys()),
}

VER = "_v3"#"_v6"

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
    input_path  = os.path.join(base_path, "input",  input_name)
    output_path = os.path.join(base_path, "output", output_name)
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
    k_for_plin = ks / h
    if use_eh:
        pk_fid_hmpc = get_eisensteinhu_nw(k_for_plin, As, Om, Ob, h, ns, mnu=0.06, w0=w0, wa=wa)
    else:
        pk_fid_hmpc = plin_emulated(k_for_plin, Om, Ob, h, ns, As=As, w0=w0, wa=wa)
    a_array = 1.0 / (zs + 1)
    D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
    Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
    R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
    Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
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

    # k in h/Mpc for syren calls
    k_hmpc = ks / h

    # Linear Pk in (Mpc/h)^3 from syren
    pk_lin_hmpc = plin_emulated(k_hmpc, Om, Ob, h, ns, As=As, w0=w0, wa=wa)  # shape (N_K,)

    a_array = 1.0 / (np.asarray(zs) + 1.0)

    # Apply growth factor to get P_lin(k, z) in (Mpc/h)^3
    D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
    Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
    R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
    Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
    growth_factors = (Dz / D0) ** 2 * (Rz / R0)          # shape (N_ZS,)

    pk_lin_hmpc_z = pk_lin_hmpc[None, :] * growth_factors[:, None]  # (N_ZS, N_K)

    # sigma8 at z=0 for halofit (halofit uses sigma8 at z=0 + a(z) internally)
    sigma8_z0 = As_to_sigma8(As, Om, Ob, h, ns, mnu=0.06, w0=w0, wa=wa)

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

    # Convert (Mpc/h)^3 → Mpc³
    pk_nl_mpc_z = pk_nl_hmpc_z / h**3
    return pk_nl_mpc_z.astype(np.float32)


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
        if not (os.path.exists(input_path) and os.path.exists(output_path)):
            if verbose:
                print(f"⚠️  Skipping missing batch {b}  (prior='{prior_type}', nl_type='{nl_type}')")
                print(f"    expected input : {input_path}")
                print(f"    expected output: {output_path}")
            continue

        x = np.load(input_path,  mmap_mode="r")
        y = np.load(output_path, mmap_mode="r")
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


def update_metadata_bundle_npce(bundle_path: str, npce_indices: np.ndarray) -> None:
    """Patch an existing metadata bundle with NPCE-specific fields."""
    bundle = load_metadata_bundle(bundle_path)
    bundle["npce_indices"] = npce_indices
    bundle["model_type"]   = "npce"
    save_metadata_bundle(bundle, bundle_path)
    print(
        f"   Bundle updated: model_type='npce', "
        f"npce_indices shape={npce_indices.shape}  →  {bundle_path}"
    )


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
        self.use_boost = (nl_type != "lin")

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
            self.pks_lin = None   # not used in NL mode; syren provides the baseline

            print(
                f"Computing NL/syren_halofit boost "
                f"(frac_pks = P_{nl_type}_camb / P_nl_syren) "
                f"over full k grid ({len(self.ks)} points)"
            )

            self.mps_approxes_nl = np.array([
                _compute_mps_nl_approximation(self.ks, self.z, p)[0]
                if len(self.z) == 1 else
                _compute_mps_nl_approximation(self.ks, self.z, p)
                for p in tqdm(self.lhs, desc="syren halofit approx", unit="cosmo")
            ])
            print(f"mps_approxes_nl shape = {self.mps_approxes_nl.shape}")

            with np.errstate(divide="ignore", invalid="ignore"):
                self.frac_pks = self.pks_target / self.mps_approxes_nl

            # Keep mps_approxes as an alias pointing to the NL approximation
            # so downstream code that references either name keeps working.
            self.mps_approxes = self.mps_approxes_nl

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

            self.mps_approxes_nl = None   # not used in linear mode

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
            self.mps_approxes_nl = self.mps_approxes_nl[good_mask]
            self.mps_approxes    = self.mps_approxes_nl   # keep alias in sync
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
        # kmin_tag removed: k-truncation no longer applied
        return (
            f"{self.cosmo_type}_{self.prior_type}{w0_tag}{w0wa_tag}"
            f"_{self.nl_type}{mode_tag}_nTrain{self.n_batches}{VER}"
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
    def prepare(self, num_pcs, num_pcs_z, metadata_dir="mps_emu/metadata"):
        """
        Fit scalers and PCA for each redshift, then run tPCA and normalise
        the resulting t-components to zero mean / unit variance per column.
        """
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

        # --- tPCA ---
        pca_z = PCA(n_components=num_pcs_z)
        pca_z.fit(pcs_flat)
        self.tpca         = pca_z
        self.t_components = pca_z.transform(pcs_flat)

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
        self.t_comp_scaler     = TComponentScaler().fit(self.t_components)
        self.t_components_norm = self.t_comp_scaler.transform(self.t_components)

        stds = self.t_components.std(axis=0)
        print(f"\n✅ Prepared all cosmologies with {num_pcs_z} tPCA components each.")
        print(f"   t-component std range (raw):  {stds.min():.3f} → {stds.max():.3f}  "
              f"(ratio = {stds.max()/stds.min():.1f}x)")
        print(f"   t-component std range (norm): "
              f"{self.t_components_norm.std(axis=0).min():.3f} → "
              f"{self.t_components_norm.std(axis=0).max():.3f}  (should be ~1.0)")

        bundle = _build_metadata_bundle(
            param_scaler  = self.param_scaler,
            t_comp_pca    = self.tpca,
            t_comp_scaler = self.t_comp_scaler,
            pcas_dict     = pcas_dict,
            scalers_dict  = scalers_dict,
            npce_indices  = None,
            model_type    = "mlp",
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
        func = tf.add(
            self.gamma,
            tf.multiply(tf.sigmoid(tf.multiply(self.beta, x)), tf.subtract(1.0, self.gamma))
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

    outputs = layers.Dense(output_shape)(x)

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
    class PrintCallback(tf.keras.callbacks.Callback):
        epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.epoch = epoch

        def on_epoch_end(self, batch, logs=None):
            print(f'Epoch: {self.epoch} => Loss = {logs["loss"]}', end="\r")

    callbacks = [PrintCallback()]

    if decayevery is not None and decayrate is not None:
        def scheduler(epoch, lr):
            return lr / decayrate if (epoch != 0 and epoch % decayevery == 0) else lr
        callbacks.append(keras.callbacks.LearningRateScheduler(scheduler))

    fit_kwargs = dict(
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )
    if validation_features is not None and validation_truths is not None:
        fit_kwargs["validation_data"] = (validation_features, validation_truths)

    history = model.fit(input_data, truths, **fit_kwargs)
    return history.history['loss'][-1]