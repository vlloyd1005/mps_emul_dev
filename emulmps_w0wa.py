# # # # Author: Victoria Lloyd (2025) & V. Miranda (as seen in model evaluation)
# # # import os
# # # import numpy as np
# # # import joblib
# # # from typing import Dict, List, Tuple, Optional
# # # import logging
# # # from pathlib import Path
# # # import sys
# # # import train_utils_pk_emulator_v2 as utils
# # # sys.modules['train_utils_pk_emulator'] = utils
# # # from train_utils_pk_emulator_v2 import CustomActivationLayer, TComponentScaler
# # # from keras.losses import MeanSquaredError, Huber
# # # import tensorflow as tf


# # # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# # # # ----------------------------------------------------------------------------------------------------
# # # # Path helpers
# # # # ----------------------------------------------------------------------------------------------------

# # # def _get_project_root() -> Path:
# # #     """Returns the directory containing this file."""
# # #     return Path(__file__).resolve().parent

# # # ROOT = _get_project_root()

# # # # ----------------------------------------------------------------------------------------------------
# # # # Dependency guard
# # # # ----------------------------------------------------------------------------------------------------

# # # try:
# # #     from tensorflow import keras
# # #     import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk")
# # #     from symbolic_pofk.linear_VL import plin_emulated, get_approximate_D, growth_correction_R, get_eisensteinhu_nw
# # #     _DEPENDENCIES_LOADED = True
# # # except ImportError as e:
# # #     logging.error("FATAL ERROR: A required dependency could not be imported.")
# # #     logging.error(f"Missing component: {e.name}")
# # #     logging.error("If running locally, ensure the symbolic_pofk library is accessible.")
# # #     _DEPENDENCIES_LOADED = False

# # # try:
# # #     from sklearn.decomposition import PCA
# # #     from sklearn.preprocessing import StandardScaler
# # # except ImportError:
# # #     pass


# # # # ----------------------------------------------------------------------------------------------------
# # # # Registry
# # # # ----------------------------------------------------------------------------------------------------

# # # NL_TYPE_REGISTRY = {
# # #     "lin":                      ("pklin",                           "Linear P(k)"),
# # #     "halofit":                  ("pknonlin",                        "Non-linear P(k) via HaloFit"),
# # #     "mead2020":                 ("mead2020_pknonlin",               "Non-linear P(k) via HMcode Mead2020"),
# # #     "mead2020_feedback":        ("mead2020_feedback_pknonlin",      "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (fixed T_AGN)"),
# # #     "mead2020_feedback_Tfree":  ("mead2020_feedback_Tfree_pknonlin","Non-linear P(k) via HMcode Mead2020 + baryonic feedback (free T_AGN)"),
# # # }

# # # VALID_NL_TYPES_BY_PRIOR = {
# # #     "expanded":    {"lin", "halofit"},
# # #     "constrained": set(NL_TYPE_REGISTRY.keys()),
# # # }

# # # VALID_COSMO_TYPES  = {"lcdm", "w0wacdm"}
# # # VALID_MODEL_TYPES  = {"mlp", "npce"}

# # # VER = "_v3"


# # # def _validate_config(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     model_type: str = "mlp",
# # # ) -> None:
# # #     if cosmo_type not in VALID_COSMO_TYPES:
# # #         raise ValueError(
# # #             f"Unknown cosmo_type '{cosmo_type}'. Must be one of: {sorted(VALID_COSMO_TYPES)}"
# # #         )
# # #     if prior_type not in VALID_NL_TYPES_BY_PRIOR:
# # #         raise ValueError(
# # #             f"Unknown prior_type '{prior_type}'. Must be one of: {sorted(VALID_NL_TYPES_BY_PRIOR)}"
# # #         )
# # #     if nl_type not in NL_TYPE_REGISTRY:
# # #         raise ValueError(
# # #             f"Unknown nl_type '{nl_type}'. Must be one of: {sorted(NL_TYPE_REGISTRY)}"
# # #         )
# # #     if nl_type not in VALID_NL_TYPES_BY_PRIOR[prior_type]:
# # #         raise ValueError(
# # #             f"nl_type '{nl_type}' is not available for prior_type='{prior_type}'. "
# # #             f"Valid choices: {sorted(VALID_NL_TYPES_BY_PRIOR[prior_type])}"
# # #         )
# # #     if cosmo_type == "lcdm" and prior_type == "expanded":
# # #         raise ValueError("prior_type='expanded' is only supported for cosmo_type='w0wacdm'.")
# # #     if model_type not in VALID_MODEL_TYPES:
# # #         raise ValueError(
# # #             f"Unknown model_type '{model_type}'. Must be one of: {sorted(VALID_MODEL_TYPES)}"
# # #         )


# # # def _metadata_tag(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> str:
# # #     w0_tag   = f"_w0min{w0_min}"     if w0_min   is not None else ""
# # #     w0wa_tag = f"_w0wamax{w0wa_max}" if w0wa_max is not None else ""
# # #     return f"{cosmo_type}_{prior_type}{w0_tag}{w0wa_tag}_{nl_type}_nTrain{num_batches}{VER}"


# # # def _model_filename(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     model_type: str = "mlp",
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> str:
# # #     """
# # #     Returns the model filename, matching the pattern written by train.py:
# # #         emulator_{model_type}_{metadata_tag}.keras
# # #     Falls back to the legacy .h5 pattern for MLP models if the .keras file
# # #     is not found (for backwards compatibility with models trained before the
# # #     format change).
# # #     """
# # #     tag  = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
# # #     return f"emulator_{model_type}_{tag}.keras"


# # # def _model_filename_legacy(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> str:
# # #     """Legacy .h5 filename for MLP models saved before the format change."""
# # #     tag = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
# # #     return f"emulator_{tag}.h5"


# # # # ----------------------------------------------------------------------------------------------------
# # # # Core emulator class
# # # # ----------------------------------------------------------------------------------------------------

# # # class PkEmulator:
# # #     """
# # #     Cosmology emulator for the matter power spectrum P(k, z).

# # #     Supports multiple cosmological models, prior widths, nonlinear prescriptions,
# # #     and both MLP and NPCE model architectures.

# # #     Parameters
# # #     ----------
# # #     cosmo_type : str
# # #         Cosmological model: 'w0wacdm' or 'lcdm'.
# # #     prior_type : str
# # #         Prior width used during training: 'expanded' or 'constrained'.
# # #     nl_type : str
# # #         Nonlinear prescription. See NL_TYPE_REGISTRY for valid options.
# # #     model_type : str
# # #         Architecture of the trained model: 'mlp' or 'npce'. Default 'mlp'.
# # #     base_model_path : str
# # #         Directory containing model files, relative to this file.
# # #     base_metadata_path : str
# # #         Parent directory of all metadata subdirectories, relative to this file.
# # #     num_batches : int
# # #         Number of training batches — used to locate metadata files.
# # #     w0_min : float or None
# # #         Lower bound on w0 used during training (embedded in filenames).
# # #     w0wa_max : float or None
# # #         Upper bound on w0+wa used during training (embedded in filenames).
# # #     """

# # #     N_PCS     = 25
# # #     N_K_MODES = 500

# # #     K_MODES = np.logspace(-5.1, 2, N_K_MODES)
# # #     Z_MODES = np.concatenate((
# # #         np.linspace(0,  3,  33, endpoint=False),
# # #         np.linspace(3,  10,  7, endpoint=False),
# # #         np.linspace(10, 50, 12),
# # #     ))
# # #     N_ZS = len(Z_MODES)

# # #     def __init__(
# # #         self,
# # #         cosmo_type: str = "w0wacdm",
# # #         prior_type: str = "constrained",
# # #         nl_type: str = "lin",
# # #         model_type: str = "mlp",
# # #         base_model_path: str = "models",
# # #         base_metadata_path: str = "metadata",
# # #         num_batches: int = 15,
# # #         w0_min: Optional[float] = None,
# # #         w0wa_max: Optional[float] = None,
# # #     ):
# # #         if not _DEPENDENCIES_LOADED:
# # #             raise RuntimeError("Cannot initialise PkEmulator: missing dependencies.")

# # #         _validate_config(cosmo_type, prior_type, nl_type, num_batches, model_type)

# # #         self.cosmo_type  = cosmo_type
# # #         self.prior_type  = prior_type
# # #         self.nl_type     = nl_type
# # #         self.model_type  = model_type
# # #         self.num_batches = num_batches
# # #         self.w0_min      = w0_min
# # #         self.w0wa_max    = w0wa_max

# # #         logging.info(
# # #             f"[PkEmulator] Initialising: cosmo_type='{cosmo_type}', "
# # #             f"prior_type='{prior_type}', nl_type='{nl_type}', "
# # #             f"model_type='{model_type}', "
# # #             f"w0_min={w0_min}, w0wa_max={w0wa_max}"
# # #         )

# # #         self.MODEL_DIR    = ROOT / base_model_path
# # #         self.METADATA_DIR = (
# # #             ROOT / base_metadata_path
# # #             / f"metadata_{_metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)}"
# # #         )

# # #         logging.info(f"[PkEmulator] Metadata directory: {self.METADATA_DIR}")

# # #         try:
# # #             # --- Cosmological parameter scaler ---
# # #             self.param_scaler = joblib.load(
# # #                 self.METADATA_DIR / f"param_scaler_lowk_{num_batches}_batches"
# # #             )

# # #             # --- tPCA ---
# # #             self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")

# # #             # --- t-component scaler (unit-variance normalisation) ---
# # #             t_comp_scaler_path = self.METADATA_DIR / "t_comp_scaler"
# # #             if t_comp_scaler_path.exists():
# # #                 self.t_comp_scaler: Optional[TComponentScaler] = joblib.load(t_comp_scaler_path)
# # #                 logging.info("[PkEmulator] t_comp_scaler loaded.")
# # #             else:
# # #                 self.t_comp_scaler = None
# # #                 logging.warning(
# # #                     "[PkEmulator] t_comp_scaler not found. "
# # #                     "Assuming model was trained on raw (un-normalised) t-components."
# # #                 )

# # #             # --- NPCE: load PCE index matrix ---
# # #             self._pce_indices = None
# # #             if model_type == "npce":
# # #                 pce_path = self.METADATA_DIR / "npce_indices.npy"
# # #                 if not pce_path.exists():
# # #                     # Also check directly in model dir (train.py saves it there)
# # #                     pce_path = self.MODEL_DIR / "npce_indices.npy"
# # #                 if not pce_path.exists():
# # #                     raise FileNotFoundError(
# # #                         f"PCE index file not found. Expected at:\n"
# # #                         f"  {self.METADATA_DIR / 'npce_indices.npy'}\n"
# # #                         f"  {self.MODEL_DIR / 'npce_indices.npy'}\n"
# # #                         "Re-run training with --model_type npce, which calls "
# # #                         "model_obj.save_pce_metadata() automatically."
# # #                     )
# # #                 self._pce_indices = np.load(pce_path)
# # #                 logging.info(
# # #                     f"[PkEmulator] PCE indices loaded: "
# # #                     f"{self._pce_indices.shape[0]} basis terms, "
# # #                     f"{self._pce_indices.shape[1]} dims."
# # #                 )

# # #             # --- Neural network ---
# # #             model_file = self.MODEL_DIR / _model_filename(
# # #                 cosmo_type, prior_type, nl_type, num_batches,
# # #                 model_type, w0_min, w0wa_max,
# # #             )
# # #             # Fallback: legacy .h5 for MLP models saved before format change
# # #             if not model_file.exists() and model_type == "mlp":
# # #                 legacy = self.MODEL_DIR / _model_filename_legacy(
# # #                     cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max,
# # #                 )
# # #                 if legacy.exists():
# # #                     logging.warning(
# # #                         f"[PkEmulator] .keras file not found, falling back to legacy: {legacy}"
# # #                     )
# # #                     model_file = legacy

# # #             logging.info(f"[PkEmulator] Loading model: {model_file}")
# # #             # self.model = keras.models.load_model(
# # #             #     model_file,
# # #             #     custom_objects={
# # #             #         "CustomActivationLayer": CustomActivationLayer,
# # #             #         "mse": MeanSquaredError(),
# # #             #         "huber_loss": Huber(),
# # #             #         "TrimmedMSELoss": TrimmedMSELoss,
# # #             #     },
# # #             # )
# # #             self.model = keras.models.load_model(
# # #                 model_file,
# # #                 custom_objects={
# # #                     "CustomActivationLayer": CustomActivationLayer,
# # #                 },
# # #                 compile=False,   # skip loss reconstruction entirely — not needed for inference
# # #             )

# # #             @tf.function(jit_compile=False)
# # #             def _compiled_inference(x):
# # #                 return self.model(x, training=False)
# # #             self._compiled_inference = _compiled_inference

# # #             # --- Per-redshift PCA and scalers ---
# # #             self.PCAS: Dict[float, PCA] = {}
# # #             self.SCALERS: Dict[float, object] = {}
# # #             self._pcas_loaded = False
# # #             self.INVERSE_TRANSFORM_MATRICES: Optional[np.ndarray] = None
# # #             self.INVERSE_TRANSFORM_OFFSETS:  Optional[np.ndarray] = None
# # #             self._load_pcas_and_scalers()

# # #             # --- Warm-up ---
# # #             logging.info("[PkEmulator] Warming up neural network...")
# # #             dummy_params = np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, -1.0]], dtype=np.float32)
# # #             dummy_norm   = self.param_scaler.transform(dummy_params)
# # #             dummy_input  = self._make_network_input(dummy_norm)
# # #             dummy_tf     = tf.constant(dummy_input, dtype=tf.float32)
# # #             _ = self._compiled_inference(dummy_tf)
# # #             _ = self._compiled_inference(dummy_tf)

# # #             logging.info("[PkEmulator] Initialisation complete.")

# # #         except FileNotFoundError as e:
# # #             logging.error(f"Required file not found: {e.filename}")
# # #             logging.warning(
# # #                 f"Check that '{self.METADATA_DIR}' and '{self.MODEL_DIR}' "
# # #                 "contain all required files."
# # #             )
# # #             raise

# # #     # -------------------------------------------------------
# # #     def _make_network_input(self, params_norm: np.ndarray) -> np.ndarray:
# # #         """
# # #         Convert normalised cosmological parameters into the network's input
# # #         representation.

# # #         For MLP:  returns params_norm unchanged  (N, n_params)
# # #         For NPCE: evaluates the PCE basis        (N, N_terms)
# # #         """
# # #         if self.model_type == "mlp":
# # #             return params_norm.astype(np.float32)

# # #         # NPCE: expand to polynomial chaos basis features
# # #         return self._evaluate_pce_basis(params_norm)

# # #     def _evaluate_pce_basis(self, X_norm: np.ndarray) -> np.ndarray:
# # #         """
# # #         Evaluate the stored PCE multi-index basis at X_norm.

# # #         Mirrors _evaluate_pce_basis in cola_npce.py exactly, using the
# # #         vectorised power-table approach (no Python loops over N or T).

# # #         Parameters
# # #         ----------
# # #         X_norm : (N, n_params) float ndarray — MinMax-normalised inputs

# # #         Returns
# # #         -------
# # #         Phi : (N, N_terms) float32 ndarray
# # #         """
# # #         indices   = self._pce_indices                             # (T, D)
# # #         X_clipped = np.clip(X_norm, -1.0, 1.0).astype(np.float64)
# # #         N, D      = X_clipped.shape
# # #         max_deg   = int(indices.max()) if indices.size > 0 else 0

# # #         pow_table = np.empty((D, max_deg + 1, N), dtype=np.float64)
# # #         pow_table[:, 0, :] = 1.0
# # #         if max_deg >= 1:
# # #             pow_table[:, 1, :] = X_clipped.T
# # #         for p in range(2, max_deg + 1):
# # #             pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T

# # #         d_idx   = np.arange(D)[np.newaxis, :]
# # #         powered = pow_table[d_idx, indices, :]   # (T, D, N)
# # #         Phi     = powered.prod(axis=1).T          # (N, T)

# # #         return Phi.astype(np.float32)

# # #     # -------------------------------------------------------
# # #     def _load_pcas_and_scalers(self) -> None:
# # #         """Load per-redshift PCA/scaler files and pre-compute inverse transform matrices."""
# # #         if self._pcas_loaded:
# # #             return

# # #         logging.info("[PkEmulator] Loading per-redshift PCA and scaler objects...")

# # #         try:
# # #             for z in self.Z_MODES:
# # #                 z_key = float(f"{z:.3f}")
# # #                 self.PCAS[z_key]    = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
# # #                 self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")

# # #             logging.info("[PkEmulator] Pre-computing inverse transformation matrices...")

# # #             inverse_matrices, inverse_offsets = [], []
# # #             for z in self.Z_MODES:
# # #                 z_key  = float(f"{z:.3f}")
# # #                 pca    = self.PCAS[z_key]
# # #                 scaler = self.SCALERS[z_key]

# # #                 if hasattr(scaler, "scale_"):
# # #                     scale = scaler.scale_
# # #                     mean  = scaler.mean_
# # #                 elif hasattr(scaler, "std"):
# # #                     scale = scaler.std
# # #                     mean  = scaler.mean
# # #                 else:
# # #                     raise AttributeError(
# # #                         f"Scaler for z={z:.3f} has neither 'scale_' nor 'std'."
# # #                     )

# # #                 inverse_matrices.append(pca.components_ * scale[None, :])
# # #                 inverse_offsets.append(pca.mean_ * scale + mean)

# # #             self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
# # #             self.INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)

# # #             self._pcas_loaded = True
# # #             logging.info("[PkEmulator] Inverse transformation matrices ready.")

# # #         except FileNotFoundError as e:
# # #             logging.error(f"Required PCA/scaler file not found: {e.filename}")
# # #             raise

# # #     # -------------------------------------------------------
# # #     def _compute_mps_approximation(self, params: np.ndarray, use_eh=False) -> np.ndarray:
# # #         """
# # #         Analytical P_lin(k, z) approximation via symbolic_pofk + growth factors.

# # #         Parameters
# # #         ----------
# # #         params : 1-D array [10^9 A_s, ns, H0, Ob, Om, w0, wa]

# # #         Returns
# # #         -------
# # #         np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³
# # #         """
# # #         As, ns, H0_in, Ob, Om, w0, w0wa = params
# # #         wa = w0wa - w0
# # #         h  = H0_in / 100.0

# # #         k_for_plin = self.K_MODES / h
# # #         if use_eh:
# # #             pk_fid = get_eisensteinhu_nw(k_for_plin, As, Om, Ob, h, ns, mnu=0.06, w0=w0, wa=wa)
# # #         else:
# # #             pk_fid = plin_emulated(k_for_plin, Om, Ob, h, ns, As=As, w0=w0, wa=wa)

# # #         a_array = 1.0 / (self.Z_MODES + 1)
# # #         D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# # #         Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
# # #         R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# # #         Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)

# # #         growth_factors = (Dz / D0) ** 2 * (Rz / R0)
# # #         result = pk_fid[None, :] * growth_factors[:, None] / h**3
# # #         return result.astype(np.float32)

# # #     # -------------------------------------------------------
# # #     def _predict_fracs_all_z(self, params_norm: np.ndarray) -> np.ndarray:
# # #         """
# # #         NN inference: normalised params -> log-fractional differences for all z.

# # #         Inference chain
# # #         ---------------
# # #         1. [NPCE only] Expand params_norm to PCE basis features
# # #         2. NN(input)               -> t_components_norm
# # #         3. t_comp_scaler.inverse   -> t_components (raw tPCA space)
# # #         4. t_comp_pca.inverse      -> pcs_flat     (N_ZS * N_PCS)
# # #         5. Fused PCA+scaler invert -> log_frac     (N_ZS, N_K_MODES)

# # #         Parameters
# # #         ----------
# # #         params_norm : (1, N_params) ndarray

# # #         Returns
# # #         -------
# # #         np.ndarray of shape (N_ZS, N_K_MODES)
# # #         """
# # #         # Step 1: build network input (PCE expansion for NPCE, identity for MLP)
# # #         net_input = self._make_network_input(params_norm)
# # #         input_tf  = tf.constant(net_input, dtype=tf.float32)

# # #         # Step 2: NN forward pass
# # #         t_comps_norm = self._compiled_inference(input_tf).numpy()   # (1, N_T_COMPS)

# # #         # Step 3: invert unit-variance normalisation
# # #         if self.t_comp_scaler is not None:
# # #             t_comps_raw = self.t_comp_scaler.inverse_transform(t_comps_norm)
# # #         else:
# # #             t_comps_raw = t_comps_norm

# # #         # Step 4: inverse tPCA -> flat PCA coefficients
# # #         pcs_flat = self.t_comp_pca.inverse_transform(t_comps_raw).astype(np.float32)

# # #         # Step 5: reshape and fused inverse PCA + inverse logfrac scaler
# # #         pcs_z = pcs_flat.reshape(self.N_ZS, self.N_PCS)
# # #         reconstructed = (
# # #             np.einsum("zp,zpk->zk", pcs_z, self.INVERSE_TRANSFORM_MATRICES)
# # #             + self.INVERSE_TRANSFORM_OFFSETS
# # #         )
# # #         return reconstructed.astype(np.float32)

# # #     # -------------------------------------------------------
# # #     def get_pks(
# # #         self,
# # #         params: List[float],
# # #         use_approximation_only: bool = False,
# # #     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# # #         """
# # #         Return P(k, z) for a single cosmology.

# # #         Parameters
# # #         ----------
# # #         params : list or 1-D array of 7 values [10^9 A_s, ns, H0, Ob, Om, w0, wa].
# # #             For LCDM pass w0=-1.0 and wa=0.0.
# # #         use_approximation_only : bool
# # #             If True, return only the symbolic EH approximation (no NN).

# # #         Returns
# # #         -------
# # #         k_modes : (N_K_MODES,)
# # #         z_modes : (N_ZS,)
# # #         pks     : (N_ZS, N_K_MODES)
# # #         """
# # #         params_array = np.array(params, dtype=np.float32)
# # #         if params_array.ndim != 1 or len(params_array) != 7:
# # #             raise ValueError(
# # #                 f"Expected 7 parameters [10^9 A_s, ns, H0, Ob, Om, w0, wa], "
# # #                 f"got shape {params_array.shape}."
# # #             )

# # #         pk_mps = self._compute_mps_approximation(params_array)

# # #         if use_approximation_only:
# # #             return self.K_MODES, self.Z_MODES, pk_mps

# # #         params_norm = self.param_scaler.transform(params_array.reshape(1, -1))
# # #         log_frac    = self._predict_fracs_all_z(params_norm)
# # #         pks         = (np.exp(log_frac) * pk_mps).astype(np.float32)

# # #         return self.K_MODES, self.Z_MODES, pks


# # # # ----------------------------------------------------------------------------------------------------
# # # # Module-level interface with instance caching
# # # # ----------------------------------------------------------------------------------------------------

# # # _emulator_cache: Dict[Tuple, PkEmulator] = {}


# # # def get_emulator(
# # #     cosmo_type: str = "w0wacdm",
# # #     prior_type: str = "constrained",
# # #     nl_type: str = "lin",
# # #     model_type: str = "mlp",
# # #     base_model_path: str = "models",
# # #     base_metadata_path: str = "metadata",
# # #     num_batches: int = 15,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> PkEmulator:
# # #     """
# # #     Return a (cached) PkEmulator for the requested configuration.

# # #     Repeated calls with identical arguments return the same instance without
# # #     reloading files.  The cache key includes all arguments that affect which
# # #     files are loaded, including model_type, w0_min, and w0wa_max.
# # #     """
# # #     if not _DEPENDENCIES_LOADED:
# # #         raise RuntimeError("Cannot create PkEmulator: missing dependencies.")

# # #     cache_key = (cosmo_type, prior_type, nl_type, model_type, num_batches, w0_min, w0wa_max)
# # #     if cache_key in _emulator_cache:
# # #         logging.info(f"[get_emulator] Returning cached emulator for {cache_key}")
# # #         return _emulator_cache[cache_key]

# # #     logging.info(f"[get_emulator] Creating new emulator for {cache_key}")
# # #     emulator = PkEmulator(
# # #         cosmo_type=cosmo_type,
# # #         prior_type=prior_type,
# # #         nl_type=nl_type,
# # #         model_type=model_type,
# # #         base_model_path=base_model_path,
# # #         base_metadata_path=base_metadata_path,
# # #         num_batches=num_batches,
# # #         w0_min=w0_min,
# # #         w0wa_max=w0wa_max,
# # #     )
# # #     _emulator_cache[cache_key] = emulator
# # #     return emulator


# # # def get_pks(
# # #     params: List[float],
# # #     cosmo_type: str = "w0wacdm",
# # #     prior_type: str = "constrained",
# # #     nl_type: str = "lin",
# # #     model_type: str = "mlp",
# # #     num_batches: int = 15,
# # #     use_approximation_only: bool = False,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# # #     """
# # #     Convenience function: get P(k, z) without managing emulator instances.

# # #     Parameters
# # #     ----------
# # #     params : list of 7 floats [10^9 A_s, ns, H0, Ob, Om, w0, wa]
# # #     model_type : str
# # #         'mlp' or 'npce'. Must match the architecture used during training.
# # #     w0_min : float or None
# # #         Must match the value passed to --w0_min during training.
# # #     w0wa_max : float or None
# # #         Must match the value passed to --w0wa_max during training.

# # #     Returns
# # #     -------
# # #     k_modes, z_modes, pks
# # #     """
# # #     emulator = get_emulator(
# # #         cosmo_type=cosmo_type,
# # #         prior_type=prior_type,
# # #         nl_type=nl_type,
# # #         model_type=model_type,
# # #         num_batches=num_batches,
# # #         w0_min=w0_min,
# # #         w0wa_max=w0wa_max,
# # #     )
# # #     return emulator.get_pks(params, use_approximation_only=use_approximation_only)


# # # Author: Victoria Lloyd (2025) & V. Miranda (as seen in model evaluation)
# # # import os
# # # import numpy as np
# # # import joblib
# # # from typing import Dict, List, Tuple, Optional
# # # import logging
# # # from pathlib import Path
# # # import sys
# # # import train_utils_pk_emulator_v2 as utils
# # # sys.modules['train_utils_pk_emulator'] = utils
# # # from train_utils_pk_emulator_v2 import CustomActivationLayer, TComponentScaler
# # # from keras.losses import MeanSquaredError, Huber
# # # import tensorflow as tf


# # # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# # # # ----------------------------------------------------------------------------------------------------
# # # # Path helpers
# # # # ----------------------------------------------------------------------------------------------------

# # # def _get_project_root() -> Path:
# # #     """Returns the directory containing this file."""
# # #     return Path(__file__).resolve().parent

# # # ROOT = _get_project_root()

# # # # ----------------------------------------------------------------------------------------------------
# # # # Dependency guard
# # # # ----------------------------------------------------------------------------------------------------

# # # try:
# # #     from tensorflow import keras
# # #     import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk")
# # #     from symbolic_pofk.linear_VL import plin_emulated, get_approximate_D, growth_correction_R, get_eisensteinhu_nw
# # #     _DEPENDENCIES_LOADED = True
# # # except ImportError as e:
# # #     logging.error("FATAL ERROR: A required dependency could not be imported.")
# # #     logging.error(f"Missing component: {e.name}")
# # #     logging.error("If running locally, ensure the symbolic_pofk library is accessible.")
# # #     _DEPENDENCIES_LOADED = False

# # # try:
# # #     from sklearn.decomposition import PCA
# # #     from sklearn.preprocessing import StandardScaler
# # # except ImportError:
# # #     pass


# # # # ----------------------------------------------------------------------------------------------------
# # # # Registry
# # # # ----------------------------------------------------------------------------------------------------

# # # NL_TYPE_REGISTRY = {
# # #     "lin":                      ("pklin",                           "Linear P(k)"),
# # #     "halofit":                  ("pknonlin",                        "Non-linear P(k) via HaloFit"),
# # #     "mead2020":                 ("mead2020_pknonlin",               "Non-linear P(k) via HMcode Mead2020"),
# # #     "mead2020_feedback":        ("mead2020_feedback_pknonlin",      "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (fixed T_AGN)"),
# # #     "mead2020_feedback_Tfree":  ("mead2020_feedback_Tfree_pknonlin","Non-linear P(k) via HMcode Mead2020 + baryonic feedback (free T_AGN)"),
# # # }

# # # VALID_NL_TYPES_BY_PRIOR = {
# # #     "expanded":    {"lin", "halofit"},
# # #     "constrained": set(NL_TYPE_REGISTRY.keys()),
# # # }

# # # VALID_COSMO_TYPES  = {"lcdm", "w0wacdm"}
# # # VALID_MODEL_TYPES  = {"mlp", "npce"}

# # # VER = "_v3"


# # # def _validate_config(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     model_type: str = "mlp",
# # # ) -> None:
# # #     if cosmo_type not in VALID_COSMO_TYPES:
# # #         raise ValueError(
# # #             f"Unknown cosmo_type '{cosmo_type}'. Must be one of: {sorted(VALID_COSMO_TYPES)}"
# # #         )
# # #     if prior_type not in VALID_NL_TYPES_BY_PRIOR:
# # #         raise ValueError(
# # #             f"Unknown prior_type '{prior_type}'. Must be one of: {sorted(VALID_NL_TYPES_BY_PRIOR)}"
# # #         )
# # #     if nl_type not in NL_TYPE_REGISTRY:
# # #         raise ValueError(
# # #             f"Unknown nl_type '{nl_type}'. Must be one of: {sorted(NL_TYPE_REGISTRY)}"
# # #         )
# # #     if nl_type not in VALID_NL_TYPES_BY_PRIOR[prior_type]:
# # #         raise ValueError(
# # #             f"nl_type '{nl_type}' is not available for prior_type='{prior_type}'. "
# # #             f"Valid choices: {sorted(VALID_NL_TYPES_BY_PRIOR[prior_type])}"
# # #         )
# # #     if cosmo_type == "lcdm" and prior_type == "expanded":
# # #         raise ValueError("prior_type='expanded' is only supported for cosmo_type='w0wacdm'.")
# # #     if model_type not in VALID_MODEL_TYPES:
# # #         raise ValueError(
# # #             f"Unknown model_type '{model_type}'. Must be one of: {sorted(VALID_MODEL_TYPES)}"
# # #         )


# # # def _metadata_tag(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> str:
# # #     w0_tag   = f"_w0min{w0_min}"     if w0_min   is not None else ""
# # #     w0wa_tag = f"_w0wamax{w0wa_max}" if w0wa_max is not None else ""
# # #     return f"{cosmo_type}_{prior_type}{w0_tag}{w0wa_tag}_{nl_type}_nTrain{num_batches}{VER}"


# # # def _model_filename(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     model_type: str = "mlp",
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> str:
# # #     tag = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
# # #     return f"emulator_{model_type}_{tag}.keras"


# # # def _model_filename_legacy(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> str:
# # #     """Legacy .h5 filename for MLP models saved before the format change."""
# # #     tag = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
# # #     return f"emulator_{tag}.h5"


# # # # ----------------------------------------------------------------------------------------------------
# # # # Core emulator class
# # # # ----------------------------------------------------------------------------------------------------

# # # class PkEmulator:
# # #     """
# # #     Cosmology emulator for the matter power spectrum P(k, z).

# # #     Supports multiple cosmological models, prior widths, nonlinear prescriptions,
# # #     and both MLP and NPCE model architectures.

# # #     Input convention
# # #     ----------------
# # #     All public methods expect params = [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa].
# # #     Column 6 is w0+wa (not wa).  This matches the training convention in
# # #     COLASet (col 6 of lhs is overwritten with w0+wa before fitting the scaler).
# # #     The syren approximation receives the same vector and derives wa internally
# # #     as w0wa - w0.

# # #     Emulation modes
# # #     ---------------
# # #     The mode is determined automatically from the metadata bundle:

# # #     - **syren/lin mode** (use_boost=False):
# # #         Network learns  log(P_lin / P_syren).
# # #         get_pks returns  exp(log_frac) * P_syren  ≈  P_lin.

# # #     - **boost mode** (use_boost=True):
# # #         Network learns  log(P_nonlin / P_lin).
# # #         get_pks returns  exp(log_frac) * P_syren  ≈  P_nonlin.

# # #     In both cases the syren approximation is the multiplicative base at
# # #     inference time.  The distinction is only in what the network learned.

# # #     Metadata loading
# # #     ----------------
# # #     The constructor first looks for a single ``metadata.joblib`` bundle
# # #     (written by COLASet.prepare()).  If absent it falls back to the individual
# # #     per-file layout used by older script versions (separate param_scaler,
# # #     t_components_pca, t_comp_scaler, and per-z .pca / .frac_pks_scaler files).

# # #     Parameters
# # #     ----------
# # #     cosmo_type : str
# # #         Cosmological model: 'w0wacdm' or 'lcdm'.
# # #     prior_type : str
# # #         Prior width used during training: 'expanded' or 'constrained'.
# # #     nl_type : str
# # #         Nonlinear prescription. See NL_TYPE_REGISTRY for valid options.
# # #     model_type : str
# # #         Architecture of the trained model: 'mlp' or 'npce'. Default 'mlp'.
# # #     base_model_path : str
# # #         Directory containing model files, relative to this file.
# # #     base_metadata_path : str
# # #         Parent directory of all metadata subdirectories, relative to this file.
# # #     num_batches : int
# # #         Number of training batches — used to locate metadata files.
# # #     w0_min : float or None
# # #         Lower bound on w0 used during training (embedded in filenames).
# # #     w0wa_max : float or None
# # #         Upper bound on w0+wa used during training (embedded in filenames).
# # #     """

# # #     N_PCS     = 25
# # #     N_K_MODES = 500

# # #     K_MODES = np.logspace(-5.1, 2, N_K_MODES)
# # #     Z_MODES = np.concatenate((
# # #         np.linspace(0,  3,  33, endpoint=False),
# # #         np.linspace(3,  10,  7, endpoint=False),
# # #         np.linspace(10, 50, 12),
# # #     ))
# # #     N_ZS = len(Z_MODES)

# # #     def __init__(
# # #         self,
# # #         cosmo_type: str = "w0wacdm",
# # #         prior_type: str = "constrained",
# # #         nl_type: str = "lin",
# # #         model_type: str = "mlp",
# # #         base_model_path: str = "models",
# # #         base_metadata_path: str = "metadata",
# # #         num_batches: int = 15,
# # #         w0_min: Optional[float] = None,
# # #         w0wa_max: Optional[float] = None,
# # #     ):
# # #         if not _DEPENDENCIES_LOADED:
# # #             raise RuntimeError("Cannot initialise PkEmulator: missing dependencies.")

# # #         _validate_config(cosmo_type, prior_type, nl_type, num_batches, model_type)

# # #         self.cosmo_type  = cosmo_type
# # #         self.prior_type  = prior_type
# # #         self.nl_type     = nl_type
# # #         self.model_type  = model_type
# # #         self.num_batches = num_batches
# # #         self.w0_min      = w0_min
# # #         self.w0wa_max    = w0wa_max

# # #         logging.info(
# # #             f"[PkEmulator] Initialising: cosmo_type='{cosmo_type}', "
# # #             f"prior_type='{prior_type}', nl_type='{nl_type}', "
# # #             f"model_type='{model_type}', "
# # #             f"w0_min={w0_min}, w0wa_max={w0wa_max}"
# # #         )

# # #         self.MODEL_DIR    = ROOT / base_model_path
# # #         self.METADATA_DIR = (
# # #             ROOT / base_metadata_path
# # #             / f"metadata_{_metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)}"
# # #         )

# # #         logging.info(f"[PkEmulator] Metadata directory: {self.METADATA_DIR}")

# # #         try:
# # #             self._load_metadata()
# # #             self._load_model(cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max)

# # #             # Per-redshift PCA and scalers
# # #             self.PCAS:    Dict[float, PCA]    = {}
# # #             self.SCALERS: Dict[float, object] = {}
# # #             self._pcas_loaded                 = False
# # #             self.INVERSE_TRANSFORM_MATRICES: Optional[np.ndarray] = None
# # #             self.INVERSE_TRANSFORM_OFFSETS:  Optional[np.ndarray] = None
# # #             self._load_pcas_and_scalers()

# # #             # Warm-up: two forward passes to trigger XLA/TF tracing.
# # #             # Uses LCDM-like params; w0+wa = -1.0 is valid for both modes.
# # #             logging.info("[PkEmulator] Warming up neural network...")
# # #             dummy_params = np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, -1.0]], dtype=np.float32)
# # #             dummy_norm   = self.param_scaler.transform(dummy_params)
# # #             dummy_input  = self._make_network_input(dummy_norm)
# # #             dummy_tf     = tf.constant(dummy_input, dtype=tf.float32)
# # #             _ = self._compiled_inference(dummy_tf)
# # #             _ = self._compiled_inference(dummy_tf)

# # #             logging.info(
# # #                 f"[PkEmulator] Initialisation complete. "
# # #                 f"use_boost={self.use_boost} "
# # #                 f"({'nonlinear boost' if self.use_boost else 'syren/lin ratio'} mode)"
# # #             )

# # #         except FileNotFoundError as e:
# # #             logging.error(f"Required file not found: {e.filename}")
# # #             logging.warning(
# # #                 f"Check that '{self.METADATA_DIR}' and '{self.MODEL_DIR}' "
# # #                 "contain all required files."
# # #             )
# # #             raise

# # #     # -------------------------------------------------------
# # #     # Metadata loading — bundle first, per-file fallback
# # #     # -------------------------------------------------------

# # #     def _load_metadata(self) -> None:
# # #         """
# # #         Load scalers, tPCA, use_boost flag, and (optionally) NPCE indices.

# # #         Strategy
# # #         --------
# # #         1. Look for ``metadata.joblib`` in METADATA_DIR (bundle written by
# # #            COLASet.prepare()).  This is the preferred path.
# # #         2. If absent, fall back to individual files written by older versions:
# # #                param_scaler_lowk_{n}_batches
# # #                t_components_pca_lowk
# # #                t_comp_scaler          (optional)
# # #                npce_indices.npy       (NPCE only, also checked in MODEL_DIR)

# # #         In both cases the same instance attributes are set so all downstream
# # #         methods are layout-agnostic.
# # #         """
# # #         bundle_path = self.METADATA_DIR / "metadata.joblib"

# # #         if bundle_path.exists():
# # #             logging.info(f"[PkEmulator] Loading metadata from bundle: {bundle_path}")
# # #             bundle = joblib.load(bundle_path)

# # #             self.param_scaler  = bundle["param_scaler"]
# # #             self.t_comp_pca    = bundle["t_comp_pca"]
# # #             self.t_comp_scaler = bundle.get("t_comp_scaler", None)

# # #             # use_boost: written by train_utils_pk_emulator_v2 for nonlinear
# # #             # models.  Fall back to nl_type check for bundles that pre-date
# # #             # this field.
# # #             if "use_boost" in bundle:
# # #                 self.use_boost = bundle["use_boost"]
# # #             else:
# # #                 self.use_boost = (self.nl_type != "lin")
# # #                 if self.use_boost:
# # #                     logging.warning(
# # #                         "[PkEmulator] 'use_boost' not in bundle; "
# # #                         f"inferred use_boost=True from nl_type='{self.nl_type}'."
# # #                     )

# # #             # PCE index matrix (None for MLP bundles)
# # #             self._pce_indices = bundle.get("npce_indices", None)

# # #             # Per-z PCAs and scalers stored inside the bundle as z-keyed dicts.
# # #             # Stash them here so _load_pcas_and_scalers skips disk I/O.
# # #             self._bundle_pcas    = bundle.get("pcas",    None)
# # #             self._bundle_scalers = bundle.get("scalers", None)

# # #             self._metadata_source = "bundle"

# # #         else:
# # #             logging.warning(
# # #                 f"[PkEmulator] Bundle not found at {bundle_path}. "
# # #                 "Falling back to individual metadata files."
# # #             )

# # #             self.param_scaler = joblib.load(
# # #                 self.METADATA_DIR / f"param_scaler_lowk_{self.num_batches}_batches"
# # #             )
# # #             self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")

# # #             t_comp_scaler_path = self.METADATA_DIR / "t_comp_scaler"
# # #             if t_comp_scaler_path.exists():
# # #                 self.t_comp_scaler = joblib.load(t_comp_scaler_path)
# # #                 logging.info("[PkEmulator] t_comp_scaler loaded from file.")
# # #             else:
# # #                 self.t_comp_scaler = None
# # #                 logging.warning(
# # #                     "[PkEmulator] t_comp_scaler not found. "
# # #                     "Assuming model was trained on raw (un-normalised) t-components."
# # #                 )

# # #             # No bundle to read use_boost from — infer from nl_type.
# # #             self.use_boost       = (self.nl_type != "lin")
# # #             self._bundle_pcas    = None   # will be loaded from disk
# # #             self._bundle_scalers = None
# # #             self._pce_indices    = None

# # #             self._metadata_source = "files"

# # #         # NPCE: PCE index matrix — may already be populated from the bundle.
# # #         if self.model_type == "npce" and self._pce_indices is None:
# # #             for pce_path in [
# # #                 self.METADATA_DIR / "npce_indices.npy",
# # #                 self.MODEL_DIR    / "npce_indices.npy",
# # #             ]:
# # #                 if pce_path.exists():
# # #                     self._pce_indices = np.load(pce_path)
# # #                     logging.info(
# # #                         f"[PkEmulator] PCE indices loaded from {pce_path}: "
# # #                         f"{self._pce_indices.shape[0]} terms, "
# # #                         f"{self._pce_indices.shape[1]} dims."
# # #                     )
# # #                     break
# # #             if self._pce_indices is None:
# # #                 raise FileNotFoundError(
# # #                     "PCE index file not found. Expected at:\n"
# # #                     f"  {self.METADATA_DIR / 'npce_indices.npy'}\n"
# # #                     f"  {self.MODEL_DIR    / 'npce_indices.npy'}\n"
# # #                     "Re-run training with --model_type npce."
# # #                 )

# # #         if self.t_comp_scaler is None:
# # #             logging.warning(
# # #                 "[PkEmulator] t_comp_scaler is None — "
# # #                 "assuming model was trained on raw (un-normalised) t-components."
# # #             )

# # #         logging.info(
# # #             f"[PkEmulator] Metadata loaded (source={self._metadata_source}, "
# # #             f"use_boost={self.use_boost})."
# # #         )

# # #     # -------------------------------------------------------

# # #     def _load_model(
# # #         self,
# # #         cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max
# # #     ) -> None:
# # #         """Locate and load the Keras model, with legacy .h5 fallback for MLP."""
# # #         model_file = self.MODEL_DIR / _model_filename(
# # #             cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max,
# # #         )
# # #         if not model_file.exists() and model_type == "mlp":
# # #             legacy = self.MODEL_DIR / _model_filename_legacy(
# # #                 cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max,
# # #             )
# # #             if legacy.exists():
# # #                 logging.warning(
# # #                     f"[PkEmulator] .keras file not found, falling back to legacy: {legacy}"
# # #                 )
# # #                 model_file = legacy

# # #         logging.info(f"[PkEmulator] Loading model: {model_file}")
# # #         self.model = keras.models.load_model(
# # #             model_file,
# # #             custom_objects={"CustomActivationLayer": CustomActivationLayer},
# # #             compile=False,
# # #         )

# # #         @tf.function(jit_compile=False)
# # #         def _compiled_inference(x):
# # #             return self.model(x, training=False)
# # #         self._compiled_inference = _compiled_inference

# # #     # -------------------------------------------------------
# # #     # Inference helpers
# # #     # -------------------------------------------------------

# # #     def _make_network_input(self, params_norm: np.ndarray) -> np.ndarray:
# # #         """
# # #         For MLP:  returns params_norm unchanged  (N, n_params).
# # #         For NPCE: evaluates the PCE basis        (N, N_terms).
# # #         """
# # #         if self.model_type == "mlp":
# # #             return params_norm.astype(np.float32)
# # #         return self._evaluate_pce_basis(params_norm)

# # #     def _evaluate_pce_basis(self, X_norm: np.ndarray) -> np.ndarray:
# # #         """
# # #         Evaluate the stored PCE multi-index basis at X_norm.

# # #         Parameters
# # #         ----------
# # #         X_norm : (N, n_params) float ndarray — MinMax-normalised inputs in [-1, 1]

# # #         Returns
# # #         -------
# # #         Phi : (N, N_terms) float32 ndarray
# # #         """
# # #         indices   = self._pce_indices                             # (T, D)
# # #         X_clipped = np.clip(X_norm, -1.0, 1.0).astype(np.float64)
# # #         N, D      = X_clipped.shape
# # #         max_deg   = int(indices.max()) if indices.size > 0 else 0

# # #         pow_table = np.empty((D, max_deg + 1, N), dtype=np.float64)
# # #         pow_table[:, 0, :] = 1.0
# # #         if max_deg >= 1:
# # #             pow_table[:, 1, :] = X_clipped.T
# # #         for p in range(2, max_deg + 1):
# # #             pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T

# # #         d_idx   = np.arange(D)[np.newaxis, :]
# # #         powered = pow_table[d_idx, indices, :]   # (T, D, N)
# # #         Phi     = powered.prod(axis=1).T          # (N, T)

# # #         return Phi.astype(np.float32)

# # #     def _load_pcas_and_scalers(self) -> None:
# # #         """
# # #         Load per-redshift PCA/scaler objects and pre-compute the fused inverse
# # #         transform matrices for fast batched inference.

# # #         Uses bundle dicts directly if available; otherwise loads individual
# # #         .pca and .frac_pks_scaler files from METADATA_DIR.
# # #         """
# # #         if self._pcas_loaded:
# # #             return

# # #         if self._bundle_pcas is not None and self._bundle_scalers is not None:
# # #             logging.info("[PkEmulator] Loading per-z PCA/scalers from bundle dicts.")
# # #             self.PCAS    = self._bundle_pcas
# # #             self.SCALERS = self._bundle_scalers
# # #         else:
# # #             logging.info("[PkEmulator] Loading per-z PCA/scaler files from disk.")
# # #             for z in self.Z_MODES:
# # #                 z_key = float(f"{z:.3f}")
# # #                 self.PCAS[z_key]    = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
# # #                 self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")

# # #         logging.info("[PkEmulator] Pre-computing inverse transformation matrices...")

# # #         inverse_matrices, inverse_offsets = [], []
# # #         for z in self.Z_MODES:
# # #             z_key  = float(f"{z:.3f}")
# # #             pca    = self.PCAS[z_key]
# # #             scaler = self.SCALERS[z_key]

# # #             if hasattr(scaler, "scale_"):
# # #                 scale = scaler.scale_
# # #                 mean  = scaler.mean_
# # #             elif hasattr(scaler, "std"):
# # #                 scale = scaler.std
# # #                 mean  = scaler.mean
# # #             else:
# # #                 raise AttributeError(
# # #                     f"Scaler for z={z:.3f} has neither 'scale_' nor 'std'."
# # #                 )

# # #             inverse_matrices.append(pca.components_ * scale[None, :])
# # #             inverse_offsets.append(pca.mean_ * scale + mean)

# # #         self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
# # #         self.INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)

# # #         self._pcas_loaded = True
# # #         logging.info("[PkEmulator] Inverse transformation matrices ready.")

# # #     def _compute_mps_approximation(self, params: np.ndarray, use_eh: bool = False) -> np.ndarray:
# # #         """
# # #         Analytical P_lin(k, z) approximation via symbolic_pofk + growth factors.

# # #         Parameters
# # #         ----------
# # #         params : 1-D array [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
# # #             Column 6 is w0+wa; wa is derived internally as w0wa - w0.

# # #         Returns
# # #         -------
# # #         np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³
# # #         """
# # #         As, ns, H0_in, Ob, Om, w0, w0wa = params
# # #         wa = w0wa - w0
# # #         h  = H0_in / 100.0

# # #         k_for_plin = self.K_MODES / h
# # #         if use_eh:
# # #             pk_fid = get_eisensteinhu_nw(k_for_plin, As, Om, Ob, h, ns, mnu=0.06, w0=w0, wa=wa)
# # #         else:
# # #             pk_fid = plin_emulated(k_for_plin, Om, Ob, h, ns, As=As, w0=w0, wa=wa)

# # #         a_array = 1.0 / (self.Z_MODES + 1)
# # #         D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# # #         Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
# # #         R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# # #         Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)

# # #         growth_factors = (Dz / D0) ** 2 * (Rz / R0)
# # #         result = pk_fid[None, :] * growth_factors[:, None] / h**3
# # #         return result.astype(np.float32)

# # #     def _predict_fracs_all_z(self, params_norm: np.ndarray) -> np.ndarray:
# # #         """
# # #         NN inference: normalised params -> log-fractional differences for all z.

# # #         Inference chain
# # #         ---------------
# # #         1. [NPCE only] Expand params_norm to PCE basis features
# # #         2. NN(input)               -> t_components_norm  (1, N_T_COMPS)
# # #         3. t_comp_scaler.inverse   -> t_components (raw tPCA space)
# # #         4. t_comp_pca.inverse      -> pcs_flat     (N_ZS * N_PCS,)
# # #         5. Fused PCA+scaler invert -> log_frac     (N_ZS, N_K_MODES)

# # #         Returns
# # #         -------
# # #         np.ndarray of shape (N_ZS, N_K_MODES)
# # #         """
# # #         net_input = self._make_network_input(params_norm)
# # #         input_tf  = tf.constant(net_input, dtype=tf.float32)

# # #         t_comps_norm = self._compiled_inference(input_tf).numpy()

# # #         if self.t_comp_scaler is not None:
# # #             t_comps_raw = self.t_comp_scaler.inverse_transform(t_comps_norm)
# # #         else:
# # #             t_comps_raw = t_comps_norm

# # #         pcs_flat = self.t_comp_pca.inverse_transform(t_comps_raw).astype(np.float32)
# # #         pcs_z    = pcs_flat.reshape(self.N_ZS, self.N_PCS)

# # #         reconstructed = (
# # #             np.einsum("zp,zpk->zk", pcs_z, self.INVERSE_TRANSFORM_MATRICES)
# # #             + self.INVERSE_TRANSFORM_OFFSETS
# # #         )
# # #         return reconstructed.astype(np.float32)

# # #     # -------------------------------------------------------
# # #     # Public interface
# # #     # -------------------------------------------------------

# # #     def get_pks(
# # #         self,
# # #         params: List[float],
# # #         use_approximation_only: bool = False,
# # #     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# # #         """
# # #         Return P(k, z) for a single cosmology.

# # #         Parameters
# # #         ----------
# # #         params : list or 1-D array of 7 values
# # #             [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
# # #             Column 6 is w0+wa (not wa).  For LCDM pass w0=-1.0, w0+wa=-1.0.
# # #         use_approximation_only : bool
# # #             If True, skip the NN and return only the syren P_lin approximation.

# # #         Returns
# # #         -------
# # #         k_modes : (N_K_MODES,)
# # #         z_modes : (N_ZS,)
# # #         pks     : (N_ZS, N_K_MODES)

# # #         Notes on emulation mode
# # #         -----------------------
# # #         use_boost=False  (syren/lin mode, nl_type='lin'):
# # #             Network predicts log(P_lin / P_syren).
# # #             Returned pks ≈ P_lin = exp(log_frac) * P_syren.

# # #         use_boost=True  (boost mode, nl_type != 'lin'):
# # #             Network predicts log(P_nonlin / P_lin).
# # #             Returned pks ≈ P_nonlin = exp(log_frac) * P_syren.

# # #         In both cases P_syren is the multiplicative base.
# # #         """
# # #         params_array = np.array(params, dtype=np.float32)
# # #         if params_array.ndim != 1 or len(params_array) != 7:
# # #             raise ValueError(
# # #                 f"Expected 7 parameters [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa], "
# # #                 f"got shape {params_array.shape}."
# # #             )

# # #         # P_syren is always needed: it is the return value when
# # #         # use_approximation_only=True and the multiplicative base otherwise.
# # #         pk_mps = self._compute_mps_approximation(params_array)

# # #         if use_approximation_only:
# # #             return self.K_MODES, self.Z_MODES, pk_mps

# # #         # col 6 is already w0+wa — pass directly to the scaler (no conversion).
# # #         params_norm = self.param_scaler.transform(params_array.reshape(1, -1))
# # #         log_frac    = self._predict_fracs_all_z(params_norm)
# # #         pks         = (np.exp(log_frac) * pk_mps).astype(np.float32)

# # #         return self.K_MODES, self.Z_MODES, pks


# # # # ----------------------------------------------------------------------------------------------------
# # # # Module-level interface with instance caching
# # # # ----------------------------------------------------------------------------------------------------

# # # _emulator_cache: Dict[Tuple, PkEmulator] = {}


# # # def get_emulator(
# # #     cosmo_type: str = "w0wacdm",
# # #     prior_type: str = "constrained",
# # #     nl_type: str = "lin",
# # #     model_type: str = "mlp",
# # #     base_model_path: str = "models",
# # #     base_metadata_path: str = "metadata",
# # #     num_batches: int = 15,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> PkEmulator:
# # #     """
# # #     Return a (cached) PkEmulator for the requested configuration.

# # #     Repeated calls with identical arguments return the same instance without
# # #     reloading files.  The cache key covers all arguments that affect which
# # #     files are loaded.
# # #     """
# # #     if not _DEPENDENCIES_LOADED:
# # #         raise RuntimeError("Cannot create PkEmulator: missing dependencies.")

# # #     cache_key = (cosmo_type, prior_type, nl_type, model_type, num_batches, w0_min, w0wa_max)
# # #     if cache_key in _emulator_cache:
# # #         logging.info(f"[get_emulator] Returning cached emulator for {cache_key}")
# # #         return _emulator_cache[cache_key]

# # #     logging.info(f"[get_emulator] Creating new emulator for {cache_key}")
# # #     emulator = PkEmulator(
# # #         cosmo_type=cosmo_type,
# # #         prior_type=prior_type,
# # #         nl_type=nl_type,
# # #         model_type=model_type,
# # #         base_model_path=base_model_path,
# # #         base_metadata_path=base_metadata_path,
# # #         num_batches=num_batches,
# # #         w0_min=w0_min,
# # #         w0wa_max=w0wa_max,
# # #     )
# # #     _emulator_cache[cache_key] = emulator
# # #     return emulator


# # # def get_pks(
# # #     params: List[float],
# # #     cosmo_type: str = "w0wacdm",
# # #     prior_type: str = "constrained",
# # #     nl_type: str = "lin",
# # #     model_type: str = "mlp",
# # #     num_batches: int = 15,
# # #     use_approximation_only: bool = False,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# # #     """
# # #     Convenience function: get P(k, z) without managing emulator instances.

# # #     Parameters
# # #     ----------
# # #     params : list of 7 floats [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
# # #         Column 6 is w0+wa (not wa).
# # #     model_type : str
# # #         'mlp' or 'npce'. Must match the architecture used during training.
# # #     w0_min : float or None
# # #         Must match the value passed to --w0_min during training.
# # #     w0wa_max : float or None
# # #         Must match the value passed to --w0wa_max during training.

# # #     Returns
# # #     -------
# # #     k_modes, z_modes, pks
# # #     """
# # #     emulator = get_emulator(
# # #         cosmo_type=cosmo_type,
# # #         prior_type=prior_type,
# # #         nl_type=nl_type,
# # #         model_type=model_type,
# # #         num_batches=num_batches,
# # #         w0_min=w0_min,
# # #         w0wa_max=w0wa_max,
# # #     )
# # #     return emulator.get_pks(params, use_approximation_only=use_approximation_only)








# # # # Author: Victoria Lloyd (2025) & V. Miranda (as seen in model evaluation)
# # # import os
# # # import numpy as np
# # # import joblib
# # # from typing import Dict, List, Tuple, Optional
# # # import logging
# # # from pathlib import Path
# # # import sys
# # # import train_utils_pk_emulator_v2 as utils
# # # sys.modules['train_utils_pk_emulator'] = utils
# # # from train_utils_pk_emulator_v2 import CustomActivationLayer, TComponentScaler
# # # from keras.losses import MeanSquaredError, Huber
# # # import tensorflow as tf


# # # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# # # # ----------------------------------------------------------------------------------------------------
# # # # Path helpers
# # # # ----------------------------------------------------------------------------------------------------

# # # def _get_project_root() -> Path:
# # #     """Returns the directory containing this file."""
# # #     return Path(__file__).resolve().parent

# # # ROOT = _get_project_root()

# # # # ----------------------------------------------------------------------------------------------------
# # # # Dependency guard
# # # # ----------------------------------------------------------------------------------------------------

# # # try:
# # #     from tensorflow import keras
# # #     import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk")
# # #     from symbolic_pofk.linear_VL import plin_emulated, get_approximate_D, growth_correction_R, get_eisensteinhu_nw
# # #     _DEPENDENCIES_LOADED = True
# # # except ImportError as e:
# # #     logging.error("FATAL ERROR: A required dependency could not be imported.")
# # #     logging.error(f"Missing component: {e.name}")
# # #     logging.error("If running locally, ensure the symbolic_pofk library is accessible.")
# # #     _DEPENDENCIES_LOADED = False

# # # try:
# # #     from sklearn.decomposition import PCA
# # #     from sklearn.preprocessing import StandardScaler
# # # except ImportError:
# # #     pass


# # # # ----------------------------------------------------------------------------------------------------
# # # # Registry
# # # # ----------------------------------------------------------------------------------------------------

# # # NL_TYPE_REGISTRY = {
# # #     "lin":                      ("pklin",                           "Linear P(k)"),
# # #     "halofit":                  ("pknonlin",                        "Non-linear P(k) via HaloFit"),
# # #     "mead2020":                 ("mead2020_pknonlin",               "Non-linear P(k) via HMcode Mead2020"),
# # #     "mead2020_feedback":        ("mead2020_feedback_pknonlin",      "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (fixed T_AGN)"),
# # #     "mead2020_feedback_Tfree":  ("mead2020_feedback_Tfree_pknonlin","Non-linear P(k) via HMcode Mead2020 + baryonic feedback (free T_AGN)"),
# # # }

# # # VALID_NL_TYPES_BY_PRIOR = {
# # #     "expanded":    {"lin", "halofit"},
# # #     "constrained": set(NL_TYPE_REGISTRY.keys()),
# # # }

# # # VALID_COSMO_TYPES  = {"lcdm", "w0wacdm"}
# # # VALID_MODEL_TYPES  = {"mlp", "npce"}

# # # VER = "_v5"


# # # def _validate_config(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     model_type: str = "mlp",
# # # ) -> None:
# # #     if cosmo_type not in VALID_COSMO_TYPES:
# # #         raise ValueError(
# # #             f"Unknown cosmo_type '{cosmo_type}'. Must be one of: {sorted(VALID_COSMO_TYPES)}"
# # #         )
# # #     if prior_type not in VALID_NL_TYPES_BY_PRIOR:
# # #         raise ValueError(
# # #             f"Unknown prior_type '{prior_type}'. Must be one of: {sorted(VALID_NL_TYPES_BY_PRIOR)}"
# # #         )
# # #     if nl_type not in NL_TYPE_REGISTRY:
# # #         raise ValueError(
# # #             f"Unknown nl_type '{nl_type}'. Must be one of: {sorted(NL_TYPE_REGISTRY)}"
# # #         )
# # #     if nl_type not in VALID_NL_TYPES_BY_PRIOR[prior_type]:
# # #         raise ValueError(
# # #             f"nl_type '{nl_type}' is not available for prior_type='{prior_type}'. "
# # #             f"Valid choices: {sorted(VALID_NL_TYPES_BY_PRIOR[prior_type])}"
# # #         )
# # #     if cosmo_type == "lcdm" and prior_type == "expanded":
# # #         raise ValueError("prior_type='expanded' is only supported for cosmo_type='w0wacdm'.")
# # #     if model_type not in VALID_MODEL_TYPES:
# # #         raise ValueError(
# # #             f"Unknown model_type '{model_type}'. Must be one of: {sorted(VALID_MODEL_TYPES)}"
# # #         )


# # # def _metadata_tag(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> str:
# # #     """
# # #     Reproduce the tag written by COLASet._metadata_tag() during training.

# # #     Format: {cosmo_type}_{prior_type}[_w0min{w0_min}][_w0wamax{w0wa_max}]
# # #             _{nl_type}[_boost][_kmin{BOOST_K_MIN}]_nTrain{num_batches}{VER}

# # #     For nonlinear nl_types the '_boost' and '_kmin' suffixes are appended to
# # #     match the updated COLASet._metadata_tag() convention.  Linear models keep
# # #     the original tag format unchanged.
# # #     """
# # #     w0_tag   = f"_w0min{w0_min}"     if w0_min   is not None else ""
# # #     w0wa_tag = f"_w0wamax{w0wa_max}" if w0wa_max is not None else ""
# # #     mode_tag = "_boost"                          if nl_type != "lin" else ""
# # #     kmin_tag = f"_kmin{utils.BOOST_K_MIN:.0e}"  if nl_type != "lin" else ""
# # #     return (
# # #         f"{cosmo_type}_{prior_type}{w0_tag}{w0wa_tag}"
# # #         f"_{nl_type}{mode_tag}{kmin_tag}_nTrain{num_batches}{VER}"
# # #     )


# # # def _model_filename(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     model_type: str = "mlp",
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> str:
# # #     tag = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
# # #     return f"emulator_{model_type}_{tag}.keras"


# # # def _model_filename_legacy(
# # #     cosmo_type: str,
# # #     prior_type: str,
# # #     nl_type: str,
# # #     num_batches: int,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> str:
# # #     """Legacy .h5 filename for MLP models saved before the format change."""
# # #     tag = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
# # #     return f"emulator_{tag}.h5"


# # # # ----------------------------------------------------------------------------------------------------
# # # # Core emulator class
# # # # ----------------------------------------------------------------------------------------------------

# # # class PkEmulator:
# # #     """
# # #     Cosmology emulator for the matter power spectrum P(k, z).

# # #     Supports multiple cosmological models, prior widths, nonlinear prescriptions,
# # #     and both MLP and NPCE model architectures.

# # #     Input convention
# # #     ----------------
# # #     All public methods expect params = [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa].
# # #     Column 6 is w0+wa (not wa).  This matches the training convention in
# # #     COLASet (col 6 of lhs is overwritten with w0+wa before fitting the scaler).
# # #     The syren approximation receives the same vector and derives wa internally
# # #     as w0wa - w0.

# # #     Emulation modes
# # #     ---------------
# # #     The mode is determined automatically from the metadata bundle:

# # #     - **syren/lin mode** (use_boost=False):
# # #         Network learns  log(P_lin / P_syren).
# # #         get_pks returns  exp(log_frac) * P_syren  ≈  P_lin.
# # #         Full k-grid is used throughout; no truncation or padding.

# # #     - **boost mode** (use_boost=True):
# # #         Network learns  log(P_nonlin / P_lin).
# # #         The emulator was trained on the truncated k-grid k >= BOOST_K_MIN.
# # #         _predict_fracs_all_z returns (N_ZS, N_K_BOOST); get_pks pads with
# # #         0.0 at k < BOOST_K_MIN (i.e. boost = 1 there) before multiplying
# # #         the full P_syren base to return (N_ZS, N_K_MODES).

# # #     N_PCS is inferred automatically from the first per-z PCA stored in the
# # #     metadata bundle, so different nonlinear prescriptions trained with
# # #     different numbers of PCs are handled transparently.

# # #     Metadata loading
# # #     ----------------
# # #     The constructor first looks for a single ``metadata.joblib`` bundle
# # #     (written by COLASet.prepare()).  If absent it falls back to the individual
# # #     per-file layout used by older script versions (separate param_scaler,
# # #     t_components_pca, t_comp_scaler, and per-z .pca / .frac_pks_scaler files).

# # #     Parameters
# # #     ----------
# # #     cosmo_type : str
# # #         Cosmological model: 'w0wacdm' or 'lcdm'.
# # #     prior_type : str
# # #         Prior width used during training: 'expanded' or 'constrained'.
# # #     nl_type : str
# # #         Nonlinear prescription. See NL_TYPE_REGISTRY for valid options.
# # #     model_type : str
# # #         Architecture of the trained model: 'mlp' or 'npce'. Default 'mlp'.
# # #     base_model_path : str
# # #         Directory containing model files, relative to this file.
# # #     base_metadata_path : str
# # #         Parent directory of all metadata subdirectories, relative to this file.
# # #     num_batches : int
# # #         Number of training batches — used to locate metadata files.
# # #     w0_min : float or None
# # #         Lower bound on w0 used during training (embedded in filenames).
# # #     w0wa_max : float or None
# # #         Upper bound on w0+wa used during training (embedded in filenames).
# # #     """

# # #     # k-grid and z-grid are shared across all instances and prescriptions.
# # #     # N_PCS and N_K_BOOST are set per-instance in _load_metadata because they
# # #     # differ between linear and nonlinear prescriptions.
# # #     N_K_MODES = 500

# # #     K_MODES = np.logspace(-5.1, 2, N_K_MODES)
# # #     Z_MODES = np.concatenate((
# # #         np.linspace(0,  3,  33, endpoint=False),
# # #         np.linspace(3,  10,  7, endpoint=False),
# # #         np.linspace(10, 50, 12),
# # #     ))
# # #     N_ZS = len(Z_MODES)

# # #     def __init__(
# # #         self,
# # #         cosmo_type: str = "w0wacdm",
# # #         prior_type: str = "constrained",
# # #         nl_type: str = "lin",
# # #         model_type: str = "mlp",
# # #         base_model_path: str = "models",
# # #         base_metadata_path: str = "metadata",
# # #         num_batches: int = 15,
# # #         w0_min: Optional[float] = None,
# # #         w0wa_max: Optional[float] = None,
# # #     ):
# # #         if not _DEPENDENCIES_LOADED:
# # #             raise RuntimeError("Cannot initialise PkEmulator: missing dependencies.")

# # #         _validate_config(cosmo_type, prior_type, nl_type, num_batches, model_type)

# # #         self.cosmo_type  = cosmo_type
# # #         self.prior_type  = prior_type
# # #         self.nl_type     = nl_type
# # #         self.model_type  = model_type
# # #         self.num_batches = num_batches
# # #         self.w0_min      = w0_min
# # #         self.w0wa_max    = w0wa_max

# # #         # N_PCS and N_K_BOOST are resolved in _load_metadata once the PCA
# # #         # objects are available.  Setting to None here makes it explicit that
# # #         # they are not yet known.
# # #         self.N_PCS    = None
# # #         self.N_K_BOOST = None

# # #         logging.info(
# # #             f"[PkEmulator] Initialising: cosmo_type='{cosmo_type}', "
# # #             f"prior_type='{prior_type}', nl_type='{nl_type}', "
# # #             f"model_type='{model_type}', "
# # #             f"w0_min={w0_min}, w0wa_max={w0wa_max}"
# # #         )

# # #         self.MODEL_DIR    = ROOT / base_model_path
# # #         self.METADATA_DIR = (
# # #             ROOT / base_metadata_path
# # #             / f"metadata_{_metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)}"
# # #         )

# # #         logging.info(f"[PkEmulator] Metadata directory: {self.METADATA_DIR}")

# # #         try:
# # #             self._load_metadata()
# # #             self._load_model(cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max)

# # #             # Per-redshift PCA and scalers
# # #             self.PCAS:    Dict[float, PCA]    = {}
# # #             self.SCALERS: Dict[float, object] = {}
# # #             self._pcas_loaded                 = False
# # #             self.INVERSE_TRANSFORM_MATRICES: Optional[np.ndarray] = None
# # #             self.INVERSE_TRANSFORM_OFFSETS:  Optional[np.ndarray] = None
# # #             self._load_pcas_and_scalers()

# # #             # Warm-up: two forward passes to trigger XLA/TF tracing.
# # #             # Uses LCDM-like params; w0+wa = -1.0 is valid for both modes.
# # #             logging.info("[PkEmulator] Warming up neural network...")
# # #             dummy_params = np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, -1.0]], dtype=np.float32)
# # #             dummy_norm   = self.param_scaler.transform(dummy_params)
# # #             dummy_input  = self._make_network_input(dummy_norm)
# # #             dummy_tf     = tf.constant(dummy_input, dtype=tf.float32)
# # #             _ = self._compiled_inference(dummy_tf)
# # #             _ = self._compiled_inference(dummy_tf)

# # #             logging.info(
# # #                 f"[PkEmulator] Initialisation complete. "
# # #                 f"use_boost={self.use_boost}, N_PCS={self.N_PCS}, "
# # #                 f"N_K_BOOST={self.N_K_BOOST} "
# # #                 f"({'nonlinear boost' if self.use_boost else 'syren/lin ratio'} mode)"
# # #             )

# # #         except FileNotFoundError as e:
# # #             logging.error(f"Required file not found: {e.filename}")
# # #             logging.warning(
# # #                 f"Check that '{self.METADATA_DIR}' and '{self.MODEL_DIR}' "
# # #                 "contain all required files."
# # #             )
# # #             raise

# # #     # -------------------------------------------------------
# # #     # Metadata loading — bundle first, per-file fallback
# # #     # -------------------------------------------------------

# # #     def _load_metadata(self) -> None:
# # #         """
# # #         Load scalers, tPCA, use_boost flag, and (optionally) NPCE indices.
# # #         Also infers self.N_PCS and self.N_K_BOOST from the stored PCA objects.

# # #         Strategy
# # #         --------
# # #         1. Look for ``metadata.joblib`` in METADATA_DIR (bundle written by
# # #            COLASet.prepare()).  This is the preferred path.
# # #            N_PCS is read directly from the first per-z PCA in the bundle.

# # #         2. If absent, fall back to individual files written by older versions.
# # #            N_PCS cannot be read until _load_pcas_and_scalers runs, so it is
# # #            left as None here and resolved there.

# # #         In both paths, N_K_BOOST is set once use_boost is known:
# # #           - boost mode  (nl_type != 'lin'): N_K_BOOST = len(ks_boost)
# # #           - linear mode (nl_type == 'lin'): N_K_BOOST = N_K_MODES (no truncation)
# # #         """
# # #         bundle_path = self.METADATA_DIR / "metadata.joblib"

# # #         if bundle_path.exists():
# # #             logging.info(f"[PkEmulator] Loading metadata from bundle: {bundle_path}")
# # #             bundle = joblib.load(bundle_path)

# # #             self.param_scaler  = bundle["param_scaler"]
# # #             self.t_comp_pca    = bundle["t_comp_pca"]
# # #             self.t_comp_scaler = bundle.get("t_comp_scaler", None)

# # #             # use_boost: written by train_utils_pk_emulator_v2 for nonlinear
# # #             # models.  Fall back to nl_type check for bundles that pre-date
# # #             # this field.
# # #             if "use_boost" in bundle:
# # #                 self.use_boost = bundle["use_boost"]
# # #             else:
# # #                 self.use_boost = (self.nl_type != "lin")
# # #                 if self.use_boost:
# # #                     logging.warning(
# # #                         "[PkEmulator] 'use_boost' not in bundle; "
# # #                         f"inferred use_boost=True from nl_type='{self.nl_type}'."
# # #                     )

# # #             # PCE index matrix (None for MLP bundles)
# # #             self._pce_indices = bundle.get("npce_indices", None)

# # #             # Per-z PCAs and scalers stored inside the bundle as z-keyed dicts.
# # #             # Stash them here so _load_pcas_and_scalers skips disk I/O.
# # #             self._bundle_pcas    = bundle.get("pcas",    None)
# # #             self._bundle_scalers = bundle.get("scalers", None)

# # #             # Infer N_PCS from the first per-z PCA stored in the bundle.
# # #             # All per-z PCAs are fitted with the same n_components so any z
# # #             # will do.  This means different prescriptions (halofit, mead2020,
# # #             # lin) are handled transparently without any manual configuration.
# # #             if self._bundle_pcas is not None:
# # #                 _first_pca  = next(iter(self._bundle_pcas.values()))
# # #                 self.N_PCS  = _first_pca.n_components_
# # #                 logging.info(f"[PkEmulator] N_PCS inferred from bundle: {self.N_PCS}")
# # #             else:
# # #                 # Bundle exists but has no 'pcas' key — unusual; will be
# # #                 # resolved in _load_pcas_and_scalers.
# # #                 self.N_PCS = None
# # #                 logging.warning(
# # #                     "[PkEmulator] Bundle has no 'pcas' key; "
# # #                     "N_PCS will be resolved after loading per-z PCA files."
# # #                 )

# # #             self._metadata_source = "bundle"

# # #         else:
# # #             logging.warning(
# # #                 f"[PkEmulator] Bundle not found at {bundle_path}. "
# # #                 "Falling back to individual metadata files."
# # #             )

# # #             self.param_scaler = joblib.load(
# # #                 self.METADATA_DIR / f"param_scaler_lowk_{self.num_batches}_batches"
# # #             )
# # #             self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")

# # #             t_comp_scaler_path = self.METADATA_DIR / "t_comp_scaler"
# # #             if t_comp_scaler_path.exists():
# # #                 self.t_comp_scaler = joblib.load(t_comp_scaler_path)
# # #                 logging.info("[PkEmulator] t_comp_scaler loaded from file.")
# # #             else:
# # #                 self.t_comp_scaler = None
# # #                 logging.warning(
# # #                     "[PkEmulator] t_comp_scaler not found. "
# # #                     "Assuming model was trained on raw (un-normalised) t-components."
# # #                 )

# # #             # No bundle to read use_boost from — infer from nl_type.
# # #             self.use_boost       = (self.nl_type != "lin")
# # #             self._bundle_pcas    = None   # will be loaded from disk
# # #             self._bundle_scalers = None
# # #             self._pce_indices    = None

# # #             # N_PCS cannot be read until the per-z PCA files are loaded in
# # #             # _load_pcas_and_scalers; mark as unresolved for now.
# # #             self.N_PCS = None
# # #             logging.info(
# # #                 "[PkEmulator] N_PCS deferred — will be inferred from per-z PCA files."
# # #             )

# # #             self._metadata_source = "files"

# # #         # N_K_BOOST: the number of k-modes seen by the emulator during training.
# # #         # In boost mode the k-grid was truncated to k >= BOOST_K_MIN, so the
# # #         # PCA components_ have N_K_BOOST columns rather than N_K_MODES.
# # #         # In linear mode the full k-grid is used and no padding is needed.
# # #         if self.use_boost:
# # #             self.N_K_BOOST = int(utils.BOOST_K_MASK.sum())
# # #         else:
# # #             self.N_K_BOOST = self.N_K_MODES

# # #         # NPCE: PCE index matrix — may already be populated from the bundle.
# # #         if self.model_type == "npce" and self._pce_indices is None:
# # #             for pce_path in [
# # #                 self.METADATA_DIR / "npce_indices.npy",
# # #                 self.MODEL_DIR    / "npce_indices.npy",
# # #             ]:
# # #                 if pce_path.exists():
# # #                     self._pce_indices = np.load(pce_path)
# # #                     logging.info(
# # #                         f"[PkEmulator] PCE indices loaded from {pce_path}: "
# # #                         f"{self._pce_indices.shape[0]} terms, "
# # #                         f"{self._pce_indices.shape[1]} dims."
# # #                     )
# # #                     break
# # #             if self._pce_indices is None:
# # #                 raise FileNotFoundError(
# # #                     "PCE index file not found. Expected at:\n"
# # #                     f"  {self.METADATA_DIR / 'npce_indices.npy'}\n"
# # #                     f"  {self.MODEL_DIR    / 'npce_indices.npy'}\n"
# # #                     "Re-run training with --model_type npce."
# # #                 )

# # #         if self.t_comp_scaler is None:
# # #             logging.warning(
# # #                 "[PkEmulator] t_comp_scaler is None — "
# # #                 "assuming model was trained on raw (un-normalised) t-components."
# # #             )

# # #         logging.info(
# # #             f"[PkEmulator] Metadata loaded (source={self._metadata_source}, "
# # #             f"use_boost={self.use_boost}, N_K_BOOST={self.N_K_BOOST})."
# # #         )

# # #     # -------------------------------------------------------

# # #     def _load_model(
# # #         self,
# # #         cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max
# # #     ) -> None:
# # #         """Locate and load the Keras model, with legacy .h5 fallback for MLP."""
# # #         model_file = self.MODEL_DIR / _model_filename(
# # #             cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max,
# # #         )
# # #         if not model_file.exists() and model_type == "mlp":
# # #             legacy = self.MODEL_DIR / _model_filename_legacy(
# # #                 cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max,
# # #             )
# # #             if legacy.exists():
# # #                 logging.warning(
# # #                     f"[PkEmulator] .keras file not found, falling back to legacy: {legacy}"
# # #                 )
# # #                 model_file = legacy

# # #         logging.info(f"[PkEmulator] Loading model: {model_file}")
# # #         self.model = keras.models.load_model(
# # #             model_file,
# # #             custom_objects={"CustomActivationLayer": CustomActivationLayer},
# # #             compile=False,
# # #         )

# # #         @tf.function(jit_compile=False)
# # #         def _compiled_inference(x):
# # #             return self.model(x, training=False)
# # #         self._compiled_inference = _compiled_inference

# # #     # -------------------------------------------------------
# # #     # Inference helpers
# # #     # -------------------------------------------------------

# # #     def _make_network_input(self, params_norm: np.ndarray) -> np.ndarray:
# # #         """
# # #         For MLP:  returns params_norm unchanged  (N, n_params).
# # #         For NPCE: evaluates the PCE basis        (N, N_terms).
# # #         """
# # #         if self.model_type == "mlp":
# # #             return params_norm.astype(np.float32)
# # #         return self._evaluate_pce_basis(params_norm)

# # #     def _evaluate_pce_basis(self, X_norm: np.ndarray) -> np.ndarray:
# # #         """
# # #         Evaluate the stored PCE multi-index basis at X_norm.

# # #         Parameters
# # #         ----------
# # #         X_norm : (N, n_params) float ndarray — MinMax-normalised inputs in [-1, 1]

# # #         Returns
# # #         -------
# # #         Phi : (N, N_terms) float32 ndarray
# # #         """
# # #         indices   = self._pce_indices                             # (T, D)
# # #         X_clipped = np.clip(X_norm, -1.0, 1.0).astype(np.float64)
# # #         N, D      = X_clipped.shape
# # #         max_deg   = int(indices.max()) if indices.size > 0 else 0

# # #         pow_table = np.empty((D, max_deg + 1, N), dtype=np.float64)
# # #         pow_table[:, 0, :] = 1.0
# # #         if max_deg >= 1:
# # #             pow_table[:, 1, :] = X_clipped.T
# # #         for p in range(2, max_deg + 1):
# # #             pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T

# # #         d_idx   = np.arange(D)[np.newaxis, :]
# # #         powered = pow_table[d_idx, indices, :]   # (T, D, N)
# # #         Phi     = powered.prod(axis=1).T          # (N, T)

# # #         return Phi.astype(np.float32)

# # #     def _load_pcas_and_scalers(self) -> None:
# # #         """
# # #         Load per-redshift PCA/scaler objects and pre-compute the fused inverse
# # #         transform matrices for fast batched inference.

# # #         Uses bundle dicts directly if available; otherwise loads individual
# # #         .pca and .frac_pks_scaler files from METADATA_DIR.

# # #         Also resolves self.N_PCS for the file-fallback path (where it could
# # #         not be read in _load_metadata).
# # #         """
# # #         if self._pcas_loaded:
# # #             return

# # #         if self._bundle_pcas is not None and self._bundle_scalers is not None:
# # #             logging.info("[PkEmulator] Loading per-z PCA/scalers from bundle dicts.")
# # #             self.PCAS    = self._bundle_pcas
# # #             self.SCALERS = self._bundle_scalers
# # #         else:
# # #             logging.info("[PkEmulator] Loading per-z PCA/scaler files from disk.")
# # #             for z in self.Z_MODES:
# # #                 z_key = float(f"{z:.3f}")
# # #                 self.PCAS[z_key]    = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
# # #                 self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")

# # #         # Resolve N_PCS if it wasn't available at metadata-load time.
# # #         # This covers the file-fallback path and the unusual case where the
# # #         # bundle existed but had no 'pcas' key.
# # #         if self.N_PCS is None:
# # #             _first_pca = next(iter(self.PCAS.values()))
# # #             self.N_PCS = _first_pca.n_components_
# # #             logging.info(
# # #                 f"[PkEmulator] N_PCS inferred from per-z PCA files: {self.N_PCS}"
# # #             )

# # #         logging.info("[PkEmulator] Pre-computing inverse transformation matrices...")

# # #         inverse_matrices, inverse_offsets = [], []
# # #         for z in self.Z_MODES:
# # #             z_key  = float(f"{z:.3f}")
# # #             pca    = self.PCAS[z_key]
# # #             scaler = self.SCALERS[z_key]

# # #             if hasattr(scaler, "scale_"):
# # #                 scale = scaler.scale_
# # #                 mean  = scaler.mean_
# # #             elif hasattr(scaler, "std"):
# # #                 scale = scaler.std
# # #                 mean  = scaler.mean
# # #             else:
# # #                 raise AttributeError(
# # #                     f"Scaler for z={z:.3f} has neither 'scale_' nor 'std'."
# # #                 )

# # #             inverse_matrices.append(pca.components_ * scale[None, :])
# # #             inverse_offsets.append(pca.mean_ * scale + mean)

# # #         self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
# # #         self.INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)

# # #         self._pcas_loaded = True
# # #         logging.info(
# # #             f"[PkEmulator] Inverse transformation matrices ready. "
# # #             f"Shape: {self.INVERSE_TRANSFORM_MATRICES.shape} "
# # #             f"(N_ZS={self.N_ZS}, N_PCS={self.N_PCS}, N_K_BOOST={self.N_K_BOOST})"
# # #         )

# # #     def _compute_mps_approximation(self, params: np.ndarray, use_eh: bool = False) -> np.ndarray:
# # #         """
# # #         Analytical P_lin(k, z) approximation via symbolic_pofk + growth factors.

# # #         Parameters
# # #         ----------
# # #         params : 1-D array [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
# # #             Column 6 is w0+wa; wa is derived internally as w0wa - w0.

# # #         Returns
# # #         -------
# # #         np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³
# # #         """
# # #         As, ns, H0_in, Ob, Om, w0, w0wa = params
# # #         wa = w0wa - w0
# # #         h  = H0_in / 100.0

# # #         k_for_plin = self.K_MODES / h
# # #         if use_eh:
# # #             pk_fid = get_eisensteinhu_nw(k_for_plin, As, Om, Ob, h, ns, mnu=0.06, w0=w0, wa=wa)
# # #         else:
# # #             pk_fid = plin_emulated(k_for_plin, Om, Ob, h, ns, As=As, w0=w0, wa=wa)

# # #         a_array = 1.0 / (self.Z_MODES + 1)
# # #         D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# # #         Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
# # #         R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# # #         Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)

# # #         growth_factors = (Dz / D0) ** 2 * (Rz / R0)
# # #         result = pk_fid[None, :] * growth_factors[:, None] / h**3
# # #         return result.astype(np.float32)

# # #     def _predict_fracs_all_z(self, params_norm: np.ndarray) -> np.ndarray:
# # #         """
# # #         NN inference: normalised params -> log-fractional differences for all z.

# # #         Inference chain
# # #         ---------------
# # #         1. [NPCE only] Expand params_norm to PCE basis features
# # #         2. NN(input)               -> t_components_norm  (1, N_T_COMPS)
# # #         3. t_comp_scaler.inverse   -> t_components (raw tPCA space)
# # #         4. t_comp_pca.inverse      -> pcs_flat     (N_ZS * N_PCS,)
# # #         5. Fused PCA+scaler invert -> log_frac     (N_ZS, N_K_BOOST)

# # #         Returns
# # #         -------
# # #         np.ndarray of shape (N_ZS, N_K_BOOST)
# # #             In linear mode N_K_BOOST == N_K_MODES so no padding is needed.
# # #             In boost mode N_K_BOOST < N_K_MODES; get_pks handles the padding.
# # #         """
# # #         net_input = self._make_network_input(params_norm)
# # #         input_tf  = tf.constant(net_input, dtype=tf.float32)

# # #         t_comps_norm = self._compiled_inference(input_tf).numpy()

# # #         if self.t_comp_scaler is not None:
# # #             t_comps_raw = self.t_comp_scaler.inverse_transform(t_comps_norm)
# # #         else:
# # #             t_comps_raw = t_comps_norm

# # #         pcs_flat = self.t_comp_pca.inverse_transform(t_comps_raw).astype(np.float32)
# # #         pcs_z    = pcs_flat.reshape(self.N_ZS, self.N_PCS)

# # #         reconstructed = (
# # #             np.einsum("zp,zpk->zk", pcs_z, self.INVERSE_TRANSFORM_MATRICES)
# # #             + self.INVERSE_TRANSFORM_OFFSETS
# # #         )
# # #         return reconstructed.astype(np.float32)

# # #     # -------------------------------------------------------
# # #     # Public interface
# # #     # -------------------------------------------------------

# # #     def get_pks(
# # #         self,
# # #         params: List[float],
# # #         use_approximation_only: bool = False,
# # #     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# # #         """
# # #         Return P(k, z) for a single cosmology.

# # #         Parameters
# # #         ----------
# # #         params : list or 1-D array of 7 values
# # #             [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
# # #             Column 6 is w0+wa (not wa).  For LCDM pass w0=-1.0, w0+wa=-1.0.
# # #         use_approximation_only : bool
# # #             If True, skip the NN and return only the syren P_lin approximation.

# # #         Returns
# # #         -------
# # #         k_modes : (N_K_MODES,)
# # #         z_modes : (N_ZS,)
# # #         pks     : (N_ZS, N_K_MODES)

# # #         Notes on emulation mode
# # #         -----------------------
# # #         use_boost=False  (syren/lin mode, nl_type='lin'):
# # #             Network predicts log(P_lin / P_syren) on the full k-grid.
# # #             Returned pks ≈ P_lin = exp(log_frac) * P_syren.
# # #             No truncation or padding; behaviour identical to the original code.

# # #         use_boost=True  (boost mode, nl_type != 'lin'):
# # #             Network predicts log(P_nonlin / P_lin) on the truncated k-grid
# # #             (k >= BOOST_K_MIN).  log_frac is padded with 0.0 at k < BOOST_K_MIN
# # #             so exp(log_frac) = 1 there (no boost correction).
# # #             Returned pks ≈ P_nonlin = exp(log_frac_padded) * P_syren.
# # #         """
# # #         params_array = np.array(params, dtype=np.float32)
# # #         if params_array.ndim != 1 or len(params_array) != 7:
# # #             raise ValueError(
# # #                 f"Expected 7 parameters [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa], "
# # #                 f"got shape {params_array.shape}."
# # #             )

# # #         # P_syren is always needed: it is the return value when
# # #         # use_approximation_only=True and the multiplicative base otherwise.
# # #         pk_mps = self._compute_mps_approximation(params_array)

# # #         if use_approximation_only:
# # #             return self.K_MODES, self.Z_MODES, pk_mps

# # #         params_norm = self.param_scaler.transform(params_array.reshape(1, -1))
# # #         log_frac    = self._predict_fracs_all_z(params_norm)
# # #         # log_frac is (N_ZS, N_K_BOOST); in linear mode N_K_BOOST == N_K_MODES
# # #         # so the branch below is skipped and behaviour is unchanged.

# # #         if self.use_boost:
# # #             # Network was trained on the truncated k-grid (k >= BOOST_K_MIN).
# # #             # Pad with 0.0 at k < BOOST_K_MIN so exp(log_frac) = 1 there,
# # #             # meaning no nonlinear boost correction in the low-k region.
# # #             log_frac_full = np.zeros((self.N_ZS, self.N_K_MODES), dtype=np.float32)
# # #             log_frac_full[:, utils.BOOST_K_MASK] = log_frac
# # #             log_frac = log_frac_full

# # #         pks = (np.exp(log_frac) * pk_mps).astype(np.float32)

# # #         return self.K_MODES, self.Z_MODES, pks


# # # # ----------------------------------------------------------------------------------------------------
# # # # Module-level interface with instance caching
# # # # ----------------------------------------------------------------------------------------------------

# # # _emulator_cache: Dict[Tuple, PkEmulator] = {}


# # # def get_emulator(
# # #     cosmo_type: str = "w0wacdm",
# # #     prior_type: str = "constrained",
# # #     nl_type: str = "lin",
# # #     model_type: str = "mlp",
# # #     base_model_path: str = "models",
# # #     base_metadata_path: str = "metadata",
# # #     num_batches: int = 15,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> PkEmulator:
# # #     """
# # #     Return a (cached) PkEmulator for the requested configuration.

# # #     Repeated calls with identical arguments return the same instance without
# # #     reloading files.  The cache key covers all arguments that affect which
# # #     files are loaded.

# # #     N_PCS is inferred automatically from the stored metadata bundle so it does
# # #     not need to be passed here — different prescriptions with different PC
# # #     counts are handled transparently.
# # #     """
# # #     if not _DEPENDENCIES_LOADED:
# # #         raise RuntimeError("Cannot create PkEmulator: missing dependencies.")

# # #     cache_key = (cosmo_type, prior_type, nl_type, model_type, num_batches, w0_min, w0wa_max)
# # #     if cache_key in _emulator_cache:
# # #         logging.info(f"[get_emulator] Returning cached emulator for {cache_key}")
# # #         return _emulator_cache[cache_key]

# # #     logging.info(f"[get_emulator] Creating new emulator for {cache_key}")
# # #     emulator = PkEmulator(
# # #         cosmo_type=cosmo_type,
# # #         prior_type=prior_type,
# # #         nl_type=nl_type,
# # #         model_type=model_type,
# # #         base_model_path=base_model_path,
# # #         base_metadata_path=base_metadata_path,
# # #         num_batches=num_batches,
# # #         w0_min=w0_min,
# # #         w0wa_max=w0wa_max,
# # #     )
# # #     _emulator_cache[cache_key] = emulator
# # #     return emulator


# # # def get_pks(
# # #     params: List[float],
# # #     cosmo_type: str = "w0wacdm",
# # #     prior_type: str = "constrained",
# # #     nl_type: str = "lin",
# # #     model_type: str = "mlp",
# # #     num_batches: int = 15,
# # #     use_approximation_only: bool = False,
# # #     w0_min: Optional[float] = None,
# # #     w0wa_max: Optional[float] = None,
# # # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# # #     """
# # #     Convenience function: get P(k, z) without managing emulator instances.

# # #     Parameters
# # #     ----------
# # #     params : list of 7 floats [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
# # #         Column 6 is w0+wa (not wa).
# # #     model_type : str
# # #         'mlp' or 'npce'. Must match the architecture used during training.
# # #     w0_min : float or None
# # #         Must match the value passed to --w0_min during training.
# # #     w0wa_max : float or None
# # #         Must match the value passed to --w0wa_max during training.

# # #     Returns
# # #     -------
# # #     k_modes, z_modes, pks
# # #     """
# # #     emulator = get_emulator(
# # #         cosmo_type=cosmo_type,
# # #         prior_type=prior_type,
# # #         nl_type=nl_type,
# # #         model_type=model_type,
# # #         num_batches=num_batches,
# # #         w0_min=w0_min,
# # #         w0wa_max=w0wa_max,
# # #     )
# # #     return emulator.get_pks(params, use_approximation_only=use_approximation_only)

# # # Author: Victoria Lloyd (2025) & V. Miranda (as seen in model evaluation)
# # import os
# # import numpy as np
# # import joblib
# # from typing import Dict, List, Tuple, Optional
# # import logging
# # from pathlib import Path
# # import sys
# # import train_utils_pk_emulator_v3 as utils
# # sys.modules['train_utils_pk_emulator'] = utils
# # from train_utils_pk_emulator_v3 import CustomActivationLayer, TComponentScaler
# # from keras.losses import MeanSquaredError, Huber
# # import tensorflow as tf


# # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# # # ----------------------------------------------------------------------------------------------------
# # # Path helpers
# # # ----------------------------------------------------------------------------------------------------

# # def _get_project_root() -> Path:
# #     """Returns the directory containing this file."""
# #     return Path(__file__).resolve().parent

# # ROOT = _get_project_root()

# # # ----------------------------------------------------------------------------------------------------
# # # Dependency guard
# # # ----------------------------------------------------------------------------------------------------

# # try:
# #     from tensorflow import keras
# #     import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk")
# #     from symbolic_pofk.linear_VL import (
# #         plin_emulated, get_approximate_D, growth_correction_R,
# #         get_eisensteinhu_nw, As_to_sigma8,
# #     )
# #     from symbolic_pofk.syrenhalofit import run_halofit_vec
# #     _DEPENDENCIES_LOADED = True
# # except ImportError as e:
# #     logging.error("FATAL ERROR: A required dependency could not be imported.")
# #     logging.error(f"Missing component: {e.name}")
# #     logging.error("If running locally, ensure the symbolic_pofk library is accessible.")
# #     _DEPENDENCIES_LOADED = False

# # try:
# #     from sklearn.decomposition import PCA
# #     from sklearn.preprocessing import StandardScaler
# # except ImportError:
# #     pass


# # # ----------------------------------------------------------------------------------------------------
# # # Registry
# # # ----------------------------------------------------------------------------------------------------

# # NL_TYPE_REGISTRY = {
# #     "lin":                        ("pklin",                              "Linear P(k)"),
# #     "halofit":                    ("pknonlin",                           "Non-linear P(k) via HaloFit"),
# #     "mead2020":                   ("mead2020_pknonlin",                  "Non-linear P(k) via HMcode Mead2020"),
# #     "mead2020_feedback":          ("mead2020_feedback_pknonlin",         "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (fixed T_AGN)"),
# #     "mead2020_feedback_Tfree":    ("mead2020_feedback_Tfree_pknonlin",   "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (free T_AGN)"),
# #     "mead2020_Tfree_mnufree_lin": ("mead2020_Tfree_mnufree_pklin",       "Linear P(k) via HMcode Mead2020 + free T_AGN + free mnu"),
# #     "mead2020_Tfree_mnufree":     ("mead2020_Tfree_mnufree_pknonlin",    "Non-linear P(k) via HMcode Mead2020 + free T_AGN + free mnu"),
# # }

# # VALID_NL_TYPES_BY_PRIOR = {
# #     "expanded":    {"lin", "halofit", "mead2020_Tfree_mnufree"},
# #     "constrained": set(NL_TYPE_REGISTRY.keys()),
# # }

# # # nl_types that are treated as boost mode (learn P_nl_camb / P_nl_syren)
# # # All nl_types that are NOT purely linear.
# # _BOOST_NL_TYPES = {
# #     "halofit", "mead2020", "mead2020_feedback", "mead2020_feedback_Tfree",
# #     "mead2020_Tfree_mnufree",
# # }

# # VALID_COSMO_TYPES  = {"lcdm", "w0wacdm"}
# # VALID_MODEL_TYPES  = {"mlp", "npce", "multihead_npce"}

# # VER = "_v7small"


# # def _validate_config(
# #     cosmo_type: str,
# #     prior_type: str,
# #     nl_type: str,
# #     num_batches: int,
# #     model_type: str = "mlp",
# # ) -> None:
# #     if cosmo_type not in VALID_COSMO_TYPES:
# #         raise ValueError(
# #             f"Unknown cosmo_type '{cosmo_type}'. Must be one of: {sorted(VALID_COSMO_TYPES)}"
# #         )
# #     if prior_type not in VALID_NL_TYPES_BY_PRIOR:
# #         raise ValueError(
# #             f"Unknown prior_type '{prior_type}'. Must be one of: {sorted(VALID_NL_TYPES_BY_PRIOR)}"
# #         )
# #     if nl_type not in NL_TYPE_REGISTRY:
# #         raise ValueError(
# #             f"Unknown nl_type '{nl_type}'. Must be one of: {sorted(NL_TYPE_REGISTRY)}"
# #         )
# #     if nl_type not in VALID_NL_TYPES_BY_PRIOR[prior_type]:
# #         raise ValueError(
# #             f"nl_type '{nl_type}' is not available for prior_type='{prior_type}'. "
# #             f"Valid choices: {sorted(VALID_NL_TYPES_BY_PRIOR[prior_type])}"
# #         )
# #     if cosmo_type == "lcdm" and prior_type == "expanded":
# #         raise ValueError("prior_type='expanded' is only supported for cosmo_type='w0wacdm'.")
# #     if model_type not in VALID_MODEL_TYPES:
# #         raise ValueError(
# #             f"Unknown model_type '{model_type}'. Must be one of: {sorted(VALID_MODEL_TYPES)}"
# #         )


# # def _metadata_tag(
# #     cosmo_type: str,
# #     prior_type: str,
# #     nl_type: str,
# #     num_batches: int,
# #     w0_min: Optional[float] = None,
# #     w0wa_max: Optional[float] = None,
# # ) -> str:
# #     """
# #     Reproduce the tag written by COLASet._metadata_tag() during training.

# #     Format: {cosmo_type}_{prior_type}[_w0min{w0_min}][_w0wamax{w0wa_max}]
# #             _{nl_type}[_boost]_nTrain{num_batches}{VER}

# #     The '_kmin' suffix has been removed: k-truncation is no longer applied
# #     since the syren halofit baseline is non-trivial at all k.
# #     """
# #     w0_tag   = f"_w0min{w0_min}"     if w0_min   is not None else ""
# #     w0wa_tag = f"_w0wamax{w0wa_max}" if w0wa_max is not None else ""
# #     mode_tag = "_boost"               if nl_type in _BOOST_NL_TYPES else ""
# #     return (
# #         f"{cosmo_type}_{prior_type}{w0_tag}{w0wa_tag}"
# #         f"_{nl_type}{mode_tag}_nTrain{num_batches}{VER}"
# #     )


# # def _model_filename(
# #     cosmo_type: str,
# #     prior_type: str,
# #     nl_type: str,
# #     num_batches: int,
# #     model_type: str = "mlp",
# #     w0_min: Optional[float] = None,
# #     w0wa_max: Optional[float] = None,
# # ) -> str:
# #     tag = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
# #     return f"emulator_{model_type}_{tag}.keras"


# # def _model_filename_legacy(
# #     cosmo_type: str,
# #     prior_type: str,
# #     nl_type: str,
# #     num_batches: int,
# #     w0_min: Optional[float] = None,
# #     w0wa_max: Optional[float] = None,
# # ) -> str:
# #     """Legacy .h5 filename for MLP models saved before the format change."""
# #     tag = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
# #     return f"emulator_{tag}.h5"


# # # ----------------------------------------------------------------------------------------------------
# # # Core emulator class
# # # ----------------------------------------------------------------------------------------------------

# # class PkEmulator:
# #     """
# #     Cosmology emulator for the matter power spectrum P(k, z).

# #     Emulation modes
# #     ---------------
# #     The mode is determined from nl_type:

# #     - **linear mode** (nl_type == 'lin' or 'mead2020_Tfree_mnufree_lin'):
# #         Network learns  log(P_lin_camb / P_lin_syren).
# #         get_pks returns  exp(log_frac) * P_lin_syren  ≈  P_lin.
# #         Full k-grid used throughout; no truncation or padding.

# #     - **boost mode** (all other nl_types):
# #         Network learns  log(P_nl_camb / P_nl_syren_halofit).
# #         The syren halofit prediction P_nl_syren is computed analytically at
# #         inference time and used as the multiplicative base:
# #             P_nl_emu = exp(log_frac) * P_nl_syren
# #         Full k-grid used throughout — no truncation or padding needed because
# #         the baseline is non-trivial at all k.

# #     Input convention
# #     ----------------
# #     All public methods expect params = [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa].
# #     Column 6 is w0+wa (not wa).  wa is derived internally as w0wa - w0.
# #     """

# #     N_K_MODES = 500

# #     K_MODES = np.logspace(-5.1, 2, N_K_MODES)
# #     Z_MODES = np.concatenate((
# #         np.linspace(0,  3,  33, endpoint=False),
# #         np.linspace(3,  10,  7, endpoint=False),
# #         np.linspace(10, 50, 12),
# #     ))
# #     N_ZS = len(Z_MODES)

# #     def __init__(
# #         self,
# #         cosmo_type: str = "w0wacdm",
# #         prior_type: str = "constrained",
# #         nl_type: str = "lin",
# #         model_type: str = "mlp",
# #         base_model_path: str = "models",
# #         base_metadata_path: str = "metadata",
# #         num_batches: int = 15,
# #         w0_min: Optional[float] = None,
# #         w0wa_max: Optional[float] = None,
# #     ):
# #         if not _DEPENDENCIES_LOADED:
# #             raise RuntimeError("Cannot initialise PkEmulator: missing dependencies.")

# #         _validate_config(cosmo_type, prior_type, nl_type, num_batches, model_type)

# #         self.cosmo_type  = cosmo_type
# #         self.prior_type  = prior_type
# #         self.nl_type     = nl_type
# #         self.model_type  = model_type
# #         self.num_batches = num_batches
# #         self.w0_min      = w0_min
# #         self.w0wa_max    = w0wa_max

# #         # use_boost: True for all nonlinear nl_types
# #         self.use_boost = nl_type in _BOOST_NL_TYPES

# #         self.N_PCS = None   # resolved in _load_metadata / _load_pcas_and_scalers

# #         logging.info(
# #             f"[PkEmulator] Initialising: cosmo_type='{cosmo_type}', "
# #             f"prior_type='{prior_type}', nl_type='{nl_type}', "
# #             f"model_type='{model_type}', use_boost={self.use_boost}, "
# #             f"w0_min={w0_min}, w0wa_max={w0wa_max}"
# #         )

# #         self.MODEL_DIR    = ROOT / base_model_path
# #         self.METADATA_DIR = (
# #             ROOT / base_metadata_path
# #             / f"metadata_{_metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)}"
# #         )

# #         logging.info(f"[PkEmulator] Metadata directory: {self.METADATA_DIR}")

# #         try:
# #             self._load_metadata()
# #             self._load_model(cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max)

# #             self.PCAS:    Dict[float, PCA]    = {}
# #             self.SCALERS: Dict[float, object] = {}
# #             self._pcas_loaded                 = False
# #             self.INVERSE_TRANSFORM_MATRICES: Optional[np.ndarray] = None
# #             self.INVERSE_TRANSFORM_OFFSETS:  Optional[np.ndarray] = None
# #             self._load_pcas_and_scalers()

# #             # Warm-up: two forward passes to trigger XLA/TF tracing.
# #             logging.info("[PkEmulator] Warming up neural network...")
# #             dummy_params = np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, -1.0]], dtype=np.float32)
# #             dummy_norm   = self.param_scaler.transform(dummy_params)
# #             dummy_input  = self._make_network_input(dummy_norm)
# #             dummy_tf     = tf.constant(dummy_input, dtype=tf.float32)
# #             _ = self._compiled_inference(dummy_tf)
# #             _ = self._compiled_inference(dummy_tf)

# #             logging.info(
# #                 f"[PkEmulator] Initialisation complete. "
# #                 f"use_boost={self.use_boost}, N_PCS={self.N_PCS}, "
# #                 f"N_K_MODES={self.N_K_MODES} "
# #                 f"({'NL/syren_halofit residual' if self.use_boost else 'lin/syren ratio'} mode)"
# #             )

# #         except FileNotFoundError as e:
# #             logging.error(f"Required file not found: {e.filename}")
# #             logging.warning(
# #                 f"Check that '{self.METADATA_DIR}' and '{self.MODEL_DIR}' "
# #                 "contain all required files."
# #             )
# #             raise

# #     # -------------------------------------------------------
# #     # Metadata loading
# #     # -------------------------------------------------------

# #     def _load_metadata(self) -> None:
# #         """
# #         Load scalers, tPCA, use_boost flag, and (optionally) NPCE indices.

# #         Tries metadata.joblib bundle first (written by COLASet.prepare()),
# #         falls back to individual files for older bundles.

# #         Note: N_K_BOOST is no longer stored — all PCAs are fitted on the full
# #         k-grid (N_K_MODES=500). The truncation-era field is gone.
# #         """
# #         bundle_path = self.METADATA_DIR / "metadata.joblib"

# #         if bundle_path.exists():
# #             logging.info(f"[PkEmulator] Loading metadata from bundle: {bundle_path}")
# #             bundle = joblib.load(bundle_path)

# #             self.param_scaler  = bundle["param_scaler"]
# #             self.t_comp_pca    = bundle["t_comp_pca"]
# #             self.t_comp_scaler = bundle.get("t_comp_scaler", None)

# #             # use_boost: prefer bundle field, fall back to nl_type check
# #             if "use_boost" in bundle:
# #                 self.use_boost = bundle["use_boost"]
# #             # (already set in __init__ from nl_type; keep that value if not in bundle)

# #             self._pce_indices    = bundle.get("npce_indices", None)
# #             self._bundle_pcas    = bundle.get("pcas",    None)
# #             self._bundle_scalers = bundle.get("scalers", None)

# #             if self._bundle_pcas is not None:
# #                 _first_pca = next(iter(self._bundle_pcas.values()))
# #                 self.N_PCS = _first_pca.n_components_
# #                 logging.info(f"[PkEmulator] N_PCS inferred from bundle: {self.N_PCS}")
# #             else:
# #                 self.N_PCS = None

# #             self._metadata_source = "bundle"

# #         else:
# #             logging.warning(
# #                 f"[PkEmulator] Bundle not found at {bundle_path}. "
# #                 "Falling back to individual metadata files."
# #             )

# #             self.param_scaler = joblib.load(
# #                 self.METADATA_DIR / f"param_scaler_lowk_{self.num_batches}_batches"
# #             )
# #             self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")

# #             t_comp_scaler_path = self.METADATA_DIR / "t_comp_scaler"
# #             if t_comp_scaler_path.exists():
# #                 self.t_comp_scaler = joblib.load(t_comp_scaler_path)
# #             else:
# #                 self.t_comp_scaler = None
# #                 logging.warning(
# #                     "[PkEmulator] t_comp_scaler not found; "
# #                     "assuming model trained on raw (un-normalised) t-components."
# #                 )

# #             self._bundle_pcas    = None
# #             self._bundle_scalers = None
# #             self._pce_indices    = None
# #             self.N_PCS           = None

# #             self._metadata_source = "files"

# #         # NPCE: PCE index matrix
# #         # if self.model_type == "npce" and self._pce_indices is None:
# #         #     for pce_path in [
# #         #         self.METADATA_DIR / "npce_indices.npy",
# #         #         self.MODEL_DIR    / "npce_indices.npy",
# #         #     ]:
# #         #         if pce_path.exists():
# #         #             self._pce_indices = np.load(pce_path)
# #         #             logging.info(
# #         #                 f"[PkEmulator] PCE indices loaded from {pce_path}: "
# #         #                 f"{self._pce_indices.shape[0]} terms, "
# #         #                 f"{self._pce_indices.shape[1]} dims."
# #         #             )
# #         #             break
# #         #     if self._pce_indices is None:
# #         #         raise FileNotFoundError(
# #         #             "PCE index file not found. Expected at:\n"
# #         #             f"  {self.METADATA_DIR / 'npce_indices.npy'}\n"
# #         #             f"  {self.MODEL_DIR    / 'npce_indices.npy'}\n"
# #         #             "Re-run training with --model_type npce."
# #         #         )
# #         if self.model_type in ("npce", "multihead_npce"): # if self.model_type == "npce":
# #             # --- PCE multi-index array ---
# #             if self._pce_indices is None:
# #                 raise FileNotFoundError(
# #                     "npce_indices not found in metadata bundle. "
# #                     "Re-run training with --model_type npce."
# #                 )

# #             # --- PCE coefficient matrix W ---
# #             # Stored directly in the bundle since v5.  No separate file needed.
# #             self._pce_W = bundle.get("npce_W", None)
# #             if self._pce_W is None:
# #                 raise FileNotFoundError(
# #                     "npce_W not found in metadata bundle. "
# #                     "Re-run training, or use the recover_pce_weights.py script "
# #                     "to patch W into an existing bundle without retraining."
# #                 )
# #             self._pce_W = np.asarray(self._pce_W, dtype=np.float32)
# #             logging.info(
# #                 f"[PkEmulator] PCE coefficient matrix W loaded from bundle: "
# #                 f"shape {self._pce_W.shape}  "
# #                 f"(N_t={self._pce_W.shape[0]}, N_terms={self._pce_W.shape[1]})"
# #             )
# #         else:
# #             self._pce_indices = None
# #             self._pce_W       = None

# #         if self.t_comp_scaler is None:
# #             logging.warning(
# #                 "[PkEmulator] t_comp_scaler is None — "
# #                 "assuming model was trained on raw (un-normalised) t-components."
# #             )

# #         logging.info(
# #             f"[PkEmulator] Metadata loaded (source={self._metadata_source}, "
# #             f"use_boost={self.use_boost})."
# #         )

# #     # -------------------------------------------------------

# #     def _load_model(
# #         self,
# #         cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max
# #     ) -> None:
# #         """Locate and load the Keras model, with legacy .h5 fallback for MLP."""
# #         model_file = self.MODEL_DIR / _model_filename(
# #             cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max,
# #         )
# #         if not model_file.exists() and model_type == "mlp":
# #             legacy = self.MODEL_DIR / _model_filename_legacy(
# #                 cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max,
# #             )
# #             if legacy.exists():
# #                 logging.warning(
# #                     f"[PkEmulator] .keras file not found, falling back to legacy: {legacy}"
# #                 )
# #                 model_file = legacy

# #         logging.info(f"[PkEmulator] Loading model: {model_file}")
# #         self.model = keras.models.load_model(
# #             model_file,
# #             custom_objects={"CustomActivationLayer": CustomActivationLayer},
# #             compile=False,
# #         )

# #         @tf.function(jit_compile=False)
# #         def _compiled_inference(x):
# #             return self.model(x, training=False)
# #         self._compiled_inference = _compiled_inference

# #     # -------------------------------------------------------
# #     # Inference helpers
# #     # -------------------------------------------------------

# #     # def _make_network_input(self, params_norm: np.ndarray) -> np.ndarray:
# #     #     """MLP: returns params_norm unchanged. NPCE: evaluates PCE basis."""
# #     #     if self.model_type == "mlp":
# #     #         return params_norm.astype(np.float32)
# #     #     return self._evaluate_pce_basis(params_norm)
# #     def _make_network_input(self, params_norm: np.ndarray) -> np.ndarray:
# #         """
# #         Produce the input tensor the Keras model expects.

# #         MLP:  params_norm unchanged  →  shape (N, n_params)
# #         NPCE: Stage 1 PCE regression →  shape (N, N_t)
# #             t_pce = Φ(params_norm) @ W.T
# #             The Keras model is Stage 2 only and expects t_pce, not Phi.
# #         """
# #         if self.model_type == "mlp":
# #             return params_norm.astype(np.float32)
# #         Phi   = self._evaluate_pce_basis(params_norm)   # (N, N_terms)
# #         t_pce = Phi @ self._pce_W.T                     # (N, N_t)  — Stage 1
# #         return t_pce.astype(np.float32)

# #     def _evaluate_pce_basis(self, X_norm: np.ndarray) -> np.ndarray:
# #         """Evaluate the stored PCE multi-index basis at X_norm (N, n_params) → (N, N_terms)."""
# #         indices   = self._pce_indices
# #         X_clipped = np.clip(X_norm, -1.0, 1.0).astype(np.float64)
# #         N, D      = X_clipped.shape
# #         max_deg   = int(indices.max()) if indices.size > 0 else 0

# #         pow_table = np.empty((D, max_deg + 1, N), dtype=np.float64)
# #         pow_table[:, 0, :] = 1.0
# #         if max_deg >= 1:
# #             pow_table[:, 1, :] = X_clipped.T
# #         for p in range(2, max_deg + 1):
# #             pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T

# #         d_idx   = np.arange(D)[np.newaxis, :]
# #         powered = pow_table[d_idx, indices, :]
# #         Phi     = powered.prod(axis=1).T
# #         return Phi.astype(np.float32)

# #     def _load_pcas_and_scalers(self) -> None:
# #         """
# #         Load per-redshift PCA/scaler objects and pre-compute fused inverse
# #         transform matrices for fast batched inference.

# #         The inverse matrices have shape (N_ZS, N_PCS, N_K_MODES) — full k-grid,
# #         no truncation.
# #         """
# #         if self._pcas_loaded:
# #             return

# #         if self._bundle_pcas is not None and self._bundle_scalers is not None:
# #             logging.info("[PkEmulator] Loading per-z PCA/scalers from bundle dicts.")
# #             self.PCAS    = self._bundle_pcas
# #             self.SCALERS = self._bundle_scalers
# #         else:
# #             logging.info("[PkEmulator] Loading per-z PCA/scaler files from disk.")
# #             for z in self.Z_MODES:
# #                 z_key = float(f"{z:.3f}")
# #                 self.PCAS[z_key]    = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
# #                 self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")

# #         if self.N_PCS is None:
# #             _first_pca = next(iter(self.PCAS.values()))
# #             self.N_PCS = _first_pca.n_components_
# #             logging.info(f"[PkEmulator] N_PCS inferred from per-z PCA files: {self.N_PCS}")

# #         logging.info("[PkEmulator] Pre-computing inverse transformation matrices...")

# #         inverse_matrices, inverse_offsets = [], []
# #         for z in self.Z_MODES:
# #             z_key  = float(f"{z:.3f}")
# #             pca    = self.PCAS[z_key]
# #             scaler = self.SCALERS[z_key]

# #             if hasattr(scaler, "scale_"):
# #                 scale = scaler.scale_
# #                 mean  = scaler.mean_
# #             elif hasattr(scaler, "std"):
# #                 scale = scaler.std
# #                 mean  = scaler.mean
# #             else:
# #                 raise AttributeError(
# #                     f"Scaler for z={z:.3f} has neither 'scale_' nor 'std'."
# #                 )

# #             inverse_matrices.append(pca.components_ * scale[None, :])
# #             inverse_offsets.append(pca.mean_ * scale + mean)

# #         self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
# #         self.INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)

# #         self._pcas_loaded = True
# #         logging.info(
# #             f"[PkEmulator] Inverse transformation matrices ready: "
# #             f"shape={self.INVERSE_TRANSFORM_MATRICES.shape} "
# #             f"(N_ZS={self.N_ZS}, N_PCS={self.N_PCS}, N_K={self.N_K_MODES})"
# #         )

# #     def _compute_syren_lin(self, params: np.ndarray) -> np.ndarray:
# #         """
# #         Analytical P_lin(k, z) from syren + growth factors.

# #         Used as the multiplicative base in linear mode and as an intermediate
# #         step when computing P_nl_syren for boost mode.

# #         Parameters
# #         ----------
# #         params : 1-D array [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]

# #         Returns
# #         -------
# #         np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³
# #         """
# #         As, ns, H0_in, Ob, Om, w0, w0wa = params[:7]
# #         wa = w0wa - w0
# #         h  = H0_in / 100.0

# #         k_hmpc = self.K_MODES / h
# #         pk_lin_hmpc = plin_emulated(k_hmpc, Om, Ob, h, ns, As=As, w0=w0, wa=wa)

# #         a_array = 1.0 / (self.Z_MODES + 1)
# #         D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# #         Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
# #         R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# #         Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)

# #         growth_factors = (Dz / D0) ** 2 * (Rz / R0)
# #         pk_lin_mpc = pk_lin_hmpc[None, :] * growth_factors[:, None] / h**3
# #         return pk_lin_mpc.astype(np.float32)

# #     def _compute_syren_nl(self, params: np.ndarray) -> np.ndarray:
# #         """
# #         Analytical P_nl(k, z) from syren halofit + growth factors.

# #         Used as the multiplicative base in boost mode:
# #             P_nl_emu = exp(log_frac) * P_nl_syren
# #         where log_frac = log(P_nl_camb / P_nl_syren) is what the network learns.

# #         Parameters
# #         ----------
# #         params : 1-D array [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]

# #         Returns
# #         -------
# #         np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³
# #         """
# #         As, ns, H0_in, Ob, Om, w0, w0wa = params[:7]
# #         wa = w0wa - w0
# #         h  = H0_in / 100.0

# #         k_hmpc = self.K_MODES / h

# #         # Linear Pk in (Mpc/h)^3 from syren, then evolved with growth factors
# #         pk_lin_hmpc = plin_emulated(k_hmpc, Om, Ob, h, ns, As=As, w0=w0, wa=wa)

# #         a_array = 1.0 / (self.Z_MODES + 1)
# #         D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# #         Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)
# #         R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=1)
# #         Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=0.06, w0=w0, wa=wa, a=a_array)

# #         growth_factors  = (Dz / D0) ** 2 * (Rz / R0)
# #         pk_lin_hmpc_z   = pk_lin_hmpc[None, :] * growth_factors[:, None]   # (N_ZS, N_K)

# #         sigma8_z0 = As_to_sigma8(As, Om, Ob, h, ns, mnu=0.06, w0=w0, wa=wa)

# #         pk_nl_hmpc_z = run_halofit_vec(
# #             k_hmpc,
# #             sigma8_z0,
# #             Om, Ob, h, ns,
# #             a_array,
# #             return_boost=False,
# #             Plin_in=pk_lin_hmpc_z,
# #         )   # (N_ZS, N_K)

# #         pk_nl_mpc_z = pk_nl_hmpc_z / h**3
# #         return pk_nl_mpc_z.astype(np.float32)

# #     def _predict_fracs_all_z(self, params_norm: np.ndarray) -> np.ndarray:
# #         """
# #         NN inference: normalised params -> log-fractional differences for all z.

# #         Returns
# #         -------
# #         np.ndarray of shape (N_ZS, N_K_MODES)
# #             Full k-grid — no truncation in either mode.
# #         """
# #         net_input = self._make_network_input(params_norm)
# #         input_tf  = tf.constant(net_input, dtype=tf.float32)

# #         t_comps_norm = self._compiled_inference(input_tf).numpy()

# #         if self.t_comp_scaler is not None:
# #             t_comps_raw = self.t_comp_scaler.inverse_transform(t_comps_norm)
# #         else:
# #             t_comps_raw = t_comps_norm

# #         if self.t_comp_pca is not None: #new
# #             pcs_flat = self.t_comp_pca.inverse_transform(t_comps_raw).astype(np.float32)
# #         else:
# #             pcs_flat = t_comps_raw.astype(np.float32)
        
# #         pcs_z    = pcs_flat.reshape(self.N_ZS, self.N_PCS)

# #         reconstructed = (
# #             np.einsum("zp,zpk->zk", pcs_z, self.INVERSE_TRANSFORM_MATRICES)
# #             + self.INVERSE_TRANSFORM_OFFSETS
# #         )
# #         return reconstructed.astype(np.float32)   # (N_ZS, N_K_MODES)

# #     # -------------------------------------------------------
# #     # Public interface
# #     # -------------------------------------------------------

# #     def get_pks(
# #         self,
# #         params: List[float],
# #         use_approximation_only: bool = False,
# #     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# #         """
# #         Return P(k, z) for a single cosmology.

# #         Parameters
# #         ----------
# #         params : list or 1-D array of 7 values
# #             [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
# #             Column 6 is w0+wa (not wa).
# #         use_approximation_only : bool
# #             If True, skip the NN and return the syren baseline only.
# #             In linear mode this is P_lin_syren.
# #             In boost mode this is P_nl_syren (the halofit approximation).

# #         Returns
# #         -------
# #         k_modes : (N_K_MODES,)
# #         z_modes : (N_ZS,)
# #         pks     : (N_ZS, N_K_MODES)

# #         Notes on emulation mode
# #         -----------------------
# #         Linear mode (nl_type == 'lin' or 'mead2020_Tfree_mnufree_lin'):
# #             Network predicts  log(P_lin_camb / P_lin_syren)  on full k-grid.
# #             Returned pks ≈ P_lin = exp(log_frac) * P_lin_syren.

# #         Boost mode (all other nl_types):
# #             Network predicts  log(P_nl_camb / P_nl_syren_halofit)  on full k-grid.
# #             Returned pks ≈ P_nl = exp(log_frac) * P_nl_syren.
# #             No truncation or padding — the halofit baseline is informative
# #             at all k so there is no wasteful flat region to discard.
# #         """
# #         params_array = np.array(params, dtype=np.float32)
# #         if params_array.ndim != 1 or len(params_array) < 7:
# #             raise ValueError(
# #                 f"Expected at least 7 parameters [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa], "
# #                 f"got shape {params_array.shape}."
# #             )

# #         if self.use_boost:
# #             # Boost mode: base is the syren halofit NL prediction
# #             pk_base = self._compute_syren_nl(params_array)
# #         else:
# #             # Linear mode: base is the syren linear prediction
# #             pk_base = self._compute_syren_lin(params_array)

# #         if use_approximation_only:
# #             return self.K_MODES, self.Z_MODES, pk_base

# #         params_norm = self.param_scaler.transform(params_array[:7].reshape(1, -1))
# #         log_frac    = self._predict_fracs_all_z(params_norm)   # (N_ZS, N_K_MODES)

# #         # No masking or padding: log_frac covers the full k-grid in both modes.
# #         pks = (np.exp(log_frac) * pk_base).astype(np.float32)

# #         return self.K_MODES, self.Z_MODES, pks

# #     def get_boost(
# #         self,
# #         params: List[float],
# #     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# #         """
# #         Return the emulated residual boost B(k,z) = P_nl_emu / P_nl_syren.

# #         This is exp(log_frac) directly — useful for diagnostics and for callers
# #         that want to apply the boost to their own linear P(k).

# #         Only meaningful in boost mode; raises if called in linear mode.

# #         Returns
# #         -------
# #         k_modes : (N_K_MODES,)
# #         z_modes : (N_ZS,)
# #         boost   : (N_ZS, N_K_MODES)
# #         """
# #         if not self.use_boost:
# #             raise ValueError(
# #                 "get_boost() is only available in boost mode "
# #                 f"(nl_type='{self.nl_type}' is a linear type)."
# #             )

# #         params_array = np.array(params, dtype=np.float32)
# #         params_norm  = self.param_scaler.transform(params_array[:7].reshape(1, -1))
# #         log_frac     = self._predict_fracs_all_z(params_norm)

# #         return self.K_MODES, self.Z_MODES, np.exp(log_frac).astype(np.float32)


# # # ----------------------------------------------------------------------------------------------------
# # # Module-level interface with instance caching
# # # ----------------------------------------------------------------------------------------------------

# # _emulator_cache: Dict[Tuple, PkEmulator] = {}


# # def get_emulator(
# #     cosmo_type: str = "w0wacdm",
# #     prior_type: str = "constrained",
# #     nl_type: str = "lin",
# #     model_type: str = "mlp",
# #     base_model_path: str = "models",
# #     base_metadata_path: str = "metadata",
# #     num_batches: int = 15,
# #     w0_min: Optional[float] = None,
# #     w0wa_max: Optional[float] = None,
# # ) -> PkEmulator:
# #     """Return a (cached) PkEmulator for the requested configuration."""
# #     if not _DEPENDENCIES_LOADED:
# #         raise RuntimeError("Cannot create PkEmulator: missing dependencies.")

# #     cache_key = (cosmo_type, prior_type, nl_type, model_type, num_batches, w0_min, w0wa_max)
# #     if cache_key in _emulator_cache:
# #         logging.info(f"[get_emulator] Returning cached emulator for {cache_key}")
# #         return _emulator_cache[cache_key]

# #     logging.info(f"[get_emulator] Creating new emulator for {cache_key}")
# #     emulator = PkEmulator(
# #         cosmo_type=cosmo_type,
# #         prior_type=prior_type,
# #         nl_type=nl_type,
# #         model_type=model_type,
# #         base_model_path=base_model_path,
# #         base_metadata_path=base_metadata_path,
# #         num_batches=num_batches,
# #         w0_min=w0_min,
# #         w0wa_max=w0wa_max,
# #     )
# #     _emulator_cache[cache_key] = emulator
# #     return emulator


# # def get_pks(
# #     params: List[float],
# #     cosmo_type: str = "w0wacdm",
# #     prior_type: str = "constrained",
# #     nl_type: str = "lin",
# #     model_type: str = "mlp",
# #     num_batches: int = 15,
# #     use_approximation_only: bool = False,
# #     w0_min: Optional[float] = None,
# #     w0wa_max: Optional[float] = None,
# # ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
# #     """
# #     Convenience function: get P(k, z) without managing emulator instances.

# #     Parameters
# #     ----------
# #     params : list of 7 floats [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
# #         Column 6 is w0+wa (not wa).
# #     """
# #     emulator = get_emulator(
# #         cosmo_type=cosmo_type,
# #         prior_type=prior_type,
# #         nl_type=nl_type,
# #         model_type=model_type,
# #         num_batches=num_batches,
# #         w0_min=w0_min,
# #         w0wa_max=w0wa_max,
# #     )
# #     return emulator.get_pks(params, use_approximation_only=use_approximation_only)

# Author: Victoria Lloyd (2025) & V. Miranda (as seen in model evaluation)
import os
import numpy as np
import joblib
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys
import train_utils_pk_emulator_v3 as utils
sys.modules['train_utils_pk_emulator'] = utils
from train_utils_pk_emulator_v3 import CustomActivationLayer, TComponentScaler, VER
from keras.losses import MeanSquaredError, Huber
import tensorflow as tf


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# ----------------------------------------------------------------------------------------------------
# Path helpers
# ----------------------------------------------------------------------------------------------------

def _get_project_root() -> Path:
    """Returns the directory containing this file."""
    return Path(__file__).resolve().parent

ROOT = _get_project_root()

# ----------------------------------------------------------------------------------------------------
# Dependency guard
# ----------------------------------------------------------------------------------------------------

try:
    from tensorflow import keras
    import sys; sys.path.insert(0, f"{ROOT}/symbolic_pofk")
    from symbolic_pofk.linear_VL import (
        plin_emulated, get_approximate_D, growth_correction_R,
        get_eisensteinhu_nw, As_to_sigma8,
    )
    from symbolic_pofk.syrenhalofit import run_halofit_vec
    _DEPENDENCIES_LOADED = True
except ImportError as e:
    logging.error("FATAL ERROR: A required dependency could not be imported.")
    logging.error(f"Missing component: {e.name}")
    logging.error("If running locally, ensure the symbolic_pofk library is accessible.")
    _DEPENDENCIES_LOADED = False

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ImportError:
    pass


# ----------------------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------------------

NL_TYPE_REGISTRY = {
    "lin":                        ("pklin",                              "Linear P(k)"),
    "halofit":                    ("pknonlin",                           "Non-linear P(k) via HaloFit"),
    "mead2020":                   ("mead2020_pknonlin",                  "Non-linear P(k) via HMcode Mead2020"),
    "mead2020_feedback":          ("mead2020_feedback_pknonlin",         "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (fixed T_AGN)"),
    "mead2020_feedback_Tfree":    ("mead2020_feedback_Tfree_pknonlin",   "Non-linear P(k) via HMcode Mead2020 + baryonic feedback (free T_AGN)"),
    "mead2020_Tfree_mnufree_lin": ("mead2020_Tfree_mnufree_pklin",       "Linear P(k) via HMcode Mead2020 + free T_AGN + free mnu"),
    "mead2020_Tfree_mnufree":     ("mead2020_Tfree_mnufree_pknonlin",    "Non-linear P(k) via HMcode Mead2020 + free T_AGN + free mnu"),
}

VALID_NL_TYPES_BY_PRIOR = {
    "expanded":    {"lin", "halofit", "mead2020_Tfree_mnufree", "mead2020_Tfree_mnufree_lin"},
    "constrained": set(NL_TYPE_REGISTRY.keys()),
}

# nl_types that are treated as boost mode (learn P_nl_camb / P_nl_syren)
# All nl_types that are NOT purely linear.
_BOOST_NL_TYPES = {
    "halofit", "mead2020", "mead2020_feedback", "mead2020_feedback_Tfree",
    "mead2020_Tfree_mnufree",
}

VALID_COSMO_TYPES  = {"lcdm", "w0wacdm"}
VALID_MODEL_TYPES  = {"mlp", "npce"}

# VER = "_v5"


def _validate_config(
    cosmo_type: str,
    prior_type: str,
    nl_type: str,
    num_batches: int,
    model_type: str = "mlp",
) -> None:
    if cosmo_type not in VALID_COSMO_TYPES:
        raise ValueError(
            f"Unknown cosmo_type '{cosmo_type}'. Must be one of: {sorted(VALID_COSMO_TYPES)}"
        )
    if prior_type not in VALID_NL_TYPES_BY_PRIOR:
        raise ValueError(
            f"Unknown prior_type '{prior_type}'. Must be one of: {sorted(VALID_NL_TYPES_BY_PRIOR)}"
        )
    if nl_type not in NL_TYPE_REGISTRY:
        raise ValueError(
            f"Unknown nl_type '{nl_type}'. Must be one of: {sorted(NL_TYPE_REGISTRY)}"
        )
    if nl_type not in VALID_NL_TYPES_BY_PRIOR[prior_type]:
        raise ValueError(
            f"nl_type '{nl_type}' is not available for prior_type='{prior_type}'. "
            f"Valid choices: {sorted(VALID_NL_TYPES_BY_PRIOR[prior_type])}"
        )
    if cosmo_type == "lcdm" and prior_type == "expanded":
        raise ValueError("prior_type='expanded' is only supported for cosmo_type='w0wacdm'.")
    if model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Must be one of: {sorted(VALID_MODEL_TYPES)}"
        )

def _n_params_from_nl_type(nl_type: str) -> int:
    """Infer parameter count from nl_type — 9 for tfree_mnufree variants, 7 otherwise."""
    return 9 if "Tfree_mnufree" in nl_type else 7

def _metadata_tag(
    cosmo_type: str,
    prior_type: str,
    nl_type: str,
    num_batches: int,
    w0_min: Optional[float] = None,
    w0wa_max: Optional[float] = None,
) -> str:
    """
    Reproduce the tag written by COLASet._metadata_tag() during training.

    Format: {cosmo_type}_{prior_type}[_w0min{w0_min}][_w0wamax{w0wa_max}]
            _{nl_type}[_boost]_nTrain{num_batches}{VER}

    The '_kmin' suffix has been removed: k-truncation is no longer applied
    since the syren halofit baseline is non-trivial at all k.
    """
    n_params = _n_params_from_nl_type(nl_type)
    w0_tag   = f"_w0min{w0_min}"     if w0_min   is not None else ""
    w0wa_tag = f"_w0wamax{w0wa_max}" if w0wa_max is not None else ""
    mode_tag = "_boost"               if nl_type in _BOOST_NL_TYPES else ""
    param_tag = "_tfreemnufree"        if n_params > 7 else ""
    return (
        f"{cosmo_type}_{prior_type}{w0_tag}{w0wa_tag}"
        f"_{nl_type}{mode_tag}{param_tag}_nTrain{num_batches}{VER}"
    )


def _model_filename(
    cosmo_type: str,
    prior_type: str,
    nl_type: str,
    num_batches: int,
    model_type: str = "mlp",
    w0_min: Optional[float] = None,
    w0wa_max: Optional[float] = None,
) -> str:
    tag = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
    return f"emulator_{model_type}_{tag}.keras"


def _model_filename_legacy(
    cosmo_type: str,
    prior_type: str,
    nl_type: str,
    num_batches: int,
    w0_min: Optional[float] = None,
    w0wa_max: Optional[float] = None,
) -> str:
    """Legacy .h5 filename for MLP models saved before the format change."""
    tag = _metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)
    return f"emulator_{tag}.h5"


# ----------------------------------------------------------------------------------------------------
# Core emulator class
# ----------------------------------------------------------------------------------------------------

class PkEmulator:
    """
    Cosmology emulator for the matter power spectrum P(k, z).

    Emulation modes
    ---------------
    The mode is determined from nl_type:

    - **linear mode** (nl_type == 'lin' or 'mead2020_Tfree_mnufree_lin'):
        Network learns  log(P_lin_camb / P_lin_syren).
        get_pks returns  exp(log_frac) * P_lin_syren  ≈  P_lin.
        Full k-grid used throughout; no truncation or padding.

    - **boost mode** (all other nl_types):
        Network learns  log(P_nl_camb / P_nl_syren_halofit).
        The syren halofit prediction P_nl_syren is computed analytically at
        inference time and used as the multiplicative base:
            P_nl_emu = exp(log_frac) * P_nl_syren
        Full k-grid used throughout — no truncation or padding needed because
        the baseline is non-trivial at all k.

    Input convention
    ----------------
    All public methods expect params = [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa].
    Column 6 is w0+wa (not wa).  wa is derived internally as w0wa - w0.
    """

    # ks_lin = np.logspace(-5.1, 2, 500)          # original grid (must match the saved files)
    # kmin, kmax = 0.005, 50.0
    # k_mask = (ks_lin >= kmin) & (ks_lin <= kmax)
    # K_MODES = ks_lin[k_mask]
    K_MODES = utils.ks
    N_K_MODES = len(K_MODES) 

    Z_MODES = np.asarray(utils.z_mps)
    N_ZS    = len(Z_MODES)

    def __init__(
        self,
        cosmo_type: str = "w0wacdm",
        prior_type: str = "constrained",
        nl_type: str = "lin",
        model_type: str = "mlp",
        base_model_path: str = "models",
        base_metadata_path: str = "metadata",
        num_batches: int = 15,
        w0_min: Optional[float] = None,
        w0wa_max: Optional[float] = None,
    ):
        if not _DEPENDENCIES_LOADED:
            raise RuntimeError("Cannot initialise PkEmulator: missing dependencies.")

        _validate_config(cosmo_type, prior_type, nl_type, num_batches, model_type)

        self.cosmo_type  = cosmo_type
        self.prior_type  = prior_type
        self.nl_type     = nl_type
        self.model_type  = model_type
        self.num_batches = num_batches
        self.w0_min      = w0_min
        self.w0wa_max    = w0wa_max

        # use_boost: True for all nonlinear nl_types
        self.use_boost = nl_type in _BOOST_NL_TYPES

        self.N_PCS = None   # resolved in _load_metadata / _load_pcas_and_scalers

        logging.info(
            f"[PkEmulator] Initialising: cosmo_type='{cosmo_type}', "
            f"prior_type='{prior_type}', nl_type='{nl_type}', "
            f"model_type='{model_type}', use_boost={self.use_boost}, "
            f"w0_min={w0_min}, w0wa_max={w0wa_max}"
        )

        self.MODEL_DIR    = ROOT / base_model_path
        print("N params: ", _n_params_from_nl_type(nl_type))
        self.METADATA_DIR = (
            ROOT / base_metadata_path
            / f"metadata_{_metadata_tag(cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max)}"
        )

        logging.info(f"[PkEmulator] Metadata directory: {self.METADATA_DIR}")

        try:
            self._load_metadata()
            self._load_model(cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max)

            self.PCAS:    Dict[float, PCA]    = {}
            self.SCALERS: Dict[float, object] = {}
            self._pcas_loaded                 = False
            self.INVERSE_TRANSFORM_MATRICES: Optional[np.ndarray] = None
            self.INVERSE_TRANSFORM_OFFSETS:  Optional[np.ndarray] = None
            self._load_pcas_and_scalers()

            # Warm-up: two forward passes to trigger XLA/TF tracing.
            logging.info("[PkEmulator] Warming up neural network...")
            n_features = self.param_scaler.n_features_in_
            dummy_raw = np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, -1.0, 7.3, 0.06]],
                                dtype=np.float32)[:, :n_features]
            dummy_norm  = self.param_scaler.transform(dummy_raw)
            dummy_input = self._make_network_input(dummy_norm)
            dummy_tf     = tf.constant(dummy_input, dtype=tf.float32)
            _ = self._compiled_inference(dummy_tf)
            _ = self._compiled_inference(dummy_tf)

            logging.info(
                f"[PkEmulator] Initialisation complete. "
                f"use_boost={self.use_boost}, N_PCS={self.N_PCS}, "
                f"N_K_MODES={self.N_K_MODES} "
                f"({'NL/syren_halofit residual' if self.use_boost else 'lin/syren ratio'} mode)"
            )

        except FileNotFoundError as e:
            logging.error(f"Required file not found: {e.filename}")
            logging.warning(
                f"Check that '{self.METADATA_DIR}' and '{self.MODEL_DIR}' "
                "contain all required files."
            )
            raise

    @classmethod
    def from_paths(
        cls,
        model_path: str,
        metadata_path: str,
        nl_type: str = "lin",
        model_type: str = "mlp",
    ) -> "PkEmulator":
        """
        Construct a PkEmulator directly from explicit file paths, bypassing the
        cosmo_type/prior_type/nl_type path-construction logic in __init__.

        Use this when you have a specific model.keras + metadata.joblib pair
        you want to load by path — e.g. a baseline model in a known folder.

        Parameters
        ----------
        model_path    : str — path to model.keras
        metadata_path : str — path to metadata.joblib
        nl_type       : str — needed to set use_boost correctly
        model_type    : str — 'mlp' or 'npce', needed for _make_network_input

        Returns
        -------
        Fully initialised PkEmulator instance.
        """
        if not _DEPENDENCIES_LOADED:
            raise RuntimeError("Cannot create PkEmulator: missing dependencies.")

        model_path    = Path(model_path)
        metadata_path = Path(metadata_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Baseline model not found: {model_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Baseline metadata not found: {metadata_path}")

        # Bypass __init__ — allocate instance and set only what _load_metadata
        # and _load_model expect to find on self.
        instance = cls.__new__(cls)
        instance.nl_type     = nl_type
        instance.model_type  = model_type
        instance.use_boost   = nl_type in _BOOST_NL_TYPES
        instance.num_batches = None   # not used in from_paths path
        instance.w0_min      = None
        instance.w0wa_max    = None
        instance.N_PCS       = None
        instance.PCAS        = {}
        instance.SCALERS     = {}
        instance._pcas_loaded = False
        instance.INVERSE_TRANSFORM_MATRICES = None
        instance.INVERSE_TRANSFORM_OFFSETS  = None

        # Point metadata dir at the parent folder of metadata_path so that
        # _load_metadata's bundle_path = self.METADATA_DIR / "metadata.joblib" resolves.
        instance.METADATA_DIR = metadata_path.parent
        instance.MODEL_DIR    = model_path.parent

        # _load_metadata reads from self.METADATA_DIR / "metadata.joblib"
        instance._load_metadata()

        # Load Keras model directly from the explicit path
        logging.info(f"[PkEmulator.from_paths] Loading model: {model_path}")
        instance.model = keras.models.load_model(
            model_path,
            custom_objects={"CustomActivationLayer": CustomActivationLayer},
            compile=False,
        )

        @tf.function(jit_compile=False)
        def _compiled_inference(x):
            return instance.model(x, training=False)

        instance._compiled_inference = _compiled_inference

        instance._load_pcas_and_scalers()

        # Warm up
        logging.info("[PkEmulator.from_paths] Warming up...")
        n_features = instance.param_scaler.n_features_in_
        dummy_raw  = np.array([[2.0, 0.96, 67.0, 0.05, 0.3, -1.0, -1.0, 7.3, 0.06]],
                            dtype=np.float32)[:, :n_features]
        dummy_norm  = instance.param_scaler.transform(dummy_raw)
        dummy_input = instance._make_network_input(dummy_norm)
        dummy_tf    = tf.constant(dummy_input, dtype=tf.float32)
        _ = instance._compiled_inference(dummy_tf)
        _ = instance._compiled_inference(dummy_tf)

        logging.info(
            f"[PkEmulator.from_paths] Ready. "
            f"nl_type={nl_type}, model_type={model_type}, "
            f"use_boost={instance.use_boost}, N_PCS={instance.N_PCS}, "
            f"n_features={n_features}"
        )
        return instance
    # -------------------------------------------------------
    # Metadata loading
    # -------------------------------------------------------

    def _load_metadata(self) -> None:
        """
        Load scalers, tPCA, use_boost flag, and (optionally) NPCE indices.

        Tries metadata.joblib bundle first (written by COLASet.prepare()),
        falls back to individual files for older bundles.

        Note: N_K_BOOST is no longer stored — all PCAs are fitted on the full
        k-grid (N_K_MODES=500). The truncation-era field is gone.
        """
        bundle_path = self.METADATA_DIR / "metadata.joblib"

        if bundle_path.exists():
            logging.info(f"[PkEmulator] Loading metadata from bundle: {bundle_path}")
            bundle = joblib.load(bundle_path)

            self.param_scaler  = bundle["param_scaler"]
            self.t_comp_pca    = bundle["t_comp_pca"]
            self.t_comp_scaler = bundle.get("t_comp_scaler", None)

            # ── Envelope (opt-in, absent in old bundles → graceful fallback) ──
            self._use_envelope  = bundle.get("use_envelope", False)
            _tagn_grid          = bundle.get("tagn_grid",        None)
            _envelope_mean_lf   = bundle.get("envelope_mean_lf", None)

            if self._use_envelope and _tagn_grid is not None:
                self._envelope_fns = utils.reconstruct_envelope_fns(
                    _tagn_grid, _envelope_mean_lf, self.K_MODES
                )
                print(f"  [Envelope] Loaded: {len(self._envelope_fns)} redshifts, "
                    f"{len(_tagn_grid)} T_AGN bins.")
            else:
                self._envelope_fns = None
                if self._use_envelope:
                    print("  [Envelope] use_envelope=True in bundle but no grid found — "
                        "falling back to no envelope.")

            # use_boost: prefer bundle field, fall back to nl_type check
            if "use_boost" in bundle:
                self.use_boost = bundle["use_boost"]
            # (already set in __init__ from nl_type; keep that value if not in bundle)

            self._pce_indices    = bundle.get("npce_indices", None)
            self._bundle_pcas    = bundle.get("pcas",    None)
            self._bundle_scalers = bundle.get("scalers", None)

            if self._bundle_pcas is not None:
                _first_pca = next(iter(self._bundle_pcas.values()))
                self.N_PCS = _first_pca.n_components_
                logging.info(f"[PkEmulator] N_PCS inferred from bundle: {self.N_PCS}")
            else:
                self.N_PCS = None

            self._metadata_source = "bundle"

        else:
            logging.warning(
                f"[PkEmulator] Bundle not found at {bundle_path}. "
                "Falling back to individual metadata files."
            )

            self.param_scaler = joblib.load(
                self.METADATA_DIR / f"param_scaler_lowk_{self.num_batches}_batches"
            )
            self.t_comp_pca = joblib.load(self.METADATA_DIR / "t_components_pca_lowk")

            t_comp_scaler_path = self.METADATA_DIR / "t_comp_scaler"
            if t_comp_scaler_path.exists():
                self.t_comp_scaler = joblib.load(t_comp_scaler_path)
            else:
                self.t_comp_scaler = None
                logging.warning(
                    "[PkEmulator] t_comp_scaler not found; "
                    "assuming model trained on raw (un-normalised) t-components."
                )

            self._bundle_pcas    = None
            self._bundle_scalers = None
            self._pce_indices    = None
            self.N_PCS           = None

            self._metadata_source = "files"

        # NPCE: PCE index matrix
        if self.model_type == "npce" and self._pce_indices is None:
            for pce_path in [
                self.METADATA_DIR / "npce_indices.npy",
                self.MODEL_DIR    / "npce_indices.npy",
            ]:
                if pce_path.exists():
                    self._pce_indices = np.load(pce_path)
                    logging.info(
                        f"[PkEmulator] PCE indices loaded from {pce_path}: "
                        f"{self._pce_indices.shape[0]} terms, "
                        f"{self._pce_indices.shape[1]} dims."
                    )
                    break
            if self._pce_indices is None:
                raise FileNotFoundError(
                    "PCE index file not found. Expected at:\n"
                    f"  {self.METADATA_DIR / 'npce_indices.npy'}\n"
                    f"  {self.MODEL_DIR    / 'npce_indices.npy'}\n"
                    "Re-run training with --model_type npce."
                )

        if self.t_comp_scaler is None:
            logging.warning(
                "[PkEmulator] t_comp_scaler is None — "
                "assuming model was trained on raw (un-normalised) t-components."
            )

        logging.info(
            f"[PkEmulator] Metadata loaded (source={self._metadata_source}, "
            f"use_boost={self.use_boost})."
        )

    # -------------------------------------------------------

    def _load_model(
        self,
        cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max
    ) -> None:
        """Locate and load the Keras model, with legacy .h5 fallback for MLP."""
        model_file = self.MODEL_DIR / _model_filename(
            cosmo_type, prior_type, nl_type, num_batches, model_type, w0_min, w0wa_max,
        )
        if not model_file.exists() and model_type == "mlp":
            legacy = self.MODEL_DIR / _model_filename_legacy(
                cosmo_type, prior_type, nl_type, num_batches, w0_min, w0wa_max,
            )
            if legacy.exists():
                logging.warning(
                    f"[PkEmulator] .keras file not found, falling back to legacy: {legacy}"
                )
                model_file = legacy

        logging.info(f"[PkEmulator] Loading model: {model_file}")
        self.model = keras.models.load_model(
            model_file,
            custom_objects={"CustomActivationLayer": CustomActivationLayer},
            compile=False,
        )

        @tf.function(jit_compile=False)
        def _compiled_inference(x):
            return self.model(x, training=False)
        self._compiled_inference = _compiled_inference

    # -------------------------------------------------------
    # Inference helpers
    # -------------------------------------------------------

    def _make_network_input(self, params_norm: np.ndarray) -> np.ndarray:
        """MLP: returns params_norm unchanged. NPCE: evaluates PCE basis."""
        if self.model_type == "mlp":
            return params_norm.astype(np.float32)
        return self._evaluate_pce_basis(params_norm)

    def _evaluate_pce_basis(self, X_norm: np.ndarray) -> np.ndarray:
        """Evaluate the stored PCE multi-index basis at X_norm (N, n_params) → (N, N_terms)."""
        indices   = self._pce_indices
        X_clipped = np.clip(X_norm, -1.0, 1.0).astype(np.float64)
        N, D      = X_clipped.shape
        max_deg   = int(indices.max()) if indices.size > 0 else 0

        pow_table = np.empty((D, max_deg + 1, N), dtype=np.float64)
        pow_table[:, 0, :] = 1.0
        if max_deg >= 1:
            pow_table[:, 1, :] = X_clipped.T
        for p in range(2, max_deg + 1):
            pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T

        d_idx   = np.arange(D)[np.newaxis, :]
        powered = pow_table[d_idx, indices, :]
        Phi     = powered.prod(axis=1).T
        return Phi.astype(np.float32)
    def _get_envelope_iz(self, params_array, iz):
        """
        Return the envelope correction at redshift iz for this single cosmology.
        Falls back to zeros if envelope is not loaded (backward-compatible).

        Parameters
        ----------
        params_array : (N_params,) — full param vector including T_AGN at col 7
        iz           : int — redshift index into self.Z_MODES

        Returns
        -------
        envelope : (N_k,) float32
        """
        if self._envelope_fns is None:
            return np.zeros(len(self.K_MODES), dtype=np.float32)

        tagn = float(params_array[7])   # T_AGN column — same index as training
        pts  = np.column_stack([
            np.full(len(self.K_MODES), tagn),
            self.K_MODES
        ])
        return self._envelope_fns[iz](pts).astype(np.float32)

    def _load_pcas_and_scalers(self) -> None:
        """
        Load per-redshift PCA/scaler objects and pre-compute fused inverse
        transform matrices for fast batched inference.

        The inverse matrices have shape (N_ZS, N_PCS, N_K_MODES) — full k-grid,
        no truncation.
        """
        if self._pcas_loaded:
            return

        if self._bundle_pcas is not None and self._bundle_scalers is not None:
            logging.info("[PkEmulator] Loading per-z PCA/scalers from bundle dicts.")
            self.PCAS    = self._bundle_pcas
            self.SCALERS = self._bundle_scalers
        else:
            logging.info("[PkEmulator] Loading per-z PCA/scaler files from disk.")
            for z in self.Z_MODES:
                z_key = float(f"{z:.3f}")
                self.PCAS[z_key]    = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.pca")
                self.SCALERS[z_key] = joblib.load(self.METADATA_DIR / f"Z{z:.3f}_lowk.frac_pks_scaler")

        if self.N_PCS is None:
            _first_pca = next(iter(self.PCAS.values()))
            self.N_PCS = _first_pca.n_components_
            logging.info(f"[PkEmulator] N_PCS inferred from per-z PCA files: {self.N_PCS}")

        logging.info("[PkEmulator] Pre-computing inverse transformation matrices...")

        inverse_matrices, inverse_offsets = [], []
        for z in self.Z_MODES:
            z_key  = float(f"{z:.3f}")
            pca    = self.PCAS[z_key]
            scaler = self.SCALERS[z_key]

            if hasattr(scaler, "scale_"):
                scale = scaler.scale_
                mean  = scaler.mean_
            elif hasattr(scaler, "std"):
                scale = scaler.std
                mean  = scaler.mean
            else:
                raise AttributeError(
                    f"Scaler for z={z:.3f} has neither 'scale_' nor 'std'."
                )

            inverse_matrices.append(pca.components_ * scale[None, :])
            inverse_offsets.append(pca.mean_ * scale + mean)

        self.INVERSE_TRANSFORM_MATRICES = np.stack(inverse_matrices, axis=0).astype(np.float32)
        self.INVERSE_TRANSFORM_OFFSETS  = np.stack(inverse_offsets,  axis=0).astype(np.float32)

        self._pcas_loaded = True
        logging.info(
            f"[PkEmulator] Inverse transformation matrices ready: "
            f"shape={self.INVERSE_TRANSFORM_MATRICES.shape} "
            f"(N_ZS={self.N_ZS}, N_PCS={self.N_PCS}, N_K={self.N_K_MODES})"
        )

    def _compute_syren_lin(self, params: np.ndarray) -> np.ndarray:
        """
        Analytical P_lin(k, z) from syren + growth factors.

        Used as the multiplicative base in linear mode and as an intermediate
        step when computing P_nl_syren for boost mode.

        Parameters
        ----------
        params : 1-D array [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]

        Returns
        -------
        np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³
        """
        As, ns, H0_in, Ob, Om, w0, w0wa = params[:7]
        wa = w0wa - w0
        h  = H0_in / 100.0
        mnu = float(params[8]) if len(params) > 7 else 0.06

        k_hmpc = self.K_MODES / h
        pk_lin_hmpc = plin_emulated(k_hmpc, Om, Ob, h, ns, As=As, w0=w0, wa=wa, mnu=mnu)

        a_array = 1.0 / (self.Z_MODES + 1)
        D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)
        R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)

        growth_factors = (Dz / D0) ** 2 * (Rz / R0)
        pk_lin_mpc = pk_lin_hmpc[None, :] * growth_factors[:, None] / h**3
        return pk_lin_mpc.astype(np.float32)

    def _compute_syren_nl(self, params: np.ndarray) -> np.ndarray:
        """
        Analytical P_nl(k, z) from syren halofit + growth factors.

        Used as the multiplicative base in boost mode:
            P_nl_emu = exp(log_frac) * P_nl_syren
        where log_frac = log(P_nl_camb / P_nl_syren) is what the network learns.

        Parameters
        ----------
        params : 1-D array [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]

        Returns
        -------
        np.ndarray of shape (N_ZS, N_K_MODES) in Mpc³
        """
        As, ns, H0_in, Ob, Om, w0, w0wa = params[:7]
        wa = w0wa - w0
        h  = H0_in / 100.0
        mnu = float(params[8]) if len(params) > 7 else 0.06

        k_hmpc = self.K_MODES / h

        # Linear Pk in (Mpc/h)^3 from syren, then evolved with growth factors
        pk_lin_hmpc = plin_emulated(k_hmpc, Om, Ob, h, ns, As=As, w0=w0, wa=wa, mnu=mnu)

        a_array = 1.0 / (self.Z_MODES + 1)
        D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)
        R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)

        growth_factors  = (Dz / D0) ** 2 * (Rz / R0)
        pk_lin_hmpc_z   = pk_lin_hmpc[None, :] * growth_factors[:, None]   # (N_ZS, N_K)

        sigma8_z0 = As_to_sigma8(As, Om, Ob, h, ns, mnu=mnu, w0=w0, wa=wa)

        pk_nl_hmpc_z = run_halofit_vec(
            k_hmpc,
            sigma8_z0,
            Om, Ob, h, ns,
            a_array,
            return_boost=False,
            Plin_in=pk_lin_hmpc_z,
        )   # (N_ZS, N_K)

        pk_nl_mpc_z = pk_nl_hmpc_z / h**3
        return pk_nl_mpc_z.astype(np.float32)

    def _compute_syren_boost(self, params: np.ndarray) -> np.ndarray:
        """B_syren(k,z) = P_nl_syren / P_lin_syren — the denominator used in training."""
        As, ns, H0_in, Ob, Om, w0, w0wa = params[:7]
        wa  = w0wa - w0
        h   = H0_in / 100.0
        mnu = float(params[8]) if len(params) > 7 else 0.06
        k_hmpc = self.K_MODES / h
        pk_lin_hmpc = plin_emulated(k_hmpc, Om, Ob, h, ns, As=As, w0=w0, wa=wa, mnu=mnu)
        a_array = 1.0 / (self.Z_MODES + 1)
        D0 = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Dz = get_approximate_D(k=1e-4, As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)
        R0 = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=1)
        Rz = growth_correction_R(As=As, Om=Om, Ob=Ob, h=h, ns=ns, mnu=mnu, w0=w0, wa=wa, a=a_array)
        pk_lin_hmpc_z = pk_lin_hmpc[None, :] * ((Dz / D0) ** 2 * (Rz / R0))[:, None]
        sigma8_z0 = As_to_sigma8(As, Om, Ob, h, ns, mnu=mnu, w0=w0, wa=wa)
        return run_halofit_vec(k_hmpc, sigma8_z0, Om, Ob, h, ns, a_array,
                               return_boost=True, Plin_in=pk_lin_hmpc_z).astype(np.float32)

    def _predict_fracs_all_z(self, params_norm: np.ndarray, params_raw: np.ndarray = None) -> np.ndarray:
        """
        NN inference: normalised params -> log-fractional differences for all z.

        Returns
        -------
        np.ndarray of shape (N_ZS, N_K_MODES)
            Full k-grid — no truncation in either mode.
        """
        net_input = self._make_network_input(params_norm)
        input_tf  = tf.constant(net_input, dtype=tf.float32)

        t_comps_norm = self._compiled_inference(input_tf).numpy()

        if self.t_comp_scaler is not None:
            t_comps_raw = self.t_comp_scaler.inverse_transform(t_comps_norm)
        else:
            t_comps_raw = t_comps_norm

        pcs_flat = self.t_comp_pca.inverse_transform(t_comps_raw).astype(np.float32)
        pcs_z    = pcs_flat.reshape(self.N_ZS, self.N_PCS)

        reconstructed = (
            np.einsum("zp,zpk->zk", pcs_z, self.INVERSE_TRANSFORM_MATRICES)
            + self.INVERSE_TRANSFORM_OFFSETS
        ).astype(np.float32)
        # Add T_AGN envelope back — no-op if envelope not loaded (old bundles,
        # non-envelope training runs, or 7-param models without T_AGN).
        if self._envelope_fns is not None and params_raw is not None:
            tagn    = float(params_raw[7])
            pts     = np.column_stack([
                np.full(len(self.K_MODES), tagn),
                self.K_MODES
            ])
            for iz, fn in enumerate(self._envelope_fns):
                reconstructed[iz] += fn(pts).astype(np.float32)
                
        return reconstructed   # (N_ZS, N_K_MODES)

    # -------------------------------------------------------
    # Public interface
    # -------------------------------------------------------

    def get_pks(
        self,
        params: List[float],
        use_approximation_only: bool = False,
        pk_lin=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return P(k, z) for a single cosmology.

        Parameters
        ----------
        params : list or 1-D array of 7 values
            [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
            Column 6 is w0+wa (not wa).
        use_approximation_only : bool
            If True, skip the NN and return the syren baseline only.
            In linear mode this is P_lin_syren.
            In boost mode this is P_nl_syren (the halofit approximation).

        Returns
        -------
        k_modes : (N_K_MODES,)
        z_modes : (N_ZS,)
        pks     : (N_ZS, N_K_MODES)

        Notes on emulation mode
        -----------------------
        Linear mode (nl_type == 'lin' or 'mead2020_Tfree_mnufree_lin'):
            Network predicts  log(P_lin_camb / P_lin_syren)  on full k-grid.
            Returned pks ≈ P_lin = exp(log_frac) * P_lin_syren.

        Boost mode (all other nl_types):
            Network predicts  log(P_nl_camb / P_nl_syren_halofit)  on full k-grid.
            Returned pks ≈ P_nl = exp(log_frac) * P_nl_syren.
            No truncation or padding — the halofit baseline is informative
            at all k so there is no wasteful flat region to discard.
        """
        params_array = np.array(params, dtype=np.float32)
        if params_array.ndim != 1 or len(params_array) < 7:
            raise ValueError(
                f"Expected at least 7 parameters [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa], "
                f"got shape {params_array.shape}."
            )
        n_scaler_features = self.param_scaler.n_features_in_
        params_norm  = self.param_scaler.transform(
            params_array[:n_scaler_features].reshape(1, -1)
        )

        if self.use_boost:
            if pk_lin is None:
                logging.warning("[PkEmulator] get_pks boost mode without pk_lin: "
                                "using syren P_lin, result carries syren's linear error.")
                pk_lin = self._compute_syren_lin(params_array)
            pk_base = self._compute_syren_boost(params_array) * pk_lin
        else:
            # Linear mode: base is the syren linear prediction
            pk_base = self._compute_syren_lin(params_array)

        if use_approximation_only:
            return self.K_MODES, self.Z_MODES, pk_base

        n_scaler_features = self.param_scaler.n_features_in_
        params_norm = self.param_scaler.transform(params_array[:n_scaler_features].reshape(1, -1))
        log_frac    = self._predict_fracs_all_z(params_norm, params_raw=params_array)   # (N_ZS, N_K_MODES)

        # No masking or padding: log_frac covers the full k-grid in both modes.
        pks = (np.exp(log_frac) * pk_base).astype(np.float32)

        return self.K_MODES, self.Z_MODES, pks

    def get_boost(
        self,
        params: List[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the emulated residual boost B(k,z) = P_nl_emu / P_nl_syren.

        This is exp(log_frac) directly — useful for diagnostics and for callers
        that want to apply the boost to their own linear P(k).

        Only meaningful in boost mode; raises if called in linear mode.

        Returns
        -------
        k_modes : (N_K_MODES,)
        z_modes : (N_ZS,)
        boost   : (N_ZS, N_K_MODES)
        """
        if not self.use_boost:
            raise ValueError(
                "get_boost() is only available in boost mode "
                f"(nl_type='{self.nl_type}' is a linear type)."
            )

        params_array = np.array(params, dtype=np.float32)
        n_scaler_features = self.param_scaler.n_features_in_
        params_norm = self.param_scaler.transform(
            params_array[:n_scaler_features].reshape(1, -1)
        )
        log_frac     = self._predict_fracs_all_z(params_norm, params_raw=params_array)

        return self.K_MODES, self.Z_MODES, np.exp(log_frac).astype(np.float32)


# ----------------------------------------------------------------------------------------------------
# Module-level interface with instance caching
# ----------------------------------------------------------------------------------------------------

_emulator_cache: Dict[Tuple, PkEmulator] = {}


def get_emulator(
    cosmo_type: str = "w0wacdm",
    prior_type: str = "constrained",
    nl_type: str = "lin",
    model_type: str = "mlp",
    base_model_path: str = "models",
    base_metadata_path: str = "metadata",
    num_batches: int = 15,
    w0_min: Optional[float] = None,
    w0wa_max: Optional[float] = None,
) -> PkEmulator:
    """Return a (cached) PkEmulator for the requested configuration."""
    if not _DEPENDENCIES_LOADED:
        raise RuntimeError("Cannot create PkEmulator: missing dependencies.")

    cache_key = (cosmo_type, prior_type, nl_type, model_type, num_batches, w0_min, w0wa_max)
    if cache_key in _emulator_cache:
        logging.info(f"[get_emulator] Returning cached emulator for {cache_key}")
        return _emulator_cache[cache_key]

    logging.info(f"[get_emulator] Creating new emulator for {cache_key}")
    emulator = PkEmulator(
        cosmo_type=cosmo_type,
        prior_type=prior_type,
        nl_type=nl_type,
        model_type=model_type,
        base_model_path=base_model_path,
        base_metadata_path=base_metadata_path,
        num_batches=num_batches,
        w0_min=w0_min,
        w0wa_max=w0wa_max,
    )
    _emulator_cache[cache_key] = emulator
    return emulator


def get_pks(
    params: List[float],
    cosmo_type: str = "w0wacdm",
    prior_type: str = "constrained",
    nl_type: str = "lin",
    model_type: str = "mlp",
    num_batches: int = 15,
    use_approximation_only: bool = False,
    w0_min: Optional[float] = None,
    w0wa_max: Optional[float] = None,
    pk_lin = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function: get P(k, z) without managing emulator instances.

    Parameters
    ----------
    params : list of 7 floats [10^9 A_s, ns, H0, Ob, Om, w0, w0+wa]
        Column 6 is w0+wa (not wa).
    """
    emulator = get_emulator(
        cosmo_type=cosmo_type,
        prior_type=prior_type,
        nl_type=nl_type,
        model_type=model_type,
        num_batches=num_batches,
        w0_min=w0_min,
        w0wa_max=w0wa_max,
    )
    return emulator.get_pks(params, use_approximation_only=use_approximation_only,pk_lin=pk_lin)
















# """
# train.py — MPS Emulator Training Script

# Usage examples:
#     # Minimal (required flags only):
#     python ./mps_emu/train.py --cosmo_type w0wacdm --prior_type expanded --nl_type lin

#     # Train with NPCE instead of the default MLP:
#     python ./mps_emu/train.py \
#         --cosmo_type w0wacdm --prior_type expanded --nl_type halofit \
#         --model_type npce \
#         --pce_max_degree 8 --pce_norm_q 0.75 --pce_norm_threshold 1.0

#     # Full MLP override:
#     python ./mps_emu/train.py \
#         --cosmo_type w0wacdm \
#         --prior_type expanded \
#         --nl_type lin \
#         --n_batches 20 \
#         --num_epochs 3000 \
#         --num_layers 4 \
#         --num_neurons 1024 \
#         --num_pcs 25 \
#         --num_pcs_z 15 \
#         --start_batch 0 \
#         --test_batch 100
# """

# import argparse
# import os
# import time

# import matplotlib.pyplot as plt
# import numpy as np
# import train_utils_pk_emulator_v3 as utils
# from cola_npce_v2 import COLA_NPCE_Keras

# # ---------------------------------------------------------------------------
# # Argument parsing
# # ---------------------------------------------------------------------------

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Train the MPS power-spectrum emulator neural network.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )

#     # --- Required / core configuration ---
#     required = parser.add_argument_group("required arguments")
#     required.add_argument(
#         "--cosmo_type",
#         required=True,
#         choices=["lcdm", "w0wacdm"],
#         help="Cosmological model type.",
#     )
#     required.add_argument(
#         "--prior_type",
#         required=True,
#         choices=["constrained", "expanded"],
#         help="Prior range to use for training.",
#     )
#     required.add_argument(
#         "--nl_type",
#         required=True,
#         choices=["lin", "halofit", "mead2020", "mead2020_feedback", "mead2020_feedback_Tfree"],
#         help=(
#             "Linear ('lin'), or non-linear matter power spectrum with halofit ('halofit'), "
#             "HMCode ('mead2020'), HMCode with baryonic feedback fixed to logT_AGN=7.8 "
#             "('mead2020_feedback'), or with logT_AGN free ('mead2020_feedback_Tfree')."
#         ),
#     )

#     # --- Batch / data configuration ---
#     data_group = parser.add_argument_group("data arguments")
#     data_group.add_argument(
#         "--start_batch",
#         type=int,
#         default=0,
#         help="Index of the first training batch to load.",
#     )
#     data_group.add_argument(
#         "--n_batches",
#         type=int,
#         default=20,
#         help="Number of training batches to load.",
#     )
#     data_group.add_argument(
#         "--test_batch",
#         type=int,
#         default=100,
#         help="Batch index to use for the test/validation set.",
#     )
#     data_group.add_argument(
#         "--w0_min",
#         type=float,
#         default=None,
#         help=(
#             "If set, filter training and test sets to w0 >= this value "
#             "(expanded prior only).  E.g. --w0_min -2.0"
#         ),
#     )
#     data_group.add_argument(
#         "--w0wa_max",
#         type=float,
#         default=None,
#         help=(
#             "If set, filter training and test sets to w0+wa <= this value "
#             "(expanded prior only).  E.g. --w0wa_max -0.5"
#         ),
#     )

#     # --- PCA configuration ---
#     pca_group = parser.add_argument_group("PCA arguments")
#     pca_group.add_argument(
#         "--num_pcs",
#         type=int,
#         default=25,
#         help="Number of principal components for the power-spectrum axis.",
#     )
#     pca_group.add_argument(
#         "--num_pcs_z",
#         type=int,
#         default=15,
#         help="Number of principal components for the redshift axis.",
#     )

#     # --- Model type ---
#     model_group = parser.add_argument_group("model type arguments")
#     model_group.add_argument(
#         "--model_type",
#         default="mlp",
#         choices=["mlp", "npce", "multihead", "multihead_npce"],
#         help=(
#             "Emulator architecture.  'mlp' uses the standard dense MLP. "
#             "'npce' uses Neural Polynomial Chaos Expansion. "
#             "'multihead' uses a shared MLP backbone with per-redshift output heads "
#             "(no tPCA compression). "
#             "'multihead_npce' uses a shared NPCE backbone with per-redshift heads."
#         ),
#     )

#     # --- Shared network / training configuration ---
#     nn_group = parser.add_argument_group("neural network arguments")
#     nn_group.add_argument(
#         "--num_epochs",
#         type=int,
#         default=3000,
#         help="Number of training epochs.",
#     )
#     nn_group.add_argument(
#         "--num_layers",
#         type=int,
#         default=4,
#         help="Number of hidden layers in the MLP (shared by both model types).",
#     )
#     nn_group.add_argument(
#         "--num_neurons",
#         type=int,
#         default=1024,
#         help="Number of neurons per hidden layer (shared by both model types).",
#     )
#     nn_group.add_argument(
#         "--batch_size",
#         type=int,
#         default=512,
#         help="Mini-batch size for training.",
#     )
#     nn_group.add_argument(
#         "--initial_lr",
#         type=float,
#         default=1e-3,
#         help="Initial learning rate for cosine annealing.",
#     )
#     nn_group.add_argument(
#         "--final_lr",
#         type=float,
#         default=1e-5,
#         help="Final (floor) learning rate for cosine annealing.",
#     )
#     nn_group.add_argument(
#         "--huber_delta",
#         type=float,
#         default=1.0,
#         help=(
#             "Transition point of the Huber loss (in normalised t-component units). "
#             "1.0 = quadratic below 1 sigma, linear above.  Increase to 2-3 if the "
#             "model underfits edge cosmologies; decrease to 0.5 if outliers dominate. "
#             "Ignored when --trim_top_frac > 0 (trimmed MSE is used instead)."
#         ),
#     )
#     nn_group.add_argument(
#         "--trim_top_frac",
#         type=float,
#         default=0.0,
#         help=(
#             "Fraction of highest per-sample losses to exclude from each mini-batch "
#             "before averaging.  0.0 (default) = standard Huber loss.  When > 0, "
#             "switches to trimmed MSE and ignores --huber_delta.  Recommended range: "
#             "0.01-0.10.  Requires batch_size large enough that at least one sample "
#             "is dropped per batch (i.e. batch_size * trim_top_frac >= 1)."
#         ),
#     )

#     # Legacy step-decay params — accepted but ignored when cosine annealing is active.
#     nn_group.add_argument(
#         "--decay_every",
#         type=int,
#         default=None,
#         help="(Legacy) Decay the learning rate every this many epochs. Ignored by default.",
#     )
#     nn_group.add_argument(
#         "--decay_rate",
#         type=float,
#         default=None,
#         help="(Legacy) Learning-rate decay multiplier. Ignored by default.",
#     )

#     # --- NPCE-specific hyperparameters ---
#     npce_group = parser.add_argument_group(
#         "NPCE arguments",
#         description=(
#             "These flags are only used when --model_type npce is set. "
#             "They control the polynomial chaos expansion basis."
#         ),
#     )
#     npce_group.add_argument(
#         "--pce_max_degree",
#         type=int,
#         default=6,
#         help=(
#             "Maximum per-dimension polynomial degree for the PCE basis.  "
#             "The notebook used 12 for 6 parameters; start with 6-8 for 7 parameters "
#             "to keep the basis size manageable.  Higher values give more expressive "
#             "features but a larger input dimension and slower training."
#         ),
#     )
#     npce_group.add_argument(
#         "--pce_norm_q",
#         type=float,
#         default=0.75,
#         help=(
#             "L^q norm exponent for hyperbolic-cross truncation of the PCE basis "
#             "(0 < q <= 1).  Smaller values prune more aggressively, keeping fewer "
#             "high-order interaction terms."
#         ),
#     )
#     npce_group.add_argument(
#         "--pce_norm_threshold",
#         type=float,
#         default=1.0,
#         help=(
#             "Upper bound on the L^q multi-index norm.  Multi-indices whose L^q norm "
#             "exceeds this threshold are discarded.  Default 1.0 matches the notebook."
#         ),
#     )

#     # --- Output configuration ---
#     out_group = parser.add_argument_group("output arguments")
#     out_group.add_argument(
#         "--model_dir",
#         default="/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/models",
#         help="Directory where the trained model will be saved.",
#     )
#     out_group.add_argument(
#         "--fig_dir",
#         default="/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/validation_figs",
#         help="Directory where validation figures will be saved.",
#     )

#     return parser.parse_args()

# # ---------------------------------------------------------------------------
# # Prior-cut helper (shared between train and test sets)
# # ---------------------------------------------------------------------------

# def _apply_prior_cuts(cola_set, w0_min, w0wa_max):
#     w0_col   = utils.params.index("w")
#     w0wa_col = utils.params.index("w0+wa")

#     mask = np.ones(len(cola_set.lhs), dtype=bool)

#     if w0_min is not None:
#         w0_mask = cola_set.lhs[:, w0_col] >= w0_min
#         n_cut   = (~w0_mask & mask).sum()
#         mask   &= w0_mask
#         print(f"  w0 cut   (w0 >= {w0_min}):     removed {n_cut} cosmologies.")

#     if w0wa_max is not None:
#         w0wa_mask = cola_set.lhs[:, w0wa_col] <= w0wa_max
#         n_cut     = (~w0wa_mask & mask).sum()
#         mask     &= w0wa_mask
#         print(f"  w0+wa cut (w0+wa <= {w0wa_max}): removed {n_cut} additional cosmologies.")

#     if not mask.all():
#         for attr in ("lhs", "pks_target", "frac_pks", "logfracs"):
#             val = getattr(cola_set, attr, None)
#             if val is not None:
#                 setattr(cola_set, attr, val[mask])
#         # mps_approxes is None in boost mode, only mask it when present
#         # if cola_set.mps_approxes is not None:
#         #     cola_set.mps_approxes = cola_set.mps_approxes[mask]

#         if hasattr(cola_set, "mps_approxes_boost") and cola_set.mps_approxes_boost is not None:
#             cola_set.mps_approxes_boost = cola_set.mps_approxes_boost[mask]
#             cola_set.mps_approxes       = cola_set.mps_approxes_boost  # keep alias in sync
#         elif hasattr(cola_set, "mps_approxes") and cola_set.mps_approxes is not None:
#             cola_set.mps_approxes = cola_set.mps_approxes[mask]
#         print(f"  → {len(cola_set.lhs)} cosmologies remaining.")

#     cola_set.w0_min   = w0_min
#     cola_set.w0wa_max = w0wa_max

# # ---------------------------------------------------------------------------
# # Main training routine
# # ---------------------------------------------------------------------------

# def main():
#     args = parse_args()

#     # --- Validate NPCE-specific flags ---
#     if args.model_type in ("mlp", "multihead") and any([
#         args.pce_max_degree != 6,
#         args.pce_norm_q     != 0.75,
#         args.pce_norm_threshold != 1.0,
#     ]):
#         print(
#             "[WARN] --pce_* flags were set but model_type has no NPCE backbone. "
#             "These flags will be ignored."
#         )

#     if args.w0_min is not None or args.w0wa_max is not None:
#         if args.prior_type != "expanded":
#             raise ValueError(
#                 "--w0_min and --w0wa_max are only meaningful with "
#                 "--prior_type expanded."
#             )

#     # --- Print run configuration ---
#     print("=" * 60)
#     print("[INFO] MPS Emulator Training Configuration")
#     print("=" * 60)
#     for key, val in vars(args).items():
#         print(f"  {key:<25s} = {val}")
#     print("=" * 60)

#     # --- Directory setup ---
#     os.makedirs(args.model_dir, exist_ok=True)
#     os.makedirs(args.fig_dir,   exist_ok=True)

#     # --- Load training data ---
#     start = time.perf_counter()
#     print("\n[INFO] Loading training set...")

#     train_set = utils.COLASet(
#         target_z   = utils.z_mps,
#         cosmo_type = args.cosmo_type,
#         prior_type = args.prior_type,
#         nl_type    = args.nl_type,
#         start_batch= args.start_batch,
#         n_batches  = args.n_batches,
#     )

#     if train_set.use_boost:
#         print(
#             f"\n[INFO] Boost mode: emulator will learn P_{args.nl_type} / P_lin.\n"
#             f"       Linear counterpart (nl_type='lin') loaded automatically as denominator."
#         )
#     else:
#         print(
#             f"\n[INFO] Syren mode: emulator will learn P_lin / P_syren "
#             f"(symbolic approximation)."
#         )

#     if args.w0_min is not None or args.w0wa_max is not None:
#         print("\n[INFO] Applying prior cuts to training set...")
#         _apply_prior_cuts(train_set, args.w0_min, args.w0wa_max)

#     # --- Prepare PCA ---
#     # --- prepare() call: multihead skips tPCA so num_pcs_z is unused ---
#     if args.model_type in ("multihead", "multihead_npce"):
#         print(
#             "[INFO] Multihead mode: skipping tPCA compression. "
#             f"Network will predict {len(utils.z_mps)} × {args.num_pcs} = "
#             f"{len(utils.z_mps) * args.num_pcs} PCA coefficients directly."
#         )
#         train_set.prepare(
#             num_pcs=args.num_pcs,
#             num_pcs_z=None,          # signals prepare() to skip tPCA
#             multihead=True,
#         )
#     else:
#         train_set.prepare(num_pcs=args.num_pcs, num_pcs_z=args.num_pcs_z)

#     elapsed = time.perf_counter() - start
#     print(f"[INFO] Data loaded and PCA prepared in {elapsed / 60:.2f} minutes.")
#     print(f"[INFO] Training redshifts: {train_set.z}")

#     # --- Build model ---
#     print(f"\n[INFO] Building {args.model_type.upper()} model...")

#     # --- Build model ---
#     if args.model_type == "mlp":
#         model_obj = utils.COLA_NN_Keras(
#             train_set,
#             num_layers  = args.num_layers,
#             num_neurons = args.num_neurons,
#         )
#     elif args.model_type == "npce":
#         model_obj = COLA_NPCE_Keras(
#             train_set,
#             max_degree     = args.pce_max_degree,
#             norm_q         = args.pce_norm_q,
#             norm_threshold = args.pce_norm_threshold,
#             num_layers     = args.num_layers,
#             num_neurons    = args.num_neurons,
#         )
#     elif args.model_type == "multihead":
#         model_obj = utils.COLA_MultiHead_Keras(
#             train_set,
#             num_layers  = args.num_layers,
#             num_neurons = args.num_neurons,
#         )
#     elif args.model_type == "multihead_npce":
#         model_obj = utils.COLA_MultiHead_NPCE_Keras(
#             train_set,
#             max_degree     = args.pce_max_degree,
#             norm_q         = args.pce_norm_q,
#             norm_threshold = args.pce_norm_threshold,
#             num_layers     = args.num_layers,
#             num_neurons    = args.num_neurons,
#         )

#     # --- Train ---
#     print("[INFO] Starting training...")
#     # --- Train: multihead uses fit_multihead instead of fit_t_componets ---
#     if args.model_type in ("multihead", "multihead_npce"):
#         model_obj.fit_multihead(
#             train_set,
#             num_epochs  = args.num_epochs,
#             batch_size  = args.batch_size,
#             initial_lr  = args.initial_lr,
#             final_lr    = args.final_lr,
#             huber_delta = args.huber_delta,
#             trim_top_frac  = args.trim_top_frac,
#         )
#     else:
#         model_obj.fit_t_componets(
#             train_set,
#             num_epochs    = args.num_epochs,
#             batch_size    = args.batch_size,
#             initial_lr    = args.initial_lr,
#             final_lr      = args.final_lr,
#             huber_delta   = args.huber_delta,
#             trim_top_frac = args.trim_top_frac,
#             decayevery    = args.decay_every,
#             decayrate     = args.decay_rate,
#         )

#     # --- Save model ---
#     keras_model = model_obj.models["t-component"]
#     model_tag   = train_set._metadata_tag()
#     model_name  = f"emulator_{args.model_type}_{model_tag}.keras"
#     model_path  = os.path.join(args.model_dir, model_name)
#     keras_model.save(model_path)

#     # Save PCE indices alongside the model so they can be reconstructed later
#     # if args.model_type == "npce":
#     #     model_obj.save_pce_metadata(args.model_dir)
#     # --- Save: NPCE multihead needs indices patched into bundle too ---
#     if args.model_type == "npce":
#         utils.update_metadata_bundle_npce(
#             bundle_path  = train_set._metadata_bundle_path,
#             npce_indices = model_obj._indices,
#             model_type   = "npce",
#         )
#     elif args.model_type == "multihead_npce":
#         utils.update_metadata_bundle_npce(
#             bundle_path  = train_set._metadata_bundle_path,
#             npce_indices = model_obj._indices,
#             model_type   = "multihead_npce",   # preserves correct model_type
#         )

#     total_elapsed = time.perf_counter() - start
#     print(f"\n[INFO] Model trained and saved to: {model_path}")
#     print(f"[INFO] Total runtime: {total_elapsed / 60:.2f} minutes.")

# if __name__ == "__main__":
#     main()

