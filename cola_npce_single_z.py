"""
cola_npce_single_z.py — Neural Polynomial Chaos Expansion emulator for the
single-redshift (no-tPCA) COLASet in mps_emu_single_z.

Implements COLA_NPCE_Keras, a drop-in replacement for COLA_NN_Keras that uses
a Neural PCE architecture instead of a plain MLP.

The NPCE idea (Rebouças et al. / bernardo7crf/NPCE) replaces the raw
cosmological parameters fed to the network with a polynomial chaos expansion
(PCE) basis evaluated at each input point.  The PCE basis spans all
multivariate monomials of the normalised inputs up to a chosen maximum degree,
pruned so that:

  1. No single-variable degree exceeds `max_degree`.
  2. The L^q norm of the multi-index (with q = `norm_q`) does not exceed
     `norm_threshold` (hyperbolic cross truncation).

The network (same custom-activation MLP as COLA_NN_Keras) then maps the PCE
features to the standardised PCA coefficients of log(frac_pk) at the target
redshift.

Usage
-----
    from cola_npce_single_z import COLA_NPCE_Keras

    train_set.prepare(num_pcs=25)

    model = COLA_NPCE_Keras(
        train_set,
        max_degree     = 6,
        norm_q         = 0.75,
        norm_threshold = 1.0,
        num_layers     = 3,
        num_neurons    = 512,
    )
    model.fit_t_componets(train_set, num_epochs=4000)

    # All COLAModel methods (predict, plot_errors, get_outliers) work unchanged.

Author: Victoria Lloyd (2026), based on the NPCE notebook by bernardo7crf.
"""

from itertools import product as iter_product

import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# ── import shared utilities from the single-z module ─────────────────────────
import mps_emu_single_z as utils
from mps_emu_single_z import (
    COLAModel,
    CustomActivationLayer,
    nn_model_train_keras,
    make_trimmed_huber_loss,
)

VER = utils.VER


# =============================================================================
# PCE basis helpers
# =============================================================================

def _build_multi_indices(
    n_dims: int,
    max_degree: int,
    norm_q: float,
    norm_threshold: float,
) -> np.ndarray:
    """
    Build the matrix of PCE multi-indices after hyperbolic-cross pruning.

    Keep alpha if  sum_i (alpha_i / max_degree)^q  <=  norm_threshold
    (with the convention 0^q = 0).

    Returns
    -------
    indices : (N_terms, n_dims) int ndarray
    """
    degrees = np.arange(max_degree + 1)
    indices = np.array(list(iter_product(degrees, repeat=n_dims)), dtype=np.float64)

    normalised = indices / max_degree if max_degree > 0 else indices.copy()

    lq_norms = np.sum(
        np.where(normalised > 0, normalised ** norm_q, 0.0),
        axis=1,
    )
    mask = lq_norms <= norm_threshold + 1e-12
    return indices[mask].astype(np.int32)


def _evaluate_pce_basis(indices: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    """
    Evaluate psi_alpha(x) = prod_i x_i^{alpha_i} at each row of X_norm.

    Parameters
    ----------
    indices  : (N_terms, n_dims) int ndarray
    X_norm   : (N_samples, n_dims) float ndarray -- inputs in [-1, 1]

    Returns
    -------
    Phi : (N_samples, N_terms) float32 ndarray
    """
    X_clipped = np.clip(X_norm, -1.0, 1.0).astype(np.float32)  # (N, D)
    N, D = X_clipped.shape
    max_deg = int(indices.max()) if indices.size > 0 else 0

    # pow_table[d, p, n] = X_clipped[n, d] ** p (repeated multiplication)
    pow_table = np.empty((D, max_deg + 1, N), dtype=np.float32)
    pow_table[:, 0, :] = 1.0
    if max_deg >= 1:
        pow_table[:, 1, :] = X_clipped.T
    for p in range(2, max_deg + 1):
        pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T

    # Accumulate (T, N) one dimension at a time — never materialises (T, D, N)
    T = indices.shape[0]
    Phi_T = np.ones((T, N), dtype=np.float32)
    for d in range(D):
        Phi_T *= pow_table[d, indices[:, d], :]

    return Phi_T.T.astype(np.float32)


class COLA_NPCE_Keras(COLAModel):
    """
    Neural Polynomial Chaos Expansion emulator (single redshift, no tPCA).

    Architecture
    ------------
    Input: cosmological parameters (normalised to [-1, 1])
        ↓
    PCE basis evaluation  →  (N_samples, N_pce_terms)  [no trainable weights]
        ↓
    Dense MLP with CustomActivationLayer  (same as COLA_NN_Keras)
        ↓
    Output: PCA coefficients at the target z (standardised)
    """

    def __init__(
        self,
        trainSet,
        max_degree: int       = 6,
        norm_q: float         = 0.75,
        norm_threshold: float = 1.0,
        num_layers: int       = 3,
        num_neurons: int      = 512,
    ):
        super().__init__(trainSet)
        self.trainSet       = trainSet
        self.max_degree     = max_degree
        self.norm_q         = norm_q
        self.norm_threshold = norm_threshold
        self.num_layers     = num_layers
        self.num_neurons    = num_neurons

        n_dims = trainSet.lhs_norm.shape[1]
        self._indices = _build_multi_indices(
            n_dims         = n_dims,
            max_degree     = max_degree,
            norm_q         = norm_q,
            norm_threshold = norm_threshold,
        )
        n_terms = self._indices.shape[0]
        print(
            f"\n[COLA_NPCE_Keras] PCE basis: "
            f"{n_dims} params, max_degree={max_degree}, "
            f"q={norm_q}, threshold={norm_threshold}  "
            f"→  {n_terms} basis terms"
        )

    # ------------------------------------------------------------------
    def _make_pce_features(self, lhs_norm: np.ndarray) -> np.ndarray:
        return _evaluate_pce_basis(self._indices, lhs_norm)

    # ------------------------------------------------------------------
    def _build_model(self, input_dim: int, output_dim: int) -> keras.Model:
        """MLP mapping PCE features → standardised PCA coefficients."""
        inputs = layers.Input(shape=(input_dim,))
        x      = layers.Dense(self.num_neurons)(inputs)
        x      = CustomActivationLayer(self.num_neurons)(x)

        for _ in range(self.num_layers - 1):
            x = layers.Dense(self.num_neurons)(x)
            x = CustomActivationLayer(self.num_neurons)(x)

        # float32 output so predictions aren't rounded to bfloat16 under the
        # mixed_bfloat16 policy set in mps_emu_single_z
        outputs = layers.Dense(output_dim, dtype="float32")(x)
        model   = models.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model

    # ------------------------------------------------------------------
    def fit_t_componets(
        self,
        trainSet,
        num_epochs: int,
        batch_size: int      = 512,
        initial_lr: float    = 1e-3,
        final_lr: float      = 1e-5,
        huber_delta: float   = 1.0,
        trim_top_frac: float = 0.0,
        decayevery=None,
        decayrate=None,
    ):
        """
        Train the NPCE with cosine annealing and (trimmed) Huber loss —
        the same loss as COLA_NN_Keras, so the two are directly comparable.

        trim_top_frac == 0 : plain Huber(delta=huber_delta)
        trim_top_frac  > 0 : Huber with the worst trim_top_frac of samples in
                             each mini-batch excluded from the mean.
        decayevery, decayrate : ignored (legacy API compatibility)
        """
        # 1. Build PCE features
        Phi_train  = self._make_pce_features(trainSet.lhs_norm)
        n_terms    = Phi_train.shape[1]
        n_train    = Phi_train.shape[0]
        output_dim = trainSet.t_components_norm.shape[1]   # = num_pcs

        # 2. Loss (shared with COLA_NN_Keras)
        loss_fn    = make_trimmed_huber_loss(trim_top_frac, huber_delta)
        loss_label = (
            f"TrimmedHuber(delta={huber_delta}, trim={trim_top_frac})"
            if trim_top_frac > 0 else f"Huber(delta={huber_delta})"
        )

        # 3. Build model
        mlp = self._build_model(input_dim=n_terms, output_dim=output_dim)

        # 4. Cosine annealing LR schedule
        steps_per_epoch = max(1, n_train // batch_size)
        total_steps     = num_epochs * steps_per_epoch
        lr_schedule     = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = initial_lr,
            decay_steps           = total_steps,
            alpha                 = final_lr / initial_lr,
        )

        print(
            f"\n[COLA_NPCE_Keras.fit_t_componets] Training config:\n"
            f"  PCE input dim   = {n_terms}\n"
            f"  output          = {output_dim} PCA coefficients (z = {trainSet.z})\n"
            f"  batch_size      = {batch_size}  "
            f"({steps_per_epoch} steps/epoch, {total_steps} total steps)\n"
            f"  LR schedule     = CosineDecay  "
            f"{initial_lr:.2e} → {final_lr:.2e}\n"
            f"  loss            = {loss_label}\n"
            f"  epochs          = {num_epochs}"
        )

        mlp.compile(
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
            loss      = loss_fn,
        )

        # 5. Train
        nn_model_train_keras(
            mlp,
            epochs     = num_epochs,
            input_data = Phi_train,
            truths     = trainSet.t_components_norm,
            batch_size = batch_size,
        )

        self.models["t-component"] = mlp
        return mlp

    # ------------------------------------------------------------------
    def predict_t_components(self, x: np.ndarray) -> np.ndarray:
        """
        Returns un-normalised PCA coefficients, shape (N, num_pcs).
        Raw parameters are scaled automatically if outside [-1.5, 1.5].
        """
        mlp = self.models["t-component"]
        if x.max() > 1.5 or x.min() < -1.5:
            x = self.param_scaler.transform(x)
        Phi    = self._make_pce_features(x)
        t_norm = mlp.predict(Phi, verbose=0)
        return self.t_comp_scaler.inverse_transform(t_norm)

    # ------------------------------------------------------------------
    # Persistence helpers (the train script stores indices in the metadata
    # bundle instead; these are kept for standalone use)
    # ------------------------------------------------------------------
    def save_pce_metadata(self, metadata_dir: str):
        path = os.path.join(metadata_dir, f"npce_indices{VER}.npy")
        np.save(path, self._indices)
        print(f"  PCE indices saved to: {path}")

    @classmethod
    def load_pce_metadata(cls, metadata_dir: str) -> np.ndarray:
        path = os.path.join(metadata_dir, f"npce_indices{VER}.npy")
        return np.load(path)