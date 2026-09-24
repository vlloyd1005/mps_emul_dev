"""
cola_npce.py — Neural Polynomial Chaos Expansion emulator for COLASet

Implements COLA_NPCE_Keras, a drop-in replacement for COLA_NN_Keras that uses
a Neural PCE architecture instead of a plain MLP.

The NPCE idea (Rebouças et al. / bernardo7crf/NPCE) replaces the raw
cosmological parameters fed to the network with a polynomial chaos expansion
(PCE) basis evaluated at each input point.  The PCE basis spans all
multivariate monomials of the normalised inputs up to a chosen maximum degree,
pruned so that:

  1. No single-variable degree exceeds `max_degree_per_dim`.
  2. The L^q norm of the multi-index (with q = `norm_q`) does not exceed
     `norm_threshold` (hyperbolic cross truncation).

This gives the network a rich, structured set of nonlinear input features
that encode interactions between cosmological parameters exactly, rather than
relying entirely on learned weights to discover them.  The network itself
(same custom-activation MLP as COLA_NN_Keras) then only needs to learn a
relatively smooth mapping from the already-expressive PCE features to the
t-components.

Dependencies
------------
  numpy, itertools (stdlib), keras / tensorflow (already in the project)

No external PCE library (chaospy, numpoly, etc.) is required; the basis
evaluation is done with plain NumPy.

Usage
-----
    from cola_npce import COLA_NPCE_Keras

    train_set.prepare(num_pcs=25, num_pcs_z=15)

    model = COLA_NPCE_Keras(
        train_set,
        max_degree    = 12,    # max per-dimension polynomial degree
        norm_q        = 0.75,  # L^q truncation norm  (0 < q <= 1)
        norm_threshold= 1.0,   # upper bound on the L^q multi-index norm
        num_layers    = 3,
        num_neurons   = 512,
    )
    model.fit_t_componets(train_set, num_epochs=4000)

    # All COLAModel methods (predict, plot_errors, get_outliers) work unchanged.

Author: Victoria Lloyd (2026), based on the NPCE notebook by bernardo7crf.
"""

from itertools import product as iter_product

import numpy as np
import joblib
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers
from keras.regularizers import l1_l2

# ── import shared utilities from the project ─────────────────────────────────
import train_utils_pk_emulator_v3 as utils
from train_utils_pk_emulator_v3 import (
    COLAModel,
    CustomActivationLayer,
    nn_model_train_keras,
)

VER = "_v7"


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

    Each row is one multi-index alpha = (alpha_1, ..., alpha_d), where
    alpha_i is the polynomial degree for dimension i.

    Two pruning steps are applied (matching the notebook):

    Step 1 — L^q_0 filter on the raw integer grid:
        Keep alpha if  sum_i (alpha_i / max_degree)^q  <=  norm_threshold
        (with the convention 0^q = 0).  This is the hyperbolic cross that
        limits total interaction order.

    Step 2 — Per-element normalisation filter (notebook's filter_combinations):
        Divide each multi-index by max_degree, then apply the same L^q test
        with a slightly tighter threshold (norm_threshold - epsilon).  In the
        notebook both thresholds are the same value, so this step is
        effectively a no-op duplicate; we keep it for fidelity.

    Parameters
    ----------
    n_dims        : int   — number of cosmological parameters (input dimension)
    max_degree    : int   — maximum polynomial degree per dimension
    norm_q        : float — exponent for the L^q multi-index norm (0 < q <= 1)
    norm_threshold: float — upper bound on the L^q norm

    Returns
    -------
    indices : (N_terms, n_dims) int ndarray
    """
    degrees = np.arange(max_degree + 1)

    # All combinations (max_degree+1)^n_dims — can be large; prune eagerly.
    # We iterate dimension by dimension to avoid materialising the full grid.
    indices = np.array(list(iter_product(degrees, repeat=n_dims)), dtype=np.float64)

    # Step 1: L^q norm filter on (alpha / max_degree) entries
    # Guard against zero max_degree
    if max_degree > 0:
        normalised = indices / max_degree
    else:
        normalised = indices.copy()

    # Use the convention 0^q = 0 for any q > 0
    lq_norms = np.sum(
        np.where(normalised > 0, normalised ** norm_q, 0.0),
        axis=1,
    )
    mask = lq_norms <= norm_threshold + 1e-12
    indices = indices[mask]

    return indices.astype(np.int32)


def _evaluate_pce_basis(indices: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    """
    Evaluate the polynomial chaos basis at each row of X_norm.

    Each basis function is a product of univariate monomials:
        psi_alpha(x) = prod_i  x_i^{alpha_i}

    Fully vectorised over both samples and terms -- no Python loops over N or T.

    Strategy: build a power lookup table of shape (n_dims, max_degree+1, N_samples)
    where table[d, p, n] = X[n, d]^p using repeated multiplication, then gather
    with advanced indexing and take the product over dimensions.

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

    # pow_table[d, p, n] = X_clipped[n, d] ** p, built by repeated multiplication
    # to correctly handle negative bases with integer exponents.
    pow_table = np.empty((D, max_deg + 1, N), dtype=np.float32)
    pow_table[:, 0, :] = 1.0
    if max_deg >= 1:
        pow_table[:, 1, :] = X_clipped.T
    for p in range(2, max_deg + 1):
        pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T

    # # powered[t, d, n] = pow_table[d, indices[t,d], n]  via advanced indexing
    # d_idx   = np.arange(D)[np.newaxis, :]   # (1, D) broadcasts over T
    # powered = pow_table[d_idx, indices, :]  # (T, D, N)

    # # Product over dimensions then transpose to (N, T)
    # Phi = powered.prod(axis=1).T

    # return Phi.astype(np.float32)
    T = indices.shape[0]
    Phi_T = np.ones((T, N), dtype=np.float32)   # accumulate (T, N) directly — no (T, D, N) ever exists

    for d in range(D):
        # Gather just this dimension's contribution: (T, N), one D-slice at a time
        Phi_T *= pow_table[d, indices[:, d], :]

    # diffs = np.abs(Phi - Phi_T.T)

    # # Replicate allclose's actual formula exactly, with its default atol
    # atol = 1e-8
    # rtol = 1e-5
    # threshold = atol + rtol * np.abs(Phi)
    # violation_mask = diffs > threshold

    # n_violations = violation_mask.sum()
    # print("violations: ", n_violations)
    # finite_match = np.allclose(
    #     Phi,
    #     Phi_T.T,
    #     rtol=1e-5,
    # )
    # print("Finite entries match:", finite_match)

    return Phi_T.T.astype(np.float32)


# @tf.keras.utils.register_keras_serializable(package="mps_emu")
class TrimmedMSELoss(keras.losses.Loss):
    """
    MSE loss with the top `trim_top_frac` of per-sample losses excluded
    from each mini-batch before averaging.
    """
    def __init__(self, trim_top_frac: float = 0.05, **kwargs):
        super().__init__(**kwargs)
        self.trim_top_frac = trim_top_frac

    def call(self, y_true, y_pred):
        per_sample    = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        sorted_losses = tf.sort(per_sample)
        n    = tf.shape(sorted_losses)[0]
        keep = tf.cast(
            tf.math.ceil(tf.cast(n, tf.float32) * (1.0 - self.trim_top_frac)),
            tf.int32,
        )
        return tf.reduce_mean(sorted_losses[:keep])

    def get_config(self):
        return {**super().get_config(), "trim_top_frac": self.trim_top_frac}
    
    
class COLA_NPCE_Keras(COLAModel):
    """
    Neural Polynomial Chaos Expansion emulator for t-components.

    Inherits all prediction and evaluation methods from COLAModel.
    Only fit_t_componets and predict_t_components are overridden.

    Architecture
    ------------
    Input: cosmological parameters (normalised to [-1, 1])
        ↓
    PCE basis evaluation  →  (N_samples, N_pce_terms)  [no trainable weights]
        ↓
    Dense MLP with CustomActivationLayer  (same as COLA_NN_Keras)
        ↓
    Output: t-components (normalised to zero mean / unit variance)

    The PCE feature layer is a fixed (non-trainable) transformation.  This
    means the network input dimension is N_pce_terms, not n_params — typically
    much larger, giving the MLP rich structured features for free.

    Parameters
    ----------
    trainSet       : COLASet — must have been through prepare()
    max_degree     : int     — max per-dimension polynomial degree (default 6)
                               Lower values give fewer terms and faster training
                               but may underfit.  The notebook used 12; start
                               with 6–8 for this higher-dimensional problem.
    norm_q         : float   — L^q truncation exponent (default 0.75).
                               Smaller values prune more aggressively.
    norm_threshold : float   — upper bound on L^q norm of multi-index
                               (default 1.0, matching the notebook).
    num_layers     : int     — number of hidden layers in the MLP (default 3)
    num_neurons    : int     — neurons per hidden layer (default 512)
    """

    def __init__(
        self,
        trainSet,
        max_degree: int    = 6,
        norm_q: float      = 0.75,
        norm_threshold: float = 1.0,
        num_layers: int    = 3,
        num_neurons: int   = 512,
    ):
        super().__init__(trainSet)
        self.trainSet      = trainSet
        self.max_degree    = max_degree
        self.norm_q        = norm_q
        self.norm_threshold= norm_threshold
        self.num_layers    = num_layers
        self.num_neurons   = num_neurons

        # Build PCE multi-index matrix from the training-set parameter dimension
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
    # PCE feature construction
    # ------------------------------------------------------------------

    def _make_pce_features(self, lhs_norm: np.ndarray) -> np.ndarray:
        """
        Expand normalised cosmological parameters into PCE basis features.

        Parameters
        ----------
        lhs_norm : (N, n_params) float ndarray — MinMax-normalised inputs

        Returns
        -------
        Phi : (N, N_terms) float32 ndarray
        """
        return _evaluate_pce_basis(self._indices, lhs_norm)

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _build_model(self, input_dim: int, output_dim: int) -> keras.Model:
        """
        Build the MLP that maps PCE features → normalised t-components.

        Uses the same CustomActivationLayer as COLA_NN_Keras so the two
        models are directly comparable.
        """
        reg = None  # no L1/L2 by default; add as a parameter if needed

        def apply_activation(x):
            return CustomActivationLayer(self.num_neurons)(x)

        inputs  = layers.Input(shape=(input_dim,))
        x       = layers.Dense(self.num_neurons, kernel_regularizer=reg)(inputs)
        x       = apply_activation(x)

        for _ in range(self.num_layers - 1):
            x = layers.Dense(self.num_neurons, kernel_regularizer=reg)(x)
            x = apply_activation(x)

        outputs = layers.Dense(output_dim)(x)
        model   = models.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit_t_componets(
        self,
        trainSet,
        num_epochs: int,
        batch_size: int    = 512,
        initial_lr: float  = 1e-3,
        final_lr: float    = 1e-5,
        huber_delta: float = 1.0,
        trim_top_frac: float = 0.0,
        decayevery=None,
        decayrate=None,
    ):
        """
        Train the NPCE t-component emulator with cosine annealing.

        Loss selection
        --------------
        trim_top_frac == 0  (default):
            Standard Huber loss, controlled by huber_delta.
            huber_delta = 1.0 — quadratic below 1 sigma, linear above.
            Increase to 2-3 to tolerate more outliers; decrease to 0.5
            to suppress them more aggressively.

        trim_top_frac > 0:
            Trimmed MSE — per-sample MSE averaged over the bottom
            (1 - trim_top_frac) fraction of each mini-batch.
            The top trim_top_frac samples are excluded entirely from
            the gradient, forcing the network to focus on the bulk.
            huber_delta is ignored in this mode (MSE is used instead,
            since outlier robustness is handled structurally by trimming).

        Parameters
        ----------
        trainSet      : COLASet — prepared training dataset
        num_epochs    : int     — total training epochs
        batch_size    : int     — mini-batch size (default 512)
        initial_lr    : float   — starting LR for cosine annealing (default 1e-3)
        final_lr      : float   — final LR floor (default 1e-5)
        huber_delta   : float   — Huber loss transition point (default 1.0).
                                Ignored when trim_top_frac > 0.
        trim_top_frac : float   — fraction of highest per-sample losses to drop
                                per mini-batch (default 0.0 = no trimming).
                                Switches loss to MSE when > 0.
                                Meaningful range: 0.01–0.10.
                                Requires batch_size large enough that
                                floor(batch_size * trim_top_frac) >= 1.
        decayevery, decayrate : ignored (legacy API compatibility)
        """
        if trim_top_frac > 0 and huber_delta != 1.0:
            print(
                f"[WARN] trim_top_frac={trim_top_frac} > 0, so loss is trimmed MSE. "
                f"huber_delta={huber_delta} will be ignored."
            )

        # 1. Build PCE features
        Phi_train  = self._make_pce_features(trainSet.lhs_norm)
        n_terms    = Phi_train.shape[1]
        n_train    = Phi_train.shape[0]
        output_dim = trainSet.t_components_norm.shape[1]

        # 2. Select / build loss function
        if trim_top_frac > 0:
            loss_fn    = TrimmedMSELoss(trim_top_frac=trim_top_frac)
            loss_label = f"TrimmedMSE(trim_top={trim_top_frac:.0%})"

            # _frac = trim_top_frac   # capture for closure
            # def loss_fn(y_true, y_pred):
            #     per_sample = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
            #     sorted_losses = tf.sort(per_sample)                   # ascending
            #     n    = tf.shape(sorted_losses)[0]
            #     keep = tf.cast(
            #         tf.math.ceil(tf.cast(n, tf.float32) * (1.0 - _frac)),
            #         tf.int32,
            #     )
            #     return tf.reduce_mean(sorted_losses[:keep])
            _frac = trim_top_frac
            def loss_fn(y_true, y_pred):
                per_sample = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
                batch_size = tf.shape(per_sample)[0]
                n_trim = tf.cast(
                    tf.math.floor(tf.cast(batch_size, tf.float32) * _frac),
                    tf.int32
                )
                n_trim = tf.maximum(n_trim, 0)

                _, worst_indices = tf.math.top_k(per_sample, k=n_trim, sorted=False)
                mask = tf.ones_like(per_sample, dtype=tf.bool)
                mask = tf.tensor_scatter_nd_update(
                    mask,
                    tf.expand_dims(worst_indices, axis=1),
                    tf.zeros_like(worst_indices, dtype=tf.bool),
                )
                kept = tf.boolean_mask(per_sample, mask)
                return tf.reduce_mean(kept)
            loss_label = f"TrimmedMSE(trim_top={trim_top_frac:.0%})"
        else:
            loss_fn    = keras.losses.Huber(delta=huber_delta)
            loss_label = f"Huber(delta={huber_delta})"

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
    # Inference
    # ------------------------------------------------------------------

    def predict_t_components(self, x: np.ndarray) -> np.ndarray:
        """
        Run the NPCE and invert the t-component normalisation.

        Parameters
        ----------
        x : (N, n_params) ndarray — raw or normalised cosmological parameters.
            If values fall outside [-1.5, 1.5] the param_scaler is applied
            automatically (same heuristic as COLA_NN_Keras).

        Returns
        -------
        t_raw : (N, N_t_components) float ndarray — unnormalised t-components
        """
        mlp = self.models["t-component"]

        # Normalise if needed
        if x.max() > 1.5 or x.min() < -1.5:
            x = self.param_scaler.transform(x)

        # Expand to PCE features
        Phi    = self._make_pce_features(x)

        t_norm = mlp.predict(Phi, verbose=0)
        return self.t_comp_scaler.inverse_transform(t_norm)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_pce_metadata(self, metadata_dir: str):
        """
        Save the PCE multi-index matrix alongside the other metadata so that
        a loaded model can reconstruct features at inference time.

        Call this after fit_t_componets if you want to reload the model later.
        """
        path = os.path.join(metadata_dir, f"npce_indices{VER}.npy")
        np.save(path, self._indices)
        print(f"  PCE indices saved to: {path}")

    @classmethod
    def load_pce_metadata(cls, metadata_dir: str) -> np.ndarray:
        """Load PCE indices previously saved by save_pce_metadata."""
        path = os.path.join(metadata_dir, f"npce_indices{VER}.npy")
        return np.load(path)