"""
cola_npce.py — Neural Polynomial Chaos Expansion emulator for COLASet

Implements COLA_NPCE_Keras: a true two-stage NPCE emulator that mirrors the
architecture in the reference notebook (bernardo7crf/NPCE).

ARCHITECTURE OVERVIEW
---------------------
The original cola_npce.py was incorrect: it expanded the raw cosmological
parameters into a PCE basis and fed that directly into an MLP.  That is just
a feature-engineered MLP — it does not implement NPCE.

True NPCE is a two-stage pipeline:

  Stage 1 — PCE regression (elastic-net):
      Input:  cosmological parameters → PCE basis features  Φ(x)  (N × T)
      Fit:    elastic-net multi-task regression  W  s.t.  Φ W ≈ t_components
      Output: PCE-predicted t-components  t_pce = Φ(x) @ W   (N × N_t)

  Stage 2 — Neural network correction:
      Input:  t_pce  (the PCE prediction, N × N_t)
      Fit:    small MLP  f_θ  s.t.  f_θ(t_pce) ≈ t_components_true
      Output: corrected t-components  t_npce = f_θ(t_pce)   (N × N_t)

  Crucially, the NN operates in the PCE latent space, not on the raw
  cosmological parameters.  It learns only the residual between the PCE
  prediction and the true t-components — a much smoother, easier target.
  The residual connection (output += input, i.e. t_npce = t_pce + delta)
  further enforces this by making the network learn delta ≈ 0 by default.

  The per-component normalisation from train_utils_pk_emulator is preserved:
  both stages work on zero-mean / unit-variance t-components so that all
  latent dimensions contribute equally to the loss.

WHY THE OLD APPROACH WAS WRONG
-------------------------------
The previous implementation computed Φ(x) (PCE features of the raw parameters)
and fed them into an MLP.  This bypasses the PCE regression stage entirely:
there is no W matrix, no "PCE prediction", and therefore no residual for the NN
to correct.  It is equivalent to a polynomial-feature MLP — expressive, but not
the NPCE architecture described in the notebook or train_utils_pk_emulator's
COLA_MultiHead_NPCE_Keras.

RELATIONSHIP TO train_utils_pk_emulator.COLA_MultiHead_NPCE_Keras
------------------------------------------------------------------
COLA_MultiHead_NPCE_Keras builds the PCE basis inside the Keras graph using
TensorFlow ops (_pce_layer) and feeds them straight to a shared backbone with
per-redshift heads.  It is a single-stage learned model; the "PCE" part is the
feature expansion, not a separate fitted regression.

COLA_NPCE_Keras here implements the strict two-stage interpretation from the
notebook:
  • Stage 1 uses a numpy/ElNetFortran-style multi-task elastic-net (or a
    fast numpy fallback) to fit PCE coefficients W in closed form / iteratively.
  • Stage 2 is a small Keras MLP that only sees t_pce and corrects it.

This makes the PCE stage interpretable and independently evaluable.

Dependencies
------------
  numpy, itertools (stdlib), sklearn, keras/tensorflow

Optional: ElNetFortran (Fortran elastic-net solver).  If not available the code
falls back to sklearn.linear_model.MultiTaskElasticNet, which is slower but
produces identical results.

Usage
-----
    from cola_npce import COLA_NPCE_Keras

    # trainSet must already have been through prepare()
    model = COLA_NPCE_Keras(
        trainSet,
        max_degree      = 6,
        norm_q          = 0.75,
        norm_threshold  = 1.0,
        alpha           = 1e-4,
        l1_ratio        = 0.9,
        num_layers      = 3,
        num_neurons     = 256,
    )

    # Stage 1: fit the elastic-net PCE regression
    model.fit_pce(trainSet)

    # Stage 2: fit the NN correction on top of the PCE predictions
    model.fit_t_componets(trainSet, num_epochs=2000)

    # All COLAModel methods work unchanged (predict, plot_errors, get_outliers)
    preds = model.predict(test_lhs, z_idx=0)

Author: based on the reference NPCE notebook by bernardo7crf and
        train_utils_pk_emulator by João Victor Silva Rebouças / Victoria Lloyd.
        Corrected two-stage implementation by Victoria Lloyd (2026).
"""

from itertools import product as iter_product

import numpy as np
import os

import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from typing import Optional

# ── shared utilities from the project ────────────────────────────────────────
from train_utils_pk_emulator_v3 import (
    COLAModel,
    CustomActivationLayer,
    nn_model_train_keras,
    make_trimmed_huber_loss,
)

VER = "_v7small"


# =============================================================================
# PCE basis helpers  (Stage 1)
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
    alpha_i is the polynomial degree in dimension i.

    Pruning criterion (hyperbolic cross / L^q truncation):
        Keep alpha  iff  sum_i (alpha_i / max_degree)^q  <=  norm_threshold
        (with the convention 0^q = 0 for any q > 0).

    This is identical to the two-step filter in the reference notebook:
        filter_q0_norm  +  filter_combinations.

    Parameters
    ----------
    n_dims         : int   — cosmological parameter dimension (e.g. 7)
    max_degree     : int   — maximum polynomial degree per dimension
    norm_q         : float — L^q exponent (0 < q <= 1; smaller → more pruning)
    norm_threshold : float — upper bound on the L^q multi-index norm

    Returns
    -------
    indices : (N_terms, n_dims) int32 ndarray
    """
    degrees = np.arange(max_degree + 1, dtype=np.float64)
    all_combinations = np.array(
        list(iter_product(degrees, repeat=n_dims)), dtype=np.float64
    )

    if max_degree > 0:
        normalised = all_combinations / max_degree
    else:
        normalised = all_combinations.copy()

    # L^q norm per multi-index: 0^q = 0 for any q > 0
    lq_norms = np.sum(
        np.where(normalised > 0, normalised ** norm_q, 0.0),
        axis=1,
    )
    mask = lq_norms <= norm_threshold + 1e-12
    return all_combinations[mask].astype(np.int32)


def _evaluate_pce_basis(indices: np.ndarray, X_norm: np.ndarray) -> np.ndarray:
    """
    Evaluate the polynomial chaos basis at each row of X_norm.

    Each basis function psi_alpha(x) = prod_i  x_i^{alpha_i} is a
    multivariate monomial. Fully vectorised — no Python loops over N or T.

    Parameters
    ----------
    indices  : (N_terms, n_dims) int ndarray
    X_norm   : (N_samples, n_dims) float ndarray — inputs clipped to [-1, 1]

    Returns
    -------
    Phi : (N_samples, N_terms) float32 ndarray
        Design matrix of polynomial basis evaluations.
    """
    X_clipped = np.clip(X_norm, -1.0, 1.0).astype(np.float64)
    N, D = X_clipped.shape
    max_deg = int(indices.max()) if indices.size > 0 else 0

    # pow_table[d, p, n] = X[n, d]^p — built by repeated multiplication so
    # that negative bases with integer exponents are handled correctly.
    pow_table = np.empty((D, max_deg + 1, N), dtype=np.float64)
    pow_table[:, 0, :] = 1.0
    if max_deg >= 1:
        pow_table[:, 1, :] = X_clipped.T
    for p in range(2, max_deg + 1):
        pow_table[:, p, :] = pow_table[:, p - 1, :] * X_clipped.T

    # Advanced indexing: powered[t, d, n] = pow_table[d, indices[t,d], n]
    d_idx   = np.arange(D)[np.newaxis, :]       # (1, D) broadcasts over T
    powered = pow_table[d_idx, indices, :]      # (T, D, N)

    # Product over dimensions → (T, N), then transpose → (N, T)
    Phi = powered.prod(axis=1).T
    return Phi.astype(np.float32)


def _fit_elastic_net_pce(
    Phi_train: np.ndarray,
    T_train: np.ndarray,
    alpha: float,
    l1_ratio: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Fit a multi-task elastic-net regression:

        minimise  (1/2n) ||Phi_train @ W.T - T_train||^2_F
                  + alpha * l1_ratio       * ||W||_1
                  + alpha * (1-l1_ratio)/2 * ||W||^2_F

    Tries ElNetFortran (the fast Fortran solver used in the notebook) first;
    falls back to sklearn.linear_model.MultiTaskElasticNet if unavailable.

    Parameters
    ----------
    Phi_train : (N_train, N_terms) float32  — PCE design matrix
    T_train   : (N_train, N_t)    float     — target t-components (normalised)
    alpha     : float  — overall regularisation strength
    l1_ratio  : float  — blend between L1 (1.0 = pure lasso) and L2 (0.0 = ridge)
    max_iter  : int    — number of coordinate-descent passes (Fortran solver)
    tol       : float  — convergence tolerance

    Returns
    -------
    W : (N_t, N_terms) float64 ndarray
        Coefficient matrix.  Predictions are Phi @ W.T → (N, N_t).
    """
    Phi = Phi_train.astype(np.float64)
    T   = T_train.astype(np.float64)
    N, n_features = Phi.shape
    n_targets     = T.shape[1]

    try:
        import ElNetFortran

        l1_reg = alpha * l1_ratio       * N
        l2_reg = alpha * (1.0 - l1_ratio) * N

        # Coefficient matrix in Fortran order (required by the solver)
        W   = np.asfortranarray(np.zeros((n_targets, n_features), dtype=np.float64))
        Pf  = np.asfortranarray(Phi)
        R   = np.asfortranarray(T - Phi @ W.T)

        # Column norms of Phi (needed by coordinate descent)
        norm_cols = np.asfortranarray((Phi ** 2).sum(axis=0))

        for _ in range(max_iter):
            ElNetFortran.fit(n_features, 1, l1_reg, l2_reg, W, Pf, R, norm_cols)

        print(f"  [PCE] ElNetFortran solver: {max_iter} iterations complete.")
        return W

    except ImportError:
        print("  [PCE] ElNetFortran not found — falling back to sklearn MultiTaskElasticNet.")
        from sklearn.linear_model import MultiTaskElasticNet
        reg = MultiTaskElasticNet(
            alpha      = alpha,
            l1_ratio   = l1_ratio,
            max_iter   = max_iter,
            tol        = tol,
            fit_intercept=False,
        )
        reg.fit(Phi, T)
        # sklearn returns W with shape (n_targets, n_features) — same convention
        return reg.coef_


# =============================================================================
# Neural network correction  (Stage 2)
# =============================================================================

def _build_correction_mlp(
    input_dim: int,
    output_dim: int,
    num_layers: int,
    num_neurons: int,
) -> keras.Model:
    """
    Build the residual MLP that maps t_pce → corrected t-components.

    Architecture:
        Input (t_pce, shape N_t)
          → Dense → CustomActivation   ×  num_layers
          → Dense (output_dim)
          + skip connection: output += input

    The skip / residual connection is the key design choice: it forces the
    network to learn only the *correction* (delta = t_true - t_pce) rather
    than the full mapping from scratch.  At initialisation the correction is
    approximately zero, so training starts from the PCE prediction.

    This mirrors the Net class in the reference notebook exactly.

    Parameters
    ----------
    input_dim   : int  — N_t (number of t-components; same as output_dim)
    output_dim  : int  — N_t (must equal input_dim for the skip connection)
    num_layers  : int  — number of hidden Dense→Activation pairs
    num_neurons : int  — width of each hidden layer
    """
    if input_dim != output_dim:
        raise ValueError(
            f"Residual NPCE network requires input_dim == output_dim "
            f"(both are N_t), got {input_dim} vs {output_dim}."
        )

    inputs = layers.Input(shape=(input_dim,), name="t_pce_input")
    x      = layers.Dense(num_neurons, name="dense_0")(inputs)
    x      = CustomActivationLayer(num_neurons, name="act_0")(x)

    for i in range(1, num_layers):
        x = layers.Dense(num_neurons, name=f"dense_{i}")(x)
        x = CustomActivationLayer(num_neurons, name=f"act_{i}")(x)

    # Project back to t-component space
    delta = layers.Dense(output_dim, name="delta_output")(x)

    # Residual: t_corrected = t_pce + delta
    # This means the network explicitly learns only the residual error of the PCE.
    outputs = layers.Add(name="residual_add")([inputs, delta])

    model = keras.Model(inputs=inputs, outputs=outputs, name="NPCE_correction_MLP")
    model.summary()
    return model


# =============================================================================
# Main class
# =============================================================================

class COLA_NPCE_Keras(COLAModel):
    """
    True two-stage Neural Polynomial Chaos Expansion emulator.

    Stage 1 — PCE elastic-net regression
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A multi-task elastic-net is fitted that maps polynomial basis features
    Phi(x) of the cosmological parameters to normalised t-components:

        W* = argmin  ||Phi(X_train) @ W.T - T_train||^2  + elastic-net penalty

    This gives a sparse, interpretable surrogate.  At inference time:

        t_pce = Phi(x) @ W*.T      (shape: N × N_t)

    Stage 2 — Neural network residual correction
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A small residual MLP is fitted that maps the PCE predictions to the true
    t-components:

        f_θ* = argmin  ||f_θ(t_pce_train) - T_train||^2

    with f_θ(t_pce) = t_pce + delta_θ(t_pce)  (skip connection).

    At inference time:

        t_npce = f_θ*(t_pce)       (shape: N × N_t)

    The NN operates entirely in the PCE latent space — it never sees the raw
    cosmological parameters.  This is the architecture described in the
    reference notebook (bernardo7crf/NPCE).

    Parameters
    ----------
    trainSet       : COLASet  — must have been through prepare()
    max_degree     : int      — max PCE degree per dimension (default 6)
    norm_q         : float    — L^q truncation exponent (default 0.75)
    norm_threshold : float    — L^q norm upper bound (default 1.0)
    alpha          : float    — elastic-net regularisation strength (default 1e-4)
    l1_ratio       : float    — L1/L2 blend: 1.0 = pure lasso (default 0.9)
    pce_max_iter   : int      — coordinate-descent iterations for Stage 1 (default 1000)
    num_layers     : int      — hidden layers in the correction MLP (default 3)
    num_neurons    : int      — neurons per hidden layer (default 256)
    """

    def __init__(
        self,
        trainSet,
        max_degree: int       = 6,
        norm_q: float         = 0.75,
        norm_threshold: float = 1.0,
        alpha: float          = 1e-4,
        l1_ratio: float       = 0.9,
        pce_max_iter: int     = 1000,
        num_layers: int       = 3,
        num_neurons: int      = 256,
    ):
        super().__init__(trainSet)
        self.trainSet       = trainSet
        self.max_degree     = max_degree
        self.norm_q         = norm_q
        self.norm_threshold = norm_threshold
        self.alpha          = alpha
        self.l1_ratio       = l1_ratio
        self.pce_max_iter   = pce_max_iter
        self.num_layers     = num_layers
        self.num_neurons    = num_neurons

        # W: PCE coefficient matrix, shape (N_t, N_terms).  Set by fit_pce().
        self._W: np.ndarray | None = None

        # PCE multi-index set, built once at construction time.
        n_dims = trainSet.lhs_norm.shape[1]
        self._indices = _build_multi_indices(
            n_dims         = n_dims,
            max_degree     = max_degree,
            norm_q         = norm_q,
            norm_threshold = norm_threshold,
        )
        n_terms = self._indices.shape[0]
        print(
            f"\n[COLA_NPCE_Keras] PCE multi-index set:\n"
            f"  {n_dims} params, max_degree={max_degree}, "
            f"q={norm_q}, threshold={norm_threshold}\n"
            f"  → {n_terms} basis terms"
        )

    # ------------------------------------------------------------------
    # PCE feature construction (shared by Stage 1 and Stage 2)
    # ------------------------------------------------------------------

    def _make_pce_features(self, lhs_norm: np.ndarray) -> np.ndarray:
        """
        Expand normalised cosmological parameters into PCE basis features.

        Parameters
        ----------
        lhs_norm : (N, n_params) float ndarray — inputs normalised to [-1, 1]

        Returns
        -------
        Phi : (N, N_terms) float32 ndarray — design matrix
        """
        return _evaluate_pce_basis(self._indices, lhs_norm)

    # ------------------------------------------------------------------
    # Stage 1: PCE elastic-net regression
    # ------------------------------------------------------------------

    def fit_pce(
        self,
        trainSet,
        alpha: Optional[float] = None,
        l1_ratio: Optional[float] = None,
        max_iter: Optional[int] = None,
        tol: float             = 1e-6,
    ) -> np.ndarray:
        """
        Fit Stage 1: elastic-net PCE regression on the training t-components.

        After calling this, self._W holds the coefficient matrix and
        self.t_pce_train holds the PCE-predicted t-components for the
        training set — which is the input to Stage 2 (fit_t_componets).

        Parameters
        ----------
        trainSet  : COLASet  — training set (must be through prepare())
        alpha     : override for self.alpha
        l1_ratio  : override for self.l1_ratio
        max_iter  : override for self.pce_max_iter
        tol       : elastic-net convergence tolerance

        Returns
        -------
        W : (N_t, N_terms) float64 ndarray — PCE coefficient matrix
        """
        alpha    = alpha    if alpha    is not None else self.alpha
        l1_ratio = l1_ratio if l1_ratio is not None else self.l1_ratio
        max_iter = max_iter if max_iter is not None else self.pce_max_iter

        print(
            f"\n[COLA_NPCE_Keras] Stage 1 — PCE elastic-net regression\n"
            f"  N_train   = {len(trainSet.lhs_norm)}\n"
            f"  N_terms   = {self._indices.shape[0]}\n"
            f"  N_t       = {trainSet.t_components_norm.shape[1]}\n"
            f"  alpha     = {alpha:.2e}\n"
            f"  l1_ratio  = {l1_ratio}\n"
            f"  max_iter  = {max_iter}"
        )

        # Build PCE design matrix for training data
        Phi_train = self._make_pce_features(trainSet.lhs_norm)
        print(f"  Phi_train shape: {Phi_train.shape}")

        # Fit the elastic-net; W shape: (N_t, N_terms)
        W = _fit_elastic_net_pce(
            Phi_train = Phi_train,
            T_train   = trainSet.t_components_norm,
            alpha     = alpha,
            l1_ratio  = l1_ratio,
            max_iter  = max_iter,
            tol       = tol,
        )
        self._W = W

        # Stage 1 training-set predictions — used as input to Stage 2
        # Shape: (N_train, N_t)
        self.t_pce_train = (Phi_train @ W.T).astype(np.float32)

        # Diagnostics: PCE-only MSE on training data
        pce_mse_train = float(np.mean((self.t_pce_train - trainSet.t_components_norm) ** 2))
        sparsity = float((W == 0).mean() * 100)
        print(
            f"\n  Stage 1 complete:\n"
            f"    PCE train MSE (normalised t-space): {pce_mse_train:.6f}\n"
            f"    W sparsity: {sparsity:.1f}% zero coefficients\n"
            f"    Non-zero basis terms used: {int((W != 0).any(axis=0).sum())} / {W.shape[1]}"
        )
        return W

    def predict_t_pce(self, lhs_norm: np.ndarray) -> np.ndarray:
        """
        Apply Stage 1 only: return PCE-predicted (normalised) t-components.

        Parameters
        ----------
        lhs_norm : (N, n_params) — normalised cosmological parameters

        Returns
        -------
        t_pce_norm : (N, N_t) float32 — PCE prediction in normalised t-space
        """
        if self._W is None:
            raise RuntimeError("Call fit_pce() before predict_t_pce().")
        Phi = self._make_pce_features(lhs_norm)
        return (Phi @ self._W.T).astype(np.float32)

    # ------------------------------------------------------------------
    # Stage 2: Neural network correction
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
        decayevery=None,     # legacy API, ignored
        decayrate=None,      # legacy API, ignored
    ):
        """
        Fit Stage 2: train the residual MLP correction on top of the PCE.

        The NN takes the PCE-predicted t-components (t_pce) as input and
        produces corrected t-components.  The skip connection means it only
        needs to learn the residual error of the PCE.

        If fit_pce() has not been called yet it will be run automatically with
        the alpha / l1_ratio stored on the instance (set at construction time).
        You can also call fit_pce() explicitly beforehand if you want to tune
        the elastic-net hyper-parameters or inspect the Stage 1 predictions
        before committing to Stage 2 training.

        Parameters
        ----------
        trainSet      : COLASet — training set (must be through prepare())
        num_epochs    : int     — total training epochs
        batch_size    : int     — mini-batch size (default 512)
        initial_lr    : float   — starting LR for cosine annealing (default 1e-3)
        final_lr      : float   — ending LR (default 1e-5)
        huber_delta   : float   — Huber loss delta (default 1.0).
                                  Ignored when trim_top_frac > 0.
        trim_top_frac : float   — fraction of highest per-sample losses to drop
                                  per mini-batch (0.0 = disabled).
        """
        if self._W is None:
            print(
                "\n[COLA_NPCE_Keras] fit_pce() has not been called yet — "
                "running Stage 1 automatically before Stage 2."
            )
            self.fit_pce(trainSet)

        n_t       = trainSet.t_components_norm.shape[1]
        n_train   = len(trainSet.lhs_norm)

        # ── Input to Stage 2: PCE-predicted t-components on training data ──
        # These were computed and cached by fit_pce(); shape (N_train, N_t).
        t_pce_train = self.t_pce_train
        # ── Target for Stage 2: true normalised t-components ───────────────
        t_true_norm = trainSet.t_components_norm.astype(np.float32)

        residual_mse = float(np.mean((t_pce_train - t_true_norm) ** 2))
        print(
            f"\n[COLA_NPCE_Keras] Stage 2 — NN residual correction\n"
            f"  Input:  t_pce_train  shape {t_pce_train.shape}  "
            f"(PCE predictions in normalised t-space)\n"
            f"  Target: t_true_norm  shape {t_true_norm.shape}\n"
            f"  Residual MSE the NN must learn to reduce: {residual_mse:.6f}"
        )

        # ── Build the correction MLP ────────────────────────────────────────
        # input_dim == output_dim == N_t (required by the residual connection)
        mlp = _build_correction_mlp(
            input_dim   = n_t,
            output_dim  = n_t,
            num_layers  = self.num_layers,
            num_neurons = self.num_neurons,
        )

        # ── Loss function ───────────────────────────────────────────────────
        if trim_top_frac > 0:
            loss_fn    = make_trimmed_huber_loss(trim_top_frac, huber_delta)
            loss_label = f"TrimmedHuber(delta={huber_delta}, trim={trim_top_frac:.0%})"
        else:
            loss_fn    = keras.losses.Huber(delta=huber_delta)
            loss_label = f"Huber(delta={huber_delta})"

        # ── Cosine-decay LR schedule ────────────────────────────────────────
        steps_per_epoch = max(1, n_train // batch_size)
        total_steps     = num_epochs * steps_per_epoch
        lr_schedule     = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate = initial_lr,
            decay_steps           = total_steps,
            alpha                 = final_lr / initial_lr,
        )

        print(
            f"  batch_size      = {batch_size}  "
            f"({steps_per_epoch} steps/epoch, {total_steps} total steps)\n"
            f"  LR schedule     = CosineDecay {initial_lr:.2e} → {final_lr:.2e}\n"
            f"  loss            = {loss_label}\n"
            f"  epochs          = {num_epochs}"
        )

        mlp.compile(
            optimizer = keras.optimizers.Adam(learning_rate=lr_schedule),
            loss      = loss_fn,
        )

        # ── Train: NN maps t_pce → t_true ──────────────────────────────────
        nn_model_train_keras(
            mlp,
            epochs     = num_epochs,
            input_data = t_pce_train,
            truths     = t_true_norm,
            batch_size = batch_size,
        )

        self.models["t-component"] = mlp
        return mlp

    # ------------------------------------------------------------------
    # Inference  (overrides COLAModel.predict_t_components)
    # ------------------------------------------------------------------

    def predict_t_components(self, x: np.ndarray) -> np.ndarray:
        """
        Full two-stage NPCE prediction:

            1. Normalise x if needed
            2. Compute PCE features Phi(x)
            3. Stage 1: t_pce = Phi(x) @ W.T
            4. Stage 2: t_npce = f_θ(t_pce)    [residual NN correction]
            5. Invert t-component normalisation

        Parameters
        ----------
        x : (N, n_params) ndarray — raw or normalised cosmological parameters.
            Automatically normalised if values fall outside [-1.5, 1.5].

        Returns
        -------
        t_raw : (N, N_t) float ndarray — unnormalised (physical) t-components
        """
        if self._W is None:
            raise RuntimeError("Call fit_pce() before calling predict_t_components().")
        if "t-component" not in self.models:
            raise RuntimeError("Call fit_t_componets() before calling predict_t_components().")

        mlp = self.models["t-component"]

        # Normalise cosmological parameters to [-1, 1] if needed
        if x.max() > 1.5 or x.min() < -1.5:
            x = self.param_scaler.transform(x)

        # Stage 1: PCE prediction in normalised t-space
        t_pce = self.predict_t_pce(x)        # (N, N_t)

        # Stage 2: NN correction (residual; skip connection is inside the MLP)
        t_npce_norm = mlp.predict(t_pce, verbose=0)  # (N, N_t)

        # Invert the per-column normalisation applied in prepare()
        return self.t_comp_scaler.inverse_transform(t_npce_norm)

    def predict_t_components_pce_only(self, x: np.ndarray) -> np.ndarray:
        """
        Stage 1 only prediction (no NN correction).  Useful for ablation /
        comparing PCE accuracy vs. full NPCE.

        Returns unnormalised t-components from the PCE regression alone.
        """
        if self._W is None:
            raise RuntimeError("Call fit_pce() first.")

        if x.max() > 1.5 or x.min() < -1.5:
            x = self.param_scaler.transform(x)

        t_pce_norm = self.predict_t_pce(x)
        return self.t_comp_scaler.inverse_transform(t_pce_norm)

    # ------------------------------------------------------------------
    # Diagnostics helpers
    # ------------------------------------------------------------------

    def eval_pce_stage(self, testSet) -> dict:
        """
        Evaluate Stage 1 (PCE only) on testSet.

        Returns a dict with keys:
            'mse_t_norm'  — MSE in normalised t-space
            't_pce_norm'  — PCE predictions in normalised t-space (N, N_t)
        """
        lhs = testSet.lhs_norm if testSet.lhs_norm is not None else \
              self.param_scaler.transform(testSet.lhs)
        t_pce_norm = self.predict_t_pce(lhs)
        t_true_norm = testSet.t_components_norm
        mse = float(np.mean((t_pce_norm - t_true_norm) ** 2))
        print(f"[PCE stage] Test MSE (normalised t-space): {mse:.6f}")
        return {"mse_t_norm": mse, "t_pce_norm": t_pce_norm}

    def eval_npce_stage(self, testSet) -> dict:
        """
        Evaluate the full two-stage NPCE pipeline on testSet.

        Returns a dict with keys:
            'mse_t_norm'  — MSE in normalised t-space
            't_npce_norm' — NPCE predictions in normalised t-space (N, N_t)
        """
        lhs = testSet.lhs_norm if testSet.lhs_norm is not None else \
              self.param_scaler.transform(testSet.lhs)
        t_pce_norm  = self.predict_t_pce(lhs)
        mlp         = self.models["t-component"]
        t_npce_norm = mlp.predict(t_pce_norm, verbose=0)
        t_true_norm = testSet.t_components_norm
        mse = float(np.mean((t_npce_norm - t_true_norm) ** 2))
        print(f"[NPCE stage] Test MSE (normalised t-space): {mse:.6f}")
        return {"mse_t_norm": mse, "t_npce_norm": t_npce_norm}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_pce_weights(self, metadata_dir: str) -> None:
        """Save the PCE coefficient matrix W to disk."""
        os.makedirs(metadata_dir, exist_ok=True)
        path = os.path.join(metadata_dir, f"npce_W{VER}.npy")
        np.save(path, self._W)
        idx_path = os.path.join(metadata_dir, f"npce_indices{VER}.npy")
        np.save(idx_path, self._indices)
        print(f"  PCE weights saved to:  {path}")
        print(f"  PCE indices saved to:  {idx_path}")

    def load_pce_weights(self, metadata_dir: str) -> None:
        """Load a previously saved PCE coefficient matrix."""
        path     = os.path.join(metadata_dir, f"npce_W{VER}.npy")
        idx_path = os.path.join(metadata_dir, f"npce_indices{VER}.npy")
        self._W       = np.load(path)
        self._indices = np.load(idx_path)
        print(f"  PCE weights loaded from: {path}  (shape {self._W.shape})")

        # Recompute cached training-set PCE predictions if possible
        if hasattr(self, "trainSet") and self.trainSet.lhs_norm is not None:
            Phi = self._make_pce_features(self.trainSet.lhs_norm)
            self.t_pce_train = (Phi @ self._W.T).astype(np.float32)