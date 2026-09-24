"""
train.py — MPS Emulator Training Script

Usage examples:
    # Minimal (required flags only):
    python ./mps_emu/train.py --cosmo_type w0wacdm --prior_type expanded --nl_type lin

    # Train with NPCE instead of the default MLP:
    python ./mps_emu/train.py \
        --cosmo_type w0wacdm --prior_type expanded --nl_type halofit \
        --model_type npce \
        --pce_max_degree 8 --pce_norm_q 0.75 --pce_norm_threshold 1.0

    # Full MLP override:
    python ./mps_emu/train.py \
        --cosmo_type w0wacdm \
        --prior_type expanded \
        --nl_type lin \
        --n_batches 20 \
        --num_epochs 3000 \
        --num_layers 4 \
        --num_neurons 1024 \
        --num_pcs 25 \
        --num_pcs_z 15 \
        --start_batch 0 \
        --test_batch 100
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import train_utils_pk_emulator as utils
from cola_npce import COLA_NPCE_Keras


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the MPS power-spectrum emulator neural network.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required / core configuration ---
    required = parser.add_argument_group("required arguments")
    required.add_argument(
        "--cosmo_type",
        required=True,
        choices=["lcdm", "w0wacdm"],
        help="Cosmological model type.",
    )
    required.add_argument(
        "--prior_type",
        required=True,
        choices=["constrained", "expanded"],
        help="Prior range to use for training.",
    )
    required.add_argument(
        "--nl_type",
        required=True,
        choices=["lin", "halofit", "mead2020", "mead2020_feedback", "mead2020_feedback_Tfree"],
        help=(
            "Linear ('lin'), or non-linear matter power spectrum with halofit ('halofit'), "
            "HMCode ('mead2020'), HMCode with baryonic feedback fixed to logT_AGN=7.8 "
            "('mead2020_feedback'), or with logT_AGN free ('mead2020_feedback_Tfree')."
        ),
    )

    # --- Batch / data configuration ---
    data_group = parser.add_argument_group("data arguments")
    data_group.add_argument(
        "--start_batch",
        type=int,
        default=0,
        help="Index of the first training batch to load.",
    )
    data_group.add_argument(
        "--n_batches",
        type=int,
        default=20,
        help="Number of training batches to load.",
    )
    data_group.add_argument(
        "--test_batch",
        type=int,
        default=100,
        help="Batch index to use for the test/validation set.",
    )
    data_group.add_argument(
        "--w0_min",
        type=float,
        default=None,
        help=(
            "If set, filter training and test sets to w0 >= this value "
            "(expanded prior only).  E.g. --w0_min -2.0"
        ),
    )
    data_group.add_argument(
        "--w0wa_max",
        type=float,
        default=None,
        help=(
            "If set, filter training and test sets to w0+wa <= this value "
            "(expanded prior only).  E.g. --w0wa_max -0.5"
        ),
    )

    # --- PCA configuration ---
    pca_group = parser.add_argument_group("PCA arguments")
    pca_group.add_argument(
        "--num_pcs",
        type=int,
        default=25,
        help="Number of principal components for the power-spectrum axis.",
    )
    pca_group.add_argument(
        "--num_pcs_z",
        type=int,
        default=15,
        help="Number of principal components for the redshift axis.",
    )

    # --- Model type ---
    model_group = parser.add_argument_group("model type arguments")
    model_group.add_argument(
        "--model_type",
        default="mlp",
        choices=["mlp", "npce"],
        help=(
            "Emulator architecture.  'mlp' uses the standard dense MLP "
            "(COLA_NN_Keras).  'npce' uses Neural Polynomial Chaos Expansion "
            "(COLA_NPCE_Keras), which first expands the cosmological parameters "
            "into a polynomial basis before feeding them to the MLP."
        ),
    )

    # --- Shared network / training configuration ---
    nn_group = parser.add_argument_group("neural network arguments")
    nn_group.add_argument(
        "--num_epochs",
        type=int,
        default=3000,
        help="Number of training epochs.",
    )
    nn_group.add_argument(
        "--num_layers",
        type=int,
        default=4,
        help="Number of hidden layers in the MLP (shared by both model types).",
    )
    nn_group.add_argument(
        "--num_neurons",
        type=int,
        default=1024,
        help="Number of neurons per hidden layer (shared by both model types).",
    )
    nn_group.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Mini-batch size for training.",
    )
    nn_group.add_argument(
        "--initial_lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for cosine annealing.",
    )
    nn_group.add_argument(
        "--final_lr",
        type=float,
        default=1e-5,
        help="Final (floor) learning rate for cosine annealing.",
    )
    nn_group.add_argument(
        "--huber_delta",
        type=float,
        default=1.0,
        help=(
            "Transition point of the Huber loss (in normalised t-component units). "
            "1.0 = quadratic below 1 sigma, linear above.  Increase to 2-3 if the "
            "model underfits edge cosmologies; decrease to 0.5 if outliers dominate."
        ),
    )
    # Legacy step-decay params — accepted but ignored when cosine annealing is active.
    nn_group.add_argument(
        "--decay_every",
        type=int,
        default=None,
        help="(Legacy) Decay the learning rate every this many epochs. Ignored by default.",
    )
    nn_group.add_argument(
        "--decay_rate",
        type=float,
        default=None,
        help="(Legacy) Learning-rate decay multiplier. Ignored by default.",
    )

    # --- NPCE-specific hyperparameters ---
    npce_group = parser.add_argument_group(
        "NPCE arguments",
        description=(
            "These flags are only used when --model_type npce is set. "
            "They control the polynomial chaos expansion basis."
        ),
    )
    npce_group.add_argument(
        "--pce_max_degree",
        type=int,
        default=6,
        help=(
            "Maximum per-dimension polynomial degree for the PCE basis.  "
            "The notebook used 12 for 6 parameters; start with 6-8 for 7 parameters "
            "to keep the basis size manageable.  Higher values give more expressive "
            "features but a larger input dimension and slower training."
        ),
    )
    npce_group.add_argument(
        "--pce_norm_q",
        type=float,
        default=0.75,
        help=(
            "L^q norm exponent for hyperbolic-cross truncation of the PCE basis "
            "(0 < q <= 1).  Smaller values prune more aggressively, keeping fewer "
            "high-order interaction terms."
        ),
    )
    npce_group.add_argument(
        "--pce_norm_threshold",
        type=float,
        default=1.0,
        help=(
            "Upper bound on the L^q multi-index norm.  Multi-indices whose L^q norm "
            "exceeds this threshold are discarded.  Default 1.0 matches the notebook."
        ),
    )

    # --- Output configuration ---
    out_group = parser.add_argument_group("output arguments")
    out_group.add_argument(
        "--model_dir",
        default="/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/models",
        help="Directory where the trained model will be saved.",
    )
    out_group.add_argument(
        "--fig_dir",
        default="/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/validation_figs",
        help="Directory where validation figures will be saved.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Prior-cut helper (shared between train and test sets)
# ---------------------------------------------------------------------------

def _apply_prior_cuts(cola_set, w0_min, w0wa_max):
    """
    Remove cosmologies outside the requested prior bounds from cola_set
    in-place and update the w0_min / w0wa_max tags so _metadata_tag() reflects
    the actual filtering.

    Parameters
    ----------
    cola_set : utils.COLASet
    w0_min   : float or None
    w0wa_max : float or None
    """
    w0_col   = utils.params.index("w")
    w0wa_col = utils.params.index("w0+wa")

    mask = np.ones(len(cola_set.lhs), dtype=bool)

    if w0_min is not None:
        w0_mask = cola_set.lhs[:, w0_col] >= w0_min
        n_cut   = (~w0_mask & mask).sum()
        mask   &= w0_mask
        print(f"  w0 cut   (w0 >= {w0_min}):     removed {n_cut} cosmologies.")

    if w0wa_max is not None:
        w0wa_mask = cola_set.lhs[:, w0wa_col] <= w0wa_max
        n_cut     = (~w0wa_mask & mask).sum()
        mask     &= w0wa_mask
        print(f"  w0+wa cut (w0+wa <= {w0wa_max}): removed {n_cut} additional cosmologies.")

    if not mask.all():
        for attr in ("lhs", "pks_lin", "mps_approxes", "frac_pks", "logfracs"):
            if hasattr(cola_set, attr):
                setattr(cola_set, attr, getattr(cola_set, attr)[mask])
        print(f"  → {len(cola_set.lhs)} cosmologies remaining.")

    # Store on the set so _metadata_tag() embeds the filter in saved filenames
    cola_set.w0_min   = w0_min
    cola_set.w0wa_max = w0wa_max


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Validate NPCE-specific flags ---
    if args.model_type == "mlp" and any([
        args.pce_max_degree != 6,
        args.pce_norm_q     != 0.75,
        args.pce_norm_threshold != 1.0,
    ]):
        print(
            "[WARN] --pce_* flags were set but --model_type is 'mlp'.  "
            "These flags will be ignored."
        )

    if args.w0_min is not None or args.w0wa_max is not None:
        if args.prior_type != "expanded":
            raise ValueError(
                "--w0_min and --w0wa_max are only meaningful with "
                "--prior_type expanded."
            )

    # --- Print run configuration ---
    print("=" * 60)
    print("[INFO] MPS Emulator Training Configuration")
    print("=" * 60)
    for key, val in vars(args).items():
        print(f"  {key:<25s} = {val}")
    print("=" * 60)

    # --- Directory setup ---
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.fig_dir,   exist_ok=True)

    # --- Load training data ---
    start = time.perf_counter()
    print("\n[INFO] Loading training set...")

    train_set = utils.COLASet(
        target_z   = utils.z_mps,
        cosmo_type = args.cosmo_type,
        prior_type = args.prior_type,
        nl_type    = args.nl_type,
        start_batch= args.start_batch,
        n_batches  = args.n_batches,
    )

    if args.w0_min is not None or args.w0wa_max is not None:
        print("\n[INFO] Applying prior cuts to training set...")
        _apply_prior_cuts(train_set, args.w0_min, args.w0wa_max)

    # --- Prepare PCA ---
    train_set.prepare(num_pcs=args.num_pcs, num_pcs_z=args.num_pcs_z)

    elapsed = time.perf_counter() - start
    print(f"[INFO] Data loaded and PCA prepared in {elapsed / 60:.2f} minutes.")
    print(f"[INFO] Training redshifts: {train_set.z}")

    # --- Build model ---
    print(f"\n[INFO] Building {args.model_type.upper()} model...")

    if args.model_type == "mlp":
        model_obj = utils.COLA_NN_Keras(
            train_set,
            num_layers  = args.num_layers,
            num_neurons = args.num_neurons,
        )
    elif args.model_type == "npce":
        model_obj = COLA_NPCE_Keras(
            train_set,
            max_degree     = args.pce_max_degree,
            norm_q         = args.pce_norm_q,
            norm_threshold = args.pce_norm_threshold,
            num_layers     = args.num_layers,
            num_neurons    = args.num_neurons,
        )
    else:
        raise ValueError(f"Unknown model_type '{args.model_type}'.")

    # --- Train ---
    print("[INFO] Starting training...")
    model_obj.fit_t_componets(
        train_set,
        num_epochs  = args.num_epochs,
        batch_size  = args.batch_size,
        initial_lr  = args.initial_lr,
        final_lr    = args.final_lr,
        huber_delta = args.huber_delta,
        decayevery  = args.decay_every,
        decayrate   = args.decay_rate,
    )

    # --- Save model ---
    keras_model = model_obj.models["t-component"]
    model_tag   = train_set._metadata_tag()
    model_name  = f"emulator_{args.model_type}_{model_tag}.keras"
    model_path  = os.path.join(args.model_dir, model_name)
    keras_model.save(model_path)

    # Save PCE indices alongside the model so they can be reconstructed later
    if args.model_type == "npce":
        model_obj.save_pce_metadata(args.model_dir)

    total_elapsed = time.perf_counter() - start
    print(f"\n[INFO] Model trained and saved to: {model_path}")
    print(f"[INFO] Total runtime: {total_elapsed / 60:.2f} minutes.")


if __name__ == "__main__":
    main()