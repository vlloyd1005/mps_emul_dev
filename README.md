# MPS Emulator

A neural network emulator for the matter power spectrum P(k, z), supporting multiple cosmological models (ΛCDM and w₀wₐCDM) and nonlinear prescriptions (linear, HaloFit, HMcode-2020, and HMcode-2020 with baryonic feedback).

The emulator compresses P(k, z) into a compact representation using a two-stage PCA decomposition — a per-redshift spatial PCA followed by a temporal PCA across redshifts — and trains a multilayer perceptron to map cosmological parameters to the compressed coefficients. At inference, the prediction is multiplied by an Eisenstein-Hu analytical baseline to recover the full P(k, z).

---

## Table of Contents

- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Using a Pre-trained Model](#using-a-pre-trained-model)
- [Training New Models](#training-new-models)
  - [Training a Single Model](#training-a-single-model)
  - [Training Multiple Models in Parallel](#training-multiple-models-in-parallel)
- [Evaluating Models](#evaluating-models)
  - [Evaluating a Single Model](#evaluating-a-single-model)
  - [N-train Scaling Evaluation](#n-train-scaling-evaluation)
- [Generating New Training Data](#generating-new-training-data)
  - [Sampling Cosmological Parameters](#sampling-cosmological-parameters)
  - [Generating Power Spectra with CAMB](#generating-power-spectra-with-camb)
  - [Validating the PCA Decomposition](#validating-the-pca-decomposition)
- [File Reference](#file-reference)

---

## Repository Structure

```
mps_emu/
├── models/                        # Trained emulator weights
│   └── emulator_<tag>.h5          # e.g. emulator_w0wacdm_constrained_lin_nTrain20.h5
├── metadata/                      # PCA/scaler objects saved during training
│   └── metadata_<tag>/            # e.g. metadata_lcdm_constrained_lin_nTrain15/
│       ├── param_scaler_lowk_<N>_batches
│       ├── t_components_pca_lowk
│       ├── t_comp_scaler
│       └── Z<z>_lowk.{pca,frac_pks_scaler}  (one pair per redshift)
├── emulmps_w0wa.py                # Inference interface (start here)
├── train_utils_pk_emulator.py     # Data loading, PCA, model architecture
├── train.py                       # Train a single model
├── train.sh                       # SLURM wrapper for train.py
├── train_all_parallel.sh          # Train a grid of models on one node
├── evaluate_one.py                # Detailed error plots for one model
├── evaluate_one.sh                # SLURM wrapper
├── eval_ntrain_scaling.py         # Accuracy vs. N_train plots across models
├── evaluate_baselines.sh          # SLURM wrapper for eval_ntrain_scaling.py
├── t_comp_val.py                  # Validate PCA reconstruction before training
├── t_comp_val.sh                  # SLURM wrapper
├── lhs_generator.py               # Generate LHS cosmological parameter batches
├── datageneratorMPS.py            # Run CAMB on parameter batches (external)
└── vic_env.yml                    # Conda environment specification
```

---

## Environment Setup

A conda environment file is provided:

```bash
conda env create -f vic_env.yml
conda activate vic_env
```

Key dependencies are NumPy, SciPy, scikit-learn, TensorFlow/Keras, and the internal `symbolic_pofk` library (path configured inside `train_utils_pk_emulator.py`).

---

## Using a Pre-trained Model

The primary inference interface is `emulmps_w0wa.py`. The top-level function is `get_pks`, which returns k modes, redshifts, and P(k, z) for a single set of cosmological parameters.

Parameters are passed in **ML convention**: `[10⁹ Aₛ, nₛ, H₀, Ωb, Ωm, w₀, wₐ]`. For ΛCDM, pass `w₀ = -1.0` and `wₐ = 0.0`.

```python
from emulmps_w0wa import get_pks

# Full emulator — w0waCDM with HMcode-2020 + baryonic feedback
k, z, pk = get_pks(
    [2.1, 0.965, 68.0, 0.049, 0.31, -0.9, 0.1],
    cosmo_type='w0wacdm',
    prior_type='constrained',
    nl_type='mead2020_feedback',
)

# Symbolic EH approximation only (no neural network), for LCDM
k, z, pk = get_pks(
    [2.1, 0.965, 68.0, 0.049, 0.31, -1.0, 0.0],
    cosmo_type='lcdm',
    prior_type='constrained',
    nl_type='lin',
    use_approximation_only=True,
)
```

`get_pks` caches emulator instances internally, so repeated calls with the same configuration reuse the loaded model without re-reading files from disk.

**Available configurations** are determined by which model `.h5` and `metadata/` subdirectory exist. The valid nonlinear prescriptions are:

| `nl_type` | Description |
|---|---|
| `lin` | Linear P(k) |
| `halofit` | Non-linear via HaloFit |
| `mead2020` | Non-linear via HMcode-2020 |
| `mead2020_feedback` | HMcode-2020 + baryonic feedback (fixed log T_AGN = 7.8) |
| `mead2020_feedback_Tfree` | HMcode-2020 + baryonic feedback (free T_AGN) |

Note that `expanded` prior models only support `lin` and `halofit`; `constrained` prior models support all five.

---

## Training New Models

### Training a Single Model

`train.py` trains one emulator for a specified configuration. Required arguments are `--cosmo_type`, `--prior_type`, and `--nl_type`; all other settings have sensible defaults.

```bash
# Minimal — uses default n_batches=20, num_epochs=3000, etc.
python ./mps_emu/train.py \
    --cosmo_type w0wacdm \
    --prior_type constrained \
    --nl_type halofit

# Full control
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
```

Training will save the model weights to `models/` and the PCA/scaler metadata to `metadata/`. To submit as a SLURM job:

```bash
sbatch train.sh --cosmo_type lcdm --prior_type constrained --nl_type lin --n_batches 40
```

### Training Multiple Models in Parallel

`train_all_parallel.sh` runs a full grid of `(cosmo_type × n_batches)` training jobs in parallel on a single node, pinning each process to its own CPU slice to avoid contention. Once all training jobs complete, it automatically runs `eval_ntrain_scaling.py` to produce the accuracy-vs-N_train summary plots.

To train models for a different nonlinear prescription, edit the configuration block at the top of the script:

```bash
PRIOR_TYPE="constrained"
NL_TYPE="halofit"             # change this to any valid nl_type
N_BATCHES_LIST=(5 10 15 20 30 40 50)
COSMO_TYPES=("w0wacdm" "lcdm")
```

Then submit with:

```bash
sbatch ./mps_emu/train_all_parallel.sh
```

If you have already trained models and only want to re-run the evaluation across the N-train scaling grid (e.g. to regenerate plots), you can call `eval_ntrain_scaling.py` directly or via `evaluate_baselines.sh` — see the [N-train Scaling Evaluation](#n-train-scaling-evaluation) section below.

---

## Evaluating Models

### Evaluating a Single Model

`evaluate_one.py` runs a detailed error analysis for one specific trained model, comparing emulator predictions and the Eisenstein-Hu baseline against CAMB ground truth on the held-out test set. It produces:

1. Per-cosmology EH error lines for the w₀ > −2 subset
2. Two-panel comparison plots (full parameter space vs. w₀-filtered) for both the emulator and Syren baseline
3. 2×2 histograms of the maximum absolute fractional error per cosmology
4. An EH log-fraction diagnostic coloured by w₀, useful for diagnosing where the baseline struggles

The configuration (model, prior type, nl type, redshift, w₀ threshold) is set via constants at the top of the file. Edit those, then run:

```bash
python ./mps_emu/evaluate_one.py
# or
sbatch evaluate_one.sh
```

Figures are saved to the directory specified by `FIG_DIR` inside the script.

### N-train Scaling Evaluation

`eval_ntrain_scaling.py` evaluates accuracy across the full grid of trained models (varying `n_batches` and `cosmo_type`) and shows how emulator errors decrease as training set size grows. It produces:

1. A line plot of the 95th-percentile fractional error at k ≈ 1 h/Mpc vs. N_train, for each cosmo type
2. Per-model error band plots (emulator and Syren) across the full test set

```bash
python ./mps_emu/eval_ntrain_scaling.py \
    --prior_type constrained \
    --nl_type lin

# or via SLURM
sbatch evaluate_baselines.sh
```

This script is also called automatically at the end of `train_all_parallel.sh`.

---

## Generating New Training Data

If you want to train models on a new nonlinear prescription, extend the parameter space, or simply generate more data vectors, the pipeline has two stages: sampling cosmological parameters and running CAMB on them.

### Sampling Cosmological Parameters

`lhs_generator.py` generates batches of cosmological parameter vectors via Latin Hypercube Sampling. It saves each batch in two formats:

- **ML convention** `[10⁹ Aₛ, nₛ, H₀, Ωb, Ωm, w₀, wₐ (, T_AGN)]` — used by the emulator
- **Datagen convention** `[ln Aₛ, nₛ, H₀, Ωb h², Ωc h², w₀, wₐ (, T_AGN)]` — used by the CAMB data generator

The w₀/wₐ space is sampled by drawing w₀ and w₀+wₐ independently (not w₀ and wₐ separately), which avoids the triangular sampling artefact that would otherwise cut off large swathes of the prior.

To change the prior ranges, training batch indices, or number of samples per batch, edit the constants and `generate_batches` calls at the bottom of `lhs_generator.py`. This script is typically run on a separate server (amypond) rather than the cluster:

```bash
python ./mps_emu/lhs_generator.py
```

If you want to include T_AGN as a free parameter (for the `mead2020_feedback_Tfree` model), set `include_tagn=True` in the `generate_batches` call.

### Generating Power Spectra with CAMB

`datageneratorMPS.py` (written by Yijie) takes the datagen-convention parameter batches and runs CAMB to produce the corresponding P(k, z) arrays. Separate versions of this script exist for each nonlinear prescription:

| Prescription | Script |
|---|---|
| HaloFit | `datageneratorMPS.py` |
| HMcode-2020 | `datageneratorMPS_mead2020.py` |
| HMcode-2020 + fixed T_AGN | `datageneratorMPS_mead2020_feedback.py` |
| HMcode-2020 + free T_AGN | `datageneratorMPS_mead2020_feedback_Tfree.py` |

To generate data for a new prescription, update the CAMB settings and the input/output file paths at the top of the relevant script.

> **Note:** `datageneratorMPS.py` follows a different code style from the rest of the repository, as it was contributed externally.

### Validating the PCA Decomposition

After generating new data but **before training**, you should verify that the two-stage PCA decomposition (per-redshift spatial PCA → temporal PCA across redshifts) introduces only negligible reconstruction errors for your new dataset. This is the purpose of `t_comp_val.py`.

It loads a training set, fits the full PCA/tPCA pipeline, and produces spaghetti plots of fractional reconstruction errors P_reconstructed / P_true − 1 at z ≈ 0 and z ≈ 3, for both stages separately. If the errors are large (e.g. > 0.1%), you may need to increase `NUM_PCS` or `NUM_PCS_Z`.

Edit the configuration constants at the top of the file, then run:

```bash
python ./mps_emu/t_comp_val.py
# or
sbatch t_comp_val.sh
```

---

## File Reference

| File | Purpose |
|---|---|
| `emulmps_w0wa.py` | **Inference interface.** `get_pks()` is the entry point for all predictions. |
| `train_utils_pk_emulator.py` | Core utilities: data loading (`COLASet`), PCA preparation, neural network architecture (`COLA_NN_Keras`), and the `TComponentScaler` / `Scaler` classes. |
| `train.py` | CLI script to train a single emulator model. |
| `train.sh` | SLURM wrapper for `train.py`. |
| `train_all_parallel.sh` | Trains a full `(cosmo_type × n_batches)` grid in parallel on one node, then auto-evaluates. |
| `evaluate_one.py` | Detailed error diagnostics for a single trained model. |
| `evaluate_one.sh` | SLURM wrapper for `evaluate_one.py`. |
| `eval_ntrain_scaling.py` | Accuracy-vs-N_train plots across the full model grid. |
| `evaluate_baselines.sh` | SLURM wrapper for `eval_ntrain_scaling.py`. |
| `t_comp_val.py` | PCA/tPCA reconstruction validation — run this after generating new data. |
| `t_comp_val.sh` | SLURM wrapper for `t_comp_val.py`. |
| `lhs_generator.py` | Latin Hypercube Sampling for cosmological parameter batches. |
| `datageneratorMPS.py` | CAMB-based P(k, z) data generation (external, Yijie). |
| `models/` | Trained model weights (`.h5` files). |
| `metadata/` | PCA objects and scalers saved during `COLASet.prepare()`. |
| `vic_env.yml` | Conda environment specification. |
