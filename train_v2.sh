#!/bin/bash
# =============================================================================
# submit_train.sh — MPS Emulator Single-Model Training Job Submission
#
# Usage:
#   sbatch submit_train.sh --cosmo_type w0wacdm --prior_type expanded --nl_type lin
#   sbatch submit_train.sh --cosmo_type w0wacdm --prior_type expanded --nl_type halofit \
#       --model_type npce --n_batches 50 --w0_min -2.0 --w0wa_max -4.0
#
# All flags after the sbatch options are forwarded verbatim to train_v2.py.
# Run `python ./mps_emu/train_v2.py --help` to see all available flags.
# =============================================================================

#SBATCH --job-name=mps_train
#SBATCH --output=/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/out/train_%x_%j.txt
#SBATCH --time=48:00:00 #48:00:00
#SBATCH --partition=h200x4-long #s-long #h200x4-long        # ← GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1              # ← request 1 GPU
#SBATCH --cpus-per-task=16             # ← enough for data loading workers
#SBATCH --mem=250G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

echo "============================================================"
echo " Job name       : $SLURM_JOB_NAME"
echo " Job ID         : $SLURM_JOBID"
echo " Running on     : $(hostname)"
echo " Start time     : $(date)"
echo " Working dir    : $(pwd)"
echo " CPUs allocated : $SLURM_JOB_CPUS_PER_NODE"
echo " Tasks          : $SLURM_NTASKS"
echo " CPUs per task  : $SLURM_CPUS_PER_TASK"
echo " Train args     : $@"
echo "============================================================"

module purge > /dev/null 2>&1
module load slurm

source ~/.bashrc
source /lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/miniforge/etc/profile.d/conda.sh
conda activate vic_gpu

echo "[INFO] Active conda env: $CONDA_DEFAULT_ENV"
echo "[INFO] Python path: $(which python)"
python -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print('TF:', tf.__version__, 'GPUs:', gpus)
assert len(gpus) > 0, 'No GPU detected!'
" || { echo "[FATAL] vic_gpu env not active, GPU not detected, or env broken"; exit 1; }

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=1
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE="/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa"
TRAIN_SCRIPT="${BASE}/mps_emu/train_v2.py"

# ---------------------------------------------------------------------------
# Run training — all script arguments are forwarded to train_v2.py
# ---------------------------------------------------------------------------

cd "${BASE}" || { echo "[FATAL] Could not cd to ${BASE}"; exit 1; }

python "${TRAIN_SCRIPT}" "$@"
TRAIN_EXIT=$?

if [ ${TRAIN_EXIT} -eq 0 ]; then
    echo ""
    echo "[INFO] Training completed successfully."
else
    echo ""
    echo "[ERROR] Training script exited with code ${TRAIN_EXIT}."
fi

echo "============================================================"
echo " Job finished: $(date)"
echo "============================================================"

exit ${TRAIN_EXIT}