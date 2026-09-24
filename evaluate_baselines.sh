#!/bin/bash
# =============================================================================
# train_all_parallel.sh — MPS Emulator N-train Scaling: All Jobs in One Node
#
# Runs 8 training processes (4 n_batches x 2 cosmo_types) in parallel on a
# single hbm-1tb-long-96core node, pinning each process to its own CPU slice
# to avoid contention. Once all training finishes, runs the evaluation script
# automatically.
#
# Usage:
#   sbatch ./mps_emu/train_all_parallel.sh
# =============================================================================

#SBATCH --job-name=mps_ntrain_scaling
#SBATCH --output=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/out/ntrain_scaling_%j.txt
#SBATCH --time=1:00:00
#SBATCH --partition=short-40core-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

echo "============================================================"
echo " Job name        : $SLURM_JOB_NAME"
echo " Job ID          : $SLURM_JOBID"
echo " Running on      : $(hostname)"
echo " Start time      : $(date)"
echo " Total CPUs      : $SLURM_JOB_CPUS_PER_NODE"
echo " Tasks           : $SLURM_NTASKS"
echo " CPUs per task   : $SLURM_CPUS_PER_TASK"
echo "============================================================"

module purge > /dev/null 2>&1
module load slurm

source ~/.bashrc
source /gpfs/projects/MirandaGroup/victoria/miniconda/etc/profile.d/conda.sh
conda activate vic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE="/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa"
TRAIN_SCRIPT="${BASE}/mps_emu/train_new.py"
EVAL_SCRIPT="${BASE}/mps_emu/eval_ntrain_scaling.py"
LOG_DIR="${BASE}/mps_emu/out/parallel_logs"
mkdir -p "${LOG_DIR}"

PRIOR_TYPE="constrained"
NL_TYPE="halofit"
N_BATCHES_LIST=(5 10 15 20 30 40 50)
COSMO_TYPES=("w0wacdm" "lcdm")

# Threads per training process: floor(96 / 8) = 12
THREADS_PER_JOB=${SLURM_CPUS_PER_TASK}

echo ""
echo "[INFO] Configuration:"
echo "  PRIOR_TYPE      = ${PRIOR_TYPE}"
echo "  NL_TYPE         = ${NL_TYPE}"
echo "  N_BATCHES_LIST  = ${N_BATCHES_LIST[*]}"
echo "  COSMO_TYPES     = ${COSMO_TYPES[*]}"
echo "  THREADS_PER_JOB = ${THREADS_PER_JOB}"
echo ""


echo "============================================================"
echo "[INFO] All training jobs succeeded. Starting evaluation..."
echo "       Start time: $(date)"
echo "============================================================"
echo ""

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python "${EVAL_SCRIPT}" \
    --prior_type "${PRIOR_TYPE}" \
    --nl_type    "${NL_TYPE}"

EVAL_EXIT=$?

if [ ${EVAL_EXIT} -eq 0 ]; then
    echo ""
    echo "[INFO] Evaluation completed successfully."
else
    echo ""
    echo "[ERROR] Evaluation script exited with code ${EVAL_EXIT}."
    exit ${EVAL_EXIT}
fi

echo ""
echo "============================================================"
echo " All done. Finished: $(date)"
echo "============================================================"