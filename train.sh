#!/bin/bash
# =============================================================================
# submit_train.sh — MPS Emulator Training Job Submission
#
# Usage:
#   sbatch train.sh --cosmo_type w0wacdm --prior_type expanded --nl_type lin
#   sbatch rain.sh --cosmo_type lcdm --prior_type constrained --nl_type nl --n_batches 40
#
# All flags after the sbatch options are forwarded verbatim to train.py.
# Run `python ./mps_emu/train.py --help` to see all available flags.
# =============================================================================

#SBATCH --job-name=mps_train
#SBATCH --output=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/out/train_%x_%j.txt
#SBATCH --time=48:00:00
#SBATCH --partition=hbm-1tb-long-96core
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
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
source /gpfs/projects/MirandaGroup/victoria/miniconda/etc/profile.d/conda.sh
conda activate vic

# Pass available CPUs to TensorFlow/NumPy threading backends.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ---------------------------------------------------------------------------
# Run training — all script arguments are forwarded to train.py
# ---------------------------------------------------------------------------

srun python ./mps_emu/train.py "$@"

echo "============================================================"
echo " Job finished: $(date)"
echo "============================================================"