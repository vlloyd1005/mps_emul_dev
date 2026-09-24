#!/bin/bash
# =============================================================================
# train_z_block.sh — Train single-z emulators in blocks of consecutive z_mps
# grid indices.  Array task t trains indices
#     t*BLOCK_SIZE ... t*BLOCK_SIZE + BLOCK_SIZE - 1
# one after another on a single GPU.  With BLOCK_SIZE=10:
#     task 0 -> 0-9,  task 1 -> 10-19,  task 2 -> 20-29,
#     task 3 -> 30-39, task 4 -> 40-49, task 5 -> 50-51
#
# Usage (h200x8-long allows 2 jobs per user, so submit 2 tasks at a time):
#   sbatch --array=0,1 ./mps_emu/train_z_block.sh --model_type npce --cosmo_type w0wacdm ...
#   sbatch --array=2,3 ./mps_emu/train_z_block.sh <same flags>   # when those finish
#   sbatch --array=4,5 ./mps_emu/train_z_block.sh <same flags>
#
# All flags are forwarded to train_single_z.py with --target_z added per
# redshift.  Do NOT pass --target_z yourself.
#
# Resuming: each finished redshift leaves a marker
#   mps_emu/out/single_z_runs/<RUN_ID>/done_<idx>
# (RUN_ID = hash of the training flags).  Resubmitting the same task skips
# redshifts that already finished.  Per-redshift logs: .../<RUN_ID>/z<idx>.log
# =============================================================================

#SBATCH --job-name=train_zblk
#SBATCH --output=/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/out/train_%x_%A_%a.txt
#SBATCH --array=0,1
#SBATCH --time=24:00:00
#SBATCH --partition=h200x4-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu

BLOCK_SIZE=10
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

for arg in "$@"; do
    if [[ "${arg}" == --target_z* ]]; then
        echo "[FATAL] --target_z is set per redshift by this script; remove it."
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

module purge > /dev/null 2>&1
module load slurm

source ~/.bashrc
source /lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/miniforge/etc/profile.d/conda.sh
conda activate vic_gpu

BASE="/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa"
TRAIN_SCRIPT="${BASE}/mps_emu/train_single_z.py"
cd "${BASE}" || { echo "[FATAL] Could not cd to ${BASE}"; exit 1; }

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
# Grid indices / redshifts for this block
# (z_mps definition must match mps_emu_single_z.py)
# ---------------------------------------------------------------------------

Z_TABLE=$(python -c "
import sys
import numpy as np
z_mps = np.concatenate((
    np.linspace(0, 3, 33, endpoint=False),
    np.linspace(3, 10, 7, endpoint=False),
    np.linspace(10, 50, 12),
))
task, block = int(sys.argv[1]), int(sys.argv[2])
lo, hi = task * block, min((task + 1) * block, len(z_mps))
if lo >= len(z_mps):
    sys.exit(f'task {task} starts at index {lo}, beyond the z_mps grid (0..{len(z_mps)-1})')
for i in range(lo, hi):
    print(i, repr(float(z_mps[i])))
" "${TASK_ID}" "${BLOCK_SIZE}") || { echo "[FATAL] Could not resolve block for task ${TASK_ID}"; exit 1; }

RUN_ID=$(printf '%s ' "$@" | md5sum | cut -c1-10)
RUN_DIR="${BASE}/mps_emu/out/single_z_runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"
printf '%s ' "$@" > "${RUN_DIR}/args.txt"; echo >> "${RUN_DIR}/args.txt"

echo "============================================================"
echo " Job             : $SLURM_JOB_NAME  (array ${SLURM_ARRAY_JOB_ID}, task ${TASK_ID})"
echo " Running on      : $(hostname)"
echo " Start time      : $(date)"
echo " Block           : $(echo "${Z_TABLE}" | head -1 | cut -d' ' -f1)-$(echo "${Z_TABLE}" | tail -1 | cut -d' ' -f1)"
echo " Run dir         : ${RUN_DIR}"
echo " Train args      : $@"
echo "============================================================"

# ---------------------------------------------------------------------------
# Train each redshift in the block sequentially
# ---------------------------------------------------------------------------

N_FAIL=0
while read -r idx z; do
    marker="${RUN_DIR}/done_${idx}"
    log="${RUN_DIR}/z${idx}.log"

    if [[ -f "${marker}" ]]; then
        echo "[$(date +%T)] z[${idx}]=${z} already done — skipping."
        continue
    fi

    echo "[$(date +%T)] start z[${idx}]=${z}   (log: ${log})"
    t0=$(date +%s)
    python "${TRAIN_SCRIPT}" "$@" --target_z "${z}" > "${log}" 2>&1
    rc=$?
    dt=$(( ($(date +%s) - t0) / 60 ))

    if [[ ${rc} -eq 0 ]]; then
        touch "${marker}"
        echo "[$(date +%T)] done  z[${idx}]=${z}   (${dt} min)"
    else
        N_FAIL=$(( N_FAIL + 1 ))
        echo "[$(date +%T)] FAILED z[${idx}]=${z} (exit ${rc}, ${dt} min) — see ${log}"
        echo "${idx} ${z} exit=${rc} job=${SLURM_JOB_ID}" >> "${RUN_DIR}/failed.txt"
    fi
done <<< "${Z_TABLE}"

echo "============================================================"
echo " Task ${TASK_ID} finished: $(date)   failures: ${N_FAIL}"
echo "============================================================"

[[ ${N_FAIL} -eq 0 ]]
exit $?