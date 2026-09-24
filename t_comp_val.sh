#!/bin/bash
#SBATCH --job-name=mps_val
#SBATCH --output=/lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/cocoa/Cocoa/mps_emu/out/tcomp_validation_%x_%a_%A.txt
#SBATCH --time=2:00:00
#SBATCH --partition=debug-h200x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=256G # 100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu



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


# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load slurm

source ~/.bashrc
source /lustre/nvwulf/projects/MirandaGroup-nvwulf/victoria/miniforge/etc/profile.d/conda.sh
conda activate vic_gpu
# source start_cocoa.sh

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTRAOP_THREADS=${SLURM_CPUS_PER_TASK}
export TF_NUM_INTEROP_THREADS=1
export TF_GPU_THREAD_MODE=gpu_private
export TF_GPU_THREAD_COUNT=2

python ./mps_emu/t_comp_val_nl_tagn.py