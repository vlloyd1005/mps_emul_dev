#!/bin/bash
#SBATCH --job-name=mps_eval
#SBATCH --output=/gpfs/projects/MirandaGroup/victoria/cocoa/Cocoa/mps_emu/out/evaluate_%x_%a_%A.txt
#SBATCH --time=4:00:00
#SBATCH -p hbm-1tb-long-96core
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=9
#SBATCH --mail-type=ALL
#SBATCH --mail-user=victoria.lloyd@stonybrook.edu


echo "Available CPUs:  $SLURM_JOB_CPUS_PER_NODE"
# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load slurm

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID
echo Number of task is $SLURM_NTASKS
echo Number of cpus per task is $SLURM_CPUS_PER_TASK

source ~/.bashrc
source /gpfs/projects/MirandaGroup/victoria/miniconda/etc/profile.d/conda.sh
conda init bash
conda deactivate
conda activate cocoa

mpirun -n ${SLURM_NTASKS} --oversubscribe --mca btl vader,tcp,self --bind-to core:overload-allowed --map-by numa:pe=${OMP_NUM_THREADS} python ./mps_emu/recover_weights.py
