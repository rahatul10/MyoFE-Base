#!/bin/bash
#SBATCH --time=13-00:00:00
#SBATCH --job-name=coup_marg2_p60
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3645M
#SBATCH --nodes=1
#SBATCH --no-requeue
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAC48M192_L
#SBATCH --output=/mnt/gpfs2_4m/scratch/ris237/perfusion_coupling_MARG2_p60/sim_output/output.%J.out
#SBATCH --error=/mnt/gpfs2_4m/scratch/ris237/perfusion_coupling_MARG2_p60/sim_output/output.%J.err
#SBATCH --mail-user=ris237@uky.edu
#SBATCH --mail-type=BEGIN,END,FAIL
module load ccs/singularity

cd MyoFE-Base/python_codes
singularity exec --cleanenv /home/ris237/fenics.img  mpiexec -np $SLURM_NTASKS python MyoFE.py LV_sim "/home/ris237/MyoFE-Base/demos/perfusion/perfusion_coupling/MARG/MARG2/perfusion_coupling_MARG2_60%.json"
