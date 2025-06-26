#!/bin/bash
#SBATCH --time=13-00:00:00             # Time limit for the job (REQUIRED).
#SBATCH --job-name=HMR_8u
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c).
#SBATCH --mem=180G                  # memory required per node - amount of memory (in bytes)
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAC48M192_L  # Partition/queue to run the job in. (REQUIRED)
#SBATCH --output=/mnt/gpfs2_4m/scratch/sba431/HCM_paper/growthmr8/sim_output/output.%J.out # STDOUT
#SBATCH --error=/mnt/gpfs2_4m/scratch/sba431/HCM_paper/growthmr8/sim_output/output.%J.err
# STDOUT
cd MyoFE_H/python_codes
singularity exec --cleanenv /home/sba431/fenics.img  mpiexec -np $SLURM_NTASKS  python MyoFE.py LV_sim /home/sba431/MyoFE_H/demos/HCM_paper/simulations_baseline_final/growthmr_holzapfel/sim_inputs/input_parameters.json
