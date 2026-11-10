#!/bin/bash
#SBATCH --time=13-00:00:00             # Time limit for the job (REQUIRED).
#SBATCH --job-name=auto_lad_ds0
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1      # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=3645M                 # memory required per node - amount of memory (in bytes)
#SBATCH --nodes=1              # all 32 tasks on ONE machine -- split runs crawl
#SBATCH --no-requeue           # a node failure ends the job instead of silently restarting it
#SBATCH --account=col_jfwe223_uksr
#SBATCH --partition=CAC48M192_L         # Partition/queue to run the job in. (REQUIRED)
#SBATCH --output=/mnt/gpfs2_4m/scratch/ris237/perfusion/autoregulation_LAD/ds0/sim_output/output.%J.out # STDOUT
#SBATCH --error=/mnt/gpfs2_4m/scratch/ris237/perfusion/autoregulation_LAD/ds0/sim_output/output.%J.err
#SBATCH --mail-user=ris237@uky.edu   # Where email is sent to (optional)
#SBATCH --mail-type=BEGIN,END,FAIL         # email when the job finishes or fails
# STDOUT
module load ccs/singularity

cd MyoFE-Base/python_codes
singularity exec --cleanenv /home/ris237/fenics.img  mpiexec -np $SLURM_NTASKS python MyoFE.py LV_sim /home/ris237/MyoFE-Base/demos/perfusion/autoregulation/perfusion_autoregulation_LAD_ds0.json
