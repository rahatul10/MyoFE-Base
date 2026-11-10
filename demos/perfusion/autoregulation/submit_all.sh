#!/bin/bash
# Make the output folders (SLURM cannot create them), then submit every case.
for n in 0 50 60 65 70 72 75 80 90; do
    mkdir -p /mnt/gpfs2_4m/scratch/ris237/perfusion/autoregulation_LAD/ds$n/sim_output
    sbatch job_perf_auto_lad_ds$n.sh
done
