#!/bin/bash
for p in 80 60 20 1; do
    mkdir -p /mnt/gpfs2_4m/scratch/ris237/perfusion_coupling/marg2/p$p/sim_output
    sbatch job_perf_coup_marg2_p$p.sh
done
