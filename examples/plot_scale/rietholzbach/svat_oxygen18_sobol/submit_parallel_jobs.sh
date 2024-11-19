#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach_new/svat_oxygen18_sensitivity

sbatch --partition single svat18O_pi_sa_parallel_slurm.sh
sbatch --partition single svat18O_cm_sa_parallel_slurm.sh
sbatch --partition single svat18O_adp_sa_parallel_slurm.sh
sbatch --partition single svat18O_adpt_sa_parallel_slurm.sh