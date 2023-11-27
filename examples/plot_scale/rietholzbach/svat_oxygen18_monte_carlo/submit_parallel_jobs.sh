#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach_new/svat_oxygen18_monte_carlo

sbatch --partition single svat18O_adp_parallel_slurm.sh
sbatch --partition single svat18O_adpt_parallel_slurm.sh

