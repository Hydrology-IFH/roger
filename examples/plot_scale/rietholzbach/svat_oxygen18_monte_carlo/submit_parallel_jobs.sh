#!/bin/bash

cd ~/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo

sbatch --partition single svat18O_adp_mc_parallel_slurm.sh
sbatch --partition single svat18O_adpt_mc_parallel_slurm.sh

