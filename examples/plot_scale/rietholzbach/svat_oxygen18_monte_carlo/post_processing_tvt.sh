#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -l pmem=16000mb
#PBS -N pp_d18O_mc_tvt
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo
python post_processing.py -td /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18_monte_carlo -tms time-variant-transp
