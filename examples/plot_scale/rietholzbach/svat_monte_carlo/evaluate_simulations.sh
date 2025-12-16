#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -l pmem=16000mb
#PBS -N pp_mc
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_monte_carlo

# adapt command to your available scheduler / MPI implementation
python evaluate_simulations.py -td /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_monte_carlo
