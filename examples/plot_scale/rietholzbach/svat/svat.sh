#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=2:00:00
#PBS -l pmem=1000mb
#PBS -N svat
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat

# adapt command to your available scheduler / MPI implementation
python svat.py -b numpy -d cpu
