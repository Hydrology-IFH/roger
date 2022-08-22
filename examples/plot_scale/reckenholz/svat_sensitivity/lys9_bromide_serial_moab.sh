#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=40:00:00
#PBS -l pmem=16000mb
#PBS -N svat_sa_lys9_br
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_sensitivity

# adapt command to your available scheduler / MPI implementation
python svat_crop.py -lys lys9_bromide
