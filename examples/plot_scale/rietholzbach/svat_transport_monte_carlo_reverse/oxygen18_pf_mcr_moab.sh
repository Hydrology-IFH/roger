#!/bin/bash
#PBS -l nodes=5:ppn=10
#PBS -l walltime=30:00:00
#PBS -l pmem=12000mb
#PBS -N oxygen18_pf_mcr
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
 
# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b jax -d cpu -n 50 1 -tms preferential
