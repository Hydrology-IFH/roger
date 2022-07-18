#!/bin/bash
#
#PBS -q short
#PBS -N oxygen18_ad_sa
#PBS -l nodes=2:ppn=16
#PBS -l walltime=30:00:00
#PBS -l pmem=128000mb
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
 
# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b jax -d cpu -n 32 1 -tms advection-dispersion
