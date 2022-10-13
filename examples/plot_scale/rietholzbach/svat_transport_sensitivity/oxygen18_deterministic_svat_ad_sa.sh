#!/bin/bash
#PBS -l nodes=2:ppn=16
#PBS -l walltime=120:00:00
#PBS -l pmem=4000mb
#PBS -N oxygen18_deterministic_svat_ad_sa
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_sensitivity
 
# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes -b jax -d cpu -n 32 1 -ns 1024 -tms advection-dispersion -td /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport_sensitivity -ss deterministic
