#!/bin/bash
#PBS -l nodes=4:ppn=2:gpus=2:default
#PBS -l walltime=48:00:00
#PBS -l pmem=12000mb
#PBS -N oxygen18_ad_mc
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo

# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b jax -d gpu -ns 400 -n 8 1 -tms advection-dispersion -td /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport_monte_carlo
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport_monte_carlo
