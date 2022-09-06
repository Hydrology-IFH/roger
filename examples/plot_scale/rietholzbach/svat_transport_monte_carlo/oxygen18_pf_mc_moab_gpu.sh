#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:default
#PBS -l walltime=1:00:00
#PBS -l pmem=12000mb
#PBS -N oxygen18_pf_mc
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
export OMP_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo

python svat_transport.py -b jax -d gpu -ns 200 -tms preferential -td /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo
