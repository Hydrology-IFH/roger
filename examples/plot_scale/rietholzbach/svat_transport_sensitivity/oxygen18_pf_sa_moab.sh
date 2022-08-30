#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -l pmem=12000mb
#PBS -N oxygen18_pf_sa
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_sensitivity

# adapt command to your available scheduler / MPI implementation
python svat_transport.py -b numpy -d cpu -ns 32 -tms preferential -td /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport_sensitivity
