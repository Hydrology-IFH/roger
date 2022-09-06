#!/bin/bash
#PBS -l nodes=1:ppn=2:gpus=2:default
#PBS -l walltime=48:00:00
#PBS -l pmem=12000mb
#PBS -N oxygen18_pft_mc
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

mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b jax -d gpu -n 2 1 -ns 200 -tms time-variant_preferential -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport_monte_carlo"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport_monte_carlo
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport_monte_carlo
