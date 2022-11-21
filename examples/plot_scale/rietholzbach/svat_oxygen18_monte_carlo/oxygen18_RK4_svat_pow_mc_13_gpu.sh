#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:default
#PBS -l walltime=12:00:00
#PBS -l pmem=4000mb
#PBS -N oxygen18_RK4_svat_pow_mc_13
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo
 
python svat_transport.py --id 13 -b jax -d gpu -ns 500 -tms power -td "${TMPDIR}" -ss RK4
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18_monte_carlo"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18_monte_carlo
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18_monte_carlo
