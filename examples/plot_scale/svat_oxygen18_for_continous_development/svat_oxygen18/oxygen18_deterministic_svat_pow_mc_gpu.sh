#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:default
#PBS -l walltime=6:00:00
#PBS -l pmem=4000mb
#PBS -N oxygen18_deterministic_svat_pow_mc
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo

python svat_oxygen18.py -b jax -d gpu -tms power -td "${TMPDIR}" -ss deterministic
# Move output from local SSD to workspace
echo "Move output from $TMPDIR to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/svat_oxygen18_for_continous_development/svat_oxygen18"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/svat_oxygen18_for_continous_development/svat_oxygen18
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/svat_oxygen18_for_continous_development/svat_oxygen18
