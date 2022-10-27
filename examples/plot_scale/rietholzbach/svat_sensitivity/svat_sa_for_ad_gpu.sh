#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:default
#PBS -l walltime=4:00:00
#PBS -l pmem=8000mb
#PBS -N svat_sa_for_ad_gpu
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

# load module dependencies
module purge
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_sensitivity

# adapt command to your available scheduler / MPI implementation
python svat.py -b jax -d gpu -td "${TMPDIR}" -tms advection-dispersion

# Write output to temporary SSD of computing node
echo "Write output to $TMPDIR"
# Move output from temporary SSD to workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_sensitivity"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_sensitivity
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_sensitivity
