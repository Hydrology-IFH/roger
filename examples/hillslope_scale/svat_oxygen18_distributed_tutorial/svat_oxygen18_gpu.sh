#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:default
#PBS -l walltime=24:00:00
#PBS -l pmem=4000mb
#PBS -N svat_oxygen18_gpu
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18

python svat_transport.py -b jax -d gpu -tms advection-dispersion  -td "${TMPDIR}"

# Write output to temporary SSD of computing node
echo "Write output to $TMPDIR"
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/svat_oxygen18_distributed_tutorial"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/svat_oxygen18_distributed_tutorial
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/svat_oxygen18_distributed_tutorial
