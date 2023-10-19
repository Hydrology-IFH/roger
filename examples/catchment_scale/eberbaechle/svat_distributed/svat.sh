#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=4:00:00
#PBS -l pmem=4000mb
#PBS -N svat
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de

eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/ruetlibach/svat_distributed

# adapt command to your available scheduler / MPI implementation
python svat.py -b numpy -d cpu -td "${TMPDIR}"

# Write output to temporary SSD of computing node
echo "Write output to $TMPDIR"
# Move output from temporary SSD to workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/svat_distributed_tutorial"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/svat_distributed_tutorial
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/svat_distributed_tutorial
