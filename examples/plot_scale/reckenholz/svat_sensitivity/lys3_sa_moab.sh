#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=40:00:00
#PBS -l pmem=8000mb
#PBS -N lys3_sa
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_sensitivity
 
# adapt command to your available scheduler / MPI implementation
python svat_crop.py -b numpy -d cpu -lys lys3 -td "${TMPDIR}"
# Write output to temporary SSD of computing node
echo "Write output to $TMPDIR"
# Move output from temporary SSD to workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_sensitivity"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_sensitivity
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_sensitivity
