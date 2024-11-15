#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -l pmem=8000mb
#PBS -N svat_crop_lys8_bromide_sa_ref
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_crop_sobol
 
# adapt command to your available scheduler / MPI implementation
python svat_crop.py -b numpy -d cpu -lys lys8_bromide -td "${TMPDIR}"
# Write output to temporary SSD of computing node
echo "Write output to $TMPDIR"
# Move output from temporary SSD to workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_sobol"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_sobol
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_sobol
