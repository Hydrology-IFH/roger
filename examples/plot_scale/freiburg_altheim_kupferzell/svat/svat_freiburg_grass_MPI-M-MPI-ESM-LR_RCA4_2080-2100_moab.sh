#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=4:00:00
#PBS -l pmem=80000mb
#PBS -N svat_freiburg_grass_MPI-M-MPI-ESM-LR_RCA4_2080-2100
#PBS -m a
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat
 
python svat.py -b numpy -d cpu --location freiburg --land-cover-scenario grass --climate-scenario MPI-M-MPI-ESM-LR_RCA4 --period 2080-2100 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat
mv "${TMPDIR}"/SVAT_*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat
