#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -l pmem=80000mb
#PBS -N svat_freiburg_corn_catch_crop_observed_2016-2021
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell
 
python svat_crop.py -b numpy -d cpu --land-cover-scenario corn_catch_crop --climate-scenario observed --period 2016-2021 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat
