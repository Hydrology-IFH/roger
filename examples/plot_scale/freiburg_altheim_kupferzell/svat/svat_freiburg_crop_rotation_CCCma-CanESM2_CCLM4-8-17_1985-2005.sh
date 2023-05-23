#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -l pmem=80000mb
#PBS -N svat_freiburg_crop_rotation_CCCma-CanESM2_CCLM4-8-17_1985-2005
#PBS -m a
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat
 
python svat_crop.py -b numpy -d cpu --location freiburg --land-cover-scenario crop_rotation --climate-scenario CCCma-CanESM2_CCLM4-8-17 --period 1985-2005 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat
mv "${TMPDIR}"/SVAT_*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat
