#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=96:00:00
#PBS -l pmem=4000mb
#PBS -N bromide_deterministic_svat_cm_1998
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport
 
python svat_transport.py -b jax -d cpu -tms complete-mixing -td "${TMPDIR}" -ss deterministic, -y 1998
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_transport
