#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=96:00:00
#PBS -l pmem=4000mb
#PBS -N oxygen18_deterministic_svat_pfadt
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18
 
python svat_transport.py -b jax -d cpu -tms time-variant_preferential_+_advection-dispersion -td "${TMPDIR}" -ss deterministic
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18
