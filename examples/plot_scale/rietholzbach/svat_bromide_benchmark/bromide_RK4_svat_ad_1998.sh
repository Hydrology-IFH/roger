#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=8:00:00
#PBS -l pmem=4000mb
#PBS -N bromide_RK4_svat_ad_1998
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_bromide_benchmark
 
python svat_transport.py -b jax -d cpu -tms advection-dispersion -td "${TMPDIR}" -ss RK4 -y 1998
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_bromide_benchmark"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_bromide_benchmark
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_bromide_benchmark
