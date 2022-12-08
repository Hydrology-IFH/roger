#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=6:00:00
#PBS -l pmem=4000mb
#PBS -N bromide_deterministic_svat_adt_2006
 
# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_bromide_benchmark
 
python svat_transport.py -b jax -d cpu -tms time-variant_advection-dispersion -td "${TMPDIR}" -ss deterministic -y 2006
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_bromide_benchmark"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_bromide_benchmark
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_bromide_benchmark
