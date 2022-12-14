#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:default
#PBS -l walltime=12:00:00
#PBS -l pmem=4000mb
#PBS -N oxygen18_deterministic_svat_adpt_sa_1000_2000
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity
 
python svat_transport.py --x1 1000 --x2 2000 -b jax -d gpu -tms time-variant_advection-dispersion-power -td "${TMPDIR}" -ss deterministic
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18_sensitivity"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18_sensitivity
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/rietholzbach/svat_oxygen18_sensitivity
