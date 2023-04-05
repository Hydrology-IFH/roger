#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:default
#PBS -l walltime=6:00:00
#PBS -l pmem=4000mb
#PBS -N svat_transport_altheim_corn_observed_2016-2021
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell
 
python svat_crop_transport.py -b jax -d gpu --land-cover-scenario corn --climate-scenario MPI-M-MPI-ESM-LR_RCA4 --period 2080-2100 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport
