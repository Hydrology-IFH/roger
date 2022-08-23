#!/bin/bash
#PBS -l nodes=2:ppn=16
#PBS -l walltime=30:00:00
#PBS -l pmem=4000mb
#PBS -N nitrate2_lys2_cm_sa
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_transport_sensitivity
 
# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_transport_nitrate2.py -b numpy -d cpu -n 32 1 -lys lys2 -tms complete-mixing -td "${TMPDIR}"
# Write output to temporary SSD of computing node
echo "Write output to $TMPDIR"
# Move output from temporary SSD to workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_transport_sensitivity"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_transport_sensitivity
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_transport_sensitivity
