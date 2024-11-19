#!/bin/bash
#PBS -l nodes=1:ppn=20
#PBS -l walltime=48:00:00
#PBS -l pmem=6000mb
#PBS -N svat_crop_nitrate_adp_lys8_sa
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
 
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_crop_nitrate_sobol
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/reckenholz/svat_crop_nitrate_sobol/SVATCROP_lys8.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/reckenholz/svat_crop_nitrate_sobol/SVATCROP_lys8.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
sleep 10
checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVATCROP_lys8.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_crop_nitrate.py -b jax -d cpu -n 20 1 -lys lys8 -td "${TMPDIR}"
# Write output to temporary SSD of computing node
echo "Write output to $TMPDIR"
# Move output from temporary SSD to workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_nitrate_sobol"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_nitrate_sobol
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_nitrate_sobol
