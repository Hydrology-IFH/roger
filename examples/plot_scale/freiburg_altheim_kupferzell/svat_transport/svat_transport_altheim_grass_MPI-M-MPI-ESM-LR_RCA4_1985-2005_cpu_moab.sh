#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=48:00:00
#PBS -l pmem=8000mb
#PBS -N svat_transport_altheim_grass_MPI-M-MPI-ESM-LR_RCA4_1985-2005
#PBS -m a
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat/SVAT_altheim_grass_MPI-M-MPI-ESM-LR_RCA4_1985-2005.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat/SVAT_altheim_grass_MPI-M-MPI-ESM-LR_RCA4_1985-2005.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT_altheim_grass_MPI-M-MPI-ESM-LR_RCA4_1985-2005.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
python svat_transport.py -b jax -d cpu --location altheim --land-cover-scenario grass --climate-scenario MPI-M-MPI-ESM-LR_RCA4 --period 1985-2005 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport
