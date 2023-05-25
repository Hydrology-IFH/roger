#!/bin/bash
#PBS -l nodes=1:ppn=4
#PBS -l walltime=14:00:00
#PBS -l pmem=8000mb
#PBS -N svat_transport_freiburg_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2005
#PBS -m a
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat/SVAT_freiburg_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2005.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat/SVAT_freiburg_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2005.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT_freiburg_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2005.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
mpirun --bind-to core --map-by core -report-bindings python svat_crop_transport.py -b jax -d cpu -n 4 1 --location freiburg --land-cover-scenario corn_catch_crop --climate-scenario CCCma-CanESM2_CCLM4-8-17 --period 1985-2005 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport
