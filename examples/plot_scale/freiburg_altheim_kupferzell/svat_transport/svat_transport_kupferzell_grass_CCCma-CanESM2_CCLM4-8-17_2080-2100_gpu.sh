#!/bin/bash
#PBS -l nodes=1:ppn=1:gpus=1:default
#PBS -l walltime=6:00:00
#PBS -l pmem=4000mb
#PBS -N svat_transport_kupferzell_grass_CCCma-CanESM2_CCLM4-8-17_2080-2100
#PBS -m a
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
module load lib/cudnn/8.2-cuda-11.4
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport
 
# Move fluxes and states from global workspace to local SSD
mv /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport/freiburg_altheim_kupferzell/output/svat/SVAT_kupferzell_crop_rotation_CCCma-CanESM2_CCLM4-8-17_2080-2100.nc "${TMPDIR}"/SVAT_kupferzell_crop_rotation_CCCma-CanESM2_CCLM4-8-17_2080-2100.nc
# Wait for files
checksum_gws=$(shasum -a 256 /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport/freiburg_altheim_kupferzell/output/svat/SVAT_kupferzell_crop_rotation_CCCma-CanESM2_CCLM4-8-17_2080-2100.nc | cut -f 1 -d " ")
checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT_kupferzell_crop_rotation_CCCma-CanESM2_CCLM4-8-17_2080-2100.nc | cut -f 1 -d " ")
while [ ${checksum_gws} != ${checksum_ssd} ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT_kupferzell_crop_rotation_CCCma-CanESM2_CCLM4-8-17_2080-2100.nc | cut -f 1 -d " ")
done
 
python svat_crop_transport.py -b jax -d gpu --location kupferzell --land-cover-scenario grass --climate-scenario CCCma-CanESM2_CCLM4-8-17 --period 2080-2100 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/freiburg_altheim_kupferzell/svat_transport
