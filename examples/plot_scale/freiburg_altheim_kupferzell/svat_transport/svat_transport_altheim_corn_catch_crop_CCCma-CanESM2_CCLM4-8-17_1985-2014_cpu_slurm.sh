#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_transport_altheim_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2014
#SBATCH --output=svat_transport_altheim_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2014.out
#SBATCH --error=svat_transport_altheim_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2014_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_altheim_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2014.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_altheim_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2014.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT_altheim_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_1985-2014.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
python svat_crop_transport.py -b numpy -d cpu --location altheim --land-cover-scenario corn_catch_crop --climate-scenario CCCma-CanESM2_CCLM4-8-17 --period 1985-2014 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
