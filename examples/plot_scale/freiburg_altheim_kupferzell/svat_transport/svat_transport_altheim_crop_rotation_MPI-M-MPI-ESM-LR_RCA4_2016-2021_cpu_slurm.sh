#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_transport_altheim_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2016-2021
#SBATCH --output=svat_transport_altheim_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2016-2021.out
#SBATCH --error=svat_transport_altheim_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2016-2021_err.txt
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_altheim_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2016-2021.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_altheim_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2016-2021.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT_altheim_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2016-2021.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
python svat_crop_transport.py -b jax -d cpu --location altheim --land-cover-scenario crop_rotation --climate-scenario MPI-M-MPI-ESM-LR_RCA4 --period 2016-2021 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
