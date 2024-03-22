#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_nitrate_lahr_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert
#SBATCH --output=svat_crop_nitrate_lahr_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert.out
#SBATCH --error=svat_crop_nitrate_lahr_summer-wheat_winter-wheat_winter-rape_yellow-mustard_high_Nfert_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/svat_crop_nitrate
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/boadkh/output/svat_crop/SVATCROP_lahr_summer-wheat_winter-wheat_winter-rape_yellow-mustard.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/boadkh/output/svat_crop/SVATCROP_lahr_summer-wheat_winter-wheat_winter-rape_yellow-mustard.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
sleep 10
checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVATCROP_lahr_summer-wheat_winter-wheat_winter-rape_yellow-mustard.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
python svat_crop_nitrate.py -b jax -d cpu --float-type float64 --location lahr --crop-rotation-scenario summer-wheat_winter-wheat_winter-rape_yellow-mustard --fertilization-intensity high -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/boadkh/output/svat_crop_nitrate"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/boadkh/output/svat_crop_nitrate
mv "${TMPDIR}"/SVATCROPNITRATE_*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/boadkh/output/svat_crop_nitrate
