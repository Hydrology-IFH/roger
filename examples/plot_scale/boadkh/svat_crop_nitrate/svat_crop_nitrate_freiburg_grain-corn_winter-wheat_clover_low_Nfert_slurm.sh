#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_nitrate_freiburg_grain-corn_winter-wheat_clover_low_Nfert
#SBATCH --output=svat_crop_nitrate_freiburg_grain-corn_winter-wheat_clover_low_Nfert.out
#SBATCH --error=svat_crop_nitrate_freiburg_grain-corn_winter-wheat_clover_low_Nfert_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/svat_crop_nitrate
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/output/svat_crop/SVATCROP_freiburg_grain-corn_winter-wheat_clover.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/output/svat_crop/SVATCROP_freiburg_grain-corn_winter-wheat_clover.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
sleep 10
checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVATCROP_freiburg_grain-corn_winter-wheat_clover.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
python svat_crop_nitrate.py -b jax -d cpu --float-type float64 --location freiburg --crop-rotation-scenario grain-corn_winter-wheat_clover --fertilization-intensity low -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/output/svat_crop_nitrate"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/output/svat_crop_nitrate
mv "${TMPDIR}"/SVATCROPNITRATE_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/output/svat_crop_nitrate
