#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_vellberg-kleinaltdorf_winter-wheat_silage-corn
#SBATCH --output=svat_crop_vellberg-kleinaltdorf_winter-wheat_silage-corn.out
#SBATCH --error=svat_crop_vellberg-kleinaltdorf_winter-wheat_silage-corn_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/svat_crop
 
python svat_crop.py -b numpy -d cpu --location vellberg-kleinaltdorf --crop-rotation-scenario winter-wheat_silage-corn -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/output/svat_crop"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/output/svat_crop
mv "${TMPDIR}"/SVATCROP_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/output/svat_crop
