#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_singen_grain-corn_winter-wheat_winter-barley_crop-specific-irrigation
#SBATCH --output=svat_crop_singen_grain-corn_winter-wheat_winter-barley_crop-specific-irrigation.out
#SBATCH --error=svat_crop_singen_grain-corn_winter-wheat_winter-barley_crop-specific-irrigation_err.out
#SBATCH --export=ALL
 
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/bw_cropland//irrigation
 
python svat_crop.py -b numpy -d cpu --location singen --irrigation-scenario crop-specific --crop-rotation-scenario grain-corn_winter-wheat_winter-barley -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/irrigation/crop-specific"
mkdir -p /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/irrigation/crop-specific
mv "${TMPDIR}"/SVATCROP_*.nc /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/irrigation/crop-specific
