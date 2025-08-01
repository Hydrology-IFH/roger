#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_ellwangen-rindelbach_silage-corn_yellow-mustard_no-irrigation
#SBATCH --output=svat_crop_ellwangen-rindelbach_silage-corn_yellow-mustard_no-irrigation.out
#SBATCH --error=svat_crop_ellwangen-rindelbach_silage-corn_yellow-mustard_no-irrigation_err.out
#SBATCH --export=ALL
 
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/bw_cropland//no-irrigation
 
python svat_crop.py -b numpy -d cpu --location ellwangen-rindelbach --crop-rotation-scenario silage-corn_yellow-mustard -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/no-irrigation"
mkdir -p /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/no-irrigation
mv "${TMPDIR}"/SVATCROP_*.nc /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/no-irrigation
