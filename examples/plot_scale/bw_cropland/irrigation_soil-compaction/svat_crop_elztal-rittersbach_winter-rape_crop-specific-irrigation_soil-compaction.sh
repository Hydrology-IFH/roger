#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_elztal-rittersbach_winter-rape_crop-specific-irrigation_soil-compaction
#SBATCH --output=svat_crop_elztal-rittersbach_winter-rape_crop-specific-irrigation_soil-compaction.out
#SBATCH --error=svat_crop_elztal-rittersbach_winter-rape_crop-specific-irrigation_soil-compaction_err.out
#SBATCH --export=ALL
 
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/bw_cropland//irrigation_soil-compaction
 
python svat_crop.py -b numpy -d cpu --location elztal-rittersbach --irrigation-scenario crop-specific --crop-rotation-scenario winter-rape -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/irrigation_soil-compaction/crop-specific"
mkdir -p /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/irrigation_soil-compaction/crop-specific
mv "${TMPDIR}"/SVATCROP_*.nc /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/irrigation_soil-compaction/crop-specific
