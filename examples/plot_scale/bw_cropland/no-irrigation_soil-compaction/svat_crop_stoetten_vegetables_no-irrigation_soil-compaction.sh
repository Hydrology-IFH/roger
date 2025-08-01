#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_stoetten_vegetables_no-irrigation_soil-compaction
#SBATCH --output=svat_crop_stoetten_vegetables_no-irrigation_soil-compaction.out
#SBATCH --error=svat_crop_stoetten_vegetables_no-irrigation_soil-compaction_err.out
#SBATCH --export=ALL
 
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/bw_cropland//no-irrigation_soil-compaction
 
python svat_crop.py -b numpy -d cpu --location stoetten --crop-rotation-scenario vegetables -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/no-irrigation_soil-compaction"
mkdir -p /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/no-irrigation_soil-compaction
mv "${TMPDIR}"/SVATCROP_*.nc /pfs/10/work/fr_rs1092-workspace/bw_cropland/output/no-irrigation_soil-compaction
