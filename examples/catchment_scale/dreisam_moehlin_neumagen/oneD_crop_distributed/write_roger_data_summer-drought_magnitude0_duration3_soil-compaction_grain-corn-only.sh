#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=write_roger_data_summer-drought_magnitude0_duration3_soil-compaction_grain-corn-only
#SBATCH --output=write_roger_data_summer-drought_magnitude0_duration3_soil-compaction_grain-corn-only.out
#SBATCH --error=write_roger_data_summer-drought_magnitude0_duration3_soil-compaction_grain-corn-only_err.out
#SBATCH --export=ALL

module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate roger
cd /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed

python write_roger_data_for_modflow.py --stress-test-meteo summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction soil-compaction --grain-corn-only grain-corn-only
