#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=write_data_spring-summer-drought_magnitude2_duration3_soil-compaction_yellow-mustard
#SBATCH --output=write_data_spring-summer-drought_magnitude2_duration3_soil-compaction_yellow-mustard.out
#SBATCH --error=write_data_spring-summer-drought_magnitude2_duration3_soil-compaction_yellow-mustard_err.out
#SBATCH --export=ALL
 
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate roger
cd /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/oneD_crop_distributed
python write_roger_data_for_modflow.py --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --yellow-mustard

