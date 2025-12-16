#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=write_simulations_to_csv_no-irrigation_soil-compaction
#SBATCH --output=write_simulations_to_csv_no-irrigation_soil-compaction.out
#SBATCH --error=write_simulations_to_csv_no-irrigation_soil-compaction_err.out
#SBATCH --export=ALL
 
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate roger
cd /pfs/10/work/fr_rs1092-workspace/roger/examples/plot_scale/bw_cropland/no-irrigation_soil-compaction
 
python write_simulations_of_lsv_locations_to_csv.py
# python write_simulations_to_csv.py