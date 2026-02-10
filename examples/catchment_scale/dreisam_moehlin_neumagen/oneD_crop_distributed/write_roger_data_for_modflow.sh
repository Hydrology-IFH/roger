#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=write_roger_data_for_modflow
#SBATCH --output=write_roger_data_for_modflow.out
#SBATCH --error=write_roger_data_for_modflow_err.out
#SBATCH --export=ALL

module load devel/miniforge
conda activate roger
cd /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed

python write_roger_data_for_modflow.py
