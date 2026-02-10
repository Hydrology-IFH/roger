#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=calculate_recharge_anomalies
#SBATCH --output=calculate_recharge_anomalies.out
#SBATCH --error=calculate_recharge_anomalies_err.out
#SBATCH --export=ALL

module load devel/miniforge
conda activate roger
cd /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed

python calculate_recharge_anomalies.py
