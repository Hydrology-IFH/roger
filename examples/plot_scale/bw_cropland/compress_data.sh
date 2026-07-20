#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=compress_data
#SBATCH --output=compress_data.out
#SBATCH --error=compress_data_err.out
#SBATCH --export=ALL

cd /pfs/10/work/fr_rs1092-workspace/roger/examples/plot_scale/bw_cropland/output
tar -zcvf irrigation_soil-compaction.tar.gz /pfs/10/work/fr_rs1092-workspace/roger/examples/plot_scale/bw_cropland/output/irrigation_soil-compaction
tar -zcvf no-irrigation_soil-compaction.tar.gz /pfs/10/work/fr_rs1092-workspace/roger/examples/plot_scale/bw_cropland/output/no-irrigation_soil-compaction