#!/bin/bash
#
#SBATCH --partition=single
#SBATCH --job-name=roger_rietholzbach
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=1:00:00
#SBATCH --error="error.log"

python ${HOME}/roger/examples/plot_scale/rietholzbach/svat_monte_carlo/svat.py

# displays what resources are available for immediate use for the whole partition
# sinfo_t_idle

# execute the script
# sbatch roger_single.sh
