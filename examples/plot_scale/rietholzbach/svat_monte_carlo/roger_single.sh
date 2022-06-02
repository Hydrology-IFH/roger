#!/bin/bash
#
#SBATCH --partition=single
#SBATCH --job-name=roger_rietholzbach
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=1:00:00

# load module dependencies
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1

python svat.py

# displays what resources are available for immediate use for the whole partition
# sinfo_t_idle

# execute the script
# sbatch roger_single.sh
