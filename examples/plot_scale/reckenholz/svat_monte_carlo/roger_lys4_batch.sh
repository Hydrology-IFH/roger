#!/bin/bash
#
#SBATCH --partition=single
#SBATCH --job-name=lys4_mc
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=24:00:00

# load module dependencies
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1

# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat.py lys4

# displays what resources are available for immediate use for the whole partition
# sinfo_t_idle

# execute the script
# conda activate roger-mpi
# cd ~/roger/examples/plot_scale/reckenholz/svat_monte_carlo
# chmod +x roger_lys4_batch.sh
# sbatch ./roger_lys4_batch.sh
