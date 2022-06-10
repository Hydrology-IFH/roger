#!/bin/bash
#
#SBATCH --partition=single
#SBATCH --job-name=roger_rietholzbach
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --mem-per-cpu=2000mb
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=12:00:00

# load module dependencies
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1

# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_transport.py

# displays what resources are available for immediate use for the whole partition
# sinfo_t_idle

# execute the script
# chmod +x roger_batch.sh
# sbatch ./roger_batch.sh
