#!/bin/bash
#
#SBATCH --partition=single
#SBATCH --job-name=roger_rietholzbach
#SBATCH --nodes=1
#SBATCH --ntasks=80
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2000
#SBATCH --exclusive
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=24:00:00

# load module dependencies
# module load mpi4py h5py

# adapt command to your available scheduler / MPI implementation
mpirun python ${PWD}/svat_monte_carlo/svat.py
mpirun python ${PWD}/svat_sensitivity/svat.py

# displays what resources are available for immediate use for the whole partition
# sinfo_t_idle

# execute the script
# sbatch ./roger_batch.sh
