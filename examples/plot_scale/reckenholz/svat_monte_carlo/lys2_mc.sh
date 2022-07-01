#!/bin/bash
#
#SBATCH --partition=single
#SBATCH --job-name=lys2_mc
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=24:00:00
 
# load module dependencies
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
 
# adapt command to your available scheduler / MPI implementation
conda activate roger-mpi
mpirun --bind-to core --map-by core -report-bindings python svat_crop.py -b numpy -d cpu -n 40 1 -lys lys2
