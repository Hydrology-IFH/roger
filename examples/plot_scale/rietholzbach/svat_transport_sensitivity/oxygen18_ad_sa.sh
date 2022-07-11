#!/bin/bash
#
#SBATCH --partition=multiple_e
#SBATCH --job-name=oxygen18_ad_sa
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --mem=90000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=48:00:00
 
# load module dependencies
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
 
# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b numpy -d cpu -n 32 1 -tms advection-dispersion
