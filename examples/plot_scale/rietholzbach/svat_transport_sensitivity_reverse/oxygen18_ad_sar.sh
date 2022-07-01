#!/bin/bash -l
#
#SBATCH --partition=single
#SBATCH --job-name=oxygen18_ad_sar
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=180000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=24:00:00
 
# load module dependencies
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
 
# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b numpy -d cpu -n 32 1 -tms advection-dispersion
