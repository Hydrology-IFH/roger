#!/bin/bash
#
#SBATCH --partition=gpu_8
#SBATCH --job-name=oxygen18_pi_mc
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=752000
#SBATCH --mail-type=ALL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=8:00:00
 
# load module dependencies
module load devel/cudnn/9.2
module load devel/cuda/11.4
module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1
 
python svat_transport.py -b jax -d gpu -ns 10000 -tms piston
