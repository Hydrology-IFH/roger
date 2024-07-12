#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_lys2_mc
#SBATCH --output=svat_lys2_mc.out
#SBATCH --error=svat_lys2_mc_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_monte_carlo
 
python svat.py -b numpy -d cpu --lys-experiment lys2 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/reckenholz/output/svat_monte_carlo"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/reckenholz/output/svat_monte_carlo
mv "${TMPDIR}"/SVAT_*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/reckenholz/output/svat_monte_carlo
