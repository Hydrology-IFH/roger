#!/bin/bash
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_altheim_corn_MPI-M-MPI-ESM-LR_RCA4_1985-2015
#SBATCH --output=svat_altheim_corn_MPI-M-MPI-ESM-LR_RCA4_1985-2015.out
#SBATCH --error=svat_altheim_corn_MPI-M-MPI-ESM-LR_RCA4_1985-2015_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat
 
python svat_crop.py -b numpy -d cpu --location altheim --land-cover-scenario corn --climate-scenario MPI-M-MPI-ESM-LR_RCA4 --period 1985-2015 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat
mv "${TMPDIR}"/SVAT_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat
