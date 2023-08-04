#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_lys8_bromide_mc
#SBATCH --output=svat_lys8_bromide_mc.out
#SBATCH --error=svat_lys8_bromide_mc_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_crop_monte_carlo/svat
 
python svat.py -b numpy -d cpu --lys-experiment lys8_bromide -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/output/svat_crop_monte_carlo"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/output/svat_crop_monte_carlo
mv "${TMPDIR}"/SVATCROP_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/output/svat_crop_monte_carlo
