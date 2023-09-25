#!/bin/bash
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_kupferzell_grass_CCCma-CanESM2_CCLM4-8-17_2070-2099
#SBATCH --output=svat_kupferzell_grass_CCCma-CanESM2_CCLM4-8-17_2070-2099.out
#SBATCH --error=svat_kupferzell_grass_CCCma-CanESM2_CCLM4-8-17_2070-2099_err.out
#SBATCH --export=ALL
#PBS -m a
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat
 
python svat.py -b numpy -d cpu --location kupferzell --climate-scenario CCCma-CanESM2_CCLM4-8-17 --period 2070-2099 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat
mv "${TMPDIR}"/SVAT_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat
