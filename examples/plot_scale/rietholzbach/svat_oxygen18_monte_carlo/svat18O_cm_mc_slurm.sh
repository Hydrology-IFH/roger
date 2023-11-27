#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat18O_cm_mc
#SBATCH --output=svat18O_cm_mc.out
#SBATCH --error=svat18O_cm_mc_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach_new/svat_oxygen18_monte_carlo
 
python svat_transport.py -b jax -d cpu -ns 100 -tms complete-mixing -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach_new/svat_oxygen18_monte_carlo/output"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach_new/svat_oxygen18_monte_carlo/output
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach_new/svat_oxygen18_monte_carlo/output
