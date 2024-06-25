#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=write_shapfiles_eppingen-elsenz
#SBATCH --output=write_shapfiles_eppingen-elsenz.out
#SBATCH --error=write_shapfiles_eppingen-elsenz_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh
 
python assign_simulated_values_to_polygons_for_NO3_leaching.py --location eppingen-elsenz -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/boadkh/output"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/boadkh/output
mv "${TMPDIR}"/nitrate_leaching_eppingen-elsenz.gpkg /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/boadkh/output
