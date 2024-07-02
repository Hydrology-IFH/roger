#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=write_shapefiles_stockach
#SBATCH --output=write_shapefiles_stockach.out
#SBATCH --error=write_shapefiles_stockach_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh
 
python assign_simulated_values_to_polygons_for_NO3_leaching.py --location stockach -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/output"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/output
mv "${TMPDIR}"/nitrate_leaching_stockach.gpkg /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/output
