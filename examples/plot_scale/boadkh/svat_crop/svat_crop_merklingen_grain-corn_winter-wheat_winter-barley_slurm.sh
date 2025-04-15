#!/bin/bash
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_merklingen_grain-corn_winter-wheat_winter-barley
#SBATCH --output=svat_crop_merklingen_grain-corn_winter-wheat_winter-barley.out
#SBATCH --error=svat_crop_merklingen_grain-corn_winter-wheat_winter-barley_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/svat_crop
 
python svat_crop.py -b numpy -d cpu --location merklingen --crop-rotation-scenario grain-corn_winter-wheat_winter-barley -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/output/svat_crop"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/output/svat_crop
mv "${TMPDIR}"/SVATCROP_*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/output/svat_crop
