#!/bin/bash
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_lahr_summer-wheat_winter-wheat
#SBATCH --output=svat_crop_lahr_summer-wheat_winter-wheat.out
#SBATCH --error=svat_crop_lahr_summer-wheat_winter-wheat_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/svat_crop
 
python svat_crop.py -b numpy -d cpu --location lahr --crop-rotation-scenario summer-wheat_winter-wheat -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work9/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/output/svat_crop"
mkdir -p /pfs/work9/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/output/svat_crop
mv "${TMPDIR}"/SVATCROP_*.nc /pfs/work9/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/output/svat_crop
