#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop
#SBATCH --output=svat_crop.out
#SBATCH --error=svat_crop_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_crop_distributed
  
python svat_crop.py -b jax -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_crop_distributed/output"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_crop_distributed/output
mv "${TMPDIR}"/SVAT.*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_crop_distributed/output