#!/bin/bash
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=moehlin
#SBATCH --output=moehlin.out
#SBATCH --error=moehlin_err.out
#SBATCH --export=ALL
 
eval "$(conda shell.bash hook)"
conda activate roger
cd /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/StressRes/Moehlin/oneD_free_drainage
  
python moehlin_setup.py -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/catchment_scale/StressRes/Moehlin/output/oneD_free_drainage"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/catchment_scale/StressRes/Moehlin/output/oneD_free_drainage
mv "${TMPDIR}"/ONED_Moehlin_free_drainage.*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/catchment_scale/StressRes/Moehlin/output/oneD_free_drainage