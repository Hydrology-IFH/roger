#!/bin/bash
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_altheim_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_2030-2059
#SBATCH --output=svat_altheim_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_2030-2059.out
#SBATCH --error=svat_altheim_corn_catch_crop_CCCma-CanESM2_CCLM4-8-17_2030-2059_err.out
#SBATCH --export=ALL
 
# load module dependencies
module load lib/hdf5/1.14.4-gnu-13.3-openmpi-5.0
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat
 
python svat_crop.py -b numpy -d cpu --location altheim --land-cover-scenario corn_catch_crop --climate-scenario CCCma-CanESM2_CCLM4-8-17 --period 2030-2059 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat
mv "${TMPDIR}"/SVAT_*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat
