#!/bin/bash
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_altheim_crop_rotation_CCCma-CanESM2_CCLM4-8-17_2030-2059
#SBATCH --output=svat_altheim_crop_rotation_CCCma-CanESM2_CCLM4-8-17_2030-2059.out
#SBATCH --error=svat_altheim_crop_rotation_CCCma-CanESM2_CCLM4-8-17_2030-2059_err.out
#SBATCH --export=ALL
 
# load module dependencies
module load devel/cuda/10.2
module load devel/cudnn/10.2
module load lib/hdf5/1.12.2-gnu-12.1-openmpi-4.1
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat
 
python svat_crop.py -b numpy -d cpu --location altheim --land-cover-scenario crop_rotation --climate-scenario CCCma-CanESM2_CCLM4-8-17 --period 2030-2059 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat
mv "${TMPDIR}"/SVAT_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat
