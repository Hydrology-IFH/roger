#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=moehlin
#SBATCH --output=moehlin.out
#SBATCH --error=moehlin_err.out
#SBATCH --export=ALL
 
 # load module dependencies
module load devel/cuda/10.2
module load devel/cudnn/10.2
module load lib/hdf5/1.12.2-gnu-12.1-openmpi-4.1
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/StressRes/Moehlin/oneD
  
python moehlin_setup.py -b jax -d gpu --float-type float64 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/catchment_scale/StressRes/Moehlin/output/oneD"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/catchment_scale/StressRes/Moehlin/output/oneD
mv "${TMPDIR}"/ONED_Moehlin.*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/catchment_scale/StressRes/Moehlin/output/oneD