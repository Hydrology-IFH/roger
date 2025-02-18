#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat
#SBATCH --output=svat.out
#SBATCH --error=svat_err.out
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
cd /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_distributed
  
python svat_crop.py -b jax -d gpu -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_distributed/output"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_distributed/output
mv "${TMPDIR}"/SVAT.*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_distributed/output