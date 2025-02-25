#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
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
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/StressRes/Moehlin/oneD
  
mpirun --bind-to core --map-by core -report-bindings python moehlin_setup.py -b numpy -d cpu -n 4 1 --float-type float64  -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/StressRes/Moehlin/oneD/output"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/StressRes/Moehlin/oneD/output
mv "${TMPDIR}"/ONED_Moehlin.*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/StressRes/Moehlin/oneD/output