#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --mem=24000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop
#SBATCH --output=svat_crop.out
#SBATCH --error=svat_crop_err.out
#SBATCH --export=ALL

# load module dependencies
module load lib/hdf5/1.14.4-gnu-13.3-openmpi-4.1
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_crop_distributed
  
mpirun --bind-to core --map-by core -report-bindings python svat_crop.py -b numpy -d cpu -n 6 6 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_crop_distributed/output"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_crop_distributed/output
mv "${TMPDIR}"/SVAT.*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/dreisam_moehlin_neumagen/svat_crop_distributed/output