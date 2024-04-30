#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=180000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat18O_adpt_sa
#SBATCH --output=svat18O_adpt_sa.out
#SBATCH --error=svat18O_adpt_sa_err.out
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
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity
 
mpirun --bind-to core --map-by core -report-bindings python svat_oxygen18.py -b jax -d cpu -n 32 1 --float-type float64 -ns 16384 -tms time-variant_advection-dispersion-power -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity/output"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity/output
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity/output
