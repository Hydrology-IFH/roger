#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --cpus-per-task=1
#SBATCH --mem=180000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat18O_adkt_mc
#SBATCH --output=svat18O_adkt_mc.out
#SBATCH --error=svat18O_adkt_mc_err.out
#SBATCH --export=ALL
 
# load module dependencies
module load lib/hdf5/1.14.4-gnu-13.3-openmpi-5.0
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo
 
mpirun --bind-to core --map-by core -report-bindings python svat_oxygen18.py -b jax -d cpu -n 25 1 --float-type float64 -ns 10000 -tms time-variant_advection-dispersion-kumaraswamy -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo/output"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo/output
mv "${TMPDIR}"/SVATOXYGEN18_*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/rietholzbach/svat_oxygen18_monte_carlo/output
