#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat18O_pi_sa
#SBATCH --output=svat18O_pi_sa.out
#SBATCH --error=svat18O_pi_sa_err.out
#SBATCH --export=ALL
 
# load module dependencies
module load lib/hdf5/1.14.4-gnu-13.3-openmpi-5.0
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity
 
mpirun --bind-to core --map-by core -report-bindings python svat_oxygen18.py -b jax -d cpu -n 20 1 --float-type float64 -ns 10240 -tms piston -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity/output"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity/output
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_oxygen18_sensitivity/output
