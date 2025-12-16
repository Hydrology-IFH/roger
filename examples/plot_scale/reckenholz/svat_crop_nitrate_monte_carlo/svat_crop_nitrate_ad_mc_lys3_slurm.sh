#!/bin/bash
#SBATCH --partition compute
#SBATCH --job-name=svat_crop_nitrate_ad_mc_lys3
#SBATCH --nodes=1
#SBATCH --ntasks=25
#SBATCH --mem=100000mb
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --export=ALL
#SBATCH --time=24:00:00
 
# load module dependencies
module load devel/miniforge
module load lib/hdf5/1.12-gnu-14.2-openmpi-4.1
export OMP_NUM_THREADS=1
export OMPI_MCA_btl="^uct,ofi"
export OMPI_MCA_mtl="^ofi"
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_crop_nitrate_monte_carlo
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /pfs/10/work/fr_rs1092-workspace/reckenholz/svat_crop_nitrate_monte_carlo/SVATCROP_lys3_bootstrap.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /pfs/10/work/fr_rs1092-workspace/reckenholz/svat_crop_nitrate_monte_carlo/SVATCROP_lys3_bootstrap.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
sleep 60
checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVATCROP_lys3_bootstrap.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core:overload-allowed --map-by core -report-bindings python svat_crop_nitrate.py -b jax -d cpu -n 25 1 -lys lys3 -tms advection-dispersion-power -td "${TMPDIR}"
# Write output to temporary SSD of computing node
echo "Write output to $TMPDIR"
# Move output from temporary SSD to workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/reckenholz/svat_crop_nitrate_monte_carlo"
mkdir -p /pfs/10/work/fr_rs1092-workspace/reckenholz/svat_crop_nitrate_monte_carlo
mv "${TMPDIR}"/*.nc /pfs/10/work/fr_rs1092-workspace/reckenholz/svat_crop_nitrate_monte_carlo
