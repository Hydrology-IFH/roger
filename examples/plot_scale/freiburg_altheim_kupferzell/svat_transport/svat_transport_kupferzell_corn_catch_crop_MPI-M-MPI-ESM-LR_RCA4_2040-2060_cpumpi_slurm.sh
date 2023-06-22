#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_transport_kupferzell_corn_catch_crop_MPI-M-MPI-ESM-LR_RCA4_2040-2060
#SBATCH --output=svat_transport_kupferzell_corn_catch_crop_MPI-M-MPI-ESM-LR_RCA4_2040-2060
#SBATCH --error=svat_transport_kupferzell_corn_catch_crop_MPI-M-MPI-ESM-LR_RCA4_2040-2060
#SBATCH --export=ALL
 
# load module dependencies
module load devel/cuda/10.2
module load devel/cudnn/10.2
module load lib/hdf5/1.12.2-gnu-12.1-openmpi-4.1
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_kupferzell_corn_catch_crop_MPI-M-MPI-ESM-LR_RCA4_2040-2060.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_kupferzell_corn_catch_crop_MPI-M-MPI-ESM-LR_RCA4_2040-2060.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT_kupferzell_corn_catch_crop_MPI-M-MPI-ESM-LR_RCA4_2040-2060.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
mpirun --bind-to core --map-by core -report-bindings python svat_crop_transport.py -b jax -d cpu -n 4 1 --location kupferzell --land-cover-scenario corn_catch_crop --climate-scenario MPI-M-MPI-ESM-LR_RCA4 --period 2040-2060 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
