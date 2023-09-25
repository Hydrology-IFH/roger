#!/bin/bash
#SBATCH --time=21:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_transport_freiburg_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2030-2059
#SBATCH --output=svat_transport_freiburg_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2030-2059.out
#SBATCH --error=svat_transport_freiburg_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2030-2059_err.out
#SBATCH --export=ALL
 
# load module dependencies
module load devel/cudnn/10.2
module load devel/cuda/12.0
module load lib/hdf5/1.12.2-gnu-12.1-openmpi-4.1
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_freiburg_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2030-2059.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_freiburg_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2030-2059.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT_freiburg_crop_rotation_MPI-M-MPI-ESM-LR_RCA4_2030-2059.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
python svat_crop_transport.py -b jax -d gpu --location freiburg --land-cover-scenario crop_rotation --climate-scenario MPI-M-MPI-ESM-LR_RCA4 --period 2030-2059 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
