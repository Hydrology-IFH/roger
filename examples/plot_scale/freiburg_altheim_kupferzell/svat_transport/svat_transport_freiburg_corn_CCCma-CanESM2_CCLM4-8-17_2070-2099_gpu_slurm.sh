#!/bin/bash
#SBATCH --time=21:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_transport_freiburg_corn_CCCma-CanESM2_CCLM4-8-17_2070-2099
#SBATCH --output=svat_transport_freiburg_corn_CCCma-CanESM2_CCLM4-8-17_2070-2099.out
#SBATCH --error=svat_transport_freiburg_corn_CCCma-CanESM2_CCLM4-8-17_2070-2099_err.out
#SBATCH --export=ALL
 
# load module dependencies
module load lib/hdf5/1.14.4-gnu-13.3-openmpi-5.0
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-gpu
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_freiburg_corn_CCCma-CanESM2_CCLM4-8-17_2070-2099.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat/SVAT_freiburg_corn_CCCma-CanESM2_CCLM4-8-17_2070-2099.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT_freiburg_corn_CCCma-CanESM2_CCLM4-8-17_2070-2099.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
python svat_crop_transport.py -b jax -d gpu --float-type float64 --location freiburg --land-cover-scenario corn --climate-scenario CCCma-CanESM2_CCLM4-8-17 --period 2070-2099 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
mv "${TMPDIR}"/SVATTRANSPORT_*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/freiburg_altheim_kupferzell/output/svat_transport
