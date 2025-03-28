#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=14000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat18O
#SBATCH --output=svat18O.out
#SBATCH --error=svat18Oout
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
cd /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_oxygen18
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat/output/SVAT.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat/output/SVAT.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT18O.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
python svat_oxygen18.py -b jax -d gpu -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_oxygen18/output"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_oxygen18/output
mv "${TMPDIR}"/SVAT18O.*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_oxygen18/output
