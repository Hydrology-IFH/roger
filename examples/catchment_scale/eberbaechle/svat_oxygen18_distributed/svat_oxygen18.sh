#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat18O
#SBATCH --output=svat18O.out
#SBATCH --error=svat18O_err.out
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
cd /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_oxygen18_distributed
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_distributed/output/SVAT.nc | cut -f 1 -d " ")
checksum_ssd=0
cp /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_distributed/output/SVAT.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
    sleep 10
    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVAT.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
mpirun --bind-to core --map-by core -report-bindings python svat_oxygen18.py -b numpy -d cpu -n 20 1 -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_oxygen18_distributed/output"
mkdir -p /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_oxygen18_distributed/output
mv "${TMPDIR}"/SVAT18O.*.nc /home/fr/fr_fr/fr_rs1092/roger/examples/catchment_scale/eberbaechle/svat_oxygen18_distributed/output