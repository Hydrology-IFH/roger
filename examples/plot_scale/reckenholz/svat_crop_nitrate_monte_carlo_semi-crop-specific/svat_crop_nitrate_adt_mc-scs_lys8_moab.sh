#!/bin/bash
#PBS -l nodes=1:ppn=25
#PBS -l walltime=24:00:00
#PBS -l pmem=4000mb
#PBS -N svat_crop_nitrate_adt_mc-scs_lys8
#PBS -m bea
#PBS -M robin.schwemmle@hydrology.uni-freiburg.de
 
# load module dependencies
module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_crop_nitrate_monte_carlo_semi-crop-specific
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_nitrate_monte_carlo_semi-crop-specific/SVATCROP_lys8_bootstrap.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_nitrate_monte_carlo_semi-crop-specific/SVATCROP_lys8_bootstrap.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
sleep 60
checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVATCROP_lys8_bootstrap.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
# adapt command to your available scheduler / MPI implementation
mpirun --bind-to core --map-by core -report-bindings python svat_crop_nitrate.py -b jax -d cpu -n 25 1 -lys lys8 -tms time-variant_advection-dispersion-power -td "${TMPDIR}"
# Write output to temporary SSD of computing node
echo "Write output to $TMPDIR"
# Move output from temporary SSD to workspace
echo "Move output to /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_nitrate_monte_carlo_semi-crop-specific"
mkdir -p /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_nitrate_monte_carlo_semi-crop-specific
mv "${TMPDIR}"/*.nc /beegfs/work/workspace/ws/fr_rs1092-workspace-0/reckenholz/svat_crop_nitrate_monte_carlo_semi-crop-specific
