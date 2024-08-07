#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=24000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=svat_crop_nitrate_5-3_4262994_1_freiburg_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_sas
#SBATCH --output=svat_crop_nitrate_5-3_4262994_1_freiburg_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_sas.out
#SBATCH --error=svat_crop_nitrate_5-3_4262994_1_freiburg_winter-wheat_silage-corn_yellow-mustard_medium_Nfert_sas_err.out
#SBATCH --export=ALL
 
# load module dependencies
module load lib/hdf5/1.14.4-gnu-13.3-openmpi-4.1
# prevent memory issues for Open MPI 4.1.x
export OMPI_MCA_btl="self,smcuda,vader,tcp"
export OMP_NUM_THREADS=1
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/svat_crop_nitrate_sobol_sas
 
# Copy fluxes and states from global workspace to local SSD
echo "Copy fluxes and states from global workspace to local SSD"
# Compares hashes
checksum_gws=$(shasum -a 256 /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/svat_crop/SVATCROP_freiburg_winter-wheat_silage-corn_yellow-mustard.nc | cut -f 1 -d " ")
checksum_ssd=0a
cp /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/svat_crop/SVATCROP_freiburg_winter-wheat_silage-corn_yellow-mustard.nc "${TMPDIR}"
# Wait for termination of moving files
while [ "${checksum_gws}" != "${checksum_ssd}" ]; do
sleep 10
checksum_ssd=$(shasum -a 256 "${TMPDIR}"/SVATCROP_freiburg_winter-wheat_silage-corn_yellow-mustard.nc | cut -f 1 -d " ")
done
echo "Copying was successful"
 
mpirun --bind-to core --map-by core -report-bindings python svat_crop_nitrate.py -b jax -d cpu -n 8 1 --float-type float64 --row 36 --id 5-3_4262994_1 --location freiburg --crop-rotation-scenario winter-wheat_silage-corn_yellow-mustard --fertilization-intensity medium -td "${TMPDIR}"
# Move output from local SSD to global workspace
echo "Move output to /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/svat_crop_nitrate_sobol_sas"
mkdir -p /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/svat_crop_nitrate_sobol_sas
mv "${TMPDIR}"/SVATCROPNITRATE_*.nc /pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh/svat_crop_nitrate_sobol_sas
