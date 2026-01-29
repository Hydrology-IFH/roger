#!/bin/bash
#SBATCH --time=13:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=24000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=oneD_crop_base_soil-compaction
#SBATCH --output=oneD_crop_base_soil-compaction.out
#SBATCH --error=oneD_crop_base_soil-compaction_err.out
#SBATCH --export=ALL
module load lib/hdf5/1.12-gnu-14.2-openmpi-4.1
module load devel/cuda/12.6
module load devel/miniforge
conda activate roger-gpu
cd /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed

mkdir ${TMPDIR}/roger
mkdir ${TMPDIR}/roger/examples
mkdir ${TMPDIR}/roger/examples/catchment_scale
mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen
mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed
mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/oneD_crop.py ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/parameters_roger.nc ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/config.yml ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/input ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed
sleep 120
cd ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed
python oneD_crop.py -b jax -d gpu --stress-test-meteo base --soil-compaction soil-compaction
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output"
mkdir -p /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output
mv -v "${TMPDIR}"/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/ONEDCROP_*.nc /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output