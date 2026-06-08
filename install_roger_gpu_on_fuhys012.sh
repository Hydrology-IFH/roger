#!/bin/sh
mamba env create --file=conda-environment-gpu-test.yml
singularity run --nv /exchange/nvhpc_25.7_devel.sif
conda activate roger-gpu
pip install -U "jax[cuda12]"
exit
conda activate roger-gpu
mamba install h5py
mamba install h5netcdf
mamba install rasterio
pip cache purge
mamba clean --all
