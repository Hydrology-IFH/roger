#!/bin/bash

cd ~/roger/examples/plot_scale/boadkh

sbatch --partition=single write_shapefiles_freiburg_slurm.sh
sbatch --partition=single write_shapefiles_muellheim_slurm.sh
sbatch --partition=single write_shapefiles_lahr_slurm.sh
sbatch --partition=single write_shapefiles_stockach_slurm.sh
sbatch --partition=single write_shapefiles_gottmadingen_slurm.sh
sbatch --partition=single write_shapefiles_weingarten_slurm.sh
sbatch --partition=single write_shapefiles_eppingen-elsenz_slurm.sh
sbatch --partition=single write_shapefiles_bruchsal-heidelsheim_slurm.sh
sbatch --partition=single write_shapefiles_bretten_slurm.sh
sbatch --partition=single write_shapefiles_ehingen-kirchen_slurm.sh
sbatch --partition=single write_shapefiles_merklingen_slurm.sh
sbatch --partition=single write_shapefiles_hayingen_slurm.sh
sbatch --partition=single write_shapefiles_kupferzell_slurm.sh
sbatch --partition=single write_shapefiles_oehringen_slurm.sh
sbatch --partition=single write_shapefiles_vellberg-kleinaltdorf_slurm.sh
