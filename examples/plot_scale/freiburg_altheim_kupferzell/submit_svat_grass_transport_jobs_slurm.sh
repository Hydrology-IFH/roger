#!/bin/bash

cd ~/roger/examples/plot_scale/freiburg_altheim_kupferzell/svat_transport

FILES="$PWD/svat_transport_*_grass_*_1985-2014_cpumpi_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done

FILES="$PWD/svat_transport_*_grass_*_2030-2059_cpumpi_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done

FILES="$PWD/svat_transport_*_grass_*_2070-2099_cpumpi_slurm.sh"
for f in $FILES
do
  sbatch --partition=single $f
done



