#!/bin/sh

INFILES=$(find $PWD/../benchmarks/var_size/svat/notebook -type f -maxdepth 1 -name "benchmark_*.json")
python plot_energy_footprint_notebook.py --xaxis size --nitt 10000 -- $INFILES

INFILES=$(find $PWD/../benchmarks/var_size/svat_oxygen18/notebook -type f -maxdepth 1 -name "benchmark_*.json")
python plot_energy_footprint_notebook.py --xaxis size --nitt 365 -- $INFILES
