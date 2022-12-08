#!/bin/sh

INFILES=$(find $PWD/../benchmarks/var_size/svat/notebook -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis size -- $INFILES
python plot_benchmarks.py --xaxis size --norm-component numpy -- $INFILES

INFILES=$(find $PWD/../benchmarks/var_size/svat_oxygen18/notebook -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis size -- $INFILES
python plot_benchmarks.py --xaxis size --norm-component numpy -- $INFILES
