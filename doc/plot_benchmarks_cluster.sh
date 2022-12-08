#!/bin/sh

# INFILES=$(find $PWD/../benchmarks/var_size/oneD/cluster -type f -maxdepth 1 -name "benchmark_*.json")
# python plot_benchmarks.py --xaxis size -- $INFILES
# python plot_benchmarks.py --xaxis size --norm-component numpy -- $INFILES
# INFILES=$(find $PWD/../benchmarks/var_proc/oneD/cluster -type f -maxdepth 1 -name "benchmark_*.json")
# python plot_benchmarks.py --xaxis nproc -- $INFILES

INFILES=$(find $PWD/../benchmarks/var_size/svat/cluster -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis size -- $INFILES
python plot_benchmarks.py --xaxis size --norm-component numpy -- $INFILES
INFILES=$(find $PWD/../benchmarks/var_proc/svat/cluster -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis nproc -- $INFILES

INFILES=$(find $PWD/../benchmarks/var_size/svat_oxygen18/cluster -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis size -- $INFILES
python plot_benchmarks.py --xaxis size --norm-component numpy -- $INFILES
INFILES=$(find $PWD/../benchmarks/var_proc/svat_oxygen18/cluster  -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis nproc -- $INFILES
