#!/bin/sh

INFILES=$(find $PWD/../benchmarks/var_size/oneD -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis size -- $INFILES
python plot_benchmarks.py --xaxis size --norm-component numpy -- $INFILES
INFILES=$(find $PWD/../benchmarks/var_proc/oneD -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis nproc -- $INFILES

INFILES=$(find $PWD/../benchmarks/var_size/svat -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis size -- $INFILES
python plot_benchmarks.py --xaxis size --norm-component numpy -- $INFILES
INFILES=$(find $PWD/../benchmarks/var_proc/svat -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis nproc -- $INFILES
