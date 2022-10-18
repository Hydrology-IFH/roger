#!/bin/sh

INFILES=$(find $PWD/../benchmarks -type f -maxdepth 1 -name "benchmark_*.json")
python plot_benchmarks.py --xaxis size -- $INFILES
python plot_benchmarks.py --xaxis size --norm-component numpy -- $INFILES
