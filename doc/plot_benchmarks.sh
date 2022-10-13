#!/bin/sh

INFILES=$(find $PWD/../benchmarks -type f -maxdepth 1 -name "benchmarks_*.json")
python plot_benchmarks.py -- $INFILES
