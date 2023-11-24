#!/bin/bash

# varying problem size and fixed number of CPU cores
python run_benchmarks.py --sizes 8. --sizes 100. --sizes 1000. --sizes 10000. --sizes 100000. --sizes 1000000. --sizes 10000000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --nproc 8 --only oneD_benchmark.py --debug --local --outfile $PWD/var_size/oneD/macbookpro14_2023/benchmark_21112023.json
