#!/bin/bash

# varying problem size and fixed number of CPU cores
python run_benchmarks.py --sizes 1000. --sizes 10000. --sizes 100000. --sizes 500000. --sizes 1000000. --sizes 5000000. --sizes 10000000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --backends jax-gpu --nproc 40 --pmem 4500 --only oneD_benchmark.py --debug
