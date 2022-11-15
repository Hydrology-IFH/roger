#!/bin/bash

# varying problem size and fixed number of CPU cores
# python run_benchmarks.py --sizes 1000. --sizes 10000. --sizes 100000. --sizes 500000. --sizes 1000000. --sizes 2000000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --backends jax-gpu --nproc 28 --pmem 4000 --only SVAT_benchmark.py --debug
# python run_benchmarks.py --sizes 1000. --sizes 10000. --sizes 100000. --sizes 500000. --sizes 1000000. --sizes 2000000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --backends jax-gpu --nproc 28 --pmem 4000 --only oneD_benchmark.py --debug
python run_benchmarks.py --sizes 100. --sizes 1000. --sizes 10000. --sizes 100000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --backends jax-gpu --nproc 20 --pmem 4000 --only SVATOXYGEN18_benchmark.py --debug
