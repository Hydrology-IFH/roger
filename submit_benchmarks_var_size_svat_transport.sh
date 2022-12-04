#!/bin/bash

# varying problem size and fixed number of CPU cores
python run_benchmarks.py --sizes 24. --sizes 100. --sizes 1000. --sizes 10000. --sizes 20000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --backends jax-gpu --nproc 28 --pmem 4000 --only SVATOXYGEN18_benchmark.py --timesteps 20 --debug
