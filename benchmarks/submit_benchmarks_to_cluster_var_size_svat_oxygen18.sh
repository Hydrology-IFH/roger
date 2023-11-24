#!/bin/bash

# varying problem size and fixed number of CPU cores
python run_benchmarks.py --sizes 40. --sizes 160. --sizes 1600. --sizes 16000. --sizes 24000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --backends jax-gpu --nproc 40 --pmem 4500 --only SVATOXYGEN18_benchmark.py --timesteps 20 --debug
