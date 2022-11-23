#!/bin/bash

# fixed problem size and varying number of CPU cores
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 4 --pmem 32000 --only SVAT_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 8 --pmem 16000 --only SVAT_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 16 --pmem 8000 --only SVAT_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 24 --pmem 4000 --only SVAT_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 96 --pmem 4000 --only SVAT_benchmark.py --debug
python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 192 --pmem 4000 --only SVAT_benchmark.py --debug
