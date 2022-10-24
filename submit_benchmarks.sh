#!/bin/bash

# varying problem size and fixed number of CPU cores
python run_benchmarks.py --sizes 1000. --sizes 10000. --sizes 100000. --sizes 1000000. --backends numpy --backends jax --backends numpy-mpi --backends jax-mpi --backends jax-gpu --nproc 25 --pmem 5000 --only oneD_benchmark.py --debug

# fixed problem size and varying number of CPU cores
# python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 4 --pmem 32000 --only oneD_benchmark.py --debug
# python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 8 --pmem 16000 --only oneD_benchmark.py --debug
# python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 16 --pmem 8000 --only oneD_benchmark.py --debug
# python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 25 --pmem 4000 --only oneD_benchmark.py --debug
# python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 50 --pmem 4000 --only oneD_benchmark.py --debug
# python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 100 --pmem 4000 --only oneD_benchmark.py --debug
# python run_benchmarks.py --sizes 1000000. --backends numpy-mpi --backends jax-mpi --nproc 200 --pmem 4000 --only oneD_benchmark.py --debug
