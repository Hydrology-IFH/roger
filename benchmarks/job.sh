#!/bin/bash
 
eval "$(conda shell.bash hook)"
conda activate roger-mpi
cd /Users/robinschwemmle/Desktop/PhD/models/roger/benchmarks
mpirun -n 2 python oneD_benchmark.py --backend jax --device cpu -n 2 1 --size 448 448 --timesteps 5 --float-type float64
