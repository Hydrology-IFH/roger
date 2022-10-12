#!/bin/bash
 
eval "$(conda shell.bash hook)"
conda activate roger
/Users/robinschwemmle/anaconda3/envs/roger/bin/python /Users/robinschwemmle/Desktop/PhD/models/roger/benchmarks/oneD_benchmark.py --backend jax --device cpu --size 316 316 --timesteps 5 --float-type float64
