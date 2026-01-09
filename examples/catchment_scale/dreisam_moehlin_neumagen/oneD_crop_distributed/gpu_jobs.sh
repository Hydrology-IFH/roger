#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate roger-gpu
# python oneD_crop.py -b jax -d gpu --stress-test-meteo base 
# python oneD_crop.py -b jax -d gpu --stress-test-meteo base --yellow-mustard
# python oneD_crop.py -b jax -d gpu --stress-test-meteo base --soil-compaction
# python oneD_crop.py -b jax -d gpu --stress-test-meteo base --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0
# python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 2     
# python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --irrigation
# python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0
# python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --irrigation
# python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 2 
