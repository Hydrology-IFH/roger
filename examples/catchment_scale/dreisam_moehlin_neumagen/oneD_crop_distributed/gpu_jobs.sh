#!/bin/bash

python oneD_crop.py -b jax -d gpu --stress-test-meteo base   &
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0   &
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --irrigation   &
