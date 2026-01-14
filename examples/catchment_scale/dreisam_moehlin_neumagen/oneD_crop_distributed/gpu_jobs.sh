#!/bin/bash

python oneD_crop.py -b jax -d gpu --stress-test-meteo base --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo base
python oneD_crop.py -b jax -d gpu --stress-test-meteo base --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo base --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo base --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo base --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --irrigation
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --irrigation --yellow-mustard
python oneD_crop.py -b jax -d gpu --stress-test-meteo spring-summer-wet --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction --irrigation --yellow-mustard
