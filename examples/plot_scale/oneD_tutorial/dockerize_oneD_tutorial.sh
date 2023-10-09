#!/bin/bash

# move Docker files to root directory of the source code
cd ../../..
rm -f Dockerfile
rm -f .dockerignore
rm -f compose.yaml
cp examples/plot_scale/oneD_tutorial/Dockerfile Dockerfile
cp examples/plot_scale/oneD_tutorial/.dockerignore .dockerignore
cp examples/plot_scale/oneD_tutorial/.dockerignore compose.yaml

# build the container of the model setup
docker build -t roger-oneD-tutorial .
rm -f Dockerfile
rm -f .dockerignore
rm -f compose.yaml