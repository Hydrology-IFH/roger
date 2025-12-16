#!/bin/bash

# move Docker files to root directory of the source code
cd ../../..
rm -f Dockerfile
rm -f .dockerignore
rm -f compose.yaml
cp examples/plot_scale/svat_tutorial/Dockerfile Dockerfile
cp examples/plot_scale/svat_tutorial/.dockerignore .dockerignore
cp examples/plot_scale/svat_tutorial/.dockerignore compose.yaml

# build the container of the model setup
docker build -t roger-svat-tutorial .
rm -f Dockerfile
rm -f .dockerignore
rm -f compose.yaml