#!/bin/bash

cd ../../..
mkdir tmp
mkdir tmp/roger
mkdir tmp/look_up_tables
mkdir tmp/test
cp -R roger tmp/roger/
cp -R look_up_tables tmp/look_up_tables/
cp -R test tmp/test/
cp README.md tmp/README.md
cp LICENSE tmp/LICENSE
cp MANIFEST.in tmp/MANIFEST.in
cp CITATION.cff tmp/CITATION.cff
cp meta.yaml tmp/meta.yaml
cp pyproject.toml tmp/pyproject.toml
cp setup.py tmp/setup.py
cp cuda_ext.py tmp/cuda_ext.py
cp versioneer.py tmp/versioneer.py
cp setup.cfg tmp/setup.cfg
cp requirements.txt tmp/requirements.txt
cp requirements_jax.txt tmp/requirements_jax.txt
cp examples/plot_scale/svat_tutorial/svat.py tmp/svat_tutorial
cp examples/plot_scale/svat_tutorial/Dockerfile tmp/Dockerfile

cd tmp
docker build -t svat_tutorial .
cd ..

cd examples/plot_scale/svat_tutorial
docker run --rm -it -v "/Users/robinschwemmle/Desktop/PhD/models/roger/tmp/:/roger/" svat_tutorial
# rm -rf tmp