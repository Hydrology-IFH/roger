# Dreisam-Moehlin-Neumagen catchment

Simulation of the soil water balance of the Dreisam-Moehlin-Neumagen catchment (Germany) using RoGeR.

Short description of the folders:
- `oneD_crop_distributed/`: RoGeR-1D model including lateral flow (no lateral exchange between grid cells) and crop phenology
- `svat_crop_distributed/`: RoGeR-SVAT model (no lateral flow) including crop phenology

We finally used `oneD_crop_distributed/`.

In order to run the Python scripts, you have to install the anaconda environment using `../conda-environment.yml`. You can follow the instructions provided in `../README.md`.

See READMEs in the subfolders for more information. I have tried my best to document everything as good as possible and for sure the you will encounter some bugs or incomplete documentation. My advice is that for larger modelling projects a single person is not sufficient and I suggest the four eye principle. Many conceptual and technical issues can be avoided if you are working in development teams.