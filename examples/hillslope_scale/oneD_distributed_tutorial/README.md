# Tutorial for the spatially-distributed 1D-model

Brief overview of the files:

- `input`: Contains input data for a 3-year period
- `config.yml`: File to set settings and output variables
- `parameters.csv`: File that contains the model parameters
- `oneD.py`: 1D-model setup
- `merge_output.py`: Merges model output into a single file.

# Date requirements

The following information is required to run the model (see Workflow). 

## Meteorological input data
The required meteorological input data is loaded from the input folder. The input folder should contain a text file
for precipitation (`PREC.txt`; 10 minutes time steps), air temperature (`TA.txt`; daily time steps) and potential evapotranspiration (`PET.txt`; daily time steps)

## Model parameters
The model parameters (`parameters.csv`) are loaded from the same directory as `oneD.py`.

Random model parameters can be generated running the following script:
```
python write_parameters.py
```

## Model settings
Name of model experiment and spatial discretization are defined in `config.yml`.

## Model output
The variables written to the output files are defined in `config.yml`. Available variables
are listed here https://roger.readthedocs.io/en/latest/reference/variables.html#available-variables. Generally, storage variables
should be defined for `OUTPUT_COLLECT` and flux variables for `OUTPUT_RATE`.

# Workflow
If required data is ready, the following script runs the simulations:

```
python svat.py
```

After calculation is done, the simulation results can be merged in a single NetCDF-file:
```
python merge_output.py
```