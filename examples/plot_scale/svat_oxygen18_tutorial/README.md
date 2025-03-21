# Tutorial for the SVAT-model

The tutorial simulates a single grid cell.

SVAT indicates that only vertical processes are considered.

- `input`: Contains input data
- `output`: Contains output data
- `figures`: Contains figures
- `config.yml`: File to set parameters and output variables
- `svat_oxygen18.py`: SVAT-model setup for oxygen-18
- `merge_output.py`: Merges output into a single NetCDF file
- `netcdf_to_csv.py`: Writes output to csv files
- `make_figures_and_tables.py`: Produces figures and tables

# Date requirements

The following information is required to run the model. 

## Tracer input data
The required tracer input data is loaded from the input folder. The input folder should contain a tab-delimited text file
for oxygen-18 (`d18O.txt`; daily time steps).


Format of `d18O.txt` (d18O in per mille):
| YYYY  | MM    | DD    | hh    | mm    | d18O  |
| ------| ------| ------| ------| ------| ------|
| 2023  | 1     | 1     | 0     | 0     | -10   |
| 2023  | 1     | 1     | 0     | 0     | -10.1 |        
| ...   | ...   | ...   | ...   | ...   | ...   |

where YYYY is the year, MM is the month, DD is the day, hh is the hour and mm is the minute.

## Model parameters
The model parameters are defined in `config.yml`.

## Model settings
Name of model experiment and spatial discretization are defined in `config.yml`.

## Model output
The variables written to the output files are defined in `config.yml`. Available variables
are listed [here](https://roger.readthedocs.io/en/latest/reference/variables.html#available-variables). Generally, water age and tracer concentration variables
should be defined in `OUTPUT_AVERAGE`.

# Workflow

The following workflow briefly describes the model application:

1. Prepare the meteorological input data (see Meteorological input data).

2. If required data is ready, the following script runs the simulations:

```
python svat_oxygen18.py
```

3. After calculation is done, the simulation results can be merged in a single NetCDF-file:
```
python merge_output.py
```

4. (Optional) The following script converts the model output from NetCDF to csv.
```
python netcdf_to_csv.py
```

5. Figures and tables can be produced with the following script:
```
python make_figures_and_tables.py
```