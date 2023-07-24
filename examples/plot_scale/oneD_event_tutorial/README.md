# Tutorial for the 1D-event-model

The tutorial simulates a single grid cell.

1D indicates indicates that vertical processes and lateral processes are considered.

- `input`: Contains input data
- `output`: Contains output data
- `figures`: Contains figures
- `config.yml`: File to set parameters and output variables
- `oneD_event.py`: 1D-model setup for a single event
- `merge_output.py`: Merges output into a single NetCDF file
- `netcdf_to_csv.py`: Writes output to csv files
- `make_figures_and_tables.py`: Produces figures and tables

# Date requirements

The following information is required to run the model. 

## Meteorological input data
The required meteorological input data is loaded from the input folder. The input folder should contain a tab-delimited text file
for precipitation (`PREC.txt`; 10 minutes time steps) and air temperature (`TA.txt`; daily time steps).

Format of `PREC.txt` (PREC in mm/10 minutes):
| YYYY  | MM    | DD    | hh    | mm    | PREC  |
| ------| ------| ------| ------| ------| ------|
| 2023  | 1     | 1     | 0     | 0     | 0     |
| 2023  | 1     | 1     | 0     | 10    | 0.3   |        
| 2023  | 1     | 1     | 0     | 20    | 1.0   |
| 2023  | 1     | 1     | 0     | 30    | 0.5   |        
| 2023  | 1     | 1     | 0     | 40    | 0.4   |
| 2023  | 1     | 1     | 0     | 50    | 0.7   |        
| 2023  | 1     | 1     | 1     | 0     | 0.6   |
| ...   | ...   | ...   | ...   | ...   | ...   |

Format of`TA.txt` (TA in degC):
| YYYY  | MM    | DD    | hh    | mm    | TA    |
| ------| ------| ------| ------| ------| ------|
| 2023  | 1     | 1     | 0     | 0     | 2     |
| 2023  | 1     | 1     | 0     | 0     | 3     |        
| ...   | ...   | ...   | ...   | ...   | ...   |

where YYYY is the year, MM is the month, DD is the day, hh is the hour and mm is the minute.

## Model parameters
The model parameters are defined in `config.yml`.

## Model settings
Name of model experiment and spatial discretization are defined in `config.yml`.

## Model output
The variables written to the output files are defined in `config.yml`. Available variables
are listed [here](https://roger.readthedocs.io/en/latest/reference/variables.html#available-variables). Generally, storage variables
should be defined in `OUTPUT_COLLECT` and flux variables in `OUTPUT_RATE`.

# Workflow

The following workflow briefly describes the model application:

1. Prepare the meteorological input data (see Meteorological input data).

2. If required data is ready, the following script runs the simulations:

```
python oneD_event.py
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

