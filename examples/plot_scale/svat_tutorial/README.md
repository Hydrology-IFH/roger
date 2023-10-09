# Tutorial for the SVAT-model

The tutorial simulates a single grid cell.

SVAT indicates that only vertical processes are considered.

- `input`: Contains input data
- `output`: Contains output data
- `figures`: Contains figures
- `config.yml`: File to set parameters and output variables
- `parameters.csv`: Model parameters
- `svat.py`: SVAT-model setup
- `merge_output.py`: Merges output into a single NetCDF file
- `netcdf_to_csv.py`: Writes output to csv files
- `make_figures_and_tables.py`: Produces figures and tables

# Date requirements

The following information is required to run the model. 

## Meteorological input data
The required meteorological input data is loaded from the input folder. The input folder should contain a tab-delimited text file
for precipitation (`PREC.txt`; 10 minutes time steps), air temperature (`TA.txt`; daily time steps) and potential evapotranspiration (`PET.txt`; daily time steps).

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

Format of `PET.txt` (PET in mm/day):
| YYYY  | MM    | DD    | hh    | mm    | PET   |
| ------| ------| ------| ------| ------| ------|
| 2023  | 1     | 1     | 0     | 0     | 2     |
| 2023  | 1     | 1     | 0     | 0     | 2.1   |        
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
python svat.py
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

# Dockerize SVAT

We differ between:

- Dockerfile: Blueprint for building images
- Image: Template for running containers
- Container: Running process with the packaged project

## 1. Download the image or build the Docker image
We recommend to download the image from Docker Hub
```console
$ docker pull roger-svat-tutorial:latest
```

Alternatively, the image can be locally build:
```console
$ dockerize_svat_tutorial.sh
```

## 2. Run the Docker image (starts the container)

Move to the `svat_tutorial` folder using the `cd` command and run the container:

```console
$ docker run --rm -it --mount=type=bind,source="$(pwd)",target=/roger/examples/plot_scale/svat_tutorial roger-svat-tutorial
```

or set the path to the project folder manually using `SRC_PATH`:
```console
$ SRC_PATH=path_to_folder docker run --rm -it --mount=type=bind,source=SRC_PATH,target=/roger/examples/plot_scale/svat_tutorial roger-svat-tutorial
```

# Basic model interface (BMI)
Runs the simulation using the Basic model interface (BMI):
```
python bmiroger_svat.py
```