# Modelling experiments with synthetically generated parameters
SVAT indicates that only vertical processes are active.

- `input`: Contains data of single year from two different meteorological stations
- `parameters.csv`: Contains synthetically generated parameters
- `svat.py`: SVAT-Model setup
- `post_processing.py`: Produces figures and tables from data of the modelling experiment
- `plot_numerical_error.py`: Plots the numerical error

## Worklfow

1. Run the SVAT model for a single year using meteorological data of the DWD station Breitnau
```
python svat.py --meteo-station breitnau
```
2. Run the SVAT model for a single year using meteorological data of the DWD station Ihringen
```
python svat.py --meteo-station ihringen
```
3. Merge the model output and plot the results
```
python post_processing.py
```
4. Plot the numerical error
```
python plot_numerical_error.py
```