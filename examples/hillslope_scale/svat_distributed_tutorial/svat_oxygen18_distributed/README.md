# Tutorial for the spatially-distributed SVAT-Oxygen18-model
SVAT indicates that only vertical processes are considered.

- `input`: Contains input data for a 3-year period
- `config.yml`: File to set model structure, initial conditions and output variables
- `write_parameters.py`: Writes model parameters to netcdf
- `svat_oxygen18.py`: SVAT-model setup to simulate transport of oxygen18
- `svat_oxygen18.sh`: Executes `svat_oxygen18.py` on BinAC computing cluster
- `post_processing.py`: Produces figures and tables from data of the modelling experiment
