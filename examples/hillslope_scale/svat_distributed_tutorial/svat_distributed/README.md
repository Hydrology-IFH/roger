# Tutorial for the spatially-distributed SVAT-model
SVAT indicates that only vertical processes are considered.

- `input`: Contains input data for a 3-year period
- `config.yml`: File to set output variables
- `write_parameters.py`: Writes model parameters to netcdf
- `svat.py`: SVAT-model setup
- `svat.sh`: Executes `svat.py` on BinAC computing cluster
- `merge_output.py`: Merges model output into a single file


## Workflow
```
conda activate roger
python write_parameters.py
python svat.py
python merge_output.py
```