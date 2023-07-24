import os
import glob
import h5netcdf
import datetime
from pathlib import Path
import numpy as onp
import yaml
import roger

base_path = Path(__file__).parent
# directory of results
base_path_output = base_path / "output"
if not os.path.exists(base_path_output):
    os.mkdir(base_path_output)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# load configuration file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    config = yaml.safe_load(file)

# merge model output into single file
path = str(base_path / f"{config['identifier']}.*.nc")
diag_files = glob.glob(path)
states_hm_file = base_path / f"{config['identifier']}.nc"
if not os.path.exists(states_hm_file):
    with h5netcdf.File(states_hm_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR simulations (Tutorial)",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
            model_structure="1D model with free drainage",
            roger_version=f"{roger.__version__}",
        )
        # collect dimensions
        for dfs in diag_files:
            with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                if not dfs.split("/")[-1].split(".")[1] == "constant":
                    dict_dim = {
                        "x": len(df.variables["x"]),
                        "y": len(df.variables["y"]),
                        "Time": len(df.variables["Time"]),
                    }
                    time = onp.array(df.variables.get("Time"))
        for dfs in diag_files:
            with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                if not f.dimensions:
                    f.dimensions = dict_dim
                    v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                    v.attrs["long_name"] = "x"
                    v.attrs["units"] = "m"
                    v[:] = onp.arange(dict_dim["x"]) * 5
                    v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                    v.attrs["long_name"] = "y"
                    v.attrs["units"] = "m"
                    v[:] = onp.arange(dict_dim["y"]) * 5
                    v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                    var_obj = df.variables.get("Time")
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"])
                    v[:] = time
                for key in list(df.variables.keys()):
                    var_obj = df.variables.get(key)
                    if key not in list(f.dimensions.keys()) and var_obj.ndim == 3:
                        v = f.create_variable(key, ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals.swapaxes(0, 2)
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
