from pathlib import Path
import os
import glob
import datetime
import h5netcdf
import numpy as onp
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
# merge results into single file
path = str(base_path_output / "SVATOXYGEN18_power_deterministic.*.nc")
diag_files = glob.glob(path)
states_tm_file = base_path_output / "SVATOXYGEN18_power_deterministic.nc"
if not os.path.exists(states_tm_file):
    with h5netcdf.File(states_tm_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR simulations with power law distribution function as SAS function of a rectangular grassland",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
            model_structure="SVAT power law transport model with free drainage",
            sas_solver="deterministic",
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
                        "ages": len(df.variables["ages"]),
                        "nages": len(df.variables["nages"]),
                        "n_sas_params": len(df.variables["n_sas_params"]),
                    }
                    time = onp.array(df.variables.get("Time"))
        for dfs in diag_files:
            with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                if not f.dimensions:
                    f.dimensions = dict_dim
                    v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                    v.attrs["long_name"] = "model run"
                    v.attrs["units"] = ""
                    v[:] = onp.arange(dict_dim["x"])
                    v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                    v.attrs["long_name"] = ""
                    v.attrs["units"] = ""
                    v[:] = onp.arange(dict_dim["y"])
                    v = f.create_variable("ages", ("ages",), float, compression="gzip", compression_opts=1)
                    v.attrs["long_name"] = "Water ages"
                    v.attrs["units"] = "days"
                    v[:] = onp.arange(1, dict_dim["ages"] + 1)
                    v = f.create_variable("nages", ("nages",), float, compression="gzip", compression_opts=1)
                    v.attrs["long_name"] = "Water ages (cumulated)"
                    v.attrs["units"] = "days"
                    v[:] = onp.arange(0, dict_dim["nages"])
                    v = f.create_variable(
                        "n_sas_params", ("n_sas_params",), float, compression="gzip", compression_opts=1
                    )
                    v.attrs["long_name"] = "Number of SAS parameters"
                    v.attrs["units"] = ""
                    v[:] = onp.arange(0, dict_dim["n_sas_params"])
                    v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                    var_obj = df.variables.get("Time")
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"])
                    v[:] = time
                for var_sim in list(df.variables.keys()):
                    var_obj = df.variables.get(var_sim)
                    if (
                        var_sim not in list(dict_dim.keys())
                        and ("Time", "y", "x") == var_obj.dimensions
                        and var_obj.shape[0] > 2
                    ):
                        v = f.create_variable(
                            var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                        )
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals.swapaxes(0, 2)
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                        del var_obj, vals
                    elif (
                        var_sim not in list(dict_dim.keys())
                        and ("Time", "y", "x") == var_obj.dimensions
                        and var_obj.shape[0] <= 2
                    ):
                        v = f.create_variable(var_sim, ("x", "y"), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                        del var_obj, vals
                    elif (
                        var_sim not in list(dict_dim.keys())
                        and ("Time", "n_sas_params", "y", "x") == var_obj.dimensions
                    ):
                        v = f.create_variable(
                            var_sim, ("x", "y", "n_sas_params"), float, compression="gzip", compression_opts=1
                        )
                        vals = onp.array(var_obj)
                        vals = vals.swapaxes(0, 3)
                        vals = vals.swapaxes(1, 2)
                        v[:, :, :] = vals[:, :, :, 0]
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                        del var_obj, vals
                    elif var_sim not in list(dict_dim.keys()) and ("Time", "ages", "y", "x") == var_obj.dimensions:
                        v = f.create_variable(
                            var_sim, ("x", "y", "Time", "ages"), float, compression="gzip", compression_opts=1
                        )
                        vals = onp.array(var_obj)
                        vals = vals.swapaxes(0, 3)
                        vals = vals.swapaxes(1, 2)
                        vals = vals.swapaxes(2, 3)
                        v[:, :, :, :] = vals
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                        del var_obj, vals
                    elif var_sim not in list(dict_dim.keys()) and ("Time", "nages", "y", "x") == var_obj.dimensions:
                        v = f.create_variable(
                            var_sim, ("x", "y", "Time", "nages"), float, compression="gzip", compression_opts=1
                        )
                        vals = onp.array(var_obj)
                        vals = vals.swapaxes(0, 3)
                        vals = vals.swapaxes(1, 2)
                        vals = vals.swapaxes(2, 3)
                        v[:, :, :, :] = vals
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                        del var_obj, vals
