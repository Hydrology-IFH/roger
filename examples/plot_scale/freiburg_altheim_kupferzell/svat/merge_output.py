from pathlib import Path
import os
import glob
import h5netcdf
import datetime
import numpy as onp

base_path = Path(__file__).parent

# identifiers for simulations
locations = ["freiburg", "altheim", "kupferzell"]
land_cover_scenarios = ["corn", "corn_catch_crop", "crop_rotation"]
climate_scenarios = ["CCCma-CanESM2_CCLM4-8-17", "MPI-M-MPI-ESM-LR_RCA4"]
periods = ["1985-2014", "2030-2059", "2070-2099"]

# merge model output into single file
for location in locations:
    for land_cover_scenario in land_cover_scenarios:
        for climate_scenario in climate_scenarios:
            for period in periods:
                path = str(
                    base_path.parent
                    / "output"
                    / "svat"
                    / f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.*.nc"
                )
                output_hm_file = (
                    base_path.parent
                    / "output"
                    / "svat"
                    / f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc"
                )
                if not os.path.exists(output_hm_file):
                    diag_files = glob.glob(path)
                    with h5netcdf.File(output_hm_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR simulations at {location}",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="SVAT model with free drainage",
                        )
                        # collect dimensions
                        for dfs in diag_files:
                            with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                                f.attrs.update(roger_version=df.attrs["roger_version"])
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
                                    v.attrs["long_name"] = "model run"
                                    v.attrs["units"] = ""
                                    v[:] = onp.arange(dict_dim["x"])
                                    v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                                    v.attrs["long_name"] = ""
                                    v.attrs["units"] = ""
                                    v[:] = onp.arange(dict_dim["y"])
                                    v = f.create_variable(
                                        "Time", ("Time",), float, compression="gzip", compression_opts=1
                                    )
                                    var_obj = df.variables.get("Time")
                                    v.attrs.update(
                                        time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"]
                                    )
                                    v[:] = time
                                for var_sim in list(df.variables.keys()):
                                    var_obj = df.variables.get(var_sim)
                                    if (
                                        var_sim not in list(f.dimensions.keys())
                                        and var_obj.ndim == 3
                                        and var_obj.shape[0] > 2
                                    ):
                                        v = f.create_variable(
                                            var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                                        )
                                        vals = onp.array(var_obj)
                                        v[:, :, :] = vals.swapaxes(0, 2)
                                        v.attrs.update(
                                            long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"]
                                        )
                                    elif (
                                        var_sim not in list(f.dimensions.keys())
                                        and var_obj.ndim == 3
                                        and var_obj.shape[0] <= 2
                                    ):
                                        v = f.create_variable(
                                            var_sim, ("x", "y"), float, compression="gzip", compression_opts=1
                                        )
                                        vals = onp.array(var_obj)
                                        v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                                        v.attrs.update(
                                            long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"]
                                        )

for location in locations:
    for climate_scenario in climate_scenarios:
        for period in periods:
            path = str(base_path.parent / "output" / "svat" / f"SVAT_{location}_grass_{climate_scenario}_{period}.*.nc")
            output_hm_file = (
                base_path.parent / "output" / "svat" / f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
            )
            if not os.path.exists(output_hm_file):
                diag_files = glob.glob(path)
                with h5netcdf.File(output_hm_file, "w", decode_vlen_strings=False) as f:
                    f.attrs.update(
                        date_created=datetime.datetime.today().isoformat(),
                        title=f"RoGeR simulations at {location}",
                        institution="University of Freiburg, Chair of Hydrology",
                        references="",
                        comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                        model_structure="SVAT model with free drainage",
                    )
                    # collect dimensions
                    for dfs in diag_files:
                        with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                            f.attrs.update(roger_version=df.attrs["roger_version"])
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
                                v.attrs["long_name"] = "model run"
                                v.attrs["units"] = ""
                                v[:] = onp.arange(dict_dim["x"])
                                v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                                v.attrs["long_name"] = ""
                                v.attrs["units"] = ""
                                v[:] = onp.arange(dict_dim["y"])
                                v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                                var_obj = df.variables.get("Time")
                                v.attrs.update(time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"])
                                v[:] = time
                            for var_sim in list(df.variables.keys()):
                                var_obj = df.variables.get(var_sim)
                                if (
                                    var_sim not in list(f.dimensions.keys())
                                    and var_obj.ndim == 3
                                    and var_obj.shape[0] > 2
                                ):
                                    v = f.create_variable(
                                        var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                                    )
                                    vals = onp.array(var_obj)
                                    v[:, :, :] = vals.swapaxes(0, 2)
                                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                                elif (
                                    var_sim not in list(f.dimensions.keys())
                                    and var_obj.ndim == 3
                                    and var_obj.shape[0] <= 2
                                ):
                                    v = f.create_variable(
                                        var_sim, ("x", "y"), float, compression="gzip", compression_opts=1
                                    )
                                    vals = onp.array(var_obj)
                                    v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
