from pathlib import Path
import os
import glob
import h5netcdf
import datetime
import numpy as onp
from cftime import num2date
import pandas as pd
import click


@click.command("main")
def main():
    base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh")

    # identifiers for simulations
    locations = ["freiburg", "lahr", "muellheim", 
                 "stockach", "gottmadingen", "weingarten",
                 "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
                 "ehingen-kirchen", "merklingen", "hayingen",
                 "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
    locations = [
        "freiburg"
    ]
    crop_rotation_scenarios = ["winter-wheat_clover",
                               "winter-wheat_silage-corn",
                               "summer-wheat_winter-wheat",
                               "summer-wheat_clover_winter-wheat",
                               "winter-wheat_clover_silage-corn",
                               "winter-wheat_sugar-beet_silage-corn",
                               "summer-wheat_winter-wheat_silage-corn",
                               "summer-wheat_winter-wheat_winter-rape",
                               "winter-wheat_winter-rape",
                               "winter-wheat_soybean_winter-rape",
                               "sugar-beet_winter-wheat_winter-barley", 
                               "grain-corn_winter-wheat_winter-rape", 
                               "grain-corn_winter-wheat_winter-barley",
                               "grain-corn_winter-wheat_clover",
                               "winter-wheat_silage-corn_yellow-mustard",
                               "summer-wheat_winter-wheat_yellow-mustard",
                               "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                               "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                               "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                               "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                               "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                               "grain-corn_winter-wheat_winter-barley_yellow-mustard"]

    # merge model output into single file
    for location in locations:
        for crop_rotation_scenario in crop_rotation_scenarios:
            try:
                crop_rotation_scenario1 = crop_rotation_scenario.replace("-", " ").replace("_", ", ")
                path = str(base_path / "output" / "svat_crop" / f"SVATCROP_{location}_{crop_rotation_scenario}.*.nc")
                output_hm_file = base_path / "output" / "svat_crop" / f"SVATCROP_{location}_{crop_rotation_scenario}.nc"
                if not os.path.exists(output_hm_file):
                    diag_files = glob.glob(path)
                    with h5netcdf.File(output_hm_file, "w", decode_vlen_strings=False) as f:
                        f.attrs.update(
                            date_created=datetime.datetime.today().isoformat(),
                            title=f"RoGeR simulations at {location} with {crop_rotation_scenario1} for crop rotation",
                            institution="University of Freiburg, Chair of Hydrology",
                            references="",
                            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                            model_structure="SVAT model with free drainage and crop phenology",
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
                                    time_origin = df.variables['Time'].attrs['time_origin']
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
                                    if var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] > 2:
                                        v = f.create_variable(
                                            var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                                        )
                                        vals = onp.array(var_obj)
                                        v[:, :, :] = vals.swapaxes(0, 2)
                                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                                    elif (
                                        var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] <= 2
                                    ):
                                        v = f.create_variable(
                                            var_sim, ("x", "y"), float, compression="gzip", compression_opts=1
                                        )
                                        vals = onp.array(var_obj)
                                        v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                        # add year and day of year for nitrate transport model
                        dates1 = num2date(
                            time,
                            units=f"days since {time_origin}",
                            calendar="standard",
                            only_use_cftime_datetimes=False,
                        )
                        dates = pd.to_datetime(dates1)
                        vals = onp.array(dates.year)
                        v = f.create_variable(
                            "year", ("Time",), float, compression="gzip", compression_opts=1
                        )
                        v[:] = onp.array(dates.year)
                        v.attrs.update(long_name="Year", units="")
                        vals = onp.array(dates.year)
                        v = f.create_variable(
                            "doy", ("Time",), float, compression="gzip", compression_opts=1
                        )
                        v[:] = onp.array(dates.day_of_year)
                        v.attrs.update(long_name="Day of year", units="")
            except:
                print(f"sbatch --partition=single svat_crop_{location}_{crop_rotation_scenario}_slurm.sh")
                continue
    return


if __name__ == "__main__":
    main()
