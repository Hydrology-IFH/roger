from pathlib import Path
import os
import glob
import h5netcdf
import datetime
import numpy as onp
from cftime import num2date
import pandas as pd
import click
import roger
import yaml


@click.command("main")
def main():
    base_path = Path(__file__).parent.parent
    dir_name = os.path.basename(str(Path(__file__).parent))

    # load the configuration file
    with open(base_path / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    # load the subregions and crop rotations
    df_subregions_crop_rotations = pd.read_csv(base_path / "subregions_crop_rotations.csv", sep=";")

    # identifiers of the simulations
    locations = df_subregions_crop_rotations.loc[:, "subregion"].values.astype(str).tolist()
    crop_rotation_scenarios = df_subregions_crop_rotations.loc[:, "crop_rotation_type"].values.astype(str).tolist()
    stress_tests_meteo = config["climate_scenarios"]

    # merge model output into single file
    for stress_test_meteo in stress_tests_meteo:
        click.echo(f"Processing {stress_test_meteo}...")
        if stress_test_meteo == "base":
            durations_magnitudes = [(0, 0)]
        elif stress_test_meteo in ["spring-drought", "summer-drought"]:
            durations_magnitudes = [(3, 0), (3, 2)]
        elif stress_test_meteo == "long-term":
            durations_magnitudes = [(0, 2)]
        for duration_magnitude in durations_magnitudes:
            duration = duration_magnitude[0]
            magnitude = duration_magnitude[1]
            click.echo(f"Processing duration {duration} cosequent years, magnitude {magnitude}...")
            for location, crop_rotation_scenario in zip(locations, crop_rotation_scenarios):
                click.echo(f"Processing location {location} and crop rotation {crop_rotation_scenario}...")
                crop_rotation_scenario1 = crop_rotation_scenario.replace("-", " ").replace("_", ", ")
                path = str(base_path / "output" / dir_name / f"SVATCROP_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}.*.nc")
                output_hm_file = base_path / "output" / dir_name / f"SVATCROP_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}.nc"
                if not os.path.exists(output_hm_file):
                    diag_files = glob.glob(path)
                    if diag_files:
                        with h5netcdf.File(output_hm_file, "w", decode_vlen_strings=False) as f:
                            f.attrs.update(
                                date_created=datetime.datetime.today().isoformat(),
                                title=f"RoGeR simulations using DWD station {location} with the {stress_test_meteo} (duration {duration} and magnitude {magnitude}) climate scenario as input and {crop_rotation_scenario1} as crop rotation, irrigation and soil compaction.",
                                institution="University of Freiburg, Chair of Hydrology",
                                references="",
                                comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                                model_structure="SVAT model with free drainage and crop phenology",
                                roger_version=f"{roger.__version__}",
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
                                            vals = onp.array(var_obj, dtype=float)
                                            v[:, :, :] = vals.swapaxes(0, 2)
                                            v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                                        elif (
                                            var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] <= 2
                                        ):
                                            v = f.create_variable(
                                                var_sim, ("x", "y"), float, compression="gzip", compression_opts=1
                                            )
                                            vals = onp.array(var_obj, dtype=float)
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
                            # remove diagnostic files
                            for dfs in diag_files:
                                if os.path.exists(dfs):
                                    os.remove(dfs)
    return


if __name__ == "__main__":
    main()