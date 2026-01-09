import os
import glob
import h5netcdf
import xarray as xr
import datetime
from pathlib import Path
import numpy as onp
import click
import roger


@click.command("main")
def main():
    base_path = Path(__file__).parent
    # directory of results
    base_path_output = Path("/Volumes/LaCie/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed") / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)

    irrigation_scenarios = ["no-irrigation"]
    catch_crop_scenarios = ["no-yellow-mustard"]
    soil_compaction_scenarios = ["no-soil-compaction"]
    meteo_stress = ["base"]
    magnitudes = [2]
    durations = [0]

    meteo_stress_tests = []
    for meteo in meteo_stress:
        if meteo == "base" or meteo == "base_2000-2024":
            meteo_stress_tests.append(meteo_stress_tests.append(f"{meteo}-magnitude0-duration0"))
        else:
            for magnitude in magnitudes:
                for duration in durations:
                    meteo_stress_tests.append(f"{meteo}-magnitude{magnitude}-duration{duration}")

    for meteo_stress_test in meteo_stress_tests:
        for irrigation_scenario in irrigation_scenarios:
            for catch_crop_scenario in catch_crop_scenarios:
                for soil_compaction_scenario in soil_compaction_scenarios:
                    # merge model output into single file
                    path = str(base_path_output / f"ONEDCROP_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}.*.nc")
                    diag_files = glob.glob(path)
                    if diag_files:
                        output_file = base_path_output / f"ONEDCROP_{meteo_stress_test}_{irrigation_scenario}_{catch_crop_scenario}_{soil_compaction_scenario}.nc"
                        if not os.path.exists(output_file):
                            with h5netcdf.File(output_file, "w", decode_vlen_strings=False) as f:
                                f.attrs.update(
                                    date_created=datetime.datetime.today().isoformat(),
                                    title="RoGeR simulations for the Dreisam-Moehlin-Neumagen catchment",
                                    institution="University of Freiburg, Chair of Hydrology",
                                    references="",
                                    comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                                    model_structure="ONED model with free drainage and explicit crop growth dynamics",
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
                                    print(f"Merging file {dfs}...")
                                    with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                                        if not f.dimensions:
                                            f.dimensions = dict_dim
                                            v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                                            v.attrs["long_name"] = "x"
                                            v.attrs["units"] = "m"
                                            v[:] = onp.arange(dict_dim["x"]) * 25
                                            v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                                            v.attrs["long_name"] = "y"
                                            v.attrs["units"] = "m"
                                            v[:] = onp.arange(dict_dim["y"]) * 25
                                            v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                                            var_obj = df.variables.get("Time")
                                            v.attrs.update(time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"])
                                            v[:] = time
                                    with xr.open_dataset(dfs, engine="h5netcdf") as ds:
                                        for key in list(df.variables.keys()):
                                            var_obj = ds[key]
                                            if key not in list(f.dimensions.keys()) and var_obj.ndim == 3:
                                                print(f"Merging variable {key}...")
                                                v = f.create_variable(key, ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                                                vals = var_obj.values
                                                v[:, :, :] = vals.swapaxes(0, 2)
                                                v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
    return


if __name__ == "__main__":
    main()