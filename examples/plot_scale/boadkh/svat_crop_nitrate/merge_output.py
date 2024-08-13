from pathlib import Path
import os
import glob
import h5netcdf
import datetime
import numpy as onp
import click
import roger


@click.command("main")
def main():
    # base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh")
    base_path = Path("/pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh")


    # identifiers for simulations
    locations = ["freiburg", "lahr", "muellheim", 
                "stockach", "gottmadingen", "weingarten",
                "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
                "ehingen-kirchen", "merklingen", "hayingen",
                "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
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
                               "grain-corn_winter-wheat_winter-barley_yellow-mustard",
                               "miscanthus",
                               "bare-grass"]

    fertilization_intensities = ["low", "medium", "high"]

    # merge model output into single file
    for location in locations:
        for crop_rotation_scenario in crop_rotation_scenarios:
            for fertilization_intensity in fertilization_intensities:
                path = str(
                    base_path
                    / "output"
                    / "svat_crop_nitrate"
                    / f"SVATCROPNITRATE_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.*.nc"
                )
                output_tm_file = (
                    base_path
                    / "output"
                    / "svat_crop_nitrate"
                    / f"SVATCROPNITRATE_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.nc"
                )
                diag_files = glob.glob(path)
                if diag_files:
                    if not os.path.exists(output_tm_file):
                        with h5netcdf.File(output_tm_file, "w", decode_vlen_strings=False) as f:
                            crop_rotation_scenario1 = crop_rotation_scenario.replace("_", ", ")
                            f.attrs.update(
                                date_created=datetime.datetime.today().isoformat(),
                                title=f"RoGeR nitrate transport simulations at {location} with crop rotation scenario {crop_rotation_scenario1} and {fertilization_intensity} fertilization intensity",
                                institution="University of Freiburg, Chair of Hydrology",
                                references="",
                                comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                                model_structure="SVAT model with free drainage, crop phenology and time-variant power law distribution as SAS function with advective-dispersive parameters",
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
                                            "ages": len(df.variables["ages"]),
                                            "nages": len(df.variables["nages"]),
                                            "n_sas_params": len(df.variables["n_sas_params"]),
                                        }
                                        time = onp.array(df.variables.get("Time"))
                                    if not f.dimensions:
                                        f.dimensions = dict_dim
                                        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                                        v.attrs["long_name"] = "Number of model run"
                                        v.attrs["units"] = ""
                                        v[:] = onp.arange(dict_dim["x"])
                                        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                                        v.attrs["long_name"] = ""
                                        v.attrs["units"] = ""
                                        v[:] = onp.arange(dict_dim["y"])
                                        v = f.create_variable(
                                            "ages", ("ages",), float, compression="gzip", compression_opts=1
                                        )
                                        v.attrs["long_name"] = "Water ages"
                                        v.attrs["units"] = "days"
                                        v[:] = onp.arange(1, dict_dim["ages"] + 1)
                                        v = f.create_variable(
                                            "nages", ("nages",), float, compression="gzip", compression_opts=1
                                        )
                                        v.attrs["long_name"] = "Water ages (cumulated)"
                                        v.attrs["units"] = "days"
                                        v[:] = onp.arange(0, dict_dim["nages"])
                                        v = f.create_variable(
                                            "n_sas_params", ("n_sas_params",), float, compression="gzip", compression_opts=1
                                        )
                                        v.attrs["long_name"] = "Number of SAS parameters"
                                        v.attrs["units"] = ""
                                        v[:] = onp.arange(0, dict_dim["n_sas_params"])
                                        v = f.create_variable(
                                            "Time", ("Time",), float, compression="gzip", compression_opts=1
                                        )
                                        var_obj = df.variables.get("Time")
                                        v.attrs.update(
                                            time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"]
                                        )
                                        v[:] = time
                                    for var_sim in list(df.variables.keys()):
                                        _file = base_path / "output" / "svat_crop_nitrate" / f"SVATCROPNITRATE_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.collect.nc"
                                        if dfs == _file and var_sim == "C_q_ss":
                                            pass
                                        else:
                                            var_obj = df.variables.get(var_sim)
                                            if (
                                                var_sim not in list(dict_dim.keys())
                                                and ("Time", "y", "x") == var_obj.dimensions
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
                                                var_sim not in list(dict_dim.keys())
                                                and ("Time", "n_sas_params", "y", "x") == var_obj.dimensions
                                            ):
                                                v = f.create_variable(
                                                    var_sim,
                                                    ("x", "y", "n_sas_params"),
                                                    float,
                                                    compression="gzip",
                                                    compression_opts=1,
                                                )
                                                vals = onp.array(var_obj)
                                                vals = vals.swapaxes(0, 3)
                                                vals = vals.swapaxes(1, 2)
                                                v[:, :, :] = vals[:, :, :, 0]
                                                v.attrs.update(
                                                    long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"]
                                                )
                                            elif (
                                                var_sim not in list(dict_dim.keys())
                                                and ("Time", "ages", "y", "x") == var_obj.dimensions
                                            ):
                                                v = f.create_variable(
                                                    var_sim,
                                                    ("x", "y", "Time", "ages"),
                                                    float,
                                                    compression="gzip",
                                                    compression_opts=1,
                                                )
                                                vals = onp.array(var_obj)
                                                vals = vals.swapaxes(0, 3)
                                                vals = vals.swapaxes(1, 2)
                                                vals = vals.swapaxes(2, 3)
                                                v[:, :, :, :] = vals
                                                v.attrs.update(
                                                    long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"]
                                                )
                                            elif (
                                                var_sim not in list(dict_dim.keys())
                                                and ("Time", "nages", "y", "x") == var_obj.dimensions
                                            ):
                                                v = f.create_variable(
                                                    var_sim,
                                                    ("x", "y", "Time", "nages"),
                                                    float,
                                                    compression="gzip",
                                                    compression_opts=1,
                                                )
                                                vals = onp.array(var_obj)
                                                vals = vals.swapaxes(0, 3)
                                                vals = vals.swapaxes(1, 2)
                                                vals = vals.swapaxes(2, 3)
                                                v[:, :, :, :] = vals
                                                v.attrs.update(
                                                    long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"]
                                        )
                    else:
                        print(f"sbatch --partition=single svat_crop_nitrate_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert_slurm.sh")
    return


if __name__ == "__main__":
    main()
