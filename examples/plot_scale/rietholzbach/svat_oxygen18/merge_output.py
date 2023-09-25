from pathlib import Path
import os
import glob
import datetime
import h5netcdf
import numpy as onp
import click


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    transport_models = [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
    ]
    # merge model output into a single file
    for tm in transport_models:
        tms = tm.replace(" ", "_")
        path = str(base_path_output / f"SVATTRANSPORT_{tms}_deterministic.*.nc")
        diag_files = glob.glob(path)
        states_tm_file = base_path_output / f"states_{tms}.nc"
        if not os.path.exists(states_tm_file):
            click.echo(f"Merge output files of {tm} into {states_tm_file.as_posix()}")
            with h5netcdf.File(states_tm_file, "w", decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title=f"RoGeR {tm} transport model results at Rietholzbach lysimeter site",
                    institution="University of Freiburg, Chair of Hydrology",
                    references="",
                    comment=f"SVAT {tm} transport model with free drainage",
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
                            if var_sim not in list(dict_dim.keys()) and ("Time", "y", "x") == var_obj.dimensions:
                                v = f.create_variable(
                                    var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                                )
                                vals = onp.array(var_obj)
                                v[:, :, :] = vals.swapaxes(0, 2)
                                v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
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
                            elif (
                                var_sim not in list(dict_dim.keys())
                                and ("Time", "ages", "y", "x") == var_obj.dimensions
                            ):
                                v = f.create_variable(
                                    var_sim, ("x", "y", "Time", "ages"), float, compression="gzip", compression_opts=1
                                )
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                vals = vals.swapaxes(2, 3)
                                v[:, :, :, :] = vals
                                v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                            elif (
                                var_sim not in list(dict_dim.keys())
                                and ("Time", "nages", "y", "x") == var_obj.dimensions
                            ):
                                v = f.create_variable(
                                    var_sim, ("x", "y", "Time", "nages"), float, compression="gzip", compression_opts=1
                                )
                                vals = onp.array(var_obj)
                                vals = vals.swapaxes(0, 3)
                                vals = vals.swapaxes(1, 2)
                                vals = vals.swapaxes(2, 3)
                                v[:, :, :, :] = vals
                                v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
    return


if __name__ == "__main__":
    main()
