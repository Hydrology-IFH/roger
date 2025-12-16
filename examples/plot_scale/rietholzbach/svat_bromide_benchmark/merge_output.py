from pathlib import Path
import os
import glob
import h5netcdf
import datetime
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

    # merge model output into single file
    years = onp.arange(1997, 2007).tolist()
    for transport_model_structure in [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant_advection-dispersion-power",
    ]:
        tms = transport_model_structure.replace("_", " ")
        tm_file= base_path / "output" / f"SVATBROMIDE_{transport_model_structure}_bromide_benchmark.nc"
        if not os.path.exists(tm_file):
            for year in years:
                path = str(
                    base_path / "output" / f"SVATBROMIDE_{transport_model_structure}_{year}_deterministic.*.nc"
                )
                diag_files = glob.glob(path)
                with h5netcdf.File(tm_file, "a", decode_vlen_strings=False) as f:
                    click.echo(f"Merge output files of {tms}-{year} into {tm_file.as_posix()}")
                    if f"{year}" not in list(f.groups.keys()):
                        f.create_group(f"{year}")
                    f.attrs.update(
                        date_created=datetime.datetime.today().isoformat(),
                        title="RoGeR transport simulations for virtual bromide experiments at Rietholzbach Lysimeter site",
                        institution="University of Freiburg, Chair of Hydrology",
                        references="",
                        comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                        model_structure="SVAT transport model with free drainage",
                        sas_solver="deterministic",
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
                    for dfs in diag_files:
                        with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                            if not f.groups[f"{year}"].dimensions:
                                f.groups[f"{year}"].dimensions = dict_dim
                                v = f.groups[f"{year}"].create_variable(
                                    "x", ("x",), float, compression="gzip", compression_opts=1
                                )
                                v.attrs["long_name"] = "Number of model run"
                                v.attrs["units"] = ""
                                v[:] = onp.arange(dict_dim["x"])
                                v = f.groups[f"{year}"].create_variable(
                                    "y", ("y",), float, compression="gzip", compression_opts=1
                                )
                                v.attrs["long_name"] = ""
                                v.attrs["units"] = ""
                                v[:] = onp.arange(dict_dim["y"])
                                v = f.groups[f"{year}"].create_variable(
                                    "ages", ("ages",), float, compression="gzip", compression_opts=1
                                )
                                v.attrs["long_name"] = "Water ages"
                                v.attrs["units"] = "days"
                                v[:] = onp.arange(1, dict_dim["ages"] + 1)
                                v = f.groups[f"{year}"].create_variable(
                                    "nages", ("nages",), float, compression="gzip", compression_opts=1
                                )
                                v.attrs["long_name"] = "Water ages (cumulated)"
                                v.attrs["units"] = "days"
                                v[:] = onp.arange(0, dict_dim["nages"])
                                v = f.groups[f"{year}"].create_variable(
                                    "Time", ("Time",), float, compression="gzip", compression_opts=1
                                )
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
                                    v = f.groups[f"{year}"].create_variable(
                                        var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                                    )
                                    vals = onp.array(var_obj)
                                    v[:, :, :] = vals.swapaxes(0, 2)
                                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                                elif (
                                    var_sim not in list(dict_dim.keys())
                                    and ("Time", "y", "x") == var_obj.dimensions
                                    and var_obj.shape[0] <= 2
                                ):
                                    v = f.groups[f"{year}"].create_variable(
                                        var_sim, ("x", "y"), float, compression="gzip", compression_opts=1
                                    )
                                    vals = onp.array(var_obj)
                                    v[:, :, :] = vals.swapaxes(0, 2)[:, :, 0]
                                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                                elif (
                                    var_sim not in list(dict_dim.keys())
                                    and ("Time", "n_sas_params", "y", "x") == var_obj.dimensions
                                ):
                                    v = f.groups[f"{year}"].create_variable(
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
                                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                                elif (
                                    var_sim not in list(dict_dim.keys())
                                    and ("Time", "ages", "y", "x") == var_obj.dimensions
                                ):
                                    v = f.groups[f"{year}"].create_variable(
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
                                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                                elif (
                                    var_sim not in list(dict_dim.keys())
                                    and ("Time", "nages", "y", "x") == var_obj.dimensions
                                ):
                                    v = f.groups[f"{year}"].create_variable(
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
                                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
    return


if __name__ == "__main__":
    main()
