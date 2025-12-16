import os
from pathlib import Path
import datetime
import h5netcdf
import pandas as pd
import numpy as onp
import click
import matplotlib as mpl
import seaborn as sns

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

sns.set_style("ticks")


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    file = base_path_figs / "params_metrics.txt"
    df_params_metrics = pd.read_csv(file, header=0, index_col=False, sep="\t")

    # select best model run
    idx_best1 = df_params_metrics["KGE_multi"].idxmax()
    # write .txt-file
    file = base_path_figs / "params_metrics_1.txt"
    df_params_metrics.iloc[idx_best1, :].to_frame().to_csv(file, header=True, index=False, sep="\t")

    # write file of best simulation
    click.echo("Write best simulation ...")
    hm_mc_file = base_path / "output" / "SVAT.nc"
    hm_file = base_path / "output" / "SVAT_best1.nc"
    with h5netcdf.File(hm_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR best Monte Carlo simulation at Rietholzbach Lysimeter site",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
            model_structure="SVAT model with free drainage",
        )
        with h5netcdf.File(hm_mc_file, "r", decode_vlen_strings=False) as df:
            f.attrs.update(roger_version=df.attrs["roger_version"])
            # set dimensions with a dictionary
            dict_dim = {"x": 1, "y": 1, "Time": len(df.variables["Time"])}
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
                v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                var_obj = df.variables.get("Time")
                v.attrs.update(time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"])
                v[:] = onp.array(var_obj)
            for var_sim in list(df.variables.keys()):
                var_obj = df.variables.get(var_sim)
                if var_sim not in list(f.dimensions.keys()) and ("x", "y", "Time") == var_obj.dimensions:
                    v = f.create_variable(var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :, :] = vals[idx_best1, :, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                elif var_sim not in list(f.dimensions.keys()) and ("x", "y") == var_obj.dimensions:
                    v = f.create_variable(var_sim, ("x", "y"), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :] = vals[idx_best1, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])

    # select best 10 simulations
    click.echo("Write best 10 simulations ...")
    df_params_metrics10 = df_params_metrics.copy()
    df_params_metrics10.loc[:, "id"] = range(len(df_params_metrics10.index))
    df_params_metrics10 = df_params_metrics10.sort_values(by=["KGE_multi"], ascending=False)
    idx_best10 = df_params_metrics10.loc[: df_params_metrics10.index[9], "id"].values.tolist()
    # write .txt-file
    file = base_path_figs / "params_metrics_10.txt"
    df_params_metrics10.iloc[:10, :].to_csv(file, header=True, index=False, sep="\t")
    # write file of best model run
    hm_mc_file = base_path / "output" / "SVAT.nc"
    hm_file = base_path / "output" / "SVAT_best10.nc"
    with h5netcdf.File(hm_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR best 10 Monte Carlo simulations at Rietholzbach lysimeter site",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
            model_structure="SVAT model with free drainage",
        )
        with h5netcdf.File(hm_mc_file, "r", decode_vlen_strings=False) as df:
            f.attrs.update(roger_version=df.attrs["roger_version"])
            # set dimensions with a dictionary
            dict_dim = {"x": len(idx_best10), "y": 1, "Time": len(df.variables["Time"])}
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
                v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                var_obj = df.variables.get("Time")
                v.attrs.update(time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"])
                v[:] = onp.array(var_obj)
            for var_sim in list(df.variables.keys()):
                var_obj = df.variables.get(var_sim)
                if var_sim not in list(f.dimensions.keys()) and ("x", "y", "Time") == var_obj.dimensions:
                    v = f.create_variable(var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :, :] = vals[idx_best10, :, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                elif var_sim not in list(f.dimensions.keys()) and ("x", "y") == var_obj.dimensions:
                    v = f.create_variable(var_sim, ("x", "y"), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :] = vals[idx_best10, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])

            v = f.create_variable("id", ("x", "y"), float, compression="gzip", compression_opts=1)
            v[:, 0] = onp.arange(0, 10)
            v.attrs.update(long_name="id", units="")

    # select best 100 simulations
    click.echo("Write best 100 simulations ...")
    df_params_metrics100 = df_params_metrics.copy()
    df_params_metrics100.loc[:, "id"] = range(len(df_params_metrics100.index))
    df_params_metrics100 = df_params_metrics100.sort_values(by=["KGE_multi"], ascending=False)
    idx_best100 = df_params_metrics100.loc[: df_params_metrics100.index[99], "id"].values.tolist()
    # write .txt-file
    file = base_path_figs / "params_metrics_100.txt"
    df_params_metrics100.iloc[:100, :].to_csv(file, header=True, index=False, sep="\t")
    # write file of best model run
    hm_mc_file = base_path / "output" / "SVAT.nc"
    hm_file = base_path / "output" / "SVAT_best100.nc"
    with h5netcdf.File(hm_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR best 100 Monte Carlo simulations at Rietholzbach lysimeter site",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
            model_structure="SVAT model with free drainage",
        )
        with h5netcdf.File(hm_mc_file, "r", decode_vlen_strings=False) as df:
            f.attrs.update(roger_version=df.attrs["roger_version"])
            # set dimensions with a dictionary
            dict_dim = {"x": len(idx_best100), "y": 1, "Time": len(df.variables["Time"])}
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
                v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                var_obj = df.variables.get("Time")
                v.attrs.update(time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"])
                v[:] = onp.array(var_obj)
            for var_sim in list(df.variables.keys()):
                var_obj = df.variables.get(var_sim)
                if var_sim not in list(f.dimensions.keys()) and ("x", "y", "Time") == var_obj.dimensions:
                    v = f.create_variable(var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :, :] = vals[idx_best100, :, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                elif var_sim not in list(f.dimensions.keys()) and ("x", "y") == var_obj.dimensions:
                    v = f.create_variable(var_sim, ("x", "y"), float, compression="gzip", compression_opts=1)
                    vals = onp.array(var_obj)
                    v[:, :] = vals[idx_best100, :]
                    v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])

            v = f.create_variable("id", ("x", "y"), float, compression="gzip", compression_opts=1)
            v[:, 0] = onp.arange(0, 100)
            v.attrs.update(long_name="id", units="")
    return


if __name__ == "__main__":
    main()
