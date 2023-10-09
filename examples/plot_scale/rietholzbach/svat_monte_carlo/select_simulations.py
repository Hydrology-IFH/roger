import glob
import os
from pathlib import Path
import datetime
from cftime import num2date
import h5netcdf
import xarray as xr
import pandas as pd
from de import de
import numpy as onp
import click
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
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
    # directory of results
    base_path_results = base_path / "results"
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    file = base_path_figs / "params_metrics.txt"
    df_params_metrics = pd.read_csv(file, header=0, index_col=False, sep="\t")

    # select best model run
    idx_best1 = df_params_metrics["KGE_multi"].idxmax()

    # write states of best simulation
    click.echo("Write best simulation ...")
    states_hm_mc_file = base_path / "states_hm_monte_carlo.nc"
    states_hm_file = base_path / "states_hm1.nc"
    with h5netcdf.File(states_hm_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR best Monte Carlo simulation at Rietholzbach Lysimeter site",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
            model_structure="SVAT model with free drainage",
        )
        with h5netcdf.File(states_hm_mc_file, "r", decode_vlen_strings=False) as df:
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
    # write states of best model run
    states_hm_mc_file = base_path / "states_hm_monte_carlo.nc"
    states_hm_file = base_path / "states_hm10.nc"
    with h5netcdf.File(states_hm_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR best 10 Monte Carlo simulations at Rietholzbach lysimeter site",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
            model_structure="SVAT model with free drainage",
        )
        with h5netcdf.File(states_hm_mc_file, "r", decode_vlen_strings=False) as df:
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

    # select best 100 simulations
    click.echo("Write best 100 simulations ...")
    df_params_metrics100 = df_params_metrics.copy()
    df_params_metrics100.loc[:, "id"] = range(len(df_params_metrics100.index))
    df_params_metrics100 = df_params_metrics100.sort_values(by=["KGE_multi"], ascending=False)
    idx_best100 = df_params_metrics100.loc[: df_params_metrics100.index[99], "id"].values.tolist()
    # write states of best model run
    states_hm_mc_file = base_path / "states_hm_monte_carlo.nc"
    states_hm_file = base_path / "states_hm100.nc"
    with h5netcdf.File(states_hm_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR best 100 Monte Carlo simulations at Rietholzbach lysimeter site",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
            model_structure="SVAT model with free drainage",
        )
        with h5netcdf.File(states_hm_mc_file, "r", decode_vlen_strings=False) as df:
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
    return


if __name__ == "__main__":
    main()
