import os
import h5netcdf
import datetime
from pathlib import Path
from cftime import num2date
import xarray as xr
import pandas as pd
import numpy as onp
import click
import roger

onp.random.seed(42)


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")

    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = Path(__file__).parent.parent / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        # directory of results
        base_path_output = base_path / "output" / "svat_crop_monte_carlo_reference"
        if not os.path.exists(base_path_output):
            os.mkdir(base_path_output)
        # directory of figures
        base_path_figs = Path(__file__).parent.parent / "figures" / "svat_crop_monte_carlo_reference"
        if not os.path.exists(base_path_figs):
            os.mkdir(base_path_figs)

        # load simulation
        path_sim = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        ds_sim = xr.open_dataset(path_sim, engine="h5netcdf")

        # assign date
        days_sim = ds_sim["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim = num2date(
            days_sim,
            units=f"days since {ds_sim['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim = ds_sim.assign_coords(date=("Time", date_sim))

        # load .txt-file
        file = base_path_output / f"params_eff_{lys_experiment}.txt"
        df_params_metrics = pd.read_csv(file, sep="\t")

        # calculate multi-objective efficiency
        df_params_metrics["E_multi"] = 0.7 * df_params_metrics["KGE_q_ss_perc_pet_2011-2017"] + 0.3 * (1 - ((1 - df_params_metrics["r_S_perc_pet_2011-2017"])**2 + (1 - df_params_metrics["KGE_alpha_S_perc_pet_2011-2017"])**2)**(0.5))

        # select best model run
        idx_best = df_params_metrics["E_multi"].idxmax()

        ds_sim = ds_sim.close()
        del ds_sim
        # write states of best model run
        path_sim = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        path_sim_1 = base_path_output / f"SVATCROP_{lys_experiment}_best_simulation.nc"
        with h5netcdf.File(path_sim_1, "a", decode_vlen_strings=False) as f:
            if lys_experiment not in list(f.keys()):
                f.create_group(lys_experiment)
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title="RoGeR best Monte Carlo simulation at Reckenholz Lysimeter site",
                institution="University of Freiburg, Chair of Hydrology",
                references="",
                comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                model_structure="SVAT model with free drainage and crop phenology/crop rotation",
                roger_version=f"{roger.__version__}",
            )
            with h5netcdf.File(path_sim, "r", decode_vlen_strings=False) as df:
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
                        v = f.create_variable(
                            var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                        )
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals[idx_best, :, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and ("x", "y") == var_obj.dimensions:
                        v = f.create_variable(var_sim, ("x", "y"), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :] = vals[idx_best, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])

        # select best 100 model runs
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()

        # write states of best model run
        path_sim = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        path_sim_100 = base_path_output / f"SVATCROP_{lys_experiment}_best_100_simulations.nc"
        with h5netcdf.File(path_sim_100, "a", decode_vlen_strings=False) as f:
            if lys_experiment not in list(f.keys()):
                f.create_group(lys_experiment)
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title=f"RoGeR best 100 Monte Carlo simulations at Reckenholz Lysimeter ({lys_experiment})",
                institution="University of Freiburg, Chair of Hydrology",
                references="",
                comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                model_structure="SVAT model with free drainage and crop phenology/crop rotation",
                roger_version=f"{roger.__version__}",
            )
            with h5netcdf.File(path_sim, "r", decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                dict_dim = {"x": 100, "y": 1, "Time": len(df.variables["Time"])}
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
                        v = f.create_variable(
                            var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                        )
                        vals = onp.array(var_obj)
                        for i, x in enumerate(idx_best100):
                            v[i, :, :] = vals[x, :, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and ("x", "y") == var_obj.dimensions:
                        v = f.create_variable(var_sim, ("x", "y"), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        for i, x in enumerate(idx_best100):
                            v[i, :, :] = vals[x, :, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
    return


if __name__ == "__main__":
    main()
