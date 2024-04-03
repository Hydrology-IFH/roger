import glob
import os
from pathlib import Path
import datetime
from cftime import num2date
import h5netcdf
import xarray as xr
import pandas as pd
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
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load simulation
    states_hm_mc_file = base_path / "output" / "states_hm_monte_carlo.nc"
    ds_sim = xr.open_dataset(states_hm_mc_file, engine="h5netcdf")
    # assign date
    days_sim = ds_sim["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim = num2date(
        days_sim,
        units=f"days since {ds_sim['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim = ds_sim.assign_coords(date=("Time", date_sim))

    # load observations (measured data)
    path_obs = Path(__file__).parent.parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
    # assign date
    days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_obs = num2date(
        days_obs,
        units=f"days since {ds_obs['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    # average observed soil water content of previous 5 days
    window = 5
    df_thetap = pd.DataFrame(index=date_obs, columns=["doy", "theta", "sc"])
    df_thetap.loc[:, "doy"] = df_thetap.index.day_of_year
    df_thetap.loc[:, "theta"] = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
    df_thetap.loc[df_thetap.index[window - 1] :, f"theta_avg{window}"] = (
        df_thetap.loc[:, "theta"].rolling(window=window).mean().iloc[window - 1 :].values
    )
    df_thetap.iloc[:window, 2] = onp.nan
    theta_lower = df_thetap.loc[:, f"theta_avg{window}"].quantile(0.1)
    theta_upper = df_thetap.loc[:, f"theta_avg{window}"].quantile(0.9)
    cond1 = df_thetap[f"theta_avg{window}"] < theta_lower
    cond2 = (df_thetap[f"theta_avg{window}"] >= theta_lower) & (df_thetap[f"theta_avg{window}"] < theta_upper)
    cond3 = df_thetap[f"theta_avg{window}"] >= theta_upper
    df_thetap.loc[cond1, "sc"] = 1  # dry
    df_thetap.loc[cond2, "sc"] = 2  # normal
    df_thetap.loc[cond3, "sc"] = 3  # wet

    # DataFrame with sampled model parameters and the corresponding metrics
    nx = ds_sim.dims["x"]  # number of rows
    ny = ds_sim.dims["y"]  # number of columns
    file = base_path_figs / "params_metrics_fast.txt"
    if not os.path.exists(file):
        click.echo("Calculate metrics ...")
        df_params_metrics = pd.DataFrame(index=range(nx * ny))
        # sampled model parameters
        df_params_metrics.loc[:, "c1_mak"] = ds_sim["c1_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "c2_mak"] = ds_sim["c2_mak"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "dmpv"] = ds_sim["dmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "lmpv"] = ds_sim["lmpv"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "theta_eff"] = ds_sim["theta_eff"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "frac_lp"] = ds_sim["frac_lp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "frac_fp"] = ds_sim["frac_fp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "theta_ac"] = ds_sim["theta_ac"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "theta_ufc"] = ds_sim["theta_ufc"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "theta_pwp"] = ds_sim["theta_pwp"].isel(y=0).values.flatten()
        df_params_metrics.loc[:, "ks"] = ds_sim["ks"].isel(y=0).values.flatten()
        # calculate metrics
        vars_sim = ["aet", "q_ss", "dS"]
        vars_obs = ["AET", "PERC", "dWEIGHT"]
        for var_sim, var_obs in zip(vars_sim, vars_obs):
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
            df_obs.loc[:, "obs"] = obs_vals
            for nrow in range(nx * ny):
                sim_vals = ds_sim[var_sim].isel(x=nrow, y=0).values
                # join observations on simulations
                df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)

                if var_sim in ["dS"]:
                    df_eval.loc["2000-01":"2000-06", :] = onp.nan
                    df_eval = df_eval.dropna()
                    obs_vals = df_eval.loc[:, "obs"].values
                    sim_vals = df_eval.loc[:, "sim"].values
                    key_kge = "KGE_" + var_sim
                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                else:
                    # skip first seven days for warmup
                    df_eval.loc[:"1997-01-07", :] = onp.nan
                    df_eval = df_eval.dropna()
                    obs_vals = df_eval.loc[:, "obs"].values
                    sim_vals = df_eval.loc[:, "sim"].values
                    key_kge = "KGE_" + var_sim
                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
                    key_kge = "KGE_beta_" + var_sim
                    df_params_metrics.loc[nrow, key_kge] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
                    key_rbs = "RBS_" + var_sim
                    df_params_metrics.loc[nrow, key_rbs] = eval_utils.calc_rbs(obs_vals, sim_vals)

                # avoid defragmentation of DataFrame
                click.echo(f"{var_sim}: {nrow}")
                df_params_metrics = df_params_metrics.copy()
        # Calculate multi-objective metric
        df_params_metrics.loc[:, "KGE_multi"] = (
            0.2 * df_params_metrics.loc[:, "KGE_dS"]
            + 0.4 * df_params_metrics.loc[:, "KGE_aet"]
            + 0.4 * df_params_metrics.loc[:, "KGE_q_ss"]
        )

        # write .txt-file
        file = base_path_output / "params_metrics_fast.txt"
        df_params_metrics.to_csv(file, header=True, index=False, sep="\t")
    return


if __name__ == "__main__":
    main()
