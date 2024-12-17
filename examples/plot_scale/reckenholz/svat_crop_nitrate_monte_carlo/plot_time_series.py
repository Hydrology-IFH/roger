import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import click
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["legend.title_fontsize"] = 9
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 9.0,
    },
)


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")

    # directory of results
    base_path_output = base_path / "output" / "svat_crop_nitrate_monte_carlo"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = Path(__file__).parent.parent / "figures" / "svat_crop_nitrate_monte_carlo"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    crops_lys2_lys3_lys8 = {2010: "winter barley",
                            2011: "sugar beet",
                            2012: "winter wheat",
                            2013: "winter rape",
                            2014: "winter triticale",
                            2015: "silage corn",
                            2016: "winter barley",
                            2017: "sugar beet"
    }

    crops_lys4_lys9 = {2010: "winter wheat",
                       2011: "winter rape & phacelia",
                       2012: "phacelia & silage corn",
                       2013: "beet root",
                       2014: "winter barley",
                       2015: "grass",
                       2016: "grass",
                       2017: "winter wheat"
    }

    fert_lys2_lys3_lys8 = {"lys2": "130% N-fertilized",
                           "lys3": "100% N-fertilized",
                           "lys8": "70% N-fertilized"
    }

    fert_lys4_lys9 = {"lys4": "Organic",
                      "lys9": "PEP-intensive"
    }

    _lys = {"lys2": "Lys 2",
            "lys3": "Lys 3",
            "lys4": "Lys 4",
            "lys8": "Lys 8",
            "lys9": "Lys 9",
    }

    _Y_LABS_DAILY = {
        "C_q_ss_bs": r"${NO_3^-N}$ [mg/l]",
        "M_q_ss_bs": r"${NO_3^-N}$ [mg]",
    }


    lys_experiments = ["lys3"]
    for lys_experiment in lys_experiments:
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
        df_params_metrics["E_multi"] = df_params_metrics["KGE_NO3_perc_mass_bs"]
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[9], "id"].values.tolist()
        idx_best = idx_best100[0]        

        # load observations (measured data)
        path_obs = Path(__file__).parent.parent / "observations" / "reckenholz_lysimeter.nc"
        ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)
        # assign date
        days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_obs = num2date(
            days_obs,
            units=f"days since {ds_obs['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

        # load transport simulation
        states_tm_file = base_path / "output" / "svat_crop_nitrate_monte_carlo" / f"SVATCROPNITRATE_advection-dispersion-power_{lys_experiment}.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")

        # assign date
        days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))

        # compare best simulation with observations
        years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
        vars_obs = ["NO3_PERC", "NO3_PERC_MASS"]
        vars_sim = ["C_q_ss_bs", "M_q_ss_bs"]
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            fig, axs = plt.subplots(7, 1, sharey=False, sharex=False, figsize=(6, 6))
            for i, year in enumerate(years):
                obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
                df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
                df_obs.loc[:, "obs"] = obs_vals
                sim_vals = ds_sim_tm[var_sim].isel(y=0).values[idx_best100, :].T
                # join observations on simulations
                df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals, df_obs)
                df_eval = df_eval.loc[f"{year}-01-01":f"{year}-12-31", :]
                cond_na = df_eval.loc[:, "obs"].isna()
                df_eval.loc[cond_na, :] = onp.nan
                # plot observed and simulated time series
                sim_vals_min = onp.nanmin(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=1)
                sim_vals_max = onp.nanmax(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=1)
                sim_vals_median = onp.nanmedian(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=1)
                axs[i].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color="red", alpha=0.5, zorder=0)
                axs[i].plot(df_eval.index, sim_vals_median, color="red", zorder=1)
                axs[i].scatter(df_eval.index, df_eval["obs"], color="blue", s=5, zorder=2)
                axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
                axs[i].set_ylabel('')
                axs[i].set_xlabel('')
            axs[1].set_ylabel(_Y_LABS_DAILY[var_sim])
            axs[-2].set_ylabel(_Y_LABS_DAILY[var_sim])
            axs[-1].set_xlabel('Time [day-month]')
            fig.tight_layout()
            file_str = "%s_%s_%s_%s.pdf" % (var_sim, lys_experiment, years[0], years[-1])
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=300)
            plt.close("all")

        years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
        vars_obs = ["NO3_PERC_MASS"]
        vars_sim = ["M_q_ss_bs"]
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            fig, axs = plt.subplots(7, 1, sharey=False, sharex=False, figsize=(6, 6))
            for i, year in enumerate(years):
                obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
                df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
                df_obs.loc[:, "obs"] = obs_vals
                sim_vals = ds_sim_tm[var_sim].isel(y=0).values[idx_best100, :].T
                # join observations on simulations
                df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals, df_obs)
                df_eval = df_eval.loc[f"{year}-01-01":f"{year}-12-31", :]
                cond_na = df_eval.loc[:, "obs"].isna()
                df_eval.loc[cond_na, :] = onp.nan
                # plot observed and simulated time series
                sim_vals_min = onp.nanmin(onp.nancumsum(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=0), axis=1)
                sim_vals_max = onp.nanmax(onp.nancumsum(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=0), axis=1)
                sim_vals_median = onp.nanmedian(onp.nancumsum(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=0), axis=1)
                axs[i].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color="red", alpha=0.5, zorder=0)
                axs[i].plot(df_eval.index, sim_vals_median, color="red", zorder=1)
                axs[i].scatter(df_eval.index, df_eval["obs"].cumsum(), color="blue", s=5, zorder=2)
                axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
                axs[i].set_ylabel('')
                axs[i].set_xlabel('')
            axs[1].set_ylabel(_Y_LABS_DAILY[var_sim])
            axs[-2].set_ylabel(_Y_LABS_DAILY[var_sim])
            axs[-1].set_xlabel('Time [day-month]')
            fig.tight_layout()
            file_str = "%s_%s_%s_%s_cumulated.pdf" % (var_sim, lys_experiment, years[0], years[-1])
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=300)
            plt.close("all")

        vars_obs = ["NO3_PERC_MASS"]
        vars_sim = ["M_q_ss_bs"]
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            fig, axs = plt.subplots(1, 1, sharey=False, sharex=False, figsize=(6, 3))
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
            df_obs.loc[:, "obs"] = obs_vals
            sim_vals = ds_sim_tm[var_sim].isel(y=0).values[idx_best100, :].T
            # join observations on simulations
            df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals, df_obs)
            cond_na = df_eval.loc[:, "obs"].isna()
            df_eval.loc[cond_na, :] = onp.nan
            # plot observed and simulated time series
            sim_vals_min = onp.nanmin(onp.nancumsum(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=0), axis=1)
            sim_vals_max = onp.nanmax(onp.nancumsum(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=0), axis=1)
            sim_vals_median = onp.nanmedian(onp.nancumsum(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=0), axis=1)
            axs.fill_between(df_eval.index, sim_vals_min, sim_vals_max, color="red", alpha=0.5, zorder=0)
            axs.plot(df_eval.index, sim_vals_median, color="red", zorder=1)
            axs.scatter(df_eval.index, df_eval["obs"].cumsum(), color="blue", s=5, zorder=2)
            axs.set_xlim(df_eval.index[0], df_eval.index[-1])
            axs.set_ylim(0,)
            axs.set_ylabel(_Y_LABS_DAILY[var_sim])
            axs.set_xlabel('Time [day-month]')
            fig.tight_layout()
            file_str = "%s_%s_cumulated.pdf" % (var_sim, lys_experiment)
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=300)
            plt.close("all")
    return


if __name__ == "__main__":
    main()
