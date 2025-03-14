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

mpl.rcParams["font.size"]= 10
mpl.rcParams["axes.titlesize"]= 10
mpl.rcParams["axes.labelsize"]= 11
mpl.rcParams["xtick.labelsize"]= 10
mpl.rcParams["ytick.labelsize"]= 10
mpl.rcParams["legend.fontsize"]= 10
mpl.rcParams["legend.title_fontsize"]= 11
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "legend.title_fontsize": 11.0,
    },
)


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")
        # base_path = Path(__file__).parent.parent


    # directory of results
    base_path_output = base_path / "output" / "svat_crop_monte_carlo"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = Path(__file__).parent.parent / "figures" / "svat_crop_monte_carlo"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    crops_lys2_lys3_lys8 = {2010: "winter barley",
                            2011: "sugar beet",
                            2012: "winter wheat",
                            2013: "winter rape",
                            2014: "winter triticale",
                            2015: "silage corn",
                            2016: "winter barley & green manure",
                            2017: "green manure & winter wheat"
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


    # lys_experiments = ["lys2", "lys3", "lys8"]
    # for lys_experiment in lys_experiments:
    #     # load parameters and metrics
    #     df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}_bulk_samples.txt", sep="\t")
    #     df_params_metrics["E_multi"] = df_params_metrics["KGE_q_ss_2011-2015"]
    #     df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
    #     df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
    #     idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()
    #     idx_best = idx_best100[0]        
    #     # load simulation
    #     sim_hm_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
    #     ds_sim_hm = xr.open_dataset(sim_hm_file, engine="h5netcdf")
    #     # assign date
    #     days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    #     date_sim_hm = num2date(
    #         days_sim_hm,
    #         units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
    #         calendar="standard",
    #         only_use_cftime_datetimes=False,
    #     )
    #     ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

    #     # load observations (measured data)
    #     path_obs = Path(__file__).parent.parent / "observations" / "reckenholz_lysimeter.nc"
    #     ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)
    #     # assign date
    #     days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    #     date_obs = num2date(
    #         days_obs,
    #         units=f"days since {ds_obs['Time'].attrs['time_origin']}",
    #         calendar="standard",
    #         only_use_cftime_datetimes=False,
    #     )
    #     ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    #     # compare best simulation with observations
    #     years = [2011, 2012, 2013, 2014, 2015]
    #     vars_obs = ["PERC_bs"]
    #     vars_sim = ["q_ss_bs"]
    #     for var_obs, var_sim in zip(vars_obs, vars_sim):
    #         fig, axs = plt.subplots(5, 1, sharey=False, sharex=False, figsize=(6, 4))
    #         for i, year in enumerate(years):
    #             obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    #             df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    #             df_obs.loc[:, "obs"] = obs_vals
    #             sim_vals = ds_sim_hm[var_sim].isel(y=0).values[idx_best100, :].T
    #             # join observations on simulations
    #             df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
    #             # skip first seven days for warmup
    #             df_eval = df_eval.loc[f"{year}-01-01":f"{year}-12-31", :]
    #             # plot observed and simulated time series
    #             sim_vals_min = onp.nanmin(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
    #             sim_vals_max = onp.nanmax(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
    #             sim_vals_median = onp.nanmedian(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
    #             axs[i].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color="red", alpha=0.5, zorder=0)
    #             axs[i].plot(df_eval.index, sim_vals_median, color="red", zorder=1)
    #             axs[i].plot(df_eval.index, df_eval["obs"], color="blue", ls="--", zorder=2)
    #             axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
    #             axs[i].set_ylabel('')
    #             axs[i].set_xlabel('')
    #             axs[i].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d-%m'))
    #         axs[1].set_ylabel(labs._Y_LABS_DAILY[var_sim])
    #         axs[-2].set_ylabel(labs._Y_LABS_DAILY[var_sim])
    #         axs[-1].set_xlabel('[Jahr-Monat]')
    #         fig.tight_layout()
    #         file_str = "%s_%s_%s_%s.pdf" % (var_sim, lys_experiment, years[0], years[-1])
    #         path_fig = base_path_figs / file_str
    #         fig.savefig(path_fig, dpi=300)
    #         plt.close("all")


    #     # compare best simulation with observations
    #     years = [2011, 2012, 2013, 2014, 2015]
    #     vars_obs = ["PERC"]
    #     vars_sim = ["q_ss"]
    #     for var_obs, var_sim in zip(vars_obs, vars_sim):
    #         fig, axs = plt.subplots(5, 1, sharey=False, sharex=False, figsize=(6, 4))
    #         for i, year in enumerate(years):
    #             obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    #             df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    #             df_obs.loc[:, "obs"] = obs_vals
    #             sim_vals = ds_sim_hm[var_sim].isel(y=0).values[idx_best100, :].T
    #             # join observations on simulations
    #             df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
    #             # skip first seven days for warmup
    #             df_eval = df_eval.loc[f"{year}-01-01":f"{year}-12-31", :]
    #             # plot observed and simulated time series
    #             sim_vals_min = onp.nanmin(onp.nancumsum(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=0), axis=1)
    #             sim_vals_max = onp.nanmax(onp.nancumsum(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=0), axis=1)
    #             sim_vals_median = onp.nanmedian(onp.nancumsum(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=0), axis=1)
    #             axs[i].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color="red", alpha=0.5, zorder=0)
    #             axs[i].plot(df_eval.index, sim_vals_median, color="red", zorder=1)
    #             axs[i].plot(df_eval.index, df_eval["obs"].cumsum(), color="blue")
    #             axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
    #             axs[i].set_ylabel('')
    #             axs[i].set_xlabel('')
    #             axs[i].text(0.5,
    #                         1.1,
    #                         f"{year}: {crops_lys2_lys3_lys8[year]} ({fert_lys2_lys3_lys8[lys_experiment]})",
    #                         fontsize=8,
    #                         horizontalalignment="center",
    #                         verticalalignment="center",
    #                         transform=axs[i].transAxes)
    #             axs[i].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d-%m'))
    #             axs[i].set_ylim(0)
    #         axs[1].set_ylabel(labs._Y_LABS_CUM[var_sim])
    #         axs[-2].set_ylabel(labs._Y_LABS_CUM[var_sim])
    #         axs[-1].set_xlabel('[Jahr-Monat]')
    #         fig.tight_layout()
    #         file_str = "%s_%s_%s_%s_cumulated.pdf" % (var_sim, lys_experiment, years[0], years[-1])
    #         path_fig = base_path_figs / file_str
    #         fig.savefig(path_fig, dpi=300)
    #         plt.close("all")

    lys_experiments = ["lys2", "lys3", "lys8"]
    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(6, 4))
    for i, lys_experiment in enumerate(lys_experiments):
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}_bulk_samples.txt", sep="\t")
        df_params_metrics["E_multi"] = df_params_metrics["KGE_q_ss_2011-2015"]
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()
        # load simulation
        sim_hm_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        ds_sim_hm = xr.open_dataset(sim_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim_hm = num2date(
            days_sim_hm,
            units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

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
        obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm["q_ss_bs"].isel(y=0).values[idx_best100, :].T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval = df_eval.loc["2011-01-01":"2015-12-31", :]
        df_eval = df_eval.dropna()
        if lys_experiment == "lys2":
            col = "#dd1c77"
        elif lys_experiment == "lys3":
            col = "#c994c7"
        elif lys_experiment == "lys8":
            col = "#e7e1ef"
        # plot observed and simulated time series
        for j in range(100):
            sim_vals = df_eval.loc[:, f"sim{j}"].values.astype(onp.float64)
            axs[i].plot(df_eval.index, sim_vals, color=col, zorder=1, marker=".")
        axs[i].plot(df_eval.index, df_eval["obs"], color="blue", marker=".")
        axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
        axs[i].set_ylabel('[mm/14 Tage]')
        axs[i].set_xlabel('')
    axs[-1].set_xlabel('[Jahr-Monat]')
    fig.tight_layout()
    file_str = "comparison_percolation_bulk_samples_trace.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)
    plt.close("all")

    lys_experiments = ["lys2", "lys3", "lys8"]
    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(6, 4))
    for i, lys_experiment in enumerate(lys_experiments):
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}_bulk_samples.txt", sep="\t")
        df_params_metrics["E_multi"] = df_params_metrics["KGE_q_ss_2011-2015"]
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()
        # load simulation
        sim_hm_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        ds_sim_hm = xr.open_dataset(sim_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim_hm = num2date(
            days_sim_hm,
            units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

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
        obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm["q_ss_bs"].isel(y=0).values[idx_best100, :].T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval_ = df_eval.loc["2011-01-01":"2015-12-31", :].copy()
        df_eval = df_eval.loc["2011-01-01":"2015-12-31", :].bfill(limit=14)
        if lys_experiment == "lys2":
            col = "#dd1c77"
        elif lys_experiment == "lys3":
            col = "#c994c7"
        elif lys_experiment == "lys8":
            col = "#e7e1ef"
        # plot observed and simulated time series
        sim_vals_min = onp.nanmin(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_max = onp.nanmax(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_median = onp.nanmedian(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_median_ = onp.nanmedian(df_eval_.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        axs[i].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color=col, alpha=0.5, zorder=0)
        axs[i].plot(df_eval.index, sim_vals_median, color=col, zorder=1)
        axs[i].plot(df_eval.index, df_eval["obs"], color="blue")
        axs[i].scatter(df_eval_.index, sim_vals_median_, color=col, zorder=1, s=2)
        axs[i].scatter(df_eval_.index, df_eval["obs"], color="blue", s=2)
        axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
        axs[i].set_ylabel('[mm/14 Tage]')
        axs[i].set_xlabel('')
    axs[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%y'))
    axs[-1].set_xlabel('[Monat-Jahr]')
    fig.tight_layout()
    file_str = "comparison_percolation_bulk_samples.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)
    plt.close("all")


    lys_experiments = ["lys2", "lys3", "lys8"]
    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(6, 4))
    for i, lys_experiment in enumerate(lys_experiments):
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}_bulk_samples.txt", sep="\t")
        df_params_metrics["E_multi"] = df_params_metrics["KGE_q_ss_2011-2015"]
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()
        # load simulation
        sim_hm_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        ds_sim_hm = xr.open_dataset(sim_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim_hm = num2date(
            days_sim_hm,
            units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

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
        obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm["q_ss_bs"].isel(y=0).values[idx_best100, :].T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval = df_eval.loc["2011-01-01":"2015-12-31", :]
        df_eval = df_eval.dropna()
        if lys_experiment == "lys2":
            col = "#dd1c77"
        elif lys_experiment == "lys3":
            col = "#c994c7"
        elif lys_experiment == "lys8":
            col = "#e7e1ef"
        # plot observed and simulated time series
        for j in range(100):
            sim_vals = onp.nancumsum(df_eval.loc[:, f"sim{j}"].values.astype(onp.float64))
            axs[i].plot(df_eval.index, sim_vals, color=col, zorder=1, marker=".")
        axs[i].plot(df_eval.index, df_eval["obs"].cumsum(), color="blue", marker=".")
        axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
        axs[i].set_ylim(0,)
        axs[i].set_ylabel('[mm]')
        axs[i].set_xlabel('')
    axs[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%y'))
    axs[-1].set_xlabel('[Monat-Jahr]')
    fig.tight_layout()
    file_str = "comparison_percolation_bulk_samples_cumulated_trace.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)
    plt.close("all")

    lys_experiments = ["lys2", "lys3", "lys8"]
    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(6, 4))
    for i, lys_experiment in enumerate(lys_experiments):
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}_bulk_samples.txt", sep="\t")
        df_params_metrics["E_multi"] = df_params_metrics["KGE_q_ss_2011-2015"]
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()
        # load simulation
        sim_hm_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        ds_sim_hm = xr.open_dataset(sim_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim_hm = num2date(
            days_sim_hm,
            units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

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
        obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm["q_ss_bs"].isel(y=0).values[idx_best100, :].T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval = df_eval.loc["2011-01-01":"2015-12-31", :]
        df_eval = df_eval.dropna()
        if lys_experiment == "lys2":
            col = "#dd1c77"
        elif lys_experiment == "lys3":
            col = "#c994c7"
        elif lys_experiment == "lys8":
            col = "#e7e1ef"
        # plot observed and simulated time series
        sim_vals_min = onp.nanmin(onp.nancumsum(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=0), axis=1)
        sim_vals_max = onp.nanmax(onp.nancumsum(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=0), axis=1)
        sim_vals_median = onp.nanmedian(onp.nancumsum(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=0), axis=1)
        axs[i].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color=col, alpha=0.5, zorder=0)
        axs[i].plot(df_eval.index, sim_vals_median, color=col, zorder=1, marker=".", lw=2)
        axs[i].plot(df_eval.index, df_eval["obs"].cumsum(), color="blue", marker=".", lw=2)
        axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
        axs[i].set_ylim(0, 1750)
        axs[i].set_ylabel('[mm]')
        axs[i].set_xlabel('')
    axs[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%y'))
    axs[-1].set_xlabel('[Monat-Jahr]')
    fig.tight_layout()
    file_str = "comparison_percolation_bulk_samples_cumulated.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)
    plt.close("all")


    lys_experiments = ["lys2", "lys3", "lys8"]
    fig, axs = plt.subplots(6, 1, sharey=False, sharex=True, figsize=(6, 4))
    for i, lys_experiment in enumerate(lys_experiments):
        if i == 1:
            i = 2
        elif i == 2:
            i = 4
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}_bulk_samples.txt", sep="\t")
        df_params_metrics["E_multi"] = df_params_metrics["KGE_q_ss_2011-2015"]
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()
        # load simulation
        sim_hm_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        ds_sim_hm = xr.open_dataset(sim_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim_hm = num2date(
            days_sim_hm,
            units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

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
        if lys_experiment == "lys2":
            col = "#006d2c"
        elif lys_experiment == "lys3":
            col = "#74c476"
        elif lys_experiment == "lys8":
            col = "#c7e9c0"
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
        obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm["transp"].isel(y=0).values[idx_best100, :].T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval = df_eval.loc["2011-01-01":"2015-12-31", :]
        # plot observed and simulated time series
        sim_vals_min = onp.nanmin(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_max = onp.nanmax(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_median = onp.nanmedian(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        axs[i].set_yticks([])
        ax2 = axs[i].twinx()
        ax2.fill_between(df_eval.index, sim_vals_min, sim_vals_max, color=col, alpha=0.5, zorder=0)
        ax2.plot(df_eval.index, sim_vals_median, color=col, zorder=1)
        ax2.set_xlim(df_eval.index[0], df_eval.index[-1])
        ax2.set_ylim(0, 10)
        ax2.set_ylabel('[mm/Tag]', color=col)
        ax2.set_xlabel('')
        ax2.xaxis.label.set_color(col)
        ax2.tick_params(axis='y', colors=col)

        if lys_experiment == "lys2":
            col = "#dd1c77"
        elif lys_experiment == "lys3":
            col = "#c994c7"
        elif lys_experiment == "lys8":
            col = "#e7e1ef"
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
        obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm["q_ss_bs"].isel(y=0).values[idx_best100, :].T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval_ = df_eval.loc["2011-01-01":"2015-12-31", :].copy()
        df_eval = df_eval.loc["2011-01-01":"2015-12-31", :].bfill(limit=14)
        if lys_experiment == "lys2":
            col = "#dd1c77"
        elif lys_experiment == "lys3":
            col = "#c994c7"
        elif lys_experiment == "lys8":
            col = "#e7e1ef"
        # plot observed and simulated time series
        sim_vals_min = onp.nanmin(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_max = onp.nanmax(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_median = onp.nanmedian(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_median_ = onp.nanmedian(df_eval_.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        axs[i+1].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color=col, alpha=0.5, zorder=0)
        axs[i+1].plot(df_eval.index, sim_vals_median, color=col, zorder=1)
        axs[i+1].plot(df_eval.index, df_eval["obs"], color="blue")
        axs[i+1].scatter(df_eval_.index, sim_vals_median_, color=col, zorder=1, s=2)
        axs[i+1].scatter(df_eval_.index, df_eval["obs"], color="blue", s=2)
        axs[i+1].set_xlim(df_eval.index[0], df_eval.index[-1])
        axs[i+1].set_ylim(0, 150)
        axs[i+1].set_ylabel('[mm/14 Tage]')
        axs[i+1].set_xlabel('')
    axs[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%y'))
    axs[-1].set_xlabel('[Monat-Jahr]')
    fig.tight_layout()
    file_str = "comparison_percolation_bulk_samples_and_transpiration.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)
    plt.close("all")


    lys_experiments = ["lys2", "lys3", "lys8"]
    fig, axs = plt.subplots(6, 1, sharey=True, sharex=True, figsize=(6, 4))
    for i, lys_experiment in enumerate(lys_experiments):
        if i == 1:
            i = 2
        elif i == 2:
            i = 4
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}_bulk_samples.txt", sep="\t")
        df_params_metrics["E_multi"] = df_params_metrics["KGE_q_ss_2011-2015"]
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()
        # load simulation
        sim_hm_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        ds_sim_hm = xr.open_dataset(sim_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim_hm = num2date(
            days_sim_hm,
            units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

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
        if lys_experiment == "lys2":
            col = "#006d2c"
        elif lys_experiment == "lys3":
            col = "#74c476"
        elif lys_experiment == "lys8":
            col = "#e5f5e0"
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
        obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm["transp_bs"].isel(y=0).values[idx_best100, :].T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval = df_eval.loc["2011-01-01":"2015-12-31", :]
        df_eval = df_eval.dropna()
        # plot observed and simulated time series
        sim_vals_min = onp.nanmin(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_max = onp.nanmax(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_median = onp.nanmedian(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        axs[i].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color=col, alpha=0.5, zorder=0)
        axs[i].plot(df_eval.index, sim_vals_median, color=col, zorder=1, marker=".")
        axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
        axs[i].set_xlabel('')

        if lys_experiment == "lys2":
            col = "#dd1c77"
        elif lys_experiment == "lys3":
            col = "#c994c7"
        elif lys_experiment == "lys8":
            col = "#e7e1ef"
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
        obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm["q_ss_bs"].isel(y=0).values[idx_best100, :].T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval = df_eval.loc["2011-01-01":"2015-12-31", :]
        df_eval = df_eval.dropna()
        # plot observed and simulated time series
        sim_vals_min = onp.nanmin(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_max = onp.nanmax(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        sim_vals_median = onp.nanmedian(df_eval.loc[:, "sim0":"sim99"].values.astype(onp.float64), axis=1)
        axs[i+1].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color=col, alpha=0.5, zorder=0)
        axs[i+1].plot(df_eval.index, sim_vals_median, color=col, zorder=1, marker=".")
        axs[i+1].plot(df_eval.index, df_eval["obs"], color="blue", marker=".")
        axs[i+1].set_xlim(df_eval.index[0], df_eval.index[-1])
        axs[i+1].set_ylabel('[mm/14 Tage]')
        axs[i+1].set_xlabel('')
    axs[-1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%y'))
    axs[-1].set_xlabel('[Monat-Jahr]')
    fig.tight_layout()
    file_str = "comparison_percolation_bulk_samples_and_transpiration_bulk_samples.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)
    plt.close("all")
    return


if __name__ == "__main__":
    main()
