from pathlib import Path
import os
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.dates as mdates
import matplotlib.cm as cm
import click
import roger.tools.evaluation as eval_utils
import matplotlib as mpl
import seaborn as sns

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
    "talk",
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

_LABS_TM = {
    "complete-mixing": "CM",
    "piston": "PI",
    "advection-dispersion-power": "AD",
    "time-variant advection-dispersion-power": "AD-TV",
    "preferential-power": "PF",
    "older-preference-power": "OP",
    "advection-dispersion-kumaraswamy": "ADK",
    "time-variant advection-dispersion-kumaraswamy": "ADK-TV",
}


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

    # load observations (measured data)
    path_obs = Path(__file__).parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
    # assign date
    days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_obs = num2date(
        days_obs,
        units=f"days since {ds_obs['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_obs = ds_obs.assign_coords(Time=("Time", date_obs))
    df_obs = pd.DataFrame(index=date_obs)
    df_obs.loc[:, "d18O_prec"] = ds_obs["d18O_PREC"].isel(x=0, y=0).values
    df_obs.loc[:, "d18O_perc"] = ds_obs["d18O_PERC"].isel(x=0, y=0).values
    df_obs1 = df_obs.copy()
    # load data from bromide experiment
    path_obs_br = Path(__file__).parent / "observations" / "bromide_breakthrough.csv"
    df_obs_br = pd.read_csv(path_obs_br, skiprows=1, sep=";", na_values="")

    # load best monte carlo simulations
    hm1_file = base_path / "svat_monte_carlo" / "output" / "SVAT_best1.nc"
    ds_sim_hm1 = xr.open_dataset(hm1_file, engine="h5netcdf")
    # assign date
    days_sim_hm1 = ds_sim_hm1["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm1 = num2date(
        days_sim_hm1,
        units=f"days since {ds_sim_hm1['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm1 = ds_sim_hm1.assign_coords(Time=("Time", date_sim_hm1))
    # load best 10 monte carlo simulations
    hm10_file = base_path / "svat_monte_carlo" / "output" / "SVAT_best10.nc"
    ds_sim_hm10 = xr.open_dataset(hm10_file, engine="h5netcdf")
    # assign date
    days_sim_hm10 = ds_sim_hm10["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm10 = num2date(
        days_sim_hm10,
        units=f"days since {ds_sim_hm10['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm10 = ds_sim_hm10.assign_coords(Time=("Time", date_sim_hm10))

    # load best 100 monte carlo simulations
    hm100_file = base_path / "svat_monte_carlo" / "output" / "SVAT_best100.nc"
    ds_sim_hm100 = xr.open_dataset(hm100_file, engine="h5netcdf")
    # assign date
    days_sim_hm100 = ds_sim_hm100["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm100 = num2date(
        days_sim_hm100,
        units=f"days since {ds_sim_hm100['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm100 = ds_sim_hm100.assign_coords(Time=("Time", date_sim_hm100))

    hm_for_tm_file = (
        base_path / "svat_oxygen18_monte_carlo" / "output" / "SVAT_best_for_advection-dispersion-power.nc"
    )
    ds_sim_hm_for_tm = xr.open_dataset(hm_for_tm_file, engine="h5netcdf")
    # assign date
    days_sim_hm_for_tm = ds_sim_hm_for_tm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm_for_tm = num2date(
        days_sim_hm_for_tm,
        units=f"days since {ds_sim_hm_for_tm['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm_for_tm = ds_sim_hm_for_tm.assign_coords(Time=("Time", date_sim_hm_for_tm))

    data = pd.DataFrame(index=date_sim_hm_for_tm, columns=["PERC_obs [mm/day]", "PERC_sim [mm/day]"])
    data.iloc[1:, 0] = ds_obs["PERC"].isel(x=0, y=0).values
    data.iloc[:, 1] = ds_sim_hm_for_tm["q_ss"].isel(y=0).values
    file = base_path_figs / f"PERC.txt"
    data.iloc[1:, :].to_csv(file, header=True, index=True, sep="\t")

    # load HYDRUS-1D benchmarks
    # oxygen-18 simulations
    hydrus_file = base_path / "hydrus_benchmark" / "hydrus_18O.nc"
    ds_hydrus_18O = xr.open_dataset(hydrus_file, engine="h5netcdf")
    hours_hydrus_18O = ds_hydrus_18O["Time"].values / onp.timedelta64(60 * 60, "s")
    date_hydrus_18O = num2date(
        hours_hydrus_18O,
        units=f"hours since {ds_hydrus_18O['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_hydrus_18O = ds_hydrus_18O.assign_coords(Time=("Time", date_hydrus_18O))

    # travel time simulations
    hydrus_file = base_path / "hydrus_benchmark" / "hydrus_tt.nc"
    ds_hydrus_tt = xr.open_dataset(hydrus_file, engine="h5netcdf", decode_times=False)
    days_hydrus_tt = ds_hydrus_tt["Time"].values / 24
    date_hydrus_tt = num2date(
        days_hydrus_tt,
        units=f"hours since {ds_hydrus_tt['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_hydrus_tt = ds_hydrus_tt.assign_coords(Time=("Time", date_hydrus_tt))

    vars_obs = ["PREC", "PREC_corr"]
    vars_sim = ["prec", "prec_corr"]
    dict_obs = {}
    for var_obs, var_sim in zip(vars_obs, vars_sim):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm1["prec"].isel(y=0).values.T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm1, sim_vals, df_obs)
        dict_obs[var_obs] = df_eval

    vars_obs = ["AET", "PERC", "dWEIGHT"]
    vars_sim = ["aet", "q_ss", "dS"]
    vars_bench = ["aet", "perc", "dS"]
    dict_obs_sim100 = {}
    for var_obs, var_sim, var_bench in zip(vars_obs, vars_sim, vars_bench):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm100[var_sim].isel(y=0).values.T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm100, sim_vals, df_obs)
        # skip first seven days for warmup
        df_eval.loc[:"1997-01-07", :] = onp.nan
        # join benchmark simulations
        bench_vals = ds_hydrus_18O[var_bench].values
        df_bench = pd.DataFrame(index=ds_hydrus_18O["Time"].values, columns=["bench"])
        df_bench.loc[:, "bench"] = bench_vals
        df_eval = df_eval.join(df_bench)
        dict_obs_sim100[var_obs] = df_eval

    vars_obs = ["AET", "PERC", "dWEIGHT"]
    vars_sim = ["aet", "perc", "dS"]
    dict_obs_sim_hydrus = {}
    for var_obs, var_sim in zip(vars_obs, vars_sim):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_hydrus_18O[var_sim].values
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(ds_hydrus_18O["Time"].values, sim_vals, df_obs)
        df_eval = df_eval.iloc[:, :]
        dict_obs_sim_hydrus[var_obs] = df_eval

    # figures for talk
    tm_structures1 = ["complete-mixing", "advection-dispersion-power"]

    # measured oxygen-18 in precipitation and percolation
    d18O_prec_mean = onp.round(onp.nanmean(df_obs1.loc[:, "d18O_prec"].values), 2)
    d18O_perc_mean = onp.round(onp.nanmean(df_obs1.loc[:, "d18O_perc"].values), 2)
    fig, axs = plt.subplots(2, 1, figsize=(5, 3))
    axs[0].plot(df_obs1.index, df_obs1.loc[:, "d18O_prec"].fillna(method="bfill"), "-", color="blue")
    axs[0].scatter(df_obs1.index, df_obs1.loc[:, "d18O_prec"], color="blue", s=1)
    axs[0].set_ylabel(r"$\delta^{18}$O [‰]")
    axs[0].set_ylim([-20, 0])
    axs[0].set_xlim(df_obs1.index[0], df_obs1.index[-1])
    axs[1].plot(df_obs1.index, df_obs1.loc[:, "d18O_perc"].fillna(method="bfill"), "-", color="grey")
    axs[1].scatter(df_obs1.index, df_obs1.loc[:, "d18O_perc"], color="grey", s=1)
    axs[1].set_ylabel(r"$\delta^{18}$O [‰]")
    axs[1].set_xlabel("Time [year]")
    axs[1].set_ylim([-20, 0])
    axs[1].set_xlim(df_obs1.index[0], df_obs1.index[-1])
    fig.tight_layout()
    fig.text(0.15, 0.915, "(a)", ha="center", va="center", fontsize=8)
    fig.text(
        0.52, 0.905, r"$\overline{\delta^{18}O}_{prec}$: %s" % (d18O_prec_mean), ha="center", va="center", fontsize=8
    )
    fig.text(
        0.52, 0.45, r"$\overline{\delta^{18}O}_{perc}$: %s" % (d18O_perc_mean), ha="center", va="center", fontsize=8
    )
    fig.text(0.15, 0.46, "(b)", ha="center", va="center", fontsize=8)
    file = base_path_figs / "observed_d18O_prec_perc_for_talk.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    # compare best 100 simulations with observations
    nx = ds_sim_hm100.dims["x"]
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(6, 3))
    axes[0, 0].plot(
        dict_obs["PREC"].loc["1997-01-07":"1999", :].index,
        dict_obs["PREC"].loc["1997-01-07":"1999", "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    axes[0, 0].set_ylabel("PRECIP\n[mm]")
    axes[0, 0].set_xlim(
        (
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[0],
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[-1],
        )
    )
    axes[0, 0].set_ylim(
        0,
    )
    axes[0, 0].invert_yaxis()
    ax2 = axes[0, 0].twinx()
    for nrow in range(nx):
        ax2.plot(
            dict_obs_sim100["AET"].loc["1997-01-07":"1999", :].index,
            dict_obs_sim100["AET"].loc["1997-01-07":"1999", f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.7,
        )
    ax2.plot(
        dict_obs_sim100["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim100["AET"].loc["1997-01-07":"1999", "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    ax2.plot(
        dict_obs_sim_hydrus["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_hydrus["AET"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    ax2.set_ylim(
        0,
    )
    # ax2.set_ylabel('ET\n[mm]')
    axes[0, 1].plot(
        dict_obs["PREC_corr"].loc["2006":, :].index,
        dict_obs["PREC_corr"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    # axes[0, 1].set_ylabel('PRECIP\n[mm]')
    axes[0, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[0, 1].set_ylim(
        0,
    )
    axes[0, 1].invert_yaxis()
    ax2 = axes[0, 1].twinx()
    for nrow in range(nx):
        ax2.plot(
            dict_obs_sim100["AET"].loc["2006":, :].index,
            dict_obs_sim100["AET"].loc["2006":, f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.7,
        )
    ax2.plot(
        dict_obs_sim100["AET"].loc["2006":, :].index,
        dict_obs_sim100["AET"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    ax2.plot(
        dict_obs_sim_hydrus["AET"].loc["2006":, :].index,
        dict_obs_sim_hydrus["AET"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    ax2.set_ylim(
        0,
    )
    ax2.set_ylabel("ET\n[mm]")
    for nrow in range(nx):
        axes[1, 0].plot(
            dict_obs_sim100["dWEIGHT"].loc["1997-01-07":"1999", :].index,
            dict_obs_sim100["dWEIGHT"].loc["1997-01-07":"1999", f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.7,
        )
    axes[1, 0].plot(
        dict_obs_sim_hydrus["dWEIGHT"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_hydrus["dWEIGHT"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    axes[1, 0].set_ylabel("cum. $\Delta$S\n[mm]")
    axes[1, 0].set_xlim(
        (
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[0],
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[-1],
        )
    )
    for nrow in range(nx):
        axes[1, 1].plot(
            dict_obs_sim100["dWEIGHT"].loc["2006":, :].index,
            dict_obs_sim100["dWEIGHT"].loc["2006":, f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.7,
        )
    axes[1, 1].plot(
        dict_obs_sim100["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim100["dWEIGHT"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    axes[1, 1].plot(
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    # axes[1, 1].set_ylabel('cum. $\Delta$S\n[mm]')
    axes[1, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    for nrow in range(nx):
        axes[2, 0].plot(
            dict_obs_sim100["PERC"].loc["1997-01-07":"1999", :].index,
            dict_obs_sim100["PERC"].loc["1997-01-07":"1999", f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.7,
        )
    axes[2, 0].plot(
        dict_obs_sim100["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim100["PERC"].loc["1997-01-07":"1999", "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    axes[2, 0].plot(
        dict_obs_sim_hydrus["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_hydrus["PERC"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    axes[2, 0].set_ylim(
        0,
    )
    axes[2, 0].invert_yaxis()
    axes[2, 0].set_xlim(
        (
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[0],
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[-1],
        )
    )
    axes[2, 0].set_ylabel("PERC\n[mm]")
    axes[2, 0].set_xlabel(r"Time [year-month]")
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 0].tick_params(axis="x", rotation=33)
    for nrow in range(nx):
        axes[2, 1].plot(
            dict_obs_sim100["PERC"].loc["2006":, :].index,
            dict_obs_sim100["PERC"].loc["2006":, f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.7,
        )
    axes[2, 1].plot(
        dict_obs_sim100["PERC"].loc["2006":, :].index,
        dict_obs_sim100["PERC"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    axes[2, 1].plot(
        dict_obs_sim_hydrus["PERC"].loc["2006":, :].index,
        dict_obs_sim_hydrus["PERC"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    axes[2, 1].set_ylim(
        0,
    )
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    # axes[2, 1].set_ylabel('PERC\n[mm]')
    axes[2, 1].set_xlabel(r"Time [year-month]")
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 1].tick_params(axis="x", rotation=33)
    fig.tight_layout()
    file = f"prec_et_dS_perc_obs_sim_cumulated_best_100_optimized_for_talk.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"prec_et_dS_perc_obs_sim_cumulated_best_100_optimized_with_for_talk.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    # load metrics of transport simulations
    dict_params_metrics_tm_mc = {}
    for tm_structure in [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
        "preferential-power",
        "older-preference-power",
        "advection-dispersion-kumaraswamy",
        "time-variant advection-dispersion-kumaraswamy",
    ]:
        tms = tm_structure.replace(" ", "_")
        file = (
            base_path
            / "svat_oxygen18_monte_carlo"
            / "output"
            / f"params_metrics_{tms}.txt"
        )
        df_params_metrics = pd.read_csv(file, sep="\t")
        dict_params_metrics_tm_mc[tm_structure] = {}
        dict_params_metrics_tm_mc[tm_structure]["params_metrics"] = df_params_metrics

    # simulated oxygen-18 in percolation
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 4))
    df_obs = pd.DataFrame(index=date_obs)
    df_obs.loc[:, "d18O_prec"] = ds_obs["d18O_PREC"].isel(x=0, y=0).values
    ax.flatten()[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].fillna(method="bfill"), "-", color="blue")
    ax.flatten()[0].scatter(df_obs.index, df_obs.loc[:, "d18O_prec"], color="blue", s=1)
    ax.flatten()[0].set_ylabel(r"$\delta^{18}$$O_{PRECIP}$ [‰]")
    ax.flatten()[0].set_ylim([-20, 0])
    ax.flatten()[0].set_xlim(df_obs.index[0], df_obs.index[-1])
    for i, tm_structure in enumerate(tm_structures1):
        idx_best = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]["KGE_C_iso_q_ss"].idxmax()
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        tm_file = base_path / "svat_oxygen18_monte_carlo" / "output" / f"{tms}_monte_carlo.nc"
        ds_sim_tm = xr.open_dataset(tm_file, engine="h5netcdf")
        days_sim_tm = ds_sim_tm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim_tm = num2date(
            days_sim_tm,
            units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
        # join observations on simulations
        obs_vals = ds_obs["d18O_PERC"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        ax.flatten()[i + 1].plot(
            ds_sim_tm["Time"].values, ds_sim_tm["C_iso_q_ss"].isel(x=idx_best, y=0).values, color="red", lw=1
        )
        ax.flatten()[i + 1].plot(ds_hydrus_18O["Time"].values, ds_hydrus_18O["d18O_perc"].values, color="black", lw=1)
        ax.flatten()[i + 1].scatter(df_obs.index, df_obs.iloc[:, 0], color="blue", s=1)
        ax[i + 1].set_ylabel("%s\n$\delta^{18}$$O_{PERC}$ [‰]" % (_LABS_TM[tm_structure]))
        if tm_structure in ["piston"]:
            ax.flatten()[i + 1].set_ylim((-20, -4))
        else:
            ax.flatten()[i + 1].set_ylim((-15, -7))
        ax.flatten()[i + 1].set_xlim(ds_sim_tm["Time"].values[0], ds_sim_tm["Time"].values[-1])
    ax[-1].set_xlabel("Time [year]")
    fig.tight_layout()
    file = base_path_figs / "d18O_perc_sim_obs_tm_structures.png"
    fig.savefig(file, dpi=300)

    # virtual bromide experiment
    years = onp.arange(1997, 2007).tolist()
    cmap = cm.get_cmap("Reds")
    cmap_hydrus = cm.get_cmap("Greys")
    norm = Normalize(vmin=onp.min(years) - 7, vmax=onp.max(years))
    alphas = [0.8]
    for alpha in alphas:
        fig, axes = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
        for i, tm_structure in enumerate(tm_structures1):
            tms = tm_structure.replace(" ", "_")
            # add St. Gallen
            if tm_structure in [
                "complete-mixing",
                "advection-dispersion-power",
                "time-variant advection-dispersion-power",
            ]:
                df_sim_br_conc = pd.DataFrame(index=df_obs_br.index)
                br_file = (
                    base_path / "svat_bromide_benchmark" / "output" / f"{tms}_bromide_benchmark_stgallen.nc"
                )
                with xr.open_dataset(br_file, engine="h5netcdf", decode_times=False, group=f"1991") as ds:
                    sim_vals = ds["C_q_ss_mmol_bs"].isel(x=0, y=0).values[315:716]
                    sim_vals = onp.where(sim_vals < 0, onp.nan, sim_vals)
                    df_sim_br_conc.loc[:, f"1991"] = sim_vals
                axes.flatten()[i].plot(
                    df_sim_br_conc.dropna().index,
                    df_sim_br_conc.dropna()[f"1991"],
                    ls=":",
                    color="red",
                    lw=1,
                    alpha=0.8,
                    label=f"1991",
                )
            df_sim_br_conc = pd.DataFrame(index=df_obs_br.index)
            df_sim_br_mass = pd.DataFrame(index=df_obs_br.index)
            for year in years:
                br_file = base_path / "svat_bromide_benchmark" / "output" / f"{tms}_bromide_benchmark.nc"
                with xr.open_dataset(br_file, engine="h5netcdf", decode_times=False, group=f"{year}") as ds:
                    x = onp.where(
                        (onp.round(ds["alpha_transp"].isel(Time=0).values, 1) == alpha)
                        & (onp.round(ds["alpha_q"].isel(Time=0).values, 1) == alpha)
                    )[0][0]
                    sim_vals = ds["C_q_ss_mmol_bs"].isel(x=x, y=0).values[315:716]
                    sim_vals = onp.where(sim_vals < 0, onp.nan, sim_vals)
                    df_sim_br_conc.loc[:, f"{year}"] = sim_vals
                    sim_vals = ds["M_q_ss"].isel(x=x, y=0).values[315:716]
                    sim_vals = onp.where(sim_vals < 0, onp.nan, sim_vals)
                    df_sim_br_mass.loc[:, f"{year}"] = sim_vals
                axes.flatten()[i].plot(
                    df_sim_br_conc.dropna().index,
                    df_sim_br_conc.dropna()[f"{year}"],
                    ls="--",
                    color=cmap(norm(year)),
                    lw=0.8,
                    alpha=0.5,
                    label=f"{year}",
                )
            weights = df_sim_br_mass.values / df_sim_br_mass.sum(axis=1).values[:, onp.newaxis]
            df_sim_br_conc.loc[:, "avg_weighted"] = onp.sum(weights * df_sim_br_conc.values, axis=1)
            axes.flatten()[i].plot(
                df_sim_br_conc.dropna().index,
                df_sim_br_conc.dropna().loc[:, "avg_weighted"],
                color="red",
                lw=1,
                alpha=1,
                label="average",
            )
            axes.flatten()[i].plot(df_obs_br.dropna().index, df_obs_br.dropna()["Br"], color="blue", lw=1)
            axes.flatten()[i].set_xlim([0, 400])
            axes.flatten()[i].set_ylabel("%s\nBr [mmol/l]" % (_LABS_TM[tm_structure]))
        df_sim_br = pd.DataFrame(index=df_obs_br.index)
        for year in years:
            hydrus_br_file = base_path / "hydrus_benchmark" / "hydrus_bromide.nc"
            with xr.open_dataset(hydrus_br_file, engine="h5netcdf", decode_times=False, group=f"{year}") as ds:
                df_sim_br = pd.DataFrame(index=df_obs_br.index)
                df_sim_br.loc[:, f"{year}"] = ds["Br_perc_mmol"].values
            axes.flatten()[-1].plot(
                df_sim_br.dropna().index,
                df_sim_br.dropna()[f"{year}"],
                ls="--",
                color=cmap_hydrus(norm(year)),
                lw=0.8,
                alpha=0.5,
                label=f"{year}",
            )
        axes.flatten()[-1].plot(
            df_sim_br.dropna().index, df_sim_br.dropna().mean(axis=1), color="grey", lw=1, alpha=1, label="average"
        )
        axes.flatten()[-1].plot(
            df_obs_br.dropna().index, df_obs_br.dropna()["Br"], color="blue", lw=1, label="observed"
        )
        axes.flatten()[-1].set_xlim([0, 400])
        axes.flatten()[-1].set_ylabel("HYDRUS-1D\nBr [mmol/l]")
        axes.flatten()[-1].set_xlabel(r"Time [days since injection]")
        lines1, labels1 = axes.flatten()[-2].get_legend_handles_labels()
        lines2, labels2 = axes.flatten()[-1].get_legend_handles_labels()
        fig.legend(lines1, labels1, loc="upper right", fontsize=6, frameon=False, bbox_to_anchor=(0.965, 0.8))
        fig.legend(lines2, labels2, loc="lower right", fontsize=6, frameon=False, bbox_to_anchor=(0.97, 0.07))
        fig.subplots_adjust(bottom=0.1, right=0.85, hspace=0.2)
        file = base_path_figs / f"bromide_benchmark_alpha_{alpha}_for_talk.png"
        fig.savefig(file, dpi=300)

    # backward travel time distributions of transpiration and percolation
    fig, axes = plt.subplots(2, 3, sharey=True, figsize=(4, 3))
    for i, tm_structure in enumerate(tm_structures1):
        tms = tm_structure.replace(" ", "_")
        tm_file = base_path / "svat_oxygen18" / "output" / f"{tms}.nc"
        with xr.open_dataset(tm_file, engine="h5netcdf") as ds_sim_tm:
            days_sim_tm = ds_sim_tm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
            date_sim_tm = num2date(
                days_sim_tm,
                units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}",
                calendar="standard",
                only_use_cftime_datetimes=False,
            )
            ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
            for j, var_sim in enumerate(["TT_transp", "TT_q_ss"]):
                TT = ds_sim_tm[var_sim].isel(x=0, y=0).values
                TT = TT[~onp.all(TT == 0, axis=1)]
                x = onp.arange(TT.shape[-1])
                y1 = onp.nanquantile(TT, 0.05, axis=0)
                y2 = onp.nanquantile(TT, 0.95, axis=0)
                axes[j, i].fill_between(x, y1, y2, facecolor="red", alpha=0.5)
                axes[j, i].plot(onp.nanquantile(TT, 0.5, axis=0), ls="--", lw=1, color="black")
                axes[j, i].plot(onp.mean(TT, axis=0), lw=1, color="black")
                if var_sim == "TT_transp":
                    axes[j, i].set_xlim((0, 150))
                elif var_sim == "TT_q_ss":
                    axes[j, i].set_xlim((0, 1000))
                axes[j, i].set_ylim((0, 1))
                axes[j, i].set_xlabel("T [days]")
                axes[0, i].set_title(_LABS_TM[tm_structure])
                avgTT50 = int(onp.nansum(onp.diff(onp.nanmedian(TT, axis=0)) * onp.arange(1, 1501)))
                axes[j, i].text(
                    0.5,
                    0.1,
                    r"$\overline{TT}_{50}$=%s" % (avgTT50),
                    size=8,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=axes[j, i].transAxes,
                )

    for j, var_sim in enumerate(["bTT_transp", "bTT_perc"]):
        TT = onp.where(ds_hydrus_tt[var_sim].values <= 0, onp.nan, ds_hydrus_tt[var_sim].values)
        # exclude warmup and skip first 1000 days
        skipt = 1000
        x = onp.arange(TT.shape[-1])
        y1 = onp.nanquantile(TT[skipt:, :], 0.05, axis=0)
        y2 = onp.nanquantile(TT[skipt:, :], 0.95, axis=0)
        axes[j, -1].fill_between(x, y1, y2, facecolor="grey", alpha=0.5)
        axes[j, -1].plot(onp.nanmedian(TT[skipt:, :], axis=0), ls="--", lw=1, color="black")
        axes[j, -1].plot(onp.nanmean(TT[skipt:, :], axis=0), lw=1, color="black")
        if var_sim == "bTT_transp":
            axes[j, -1].set_xlim((0, 150))
        elif var_sim == "bTT_perc":
            axes[j, -1].set_xlim((0, 1000))
        axes[j, -1].set_ylim((0, 1))
        axes[j, -1].set_xlabel("T [days]")
        avgTT50 = int(onp.nansum(onp.diff(onp.nanmedian(TT, axis=0)) * onp.arange(1, 4018)))
        axes[j, -1].text(
            0.5,
            0.1,
            r"$\overline{TT}_{50}$=%s" % (avgTT50),
            size=8,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[j, -1].transAxes,
        )
    axes[0, -1].set_title("HYDRUS-1D")
    axes[0, 0].set_ylabel(r"$\overleftarrow{P}_{TRANSP}(T,t)$")
    axes[1, 0].set_ylabel(r"$\overleftarrow{P}_{PERC}(T,t)$")
    fig.tight_layout()
    file_str = "bTTD_benchmark_for_talk.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)

    plt.close("all")
    return


if __name__ == "__main__":
    main()
