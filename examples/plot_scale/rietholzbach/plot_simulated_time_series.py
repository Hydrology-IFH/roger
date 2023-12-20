from pathlib import Path
import os
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.dates as mdates
import click
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
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

    tm_structures = [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
    ]

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
    states_hm1_file = base_path / "svat_monte_carlo" / "output" / "states_hm1.nc"
    ds_sim_hm1 = xr.open_dataset(states_hm1_file, engine="h5netcdf")
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
    states_hm10_file = base_path / "svat_monte_carlo" / "output" / "states_hm10.nc"
    ds_sim_hm10 = xr.open_dataset(states_hm10_file, engine="h5netcdf")
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
    states_hm100_file = base_path / "svat_monte_carlo" / "output" / "states_hm100.nc"
    ds_sim_hm100 = xr.open_dataset(states_hm100_file, engine="h5netcdf")
    # assign date
    days_sim_hm100 = ds_sim_hm100["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm100 = num2date(
        days_sim_hm100,
        units=f"days since {ds_sim_hm100['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm100 = ds_sim_hm100.assign_coords(Time=("Time", date_sim_hm100))

    states_hm_for_tm_file = (
        base_path
        / "svat_oxygen18_monte_carlo"
        / "output"
        / "states_hm_best_for_advection-dispersion-power.nc"
    )
    ds_sim_hm_for_tm = xr.open_dataset(states_hm_for_tm_file, engine="h5netcdf")
    # assign date
    days_sim_hm_for_tm = ds_sim_hm_for_tm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm_for_tm = num2date(
        days_sim_hm_for_tm,
        units=f"days since {ds_sim_hm_for_tm['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm_for_tm = ds_sim_hm_for_tm.assign_coords(Time=("Time", date_sim_hm_for_tm))

    # load HYDRUS-1D benchmarks
    # oxygen-18 simulations
    states_hydrus_file = base_path / "hydrus_benchmark" / "states_hydrus_18O.nc"
    ds_hydrus_18O = xr.open_dataset(states_hydrus_file, engine="h5netcdf")
    hours_hydrus_18O = ds_hydrus_18O["Time"].values / onp.timedelta64(60 * 60, "s")
    date_hydrus_18O = num2date(
        hours_hydrus_18O,
        units=f"hours since {ds_hydrus_18O['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_hydrus_18O = ds_hydrus_18O.assign_coords(Time=("Time", date_hydrus_18O))

    # average observed soil water content of previous days
    window = 5
    df_thetap = pd.DataFrame(index=date_obs, columns=["doy", "theta", "sc"])
    df_thetap.loc[:, "doy"] = df_thetap.index.day_of_year
    df_thetap.loc[:, "theta"] = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
    df_thetap.loc[df_thetap.index[window - 1] :, "theta"] = (
        df_thetap.loc[:, "theta"].rolling(window=window).mean().iloc[window - 1 :].values
    )
    df_thetap.iloc[:window, 1] = onp.nan
    df_thetap_doy = df_thetap.groupby(by=["doy"], dropna=False).mean()
    theta_lower = onp.quantile(df_thetap_doy["theta"].values, 0.1)
    theta_upper = onp.quantile(df_thetap_doy["theta"].values, 0.9)
    cond1 = df_thetap["theta"] < theta_lower
    cond2 = (df_thetap["theta"] >= theta_lower) & (df_thetap["theta"] < theta_upper)
    cond3 = df_thetap["theta"] >= theta_upper
    df_thetap.loc[cond1, "sc"] = 1  # dry
    df_thetap.loc[cond2, "sc"] = 2  # normal
    df_thetap.loc[cond3, "sc"] = 3  # wet

    # compare best simulation with observations
    vars_obs = ["AET", "PERC", "dWEIGHT"]
    vars_sim = ["aet", "q_ss", "dS"]
    vars_bench = ["aet", "perc", "dS"]
    dict_obs_sim = {}
    for var_obs, var_sim, var_bench in zip(vars_obs, vars_sim, vars_bench):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm1[var_sim].isel(x=0, y=0).values
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm1, sim_vals, df_obs)
        # skip first seven days for warmup
        df_eval.loc[:"1997-01-07", :] = onp.nan
        # join benchmark simulations
        bench_vals = ds_hydrus_18O[var_bench].values
        df_bench = pd.DataFrame(index=ds_hydrus_18O["Time"].values, columns=["bench"])
        df_bench.loc[:, "bench"] = bench_vals
        df_eval = df_eval.join(df_bench)
        dict_obs_sim[var_obs] = df_eval
        # # plot observed and simulated time series
        # fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        # file_str = "%s.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # # plot cumulated observed and simulated time series
        # fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        # file_str = "%s_cum.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # fig = eval_utils.plot_obs_sim_cum_year_facet(
        #     df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        # )
        # file_str = "%s_cum_year_facet.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
    plt.close("all")

    # compare best 10 simulations with observations
    vars_obs = ["AET", "PERC", "dWEIGHT"]
    vars_sim = ["aet", "q_ss", "dS"]
    vars_bench = ["aet", "perc", "dS"]
    dict_obs_sim10 = {}
    for var_obs, var_sim, var_bench in zip(vars_obs, vars_sim, vars_bench):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm10[var_sim].isel(y=0).values.T
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm10, sim_vals, df_obs)
        # skip first seven days for warmup
        df_eval.loc[:"1997-01-07", :] = onp.nan
        # join benchmark simulations
        bench_vals = ds_hydrus_18O[var_bench].values
        df_bench = pd.DataFrame(index=ds_hydrus_18O["Time"].values, columns=["bench"])
        df_bench.loc[:, "bench"] = bench_vals
        df_eval = df_eval.join(df_bench)
        dict_obs_sim10[var_obs] = df_eval
        # # plot observed and simulated time series
        # fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        # file_str = "%s_best_10.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # # plot cumulated observed and simulated time series
        # fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        # file_str = "%s_cum_best_10.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # fig = eval_utils.plot_obs_sim_cum_year_facet(
        #     df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        # )
        # file_str = "%s_cum_year_facet_best_10.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
    plt.close("all")

    # compare best 100 simulations with observations
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
        # # plot observed and simulated time series
        # fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        # file_str = "%s_best_100.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # # plot cumulated observed and simulated time series
        # fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        # file_str = "%s_cum_best_100.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # fig = eval_utils.plot_obs_sim_cum_year_facet(
        #     df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        # )
        # file_str = "%s_cum_year_facet_best_100.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
    plt.close("all")

    # compare best simulation corresponding to best transport model with observations
    vars_obs = ["AET", "PERC", "dWEIGHT"]
    vars_sim = ["aet", "q_ss", "dS"]
    vars_bench = ["aet", "perc", "dS"]
    dict_obs_sim_for_tm = {}
    for var_obs, var_sim, var_bench in zip(vars_obs, vars_sim, vars_bench):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        sim_vals = ds_sim_hm_for_tm[var_sim].isel(y=0).values
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm1, sim_vals, df_obs)
        # skip first seven days for warmup
        df_eval.loc[:"1997-01-07", :] = onp.nan
        # join benchmark simulations
        bench_vals = ds_hydrus_18O[var_bench].values
        df_bench = pd.DataFrame(index=ds_hydrus_18O["Time"].values, columns=["bench"])
        df_bench.loc[:, "bench"] = bench_vals
        df_eval = df_eval.join(df_bench)
        dict_obs_sim_for_tm[var_obs] = df_eval
        # # plot observed and simulated time series
        # fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        # file_str = "%s_best_for_tm.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # # plot cumulated observed and simulated time series
        # fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        # file_str = "%s_cum_best_for_tm.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # fig = eval_utils.plot_obs_sim_cum_year_facet(
        #     df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        # )
        # file_str = "%s_cum_year_facet_best_for_tm.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
    plt.close("all")

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
        # # plot observed and simulated time series
        # fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        # file_str = "%s.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # # plot cumulated observed and simulated time series
        # fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        # file_str = "%s_cum.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
        # fig = eval_utils.plot_obs_sim_cum_year_facet(
        #     df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        # )
        # file_str = "%s_cum_year_facet.pdf" % (var_sim)
        # path_fig = base_path_figs / file_str
        # fig.savefig(path_fig, dpi=300)
    plt.close("all")

    # compare HYDRUS-1D simulations with observations
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

    df_params_metrics_hydrus = pd.DataFrame(index=["", "dry", "normal", "wet"])
    vars_obs = ["AET", "dWEIGHT", "PERC", "d18O_PERC"]
    vars_sim = ["aet", "dS", "perc", "d18O_perc_bs"]
    for var_obs, var_sim in zip(vars_obs, vars_sim):
        for i, sc in enumerate(["", "dry", "normal", "wet"]):
            obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
            df_obs.loc[:, "obs"] = obs_vals
            sim_vals = ds_hydrus_18O[var_sim].values
            # join observations on simulations
            df_eval = eval_utils.join_obs_on_sim(ds_hydrus_18O["Time"].values, sim_vals, df_obs)
            if i > 0:
                df_rows = pd.DataFrame(index=df_eval.index).join(df_thetap)
                rows = df_rows["sc"].values == i
                df_eval = df_eval.loc[rows, :]
            if var_sim in ["dS"]:
                df_eval.loc["2000-01":"2000-06", :] = onp.nan
            # skip first seven days for warmup
            df_eval.loc[:"1997-01-07", :] = onp.nan
            df_eval = df_eval.dropna()

            obs_vals = df_eval.loc[:, "obs"].values
            sim_vals = df_eval.loc[:, "sim"].values
            key_kge = "KGE_" + var_sim
            try:
                df_params_metrics_hydrus.loc[sc, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
            except ValueError:
                df_params_metrics_hydrus.loc[sc, key_kge] = onp.nan

    file = base_path_figs / "metrics_best_hydrus.txt"
    df_params_metrics_hydrus.to_csv(file, header=True, index=True, sep="\t")

    # plot time series for hydrus only
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(6, 3))
    axes[0, 0].plot(
        dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index,
        dict_obs["PREC_corr"].loc["1997-01-07":"1999", "obs"],
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
    ax2.plot(
        dict_obs_sim["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["AET"].loc["1997-01-07":"1999", "obs"],
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    ax2.plot(
        dict_obs_sim_hydrus["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_hydrus["AET"].loc["1997-01-07":"1999", "sim"],
        lw=1,
        color="black",
        ls="-.",
    )
    ax2.set_ylim(
        0, 15
    )
    ax2.set_ylabel("ET\n[mm]")
    axes[0, 1].plot(
        dict_obs["PREC_corr"].loc["2006":, :].index,
        dict_obs["PREC_corr"].loc["2006":, "obs"],
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    axes[0, 1].set_ylabel("PRECIP\n[mm]")
    axes[0, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[0, 1].set_ylim(
        0,
    )
    axes[0, 1].invert_yaxis()
    ax2 = axes[0, 1].twinx()
    ax2.plot(
        dict_obs_sim["AET"].loc["2006":, :].index,
        dict_obs_sim["AET"].loc["2006":, "obs"],
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    ax2.plot(
        dict_obs_sim_hydrus["AET"].loc["2006":, :].index,
        dict_obs_sim_hydrus["AET"].loc["2006":, "sim"],
        lw=1,
        color="black",
        ls="-.",
    )
    ax2.set_ylim(
        0, 15
    )
    ax2.set_ylabel("ET\n[mm]")
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
    axes[1, 1].plot(
        dict_obs_sim["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim["dWEIGHT"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[1, 1].plot(
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    axes[1, 1].set_ylabel("cum. $\Delta$S\n[mm]")
    axes[1, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[2, 0].plot(
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", "obs"],
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[2, 0].plot(
        dict_obs_sim_hydrus["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_hydrus["PERC"].loc["1997-01-07":"1999", "sim"],
        lw=1,
        color="black",
        ls="-.",
    )
    axes[2, 0].set_ylim(
        0,
    )
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 0].tick_params(axis="x", rotation=33)
    axes[2, 0].invert_yaxis()
    axes[2, 0].set_xlim(
        (
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[0],
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[-1],
        )
    )
    axes[2, 0].set_ylabel("PERC\n[mm]")
    axes[2, 0].set_xlabel(r"Time [year-month]")
    axes[2, 1].plot(
        dict_obs_sim["PERC"].loc["2006":, :].index,
        dict_obs_sim["PERC"].loc["2006":, "obs"],
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[2, 1].plot(
        dict_obs_sim_hydrus["PERC"].loc["2006":, :].index,
        dict_obs_sim_hydrus["PERC"].loc["2006":, "sim"],
        lw=1,
        color="black",
        ls="-.",
    )
    axes[2, 1].set_ylim(
        0,
    )
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 1].tick_params(axis="x", rotation=33)
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[2, 1].set_ylabel("PERC\n[mm]")
    axes[2, 1].set_xlabel(r"Time [year-month]")
    axes[0, 0].text(
        0.525,
        0.88,
        "(a)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 0].transAxes,
    )
    axes[1, 0].text(
        0.525,
        0.88,
        "(b)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 0].transAxes,
    )
    axes[2, 0].text(
        0.525,
        0.88,
        "(c)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 0].transAxes,
    )
    axes[0, 1].text(
        0.525,
        0.88,
        "(d)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 1].transAxes,
    )
    axes[1, 1].text(
        0.525,
        0.88,
        "(e)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 1].transAxes,
    )
    axes[2, 1].text(
        0.525,
        0.88,
        "(f)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 1].transAxes,
    )
    fig.tight_layout()
    file = f"prec_et_dS_perc_obs_sim_hydrus.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"prec_et_dS_perc_obs_sim_hydrus.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)


    fig, axes = plt.subplots(figsize=(6, 2))
    axes.plot(
        dict_obs_sim["PERC"].loc["2006":, :].index,
        dict_obs_sim["PERC"].loc["2006":, "obs"],
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes.plot(
        dict_obs_sim_hydrus["PERC"].loc["2006":, :].index,
        dict_obs_sim_hydrus["PERC"].loc["2006":, "sim"],
        lw=1,
        color="black",
        ls="-.",
    )
    axes.set_ylim(
        0,
    )
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes.set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes.set_ylabel("PERC\n[mm]")
    axes.set_xlabel(r"Time [year-month]")
    df_eval = pd.DataFrame()
    df_eval.loc[:, "obs"] = dict_obs_sim_hydrus["PERC"].loc["2006":, "obs"]
    df_eval.loc[:, "sim"] = dict_obs_sim_hydrus["PERC"].loc["2006":, "sim"]
    df_eval = df_eval.dropna()
    obs_vals = df_eval.loc[:, "obs"].values.astype(float)
    sim_vals = df_eval.loc[:, "sim"].values.astype(float)
    kge_val = eval_utils.calc_kge(obs_vals, sim_vals)
    axes.text(
        0.7,
        0.88,
        f"KGE = {kge_val:.2f}",
        fontsize=9,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes.transAxes,
    )
    fig.tight_layout()
    file = f"perc_obs_sim_2006_2007_hydrus.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"perc_obs_sim_2006_2007_hydrus.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)


    fig, axes = plt.subplots(figsize=(6, 2))
    axes.plot(
        dict_obs_sim["PERC"].loc["2006":, :].index,
        dict_obs_sim["PERC"].loc["2006":, "obs"],
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes.plot(
        dict_obs_sim100["PERC"].loc["2006":, :].index,
        dict_obs_sim100["PERC"].loc["2006":, "sim99"],
        lw=1,
        color="red",
        ls="-.",
    )
    axes.set_ylim(
        0,
    )
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes.set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes.set_ylabel("PERC\n[mm]")
    axes.set_xlabel(r"Time [year-month]")
    df_eval = pd.DataFrame()
    df_eval.loc[:, "obs"] = dict_obs_sim100["PERC"].loc["2006":, "obs"]
    df_eval.loc[:, "sim"] = dict_obs_sim100["PERC"].loc["2006":, "sim99"]
    df_eval = df_eval.dropna()
    obs_vals = df_eval.loc[:, "obs"].values.astype(float)
    sim_vals = df_eval.loc[:, "sim"].values.astype(float)
    kge_val = eval_utils.calc_kge(obs_vals, sim_vals)
    axes.text(
        0.7,
        0.88,
        f"KGE = {kge_val:.2f}",
        fontsize=9,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes.transAxes,
    )
    fig.tight_layout()
    file = f"perc_obs_sim_2006_2007_roger.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"perc_obs_sim_2006_2007_roger.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)


    # plot cumulated precipitation, evapotranspiration, soil storage change and percolation
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(6, 3))
    axes[0, 0].plot(
        dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index,
        dict_obs["PREC_corr"].loc["1997-01-07":"1999", "obs"].cumsum(),
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
    ax2.plot(
        dict_obs_sim["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["AET"].loc["1997-01-07":"1999", "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    ax2.plot(
        dict_obs_sim["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["AET"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    ax2.set_ylabel("ET\n[mm]")
    axes[0, 1].plot(
        dict_obs["PREC_corr"].loc["2006":, :].index,
        dict_obs["PREC_corr"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    axes[0, 1].set_ylabel("PRECIP\n[mm]")
    axes[0, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[0, 1].set_ylim(
        0,
    )
    axes[0, 1].invert_yaxis()
    ax2 = axes[0, 1].twinx()
    ax2.plot(
        dict_obs_sim["AET"].loc["2006":, :].index,
        dict_obs_sim["AET"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    ax2.plot(
        dict_obs_sim["AET"].loc["2006":, :].index,
        dict_obs_sim["AET"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[1, 0].plot(
        dict_obs_sim["dWEIGHT"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["dWEIGHT"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[1, 1].plot(
        dict_obs_sim["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim["dWEIGHT"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[1, 1].plot(
        dict_obs_sim["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim["dWEIGHT"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
    )
    axes[1, 1].plot(
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    axes[1, 1].set_ylabel("cum. $\Delta$S\n[mm]")
    axes[1, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[2, 0].plot(
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[2, 0].plot(
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 0].tick_params(axis="x", rotation=33)
    axes[2, 0].invert_yaxis()
    axes[2, 0].set_xlim(
        (
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[0],
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[-1],
        )
    )
    axes[2, 0].set_ylabel("PERC\n[mm]")
    axes[2, 0].set_xlabel(r"Time [year-month]")
    axes[2, 1].plot(
        dict_obs_sim["PERC"].loc["2006":, :].index,
        dict_obs_sim["PERC"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[2, 1].plot(
        dict_obs_sim["PERC"].loc["2006":, :].index,
        dict_obs_sim["PERC"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 1].tick_params(axis="x", rotation=33)
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[2, 1].set_ylabel("PERC\n[mm]")
    axes[2, 1].set_xlabel(r"Time [year-month]")
    axes[0, 0].text(
        0.525,
        0.88,
        "(a)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 0].transAxes,
    )
    axes[1, 0].text(
        0.525,
        0.88,
        "(b)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 0].transAxes,
    )
    axes[2, 0].text(
        0.525,
        0.88,
        "(c)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 0].transAxes,
    )
    axes[0, 1].text(
        0.525,
        0.88,
        "(d)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 1].transAxes,
    )
    axes[1, 1].text(
        0.525,
        0.88,
        "(e)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 1].transAxes,
    )
    axes[2, 1].text(
        0.525,
        0.88,
        "(f)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 1].transAxes,
    )
    fig.tight_layout()
    file = f"prec_et_dS_perc_obs_sim_cumulated_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"prec_et_dS_perc_obs_sim_cumulated_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)


    # plot cumulated precipitation, evapotranspiration, soil storage change and percolation
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(6, 3))
    axes[0, 0].plot(
        dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index,
        dict_obs["PREC_corr"].loc["1997-01-07":"1999", "obs"].cumsum(),
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
    ax2.plot(
        dict_obs_sim["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["AET"].loc["1997-01-07":"1999", "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    ax2.plot(
        dict_obs_sim["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["AET"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    ax2.set_ylabel("ET\n[mm]")
    axes[0, 1].plot(
        dict_obs["PREC_corr"].loc["2006":, :].index,
        dict_obs["PREC_corr"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    axes[0, 1].set_ylabel("PRECIP\n[mm]")
    axes[0, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[0, 1].set_ylim(
        0,
    )
    axes[0, 1].invert_yaxis()
    ax2 = axes[0, 1].twinx()
    ax2.plot(
        dict_obs_sim["AET"].loc["2006":, :].index,
        dict_obs_sim["AET"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    ax2.plot(
        dict_obs_sim["AET"].loc["2006":, :].index,
        dict_obs_sim["AET"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[1, 0].plot(
        dict_obs_sim["dWEIGHT"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["dWEIGHT"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[1, 1].plot(
        dict_obs_sim["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim["dWEIGHT"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[1, 1].plot(
        dict_obs_sim["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim["dWEIGHT"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
    )
    axes[1, 1].plot(
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    axes[1, 1].set_ylabel("cum. $\Delta$S\n[mm]")
    axes[1, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[2, 0].plot(
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[2, 0].plot(
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 0].tick_params(axis="x", rotation=33)
    axes[2, 0].invert_yaxis()
    axes[2, 0].set_xlim(
        (
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[0],
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[-1],
        )
    )
    axes[2, 0].set_ylabel("PERC\n[mm]")
    axes[2, 0].set_xlabel(r"Time [year-month]")
    axes[2, 1].plot(
        dict_obs_sim["PERC"].loc["2006":, :].index,
        dict_obs_sim["PERC"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[2, 1].plot(
        dict_obs_sim["PERC"].loc["2006":, :].index,
        dict_obs_sim["PERC"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 1].tick_params(axis="x", rotation=33)
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[2, 1].set_ylabel("PERC\n[mm]")
    axes[2, 1].set_xlabel(r"Time [year-month]")
    axes[0, 0].text(
        0.525,
        0.88,
        "(a)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 0].transAxes,
    )
    axes[1, 0].text(
        0.525,
        0.88,
        "(b)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 0].transAxes,
    )
    axes[2, 0].text(
        0.525,
        0.88,
        "(c)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 0].transAxes,
    )
    axes[0, 1].text(
        0.525,
        0.88,
        "(d)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 1].transAxes,
    )
    axes[1, 1].text(
        0.525,
        0.88,
        "(e)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 1].transAxes,
    )
    axes[2, 1].text(
        0.525,
        0.88,
        "(f)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 1].transAxes,
    )
    fig.tight_layout()
    file = f"prec_et_dS_perc_obs_sim_cumulated_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"prec_et_dS_perc_obs_sim_cumulated_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)



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
            alpha=0.8,
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
            alpha=0.8,
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
        )
    axes[1, 0].plot(
        dict_obs_sim_hydrus["dWEIGHT"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_hydrus["dWEIGHT"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
        alpha=0.8,
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
        alpha=0.8,
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
            alpha=0.8,
        )
    axes[2, 0].plot(
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", "obs"].cumsum(),
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
            alpha=0.8,
        )
    axes[2, 1].plot(
        dict_obs_sim["PERC"].loc["2006":, :].index,
        dict_obs_sim["PERC"].loc["2006":, "obs"].cumsum(),
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
    axes[0, 0].text(
        0.525,
        0.88,
        "(a)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 0].transAxes,
    )
    axes[1, 0].text(
        0.525,
        0.88,
        "(b)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 0].transAxes,
    )
    axes[2, 0].text(
        0.525,
        0.88,
        "(c)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 0].transAxes,
    )
    axes[0, 1].text(
        0.525,
        0.88,
        "(d)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 1].transAxes,
    )
    axes[1, 1].text(
        0.525,
        0.88,
        "(e)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 1].transAxes,
    )
    axes[2, 1].text(
        0.525,
        0.88,
        "(f)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 1].transAxes,
    )
    fig.tight_layout()
    file = f"prec_et_dS_perc_obs_sim_cumulated_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"prec_et_dS_perc_obs_sim_cumulated_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    # compare best 10 simulations with observations
    nx = ds_sim_hm10.dims["x"]
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
            dict_obs_sim10["AET"].loc["1997-01-07":"1999", :].index,
            dict_obs_sim10["AET"].loc["1997-01-07":"1999", f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.8,
        )
    ax2.plot(
        dict_obs_sim10["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim10["AET"].loc["1997-01-07":"1999", "obs"].cumsum(),
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
    ax2.set_ylabel("ET\n[mm]")
    axes[0, 1].plot(
        dict_obs["PREC_corr"].loc["2006":, :].index,
        dict_obs["PREC_corr"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    axes[0, 1].set_ylabel("PRECIP\n[mm]")
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
            dict_obs_sim10["AET"].loc["2006":, :].index,
            dict_obs_sim10["AET"].loc["2006":, f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.8,
        )
    ax2.plot(
        dict_obs_sim10["AET"].loc["2006":, :].index,
        dict_obs_sim10["AET"].loc["2006":, "obs"].cumsum(),
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
            dict_obs_sim10["dWEIGHT"].loc["1997-01-07":"1999", :].index,
            dict_obs_sim10["dWEIGHT"].loc["1997-01-07":"1999", f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
        )
    axes[1, 0].plot(
        dict_obs_sim_hydrus["dWEIGHT"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_hydrus["dWEIGHT"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
        alpha=0.8,
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
            dict_obs_sim10["dWEIGHT"].loc["2006":, :].index,
            dict_obs_sim10["dWEIGHT"].loc["2006":, f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
        )
    axes[1, 1].plot(
        dict_obs_sim10["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim10["dWEIGHT"].loc["2006":, "obs"].cumsum(),
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
        alpha=0.8,
    )
    axes[1, 1].set_ylabel("cum. $\Delta$S\n[mm]")
    axes[1, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    for nrow in range(nx):
        axes[2, 0].plot(
            dict_obs_sim10["PERC"].loc["1997-01-07":"1999", :].index,
            dict_obs_sim10["PERC"].loc["1997-01-07":"1999", f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.8,
        )
    axes[2, 0].plot(
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim["PERC"].loc["1997-01-07":"1999", "obs"].cumsum(),
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
            dict_obs_sim10["PERC"].loc["2006":, :].index,
            dict_obs_sim10["PERC"].loc["2006":, f"sim{nrow}"].cumsum(),
            lw=1,
            color="red",
            ls="-",
            alpha=0.8,
        )
    axes[2, 1].plot(
        dict_obs_sim["PERC"].loc["2006":, :].index,
        dict_obs_sim["PERC"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    # axes[2, 1].plot(dict_obs_sim_hydrus['PERC'].loc['2006':, :].index, dict_obs_sim_hydrus['PERC'].loc['2006':, 'sim'].cumsum(),
    #               lw=1, color='gray', ls='-.')
    axes[2, 1].set_ylim(
        0,
    )
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[2, 1].set_ylabel("PERC\n[mm]")
    axes[2, 1].set_xlabel(r"Time [year-month]")
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 1].tick_params(axis="x", rotation=33)
    axes[0, 0].text(
        0.525,
        0.88,
        "(a)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 0].transAxes,
    )
    axes[1, 0].text(
        0.525,
        0.88,
        "(b)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 0].transAxes,
    )
    axes[2, 0].text(
        0.525,
        0.88,
        "(c)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 0].transAxes,
    )
    axes[0, 1].text(
        0.525,
        0.88,
        "(d)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 1].transAxes,
    )
    axes[1, 1].text(
        0.525,
        0.88,
        "(e)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 1].transAxes,
    )
    axes[2, 1].text(
        0.525,
        0.88,
        "(f)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 1].transAxes,
    )
    fig.tight_layout()
    file = f"prec_et_dS_perc_obs_sim_cumulated_best_10_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"prec_et_dS_perc_obs_sim_cumulated_best_10_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    # plot evapotranspiration, soil storage change and percolation
    years = onp.arange(1997, 2008).tolist()
    for year in years:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 4))
        axes[0].bar(
            dict_obs["PREC_corr"].loc[f"{year}", :].index,
            dict_obs["PREC_corr"].loc[f"{year}", "obs"],
            width=0.1,
            color="blue",
            align="edge",
            edgecolor="blue",
        )
        axes[0].set_ylabel("PRECIP\n[mm/day]")
        axes[0].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[0].set_ylim(
            0,
        )
        axes[1].plot(
            dict_obs_sim["AET"].loc[f"{year}", :].index,
            dict_obs_sim["AET"].loc[f"{year}", "sim"],
            lw=1,
            color="red",
            ls="-.",
            alpha=1,
        )
        axes[1].plot(
            dict_obs_sim_hydrus["AET"].loc[f"{year}", :].index,
            dict_obs_sim_hydrus["AET"].loc[f"{year}", "sim"],
            lw=1,
            color="black",
            ls="-.",
            alpha=1,
        )
        axes[1].plot(
            dict_obs_sim["AET"].loc[f"{year}", :].index,
            dict_obs_sim["AET"].loc[f"{year}", "obs"],
            lw=1,
            color="blue",
            ls="-",
            alpha=0.5,
        )
        axes[1].set_ylabel("ET\n[mm/day]")
        axes[1].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[1].set_ylim(
            0,
        )
        axes[2].plot(
            dict_obs_sim["dWEIGHT"].loc[f"{year}", :].index,
            dict_obs_sim["dWEIGHT"].loc[f"{year}", "sim"],
            lw=1,
            color="red",
            ls="-.",
            alpha=1,
        )
        axes[2].plot(
            dict_obs_sim_hydrus["dWEIGHT"].loc[f"{year}", :].index,
            dict_obs_sim_hydrus["dWEIGHT"].loc[f"{year}", "sim"],
            lw=1,
            color="black",
            ls="-.",
            alpha=1,
        )
        axes[2].plot(
            dict_obs_sim["dWEIGHT"].loc[f"{year}", :].index,
            dict_obs_sim["dWEIGHT"].loc[f"{year}", "obs"],
            lw=1,
            color="blue",
            ls="-",
            alpha=0.5,
        )
        axes[2].set_ylabel("$\Delta$S\n[mm/day]")
        axes[2].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[3].plot(
            dict_obs_sim["PERC"].loc[f"{year}", :].index,
            dict_obs_sim["PERC"].loc[f"{year}", "sim"],
            lw=1,
            color="red",
            ls="-.",
            alpha=1,
        )
        axes[3].plot(
            dict_obs_sim_hydrus["PERC"].loc[f"{year}", :].index,
            dict_obs_sim_hydrus["PERC"].loc[f"{year}", "sim"],
            lw=1,
            color="black",
            ls="-.",
            alpha=1,
        )
        axes[3].plot(
            dict_obs_sim["PERC"].loc[f"{year}", :].index,
            dict_obs_sim["PERC"].loc[f"{year}", "obs"],
            lw=1,
            color="blue",
            ls="-",
            alpha=0.5,
        )
        axes[3].set_ylabel("PERC\n[mm/day]")
        axes[3].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[3].set_ylim(
            0,
        )
        axes[3].set_xlabel(r"Time [year-month]")
        fig.tight_layout()
        file = f"prec_et_dS_perc_obs_sim_{year}_optimized_with_KGE_multi.png"
        path = base_path_figs / file
        fig.savefig(path, dpi=300)
        file = f"prec_et_dS_perc_obs_sim_{year}_optimized_with_KGE_multi.pdf"
        path = base_path_figs / file
        fig.savefig(path, dpi=300)

    # plot evapotranspiration, soil storage change and percolation corresponding to best transport model
    years = onp.arange(1997, 2008).tolist()
    for year in years:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 4))
        axes[0].bar(
            dict_obs["PREC_corr"].loc[f"{year}", :].index,
            dict_obs["PREC_corr"].loc[f"{year}", "obs"],
            width=0.1,
            color="blue",
            align="edge",
            edgecolor="blue",
        )
        axes[0].set_ylabel("PRECIP\n[mm/day]")
        axes[0].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[0].set_ylim(
            0,
        )
        axes[1].plot(
            dict_obs_sim["AET"].loc[f"{year}", :].index,
            dict_obs_sim_for_tm["AET"].loc[f"{year}", "sim"],
            lw=1,
            color="red",
            ls="-.",
            alpha=1,
        )
        axes[1].plot(
            dict_obs_sim["AET"].loc[f"{year}", :].index,
            dict_obs_sim_for_tm["AET"].loc[f"{year}", "obs"],
            lw=1,
            color="blue",
            ls="-",
            alpha=0.5,
        )
        axes[1].set_ylabel("ET\n[mm/day]")
        axes[1].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[1].set_ylim(
            0,
        )
        axes[2].plot(
            dict_obs_sim["dWEIGHT"].loc[f"{year}", :].index,
            dict_obs_sim_for_tm["dWEIGHT"].loc[f"{year}", "sim"],
            lw=1,
            color="red",
            ls="-.",
            alpha=1,
        )
        axes[2].plot(
            dict_obs_sim["dWEIGHT"].loc[f"{year}", :].index,
            dict_obs_sim_for_tm["dWEIGHT"].loc[f"{year}", "obs"],
            lw=1,
            color="blue",
            ls="-",
            alpha=0.5,
        )
        axes[2].set_ylabel("$\Delta$S\n[mm/day]")
        axes[2].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[3].plot(
            dict_obs_sim["PERC"].loc[f"{year}", :].index,
            dict_obs_sim_for_tm["PERC"].loc[f"{year}", "sim"],
            lw=1,
            color="red",
            ls="-.",
            alpha=1,
        )
        axes[3].plot(
            dict_obs_sim["PERC"].loc[f"{year}", :].index,
            dict_obs_sim_for_tm["PERC"].loc[f"{year}", "obs"],
            lw=1,
            color="blue",
            ls="-",
            alpha=0.5,
        )
        axes[3].set_ylabel("PERC\n[mm/day]")
        axes[3].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[3].set_ylim(
            0,
        )
        axes[3].set_xlabel(r"Time [year-month]")
        fig.tight_layout()
        file = f"prec_et_dS_perc_obs_sim_{year}_optimized_with_KGE_multi_for_best_tm.png"
        path = base_path_figs / file
        fig.savefig(path, dpi=300)
        file = f"prec_et_dS_perc_obs_sim_{year}_optimized_with_KGE_multi_for_best_tm.pdf"
        path = base_path_figs / file
        fig.savefig(path, dpi=300)

    # plot cumulated precipitation, evapotranspiration, soil storage change and percolation corresponding to best transport model
    fig, axes = plt.subplots(3, 2, sharex="col", figsize=(6, 3))
    axes[0, 0].plot(
        dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index,
        dict_obs["PREC_corr"].loc["1997-01-07":"1999", "obs"].cumsum(),
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
    ax2.plot(
        dict_obs_sim_for_tm["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_for_tm["AET"].loc["1997-01-07":"1999", "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    ax2.plot(
        dict_obs_sim_for_tm["AET"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_for_tm["AET"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    ax2.set_ylabel("ET\n[mm]")
    axes[0, 1].plot(
        dict_obs["PREC_corr"].loc["2006":, :].index,
        dict_obs["PREC_corr"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=1,
    )
    axes[0, 1].set_ylabel("PRECIP\n[mm]")
    axes[0, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[0, 1].set_ylim(
        0,
    )
    axes[0, 1].invert_yaxis()
    ax2 = axes[0, 1].twinx()
    ax2.plot(
        dict_obs_sim_for_tm["AET"].loc["2006":, :].index,
        dict_obs_sim_for_tm["AET"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    ax2.plot(
        dict_obs_sim_for_tm["AET"].loc["2006":, :].index,
        dict_obs_sim_for_tm["AET"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[1, 0].plot(
        dict_obs_sim_for_tm["dWEIGHT"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_for_tm["dWEIGHT"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[1, 1].plot(
        dict_obs_sim_for_tm["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim_for_tm["dWEIGHT"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[1, 1].plot(
        dict_obs_sim_for_tm["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim_for_tm["dWEIGHT"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
    )
    axes[1, 1].plot(
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, :].index,
        dict_obs_sim_hydrus["dWEIGHT"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="black",
        ls="-.",
    )
    axes[1, 1].set_ylabel("cum. $\Delta$S\n[mm]")
    axes[1, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[2, 0].plot(
        dict_obs_sim_for_tm["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_for_tm["PERC"].loc["1997-01-07":"1999", "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[2, 0].plot(
        dict_obs_sim_for_tm["PERC"].loc["1997-01-07":"1999", :].index,
        dict_obs_sim_for_tm["PERC"].loc["1997-01-07":"1999", "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 0].tick_params(axis="x", rotation=33)
    axes[2, 0].invert_yaxis()
    axes[2, 0].set_xlim(
        (
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[0],
            dict_obs["PREC_corr"].loc["1997-01-07":"1999", :].index[-1],
        )
    )
    axes[2, 0].set_ylabel("PERC\n[mm]")
    axes[2, 0].set_xlabel(r"Time [year-month]")
    axes[2, 1].plot(
        dict_obs_sim_for_tm["PERC"].loc["2006":, :].index,
        dict_obs_sim_for_tm["PERC"].loc["2006":, "obs"].cumsum(),
        lw=1,
        color="blue",
        ls="-",
        alpha=0.5,
    )
    axes[2, 1].plot(
        dict_obs_sim_for_tm["PERC"].loc["2006":, :].index,
        dict_obs_sim_for_tm["PERC"].loc["2006":, "sim"].cumsum(),
        lw=1,
        color="red",
        ls="-.",
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
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 1].tick_params(axis="x", rotation=33)
    axes[2, 1].invert_yaxis()
    axes[2, 1].set_xlim(
        (dict_obs["PREC_corr"].loc["2006":, :].index[0], dict_obs["PREC_corr"].loc["2006":, :].index[-1])
    )
    axes[2, 1].set_ylabel("PERC\n[mm]")
    axes[2, 1].set_xlabel(r"Time [year-month]")
    axes[0, 0].text(
        0.525,
        0.88,
        "(a)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 0].transAxes,
    )
    axes[1, 0].text(
        0.525,
        0.88,
        "(b)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 0].transAxes,
    )
    axes[2, 0].text(
        0.525,
        0.88,
        "(c)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 0].transAxes,
    )
    axes[0, 1].text(
        0.525,
        0.88,
        "(d)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[0, 1].transAxes,
    )
    axes[1, 1].text(
        0.525,
        0.88,
        "(e)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[1, 1].transAxes,
    )
    axes[2, 1].text(
        0.525,
        0.88,
        "(f)",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes[2, 1].transAxes,
    )
    fig.tight_layout()
    file = "prec_et_dS_perc_obs_sim_cumulated_optimized_with_KGE_multi_for_best_tm.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = "prec_et_dS_perc_obs_sim_cumulated_optimized_with_KGE_multi_for_best_tm.pdf"
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

    # compare best model runs
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 6), gridspec_kw = {'height_ratios':[2.5,1,2,1,1]})
    df_obs = pd.DataFrame(index=date_obs)
    df_obs.loc[:, "d18O_prec"] = ds_obs["d18O_PREC"].isel(x=0, y=0).values
    ax.flatten()[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].bfill(), "-", color="blue")
    ax.flatten()[0].scatter(df_obs.index, df_obs.loc[:, "d18O_prec"], color="blue", s=1)
    ax.flatten()[0].set_ylabel(r"$\delta^{18}$$O_{PRECIP}$ [‰]")
    ax.flatten()[0].set_ylim([-20, 0])
    ax.flatten()[0].set_xlim(df_obs.index[0], df_obs.index[-1])
    for i, tm_structure in enumerate(tm_structures):
        idx_best = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]["KGE_C_iso_q_ss"].idxmax()
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        states_tm_file = (
            base_path
            / "svat_oxygen18_monte_carlo"
            / "output"
            / f"states_{tms}_monte_carlo.nc"
        )
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
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
    file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_optimized_with_KGE_multi.pdf"
    fig.savefig(file, dpi=300)

    # insets for 3 years
    years = onp.arange(1997, 2006).tolist()
    for year in years:
        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 6), gridspec_kw = {'height_ratios':[2.5,1,2,1,1]})
        df_obs = pd.DataFrame(index=date_obs)
        df_obs.loc[:, "d18O_prec"] = ds_obs["d18O_PREC"].isel(x=0, y=0).values
        df_obs = df_obs.loc[f"{year}":f"{year+2}", "d18O_prec"].to_frame()
        ax.flatten()[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].bfill(), "-", color="blue")
        ax.flatten()[0].scatter(df_obs.index, df_obs.loc[:, "d18O_prec"], color="blue", s=1)
        ax.flatten()[0].set_ylabel(r"$\delta^{18}$$O_{PRECIP}$ [‰]")
        ax.flatten()[0].set_ylim(-20, 0)
        ax.flatten()[0].set_xlim(df_obs.index[0], df_obs.index[-1])
        for i, tm_structure in enumerate(tm_structures):
            idx_best = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]["KGE_C_iso_q_ss"].idxmax()
            tms = tm_structure.replace(" ", "_")
            # load transport simulation
            states_tm_file = (
                base_path
                / "svat_oxygen18_monte_carlo"
                / "output"
                / f"states_{tms}_monte_carlo.nc"
            )
            ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
            days_sim_tm = ds_sim_tm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
            date_sim_tm = num2date(
                days_sim_tm,
                units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}",
                calendar="standard",
                only_use_cftime_datetimes=False,
            )
            ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
            ds_sim_tm_year = ds_sim_tm.sel(Time=slice(f"{year}-01-01", f"{year + 2}-12-31"))
            ds_hydrus_18O_year = ds_hydrus_18O.sel(Time=slice(f"{year}-01-01", f"{year + 2}-12-31"))
            # join observations on simulations
            obs_vals = ds_obs["d18O_PERC"].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
            df_obs.loc[:, "obs"] = obs_vals
            df_obs = df_obs.loc[f"{year}":f"{year+2}", "obs"].to_frame()
            ax.flatten()[i + 1].plot(
                ds_sim_tm_year["Time"].values,
                ds_sim_tm_year["C_iso_q_ss"].isel(x=idx_best, y=0).values,
                color="red",
                lw=1,
            )
            ax.flatten()[i + 1].plot(
                ds_hydrus_18O_year["Time"].values, ds_hydrus_18O_year["d18O_perc"].values, color="black", lw=1
            )
            ax.flatten()[i + 1].scatter(df_obs.index, df_obs.iloc[:, 0], color="blue", s=1)
            ax[i + 1].set_ylabel("%s\n$\delta^{18}$$O_{PERC}$ [‰]" % (_LABS_TM[tm_structure]))
            if tm_structure in ["piston"]:
                ax.flatten()[i + 1].set_ylim((-20, -4))
            else:
                ax.flatten()[i + 1].set_ylim((-15, -7))
            ax.flatten()[i + 1].set_xlim(ds_sim_tm_year["Time"].values[0], ds_sim_tm_year["Time"].values[-1])
        ax[-1].set_xlabel("Time [year]")
        fig.tight_layout()
        file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_optimized_with_KGE_multi_{year}_{year+2}.png"
        fig.savefig(file, dpi=300)

    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 6))
    tm_structures_extra = [
        "preferential-power",
        "older-preference-power",
        "advection-dispersion-kumaraswamy",
        "time-variant advection-dispersion-kumaraswamy",
    ]
    df_obs = pd.DataFrame(index=date_obs)
    df_obs.loc[:, "d18O_prec"] = ds_obs["d18O_PREC"].isel(x=0, y=0).values
    ax.flatten()[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].bfill(), "-", color="blue")
    ax.flatten()[0].scatter(df_obs.index, df_obs.loc[:, "d18O_prec"], color="blue", s=1)
    ax.flatten()[0].set_ylabel(r"$\delta^{18}$$O_{PRECIP}$ [‰]")
    ax.flatten()[0].set_ylim([-20, 0])
    ax.flatten()[0].set_xlim(df_obs.index[0], df_obs.index[-1])
    for i, tm_structure in enumerate(tm_structures_extra):
        idx_best = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]["KGE_C_iso_q_ss"].idxmax()
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        states_tm_file = (
            base_path
            / "svat_oxygen18_monte_carlo"
            / "output"
            / f"states_{tms}_monte_carlo.nc"
        )
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
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
        ax.flatten()[i + 1].set_ylim((-15, -7))
        ax.flatten()[i + 1].set_xlim(ds_sim_tm["Time"].values[0], ds_sim_tm["Time"].values[-1])
    ax[-1].set_xlabel("Time [year]")
    fig.tight_layout()
    file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_extra1_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6, 3))
    tm_structures_extra = ["advection-dispersion-kumaraswamy", "time-variant advection-dispersion-kumaraswamy"]
    df_obs = pd.DataFrame(index=date_obs)
    df_obs.loc[:, "d18O_prec"] = ds_obs["d18O_PREC"].isel(x=0, y=0).values
    ax.flatten()[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].bfill(), "-", color="blue")
    ax.flatten()[0].scatter(df_obs.index, df_obs.loc[:, "d18O_prec"], color="blue", s=1)
    ax.flatten()[0].set_ylabel(r"$\delta^{18}$$O_{PRECIP}$ [‰]")
    ax.flatten()[0].set_ylim([-20, 0])
    ax.flatten()[0].set_xlim(df_obs.index[0], df_obs.index[-1])
    for i, tm_structure in enumerate(tm_structures_extra):
        idx_best = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]["KGE_C_iso_q_ss"].idxmax()
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        states_tm_file = (
            base_path
            / "svat_oxygen18_monte_carlo"
            / "output"
            / f"states_{tms}_monte_carlo.nc"
        )
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
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
        ax.flatten()[i + 1].set_ylim((-15, -7))
        ax.flatten()[i + 1].set_xlim(ds_sim_tm["Time"].values[0], ds_sim_tm["Time"].values[-1])
    ax[-1].set_xlabel("Time [year]")
    fig.tight_layout()
    file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_extra2_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)

    fig, ax = plt.subplots(4, 1, sharey=False, figsize=(6, 5))
    for i, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        states_tm_file = (
            base_path
            / "svat_oxygen18_monte_carlo"
            / "output"
            / f"states_{tms}_monte_carlo.nc"
        )
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
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
        sim_vals = ds_sim_tm["C_iso_q_ss"].isel(y=0).values
        sim_vals = onp.where((sim_vals > 0) | (sim_vals < -20), onp.nan, sim_vals)
        sim_vals_avg = onp.nanmean(sim_vals, axis=0)
        sim_vals_5 = onp.nanquantile(sim_vals, 0.05, axis=0)
        sim_vals_50 = onp.nanmedian(sim_vals, axis=0)
        sim_vals_95 = onp.nanquantile(sim_vals, 0.95, axis=0)
        ax.flatten()[i].plot(ds_sim_tm["Time"].values, sim_vals_avg, ls="--", color="red", lw=1)
        ax.flatten()[i].plot(ds_sim_tm["Time"].values, sim_vals_50, ls="-", color="red", lw=1)
        ax.flatten()[i].fill_between(
            ds_sim_tm["Time"].values, sim_vals_5, sim_vals_95, color="red", edgecolor=None, alpha=0.2
        )
        ax.flatten()[i].scatter(date_obs, obs_vals, color="blue", s=1)
        ax.flatten()[i].set_title(_LABS_TM[tm_structure])
        ax[i].set_ylabel(r"$\delta^{18}$$O_{PERC}$ [‰]")
        ax.flatten()[i].set_ylim((-20, 0))
        ax.flatten()[i].set_xlim(ds_sim_tm["Time"].values[0], ds_sim_tm["Time"].values[-1])
    ax[-1].set_xlabel("Time [year]")
    fig.tight_layout()
    file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_conf_int_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)

    fig, ax = plt.subplots(4, 1, sharey=False, figsize=(6, 5))
    tm_structures_extra = [
        "preferential-power",
        "older-preference-power",
        "advection-dispersion-kumaraswamy",
        "time-variant advection-dispersion-kumaraswamy",
    ]
    for i, tm_structure in enumerate(tm_structures_extra):
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        states_tm_file = (
            base_path
            / "svat_oxygen18_monte_carlo"
            / "output"
            / f"states_{tms}_monte_carlo.nc"
        )
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
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
        sim_vals = ds_sim_tm["C_iso_q_ss"].isel(y=0).values
        sim_vals = onp.where((sim_vals > 0) | (sim_vals < -20), onp.nan, sim_vals)
        sim_vals_avg = onp.nanmean(sim_vals, axis=0)
        sim_vals_5 = onp.nanquantile(sim_vals, 0.05, axis=0)
        sim_vals_50 = onp.nanmedian(sim_vals, axis=0)
        sim_vals_95 = onp.nanquantile(sim_vals, 0.95, axis=0)
        sim_vals_hydrus = ds_hydrus_18O["d18O_perc"].values
        ax.flatten()[i].plot(ds_sim_tm["Time"].values, sim_vals_avg, ls="--", color="red", lw=1)
        ax.flatten()[i].plot(ds_sim_tm["Time"].values, sim_vals_50, ls="-", color="red", lw=1)
        ax.flatten()[i].fill_between(
            ds_sim_tm["Time"].values, sim_vals_5, sim_vals_95, color="red", edgecolor=None, alpha=0.2
        )
        ax.flatten()[i].plot(ds_hydrus_18O["Time"].values, sim_vals_hydrus, color="black", lw=1)
        ax.flatten()[i].scatter(date_obs, obs_vals, color="blue", s=1)
        ax.flatten()[i].set_title(_LABS_TM[tm_structure])
        ax[i].set_ylabel(r"$\delta^{18}$$O_{PERC}$ [‰]")
        ax.flatten()[i].set_ylim((-20, 0))
        ax.flatten()[i].set_xlim(ds_sim_tm["Time"].values[0], ds_sim_tm["Time"].values[-1])
    ax[-1].set_xlabel("Time [year]")
    fig.tight_layout()
    file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_conf_int_optimized_with_KGE_multi_extra.png"
    fig.savefig(file, dpi=300)


    plt.close("all")
    return


if __name__ == "__main__":
    main()
