from pathlib import Path
import os
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import click
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

    # load HYDRUS-1D benchmarks
    # travel time simulations
    states_hydrus_file = base_path / "hydrus_benchmark" / "states_hydrus_tt.nc"
    ds_hydrus_tt = xr.open_dataset(states_hydrus_file, engine="h5netcdf", decode_times=False)
    days_hydrus_tt = ds_hydrus_tt["Time"].values / 24
    date_hydrus_tt = num2date(
        days_hydrus_tt,
        units=f"hours since {ds_hydrus_tt['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_hydrus_tt = ds_hydrus_tt.assign_coords(Time=("Time", date_hydrus_tt))

    # average observed soil water content of previous days
    window = 5
    df_thetap = pd.DataFrame(index=date_obs, columns=["doy", "theta", "sc"])
    df_thetap.loc[:, "doy"] = df_thetap.index.day_of_year
    df_thetap.loc[:, "theta"] = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
    df_thetap.loc[df_thetap.index[window - 1] :, "theta"] = (
        df_thetap.loc[:, "theta"].rolling(window=window).mean().iloc[window - 1 :].values
    )
    df_thetap.iloc[:window, 1] = onp.nan

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
        file = base_path / "svat_oxygen18_monte_carlo" / "figures" / f"params_metrics_{tms}.txt"
        df_params_metrics = pd.read_csv(file, sep="\t")
        dict_params_metrics_tm_mc[tm_structure] = {}
        dict_params_metrics_tm_mc[tm_structure]["params_metrics"] = df_params_metrics

    # travel time benchmark
    # compare backward travel time distributions
    fig, axes = plt.subplots(2, 5, sharey=True, figsize=(6, 3))
    for i, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        states_tm_file = base_path / "svat_oxygen18" / "output" / f"states_{tms}.nc"
        with xr.open_dataset(states_tm_file, engine="h5netcdf") as ds_sim_tm:
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
                axes[j, i].plot(onp.nanquantile(TT, 0.5, axis=0), lw=1, color="black")
                axes[j, i].plot(onp.mean(TT, axis=0), lw=1, ls="--", color="black")
                if var_sim == "TT_transp":
                    axes[j, i].set_xlim((0, 150))
                elif var_sim == "TT_q_ss":
                    axes[j, i].set_xlim((0, 1000))
                axes[j, i].set_ylim((0, 1))
                axes[j, i].set_xlabel("T [days]")
                axes[0, i].set_title(_LABS_TM[tm_structure])
                avgTT50 = int(onp.nansum(onp.diff(onp.nanmedian(TT, axis=0)) * onp.arange(1, 1501)))
                axes[j, i].text(
                    0.35,
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
            0.35,
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
    file_str = "bTTD_benchmark.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)
    file_str = "bTTD_benchmark.pdf"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)

    # plot cumulative backward travel time distributions
    skipt = 1000
    TT = ds_hydrus_tt["bTT_perc"].values
    fig, axs = plt.subplots(figsize=(6, 3))
    for i in range(skipt, len(date_hydrus_tt)):
        axs.plot(TT[i, :], lw=1, color="grey")
    axs.set_xlim((0, 4000))
    axs.set_ylim((0, 1))
    axs.set_ylabel(r"$\overleftarrow{P}(T,t)$")
    axs.set_xlabel("T [days]")
    fig.tight_layout()
    file_str = "bTTD_hydrus.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)

    # plot cumulative forward travel time distributions
    skipt = 1000
    TT = ds_hydrus_tt["fTT_perc"].values
    fig, axs = plt.subplots(figsize=(6, 3))
    for i in range(0, len(date_hydrus_tt) - skipt):
        axs.plot(TT[i, :], lw=1, color="grey")
    axs.set_xlim((0, 4000))
    axs.set_ylim((0, 1))
    axs.set_ylabel(r"$\overrightarrow{P}(T,t)$")
    axs.set_xlabel("T [days]")
    fig.tight_layout()
    file_str = "fTTD_hydrus.png"
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=300)

    # compare age statistics
    df_age = pd.DataFrame(index=["MTT_transp", "MTT_perc"], columns=["CM", "PI", "AD", "AD-TV", "HYDRUS-1D"])
    for i, tm_structure in enumerate(tm_structures):
        idx_best = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]["KGE_C_iso_q_ss"].idxmax()
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        states_tm_file = base_path / "svat_oxygen18" / "output" / f"states_{tms}.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
        for j, age_metric in enumerate("ttavg_transp", "ttavg_q_ss"):
            df_age.iloc[j, i] = onp.nanmean(ds_sim_tm[age_metric].isel(x=idx_best, y=0).values)

    TT = onp.where(ds_hydrus_tt["bTT_perc"].values <= 0, onp.nan, ds_hydrus_tt["bTT_perc"].values)
    skipt = 1000
    df_age.iloc["MTT", "HYDRUS-1D"] = onp.nanmean(TT[skipt:, :], axis=0)

    plt.close("all")
    return


if __name__ == "__main__":
    main()
