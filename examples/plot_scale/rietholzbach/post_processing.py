from pathlib import Path
import os
import h5netcdf
import scipy as sp
from SALib.analyze import sobol
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.dates as mdates
import matplotlib.cm as cm
import yaml
import click
import copy
from de import de
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

_LABS_HYDRUS = {
    "n": r"$n$ [-]",
    "alpha": r"$alfa$ [-]",
    "theta_sat_m": r"$\theta_{s}^{m}$ [-]",
    "theta_sat_im": r"$\theta_{s}^{im}$ [-]",
    "ks": r"$k_{s}$ [-]",
    "omega": r"$\omega$ [-]",
    "D_l": r"$D_l$ [-]",
}

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


def kumaraswamy_cdf(x, a, b):
    return 1 - (1 - (x) ** a) ** b


def kumaraswamy_pdf(x, a, b):
    return a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)


def power_cdf(x, k):
    return x**k


def power_pdf(x, k):
    return k**k * x ** (k - 1)


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

    data = pd.DataFrame(index=date_sim_hm_for_tm, columns=["PERC_obs [mm/day]", "PERC_sim [mm/day]"])
    data.iloc[1:, 0] = ds_obs["PERC"].isel(x=0, y=0).values
    data.iloc[:, 1] = ds_sim_hm_for_tm["q_ss"].isel(y=0).values
    file = base_path_figs / f"PERC.txt"
    data.iloc[1:, :].to_csv(file, header=True, index=True, sep="\t")

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

    # plot SAS function
    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    x = onp.linspace(0, 1, num=1000)
    axs.plot(x, kumaraswamy_cdf(x, 1, 20), color="#034e7b", lw=1, label="a=1, b=20")
    axs.plot(x, kumaraswamy_cdf(x, 1.5, 20), color="#0570b0", lw=1, label="a=1.5, b=20")
    axs.plot(x, kumaraswamy_cdf(x, 1, 10), color="#3690c0", lw=1, label="a=1, b=10")
    axs.plot(x, kumaraswamy_cdf(x, 3, 1), color="#74a9cf", lw=1, label="a=3, b=1")
    axs.plot(x, kumaraswamy_cdf(x, 5, 1), color="#a6bddb", lw=1, label="a=5, b=1")
    axs.plot(x, kumaraswamy_cdf(x, 5, 1.5), color="#d0d1e6", lw=1, label="a=5, b=1.5")
    axs.set_xlim((0, 1))
    axs.set_ylim((0, 1))
    axs.set_xlabel("$P_S$ [-]")
    axs.set_ylabel("$P_Q$ [-]")
    axs.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.57, 1.05))
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.7)
    file = base_path_figs / "kumaraswamy_cdf.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "kumaraswamy_cdf.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    x = onp.linspace(0, 1, num=1000)
    axs.plot(x, kumaraswamy_pdf(x, 1, 20), color="#034e7b", lw=1, label="a=1, b=20")
    axs.plot(x, kumaraswamy_pdf(x, 1.5, 20), color="#0570b0", lw=1, label="a=1.5, b=20")
    axs.plot(x, kumaraswamy_pdf(x, 1, 10), color="#3690c0", lw=1, label="a=1, b=10")
    axs.plot(x, kumaraswamy_pdf(x, 3, 1), color="#74a9cf", lw=1, label="a=3, b=1")
    axs.plot(x, kumaraswamy_pdf(x, 5, 1), color="#a6bddb", lw=1, label="a=5, b=1")
    axs.plot(x, kumaraswamy_pdf(x, 5, 1.5), color="#d0d1e6", lw=1, label="a=5, b=1.5")
    axs.set_xlim((0, 1))
    axs.set_ylim(
        0,
    )
    axs.set_xlabel("$P_S$ [-]")
    axs.set_ylabel(r"$\omega_Q$ [-]")
    axs.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.57, 1.05))
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.7)
    file = base_path_figs / "kumaraswamy_pdf.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "kumaraswamy_pdf.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    x = onp.linspace(0, 1, num=1000)
    axs.plot(x, power_cdf(x, 0.3), color="#034e7b", lw=1, label="k=0.3")
    axs.plot(x, power_cdf(x, 0.5), color="#0570b0", lw=1, label="k=0.5")
    axs.plot(x, power_cdf(x, 0.7), color="#3690c0", lw=1, label="k=0.7")
    axs.plot(x, power_cdf(x, 1.5), color="#74a9cf", lw=1, label="k=1.5")
    axs.plot(x, power_cdf(x, 2), color="#a6bddb", lw=1, label="k=2")
    axs.plot(x, power_cdf(x, 3), color="#d0d1e6", lw=1, label="k=3")
    axs.set_xlim((0, 1))
    axs.set_ylim((0, 1))
    axs.set_xlabel("$P_S$ [-]")
    axs.set_ylabel("$P_Q$ [-]")
    axs.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.43, 1.05))
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.7)
    file = base_path_figs / "power_cdf.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "power_cdf.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    x = onp.linspace(0, 1, num=1000)
    axs.plot(x, power_pdf(x, 0.3), color="#034e7b", lw=1, label="k=0.3")
    axs.plot(x, power_pdf(x, 0.5), color="#0570b0", lw=1, label="k=0.5")
    axs.plot(x, power_pdf(x, 0.7), color="#3690c0", lw=1, label="k=0.7")
    axs.plot(x, power_pdf(x, 1.5), color="#74a9cf", lw=1, label="k=1.5")
    axs.plot(x, power_pdf(x, 2), color="#a6bddb", lw=1, label="k=2")
    axs.plot(x, power_pdf(x, 3), color="#d0d1e6", lw=1, label="k=3")
    axs.set_xlim((0, 1))
    axs.set_ylim((0, 30))
    axs.set_xlabel("$P_S$ [-]")
    axs.set_ylabel(r"$\omega_Q$ [-]")
    axs.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.43, 1.05))
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.7)
    file = base_path_figs / "power_pdf.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "power_pdf.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    # check water balance of lysimeter
    df_lys_obs = pd.DataFrame(index=date_obs)
    df_lys_obs.loc[:, "ta"] = ds_obs["TA"].isel(x=0, y=0).values
    df_lys_obs.loc[:, "prec"] = ds_obs["PREC"].isel(x=0, y=0).values
    df_lys_obs.loc[:, "prec_corr"] = ds_obs["PREC_corr"].isel(x=0, y=0).values
    df_lys_obs.loc[:, "aet"] = ds_obs["AET"].isel(x=0, y=0).values
    df_lys_obs.loc[:, "perc"] = ds_obs["PERC"].isel(x=0, y=0).values
    df_lys_obs.loc[:, "dS_weight"] = ds_obs["dWEIGHT"].isel(x=0, y=0).values
    df_lys_obs.loc[:, "hyd_year"] = df_lys_obs.index.year
    df_lys_obs.loc[(df_lys_obs.index.month >= 10), "hyd_year"] += 1
    file = base_path_figs / "lysimeter_fluxes_weight.csv"
    df_lys_obs.to_csv(file, header=True, index=True, sep=";")
    df_lys_obs.loc["1999-10":"1999-12", "dS_flux"] = 0
    df_lys_obs.loc["1999-10":"1999-12", "dS_flux_corr"] = 0
    df_lys_obs.loc["1999-10":"1999-12", "dS_weight"] = 0
    df_lys_obs.loc["1999-10":"1999-12", "dS_weight"] = onp.nan
    # data with bad quality flags have been removed
    df_lys_obs_nonan = df_lys_obs.loc[:, ["ta", "prec", "prec_corr", "aet", "perc", "dS_weight"]].dropna().copy()
    df_lys_obs_nonan.loc[:, "aet + perc"] = df_lys_obs_nonan.loc[:, "aet"] + df_lys_obs_nonan.loc[:, "perc"]
    df_lys_obs_nonan.loc[:, "dS_flux"] = (
        df_lys_obs_nonan.loc[:, "prec"] - df_lys_obs_nonan.loc[:, "aet"] - df_lys_obs_nonan.loc[:, "perc"]
    )
    df_lys_obs_nonan.loc[:, "dS_flux_corr"] = (
        df_lys_obs_nonan.loc[:, "prec_corr"] - df_lys_obs_nonan.loc[:, "aet"] - df_lys_obs_nonan.loc[:, "perc"]
    )
    df_lys_obs_nonan.loc[:, "hyd_year"] = df_lys_obs_nonan.index.year
    df_lys_obs_nonan.loc[(df_lys_obs_nonan.index.month >= 10), "hyd_year"] += 1
    # data without quality flags
    df_lys_obs_nonan_1997_1999 = df_lys_obs.loc[:, ["ta", "prec", "prec_corr", "aet", "perc"]].dropna().copy()
    df_lys_obs_nonan_1997_1999.loc[:, "aet + perc"] = (
        df_lys_obs_nonan_1997_1999.loc[:, "aet"] + df_lys_obs_nonan_1997_1999.loc[:, "perc"]
    )
    df_lys_obs_nonan_1997_1999.loc[:, "dS_flux"] = (
        df_lys_obs_nonan_1997_1999.loc[:, "prec"]
        - df_lys_obs_nonan_1997_1999.loc[:, "aet"]
        - df_lys_obs_nonan_1997_1999.loc[:, "perc"]
    )
    df_lys_obs_nonan_1997_1999.loc[:, "dS_flux_corr"] = (
        df_lys_obs_nonan_1997_1999.loc[:, "prec_corr"]
        - df_lys_obs_nonan_1997_1999.loc[:, "aet"]
        - df_lys_obs_nonan_1997_1999.loc[:, "perc"]
    )
    df_lys_obs_nonan_1997_1999.loc[:, "hyd_year"] = df_lys_obs_nonan_1997_1999.index.year
    df_lys_obs_nonan_1997_1999.loc[(df_lys_obs_nonan_1997_1999.index.month >= 10), "hyd_year"] += 1

    fig, axs = plt.subplots(1, 1, figsize=(6, 1.2))
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_flux"],
        "-",
        color="#6a51a3",
        lw=1,
        label=r"dS from fluxes",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_flux_corr"],
        "-",
        color="#9e9ac8",
        lw=0.8,
        label=r"dS from fluxes with corrected PREC",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_weight"],
        "-",
        color="#d0d1e6",
        lw=0.5,
        label=r"dS from weight",
    )
    axs.set_ylabel(r"[mm]")
    axs.set_xlabel("Time [year]")
    axs.legend(frameon=False, loc="lower right", ncol=3, fontsize=5)
    axs.set_xlim(df_lys_obs.loc["2000":, :].index[0], df_lys_obs.loc["2000":, :].index[-1])
    fig.tight_layout()
    file = base_path_figs / "dS_weight_vs_dS_flux.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(6, 1.2))
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_flux"] - df_lys_obs_nonan.loc["2000":, "dS_weight"],
        "-",
        color="blue",
        lw=1,
        label=r"dS from fluxes - dS from weight",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_flux_corr"] - df_lys_obs_nonan.loc["2000":, "dS_weight"],
        "-",
        color="red",
        lw=0.8,
        label="dS from fluxes (with corrected PREC) - dS from weight",
    )
    axs.set_ylabel(r"[mm]")
    axs.set_xlabel("Time [year]")
    axs.legend(frameon=False, loc="lower left", fontsize=5)
    axs.set_xlim(df_lys_obs.loc["2000":, :].index[0], df_lys_obs.loc["2000":, :].index[-1])
    fig.tight_layout()
    file = base_path_figs / "dS_weight_vs_dS_flux_residuals.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(6, 1.2))
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_weight"].cumsum(),
        "-",
        color="#d0d1e6",
        lw=1,
        label=r"dS from weight",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_flux"].cumsum(),
        "-",
        color="#6a51a3",
        lw=0.8,
        label=r"dS from fluxes",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_flux_corr"].cumsum(),
        "-",
        color="#9e9ac8",
        lw=0.5,
        label="dS from fluxes\n(with corrected PREC)",
    )
    axs.set_ylabel(r"[mm]")
    axs.set_xlabel("Time [year]")
    axs.legend(frameon=False, loc="upper left", fontsize=5)
    axs.set_xlim(df_lys_obs.loc["2000":, :].index[0], df_lys_obs.loc["2000":, :].index[-1])
    fig.tight_layout()
    file = base_path_figs / "dS_weight_vs_dS_flux_cumulated.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    years = onp.arange(1997, 2008).tolist()
    years1 = onp.arange(1997, 2001).tolist()
    years2 = onp.arange(2000, 2008).tolist()
    fig, axes = plt.subplots(4, 3, figsize=(6, 4.5))
    axs = axes.flatten()
    for i, year in enumerate(years):
        if year in years1:
            axs[i].plot(
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, :].index,
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, "prec"].cumsum(),
                "-",
                color="#034e7b",
                lw=1,
                label=r"PREC",
            )
            axs[i].plot(
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, :].index,
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, "prec_corr"].cumsum(),
                "-",
                color="#0570b0",
                lw=1,
                label=r"PREC (corrected)",
            )
            axs[i].plot(
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, :].index,
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, "aet + perc"].cumsum(),
                "-",
                color="#74a9cf",
                lw=1,
                label=r"AET + PERC",
            )
            axs[i].plot(
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, :].index,
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, "dS_flux"].cumsum(),
                "-",
                color="#6a51a3",
                lw=1,
                label=r"dS from fluxes",
            )
            axs[i].plot(
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, :].index,
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, "dS_flux_corr"].cumsum(),
                "-",
                color="#9e9ac8",
                lw=1,
                label=r"dS from fluxes (corrected)",
            )
            axs[i].set_xlim(
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, :].index[0],
                df_lys_obs_nonan_1997_1999.loc[df_lys_obs_nonan_1997_1999["hyd_year"] == year, :].index[-1],
            )

        elif year in years2:
            axs[i].plot(
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, :].index,
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, "prec"].cumsum(),
                "-",
                color="#034e7b",
                lw=1,
                label=r"PREC",
            )
            axs[i].plot(
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, :].index,
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, "prec_corr"].cumsum(),
                "-",
                color="#0570b0",
                lw=1,
                label=r"PREC (corrected)",
            )
            axs[i].plot(
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, :].index,
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, "aet + perc"].cumsum(),
                "-",
                color="#74a9cf",
                lw=1,
                label=r"AET + PERC",
            )
            axs[i].plot(
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, :].index,
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, "dS_flux"].cumsum(),
                "-",
                color="#6a51a3",
                lw=1,
                label=r"dS from fluxes",
            )
            axs[i].plot(
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, :].index,
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, "dS_flux_corr"].cumsum(),
                "-",
                color="#9e9ac8",
                lw=1,
                label=r"dS from fluxes (corrected)",
            )
            axs[i].plot(
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, :].index,
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, "dS_weight"].cumsum(),
                "-",
                color="#d0d1e6",
                lw=1,
                label=r"dS from weight",
            )
            axs[i].set_xlim(
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, :].index[0],
                df_lys_obs_nonan.loc[df_lys_obs_nonan["hyd_year"] == year, :].index[-1],
            )

        axs[i].tick_params(axis="x", labelrotation=90)
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%m"))
        axs[i].set_title(f"{year}")

    axes[-1, -1].axis("off")
    axes[0, 0].set_ylabel(r"[mm]")
    axes[1, 0].set_ylabel(r"[mm]")
    axes[2, 0].set_ylabel(r"[mm]")
    axes[3, 0].set_ylabel(r"[mm]")
    axes[2, 2].set_xlabel("Time [month]")
    axes[3, 0].set_xlabel("Time [month]")
    axes[3, 1].set_xlabel("Time [month]")
    lines, labels = axes[-1, 1].get_legend_handles_labels()
    fig.legend(lines, labels, loc="lower right", fontsize=8, frameon=False, bbox_to_anchor=(1.0, 0.02))
    fig.tight_layout()
    file = base_path_figs / "lys_cumulated_annually.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(6, 2))
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "prec"].cumsum(),
        "-",
        color="#034e7b",
        lw=1,
        label=r"PREC",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "prec_corr"].cumsum(),
        "-",
        color="#0570b0",
        lw=1,
        label=r"PREC (corrected)",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "aet + perc"].cumsum(),
        "-",
        color="#74a9cf",
        lw=1,
        label=r"AET + PERC",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_flux"].cumsum(),
        "-",
        color="#6a51a3",
        lw=1,
        label=r"dS from fluxes",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_flux_corr"].cumsum(),
        "-",
        color="#9e9ac8",
        lw=1,
        label=r"dS from fluxes (corrected)",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2000":, :].index,
        df_lys_obs_nonan.loc["2000":, "dS_weight"].cumsum(),
        "-",
        color="#d0d1e6",
        lw=1,
        label=r"dS from weight",
    )
    axs.tick_params(axis="x", rotation=45)
    axs.set_ylabel(r"[mm]")
    axs.set_xlabel("Time [year]")
    axs.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
    axs.set_xlim(df_lys_obs_nonan.loc["2000":, :].index[0], df_lys_obs_nonan.loc["2000":, :].index[-1])
    fig.tight_layout()
    file = base_path_figs / "lys_cumulated.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    df_lys_obs_nonan.loc[:, "hyd_year"] = df_lys_obs_nonan.index.year
    df_lys_obs_nonan.loc[(df_lys_obs_nonan.index.month >= 10), "hyd_year"] += 1
    df_lys_ann = df_lys_obs_nonan.loc["2000":, :].groupby("hyd_year").sum()
    df_lys_ann = df_lys_ann.loc[:, ["prec", "prec_corr", "aet + perc", "dS_weight"]]
    df_lys_ann.columns = ["PREC", "PREC (corrected)", "AET + PERC", "dS from weight"]
    df_lys_ann.loc[:, "year"] = df_lys_ann.index
    df_lys_ann.loc[:, "dS from fluxes"] = df_lys_ann.loc[:, "PREC"] - df_lys_ann.loc[:, "AET + PERC"]
    df_lys_ann.loc[:, "dS from fluxes (corrected)"] = (
        df_lys_ann.loc[:, "PREC (corrected)"] - df_lys_ann.loc[:, "AET + PERC"]
    )
    df_lys_ann = pd.melt(
        df_lys_ann,
        id_vars=["year"],
        value_vars=[
            "PREC",
            "PREC (corrected)",
            "AET + PERC",
            "dS from fluxes",
            "dS from fluxes (corrected)",
            "dS from weight",
        ],
    )
    fig, axs = plt.subplots(1, 1, figsize=(6, 1.5))
    g = sns.barplot(df_lys_ann, x="year", hue="variable", y="value", ax=axs, palette="PuBu_r")
    axs.set_ylabel(r"[mm]")
    axs.set_xlabel("Time [hyd. year]")
    g.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.5, 1.1))
    fig.subplots_adjust(bottom=0.3, right=0.68)
    file = base_path_figs / "lysimeter_balance_annual.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

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
    fig, axs = plt.subplots(1, 1, figsize=(6, 1.8))
    axs.axhline(y=theta_lower, color="grey", linestyle="--", lw=1)
    axs.axhline(y=theta_upper, color="grey", linestyle="--", lw=1)
    axs.plot(df_thetap.index, df_thetap["theta"], "-", color="black", lw=1)
    # axs.plot(df_thetap.index, onp.mean(ds_obs['THETA'].isel(x=0, y=0).values, axis=0), '-', color='grey')
    axs.set_ylabel(r"$\theta$ [-]")
    axs.set_xlabel("Time [year]")
    axs.set_xlim(df_thetap.index[0], df_thetap.index[-1])
    fig.tight_layout()
    file = base_path_figs / f"theta_previous_{window}_days.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    # measured oxygen-18 in precipitation and percolation
    d18O_prec_mean = onp.round(onp.nanmean(df_obs.loc[:, "d18O_prec"].values), 2)
    d18O_perc_mean = onp.round(onp.nanmean(df_obs.loc[:, "d18O_perc"].values), 2)
    fig, axs = plt.subplots(2, 1, figsize=(4, 3))
    axs[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].fillna(method="bfill"), "-", color="blue")
    axs[0].scatter(df_obs.index, df_obs.loc[:, "d18O_prec"], color="blue", s=1)
    axs[0].set_ylabel(r"$\delta^{18}$$O_{PRECIP}$ [‰]")
    axs[0].set_ylim([-20, 0])
    axs[0].set_xlim(df_obs.index[0], df_obs.index[-1])
    axs[1].plot(df_obs.index, df_obs.loc[:, "d18O_perc"].fillna(method="bfill"), "-", color="grey")
    axs[1].scatter(df_obs.index, df_obs.loc[:, "d18O_perc"], color="grey", s=1)
    axs[1].set_ylabel(r"$\delta^{18}$$O_{PERC}$ [‰]")
    axs[1].set_xlabel("Time [year]")
    axs[1].set_ylim([-20, 0])
    axs[1].set_xlim(df_obs.index[0], df_obs.index[-1])
    fig.text(0.19, 0.92, "(a)", ha="center", va="center")
    fig.text(
        0.35, 0.91, r"$\overline{\delta^{18}O}_{PRECIP}$: %s" % (d18O_prec_mean), ha="center", va="center", fontsize=9
    )
    fig.text(
        0.35, 0.45, r"$\overline{\delta^{18}O}_{PERC}$: %s" % (d18O_perc_mean), ha="center", va="center", fontsize=9
    )
    fig.text(0.19, 0.46, "(b)", ha="center", va="center")
    fig.tight_layout()
    file = base_path_figs / "observed_d18O_prec_perc.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "observed_d18O_prec_perc.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    # measured bromide in percolation
    fig, axs = plt.subplots(1, 1, figsize=(3, 1.5))
    axs.scatter(df_obs_br.dropna().index, df_obs_br.dropna()["Br"], color="grey", s=1)
    axs.plot(df_obs_br.dropna().index, df_obs_br.dropna()["Br"], color="grey", lw=1)
    axs.set_xlim([0, 400])
    axs.set_ylabel("Bromide [mmol/l]")
    axs.set_xlabel("Time [days since injection]")
    file = base_path_figs / "observed_bromide_perc.png"
    fig.tight_layout()
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    # dotty plots
    ll_param_sets = [["c1_mak", "c2_mak"], ["dmpv", "lmpv", "ks"], ["theta_ac", "theta_ufc", "theta_pwp"]]
    file = base_path / "svat_monte_carlo" / "results" / "params_metrics.txt"
    df_params_metrics = pd.read_csv(file, sep="\t")
    df_params_metrics100 = df_params_metrics.copy()
    df_params_metrics100.loc[:, "id"] = range(len(df_params_metrics100.index))
    df_params_metrics100 = df_params_metrics100.sort_values(by=[metric_for_opt], ascending=False)
    idx_best100 = df_params_metrics100.loc[: df_params_metrics100.index[99], "id"].values.tolist()
    dict_metrics_best = {}
    for sc in ['', 'dry', 'normal', 'wet']:
        dict_metrics_best[sc] = pd.DataFrame(index=range(len(idx_best100)))
    for sc, sc1 in enumerate(['', 'dry', 'normal', 'wet']):
        for ps, ps1 in enumerate(ll_param_sets):
            df_metrics = df_params_metrics.loc[:, [f'KGE_aet{sc1}', f'KGE_dS{sc1}', f'KGE_q_ss{sc1}', f'KGE_multi{sc1}']]
            df_params = df_params_metrics.loc[:, ps1]
            nrow = len(df_metrics.columns)
            ncol = len(df_params.columns)
            fig, ax = plt.subplots(nrow, ncol, sharey='row', sharex='col', figsize=(6, 5))
            for i, metric_var in enumerate(df_metrics.columns):
                for j, param_var in enumerate(df_params.columns):
                    y = df_metrics.iloc[:, i]
                    x = df_params.iloc[:, j]
                    ax[i, j].scatter(x, y, s=2, c='grey', alpha=0.5)
                    ax[i, j].set_xlabel('')
                    ax[i, j].set_ylabel('')
                    ax[i, j].set_ylim((0, 1))
                    # best parameter set for individual evaluation metric at specific storage conditions
                    df_params_metrics_sc1 = df_params_metrics.copy()
                    df_params_metrics_sc1.loc[:, 'id'] = range(len(df_params_metrics100.index))
                    df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                    idx_best_sc1 = df_params_metrics_sc1.loc[:df_params_metrics_sc1.index[99], 'id'].values.tolist()
                    for idx_best_sc in idx_best_sc1:
                        y_best_sc = df_metrics.iloc[idx_best_sc, i]
                        x_best_sc = df_params.iloc[idx_best_sc, j]
                        ax[i, j].scatter(x_best_sc, y_best_sc, s=2, c='blue', alpha=0.8)
                    # best parameter sets for multi-objective criteria
                    for ii, idx_best in enumerate(idx_best100):
                        y_best = df_metrics.iloc[idx_best, i]
                        x_best = df_params.iloc[idx_best, j]
                        ax[i, j].scatter(x_best, y_best, s=2, c='red', alpha=1)
                        dict_metrics_best[sc1].loc[dict_metrics_best[sc1].index[ii], df_metrics.columns[i]] = df_params_metrics.loc[idx_best, df_metrics.columns[i]]

            for j, param_var in enumerate(df_params.columns):
                xlabel = labs._LABS[param_var]
                ax[-1, j].set_xlabel(xlabel)

            ax[0, 0].set_ylabel('$KGE_{ET}$ [-]')
            ax[1, 0].set_ylabel(r'$KGE_{\Delta S}$ [-]')
            ax[2, 0].set_ylabel('$KGE_{PERC}$ [-]')
            ax[3, 0].set_ylabel('$KGE_{multi}$ [-]')

            fig.subplots_adjust(bottom=0.1, top=0.98, right=0.98, wspace=0.1, hspace=0.2)
            file = base_path_figs / f"dotty_plots{ps}_{sc1}_optimized_with_KGE_multi.png"
            fig.savefig(file, dpi=300)

            fig, ax = plt.subplots(nrow, ncol, sharey='row', sharex='col', figsize=(6, 5))
            for i, metric_var in enumerate(df_metrics.columns):
                for j, param_var in enumerate(df_params.columns):
                    y = df_metrics.iloc[:, i]
                    x = df_params.iloc[:, j]
                    ax[i, j].scatter(x, y, s=2, c='grey', alpha=0.5)
                    ax[i, j].set_xlabel('')
                    ax[i, j].set_ylabel('')
                    if metric_var in ['KGE_aet', 'KGE_aetwet', 'KGE_aetnormal', 'KGE_aetdry']:
                        ax[i, j].set_ylim((0.6, 1.0))
                    elif metric_var in ['KGE_dS', 'KGE_dSwet', 'KGE_dSnormal', 'KGE_dSdry']:
                        ax[i, j].set_ylim((0.6, 1.0))
                    elif metric_var in ['KGE_q_ss', 'KGE_q_sswet', 'KGE_q_ssnormal', 'KGE_q_ssdry']:
                        ax[i, j].set_ylim((0.4, 0.8))
                    elif metric_var in ['KGE_multi', 'KGE_multiwet', 'KGE_multinormal', 'KGE_multidry']:
                        ax[i, j].set_ylim((0.4, 0.8))
                    # best parameter set for individual evaluation metric at specific storage conditions
                    df_params_metrics_sc1 = df_params_metrics.copy()
                    df_params_metrics_sc1.loc[:, 'id'] = range(len(df_params_metrics100.index))
                    df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                    idx_best_sc1 = df_params_metrics_sc1.loc[:df_params_metrics_sc1.index[99], 'id'].values.tolist()
                    for idx_best_sc in idx_best_sc1:
                        y_best_sc = df_metrics.iloc[idx_best_sc, i]
                        x_best_sc = df_params.iloc[idx_best_sc, j]
                        ax[i, j].scatter(x_best_sc, y_best_sc, s=2, c='blue', alpha=0.8)
                    # best parameter sets for multi-objective criteria
                    for ii, idx_best in enumerate(idx_best100):
                        y_best = df_metrics.iloc[idx_best, i]
                        x_best = df_params.iloc[idx_best, j]
                        ax[i, j].scatter(x_best, y_best, s=2, c='red', alpha=1)
                        dict_metrics_best[sc1].loc[dict_metrics_best[sc1].index[ii], df_metrics.columns[i]] = df_params_metrics.loc[idx_best, df_metrics.columns[i]]

            for j, param_var in enumerate(df_params.columns):
                xlabel = labs._LABS[param_var]
                ax[-1, j].set_xlabel(xlabel)

            ax[0, 0].set_ylabel('$KGE_{ET}$ [-]')
            ax[1, 0].set_ylabel(r'$KGE_{\Delta S}$ [-]')
            ax[2, 0].set_ylabel('$KGE_{PERC}$ [-]')
            ax[3, 0].set_ylabel('$KGE_{multi}$ [-]')

            fig.subplots_adjust(bottom=0.1, top=0.98, right=0.98, wspace=0.1, hspace=0.2)
            file = base_path_figs / f"dotty_plots{ps}_{sc1}inset_optimized_with_KGE_multi.png"
            fig.savefig(file, dpi=300)

    # write evaluation metrics for different storage condtions to .txt
    df_avg_std = pd.DataFrame(columns=['KGE_aet', 'KGE_dS', 'KGE_q_ss', 'KGE_multi'])
    for sc in ['', 'dry', 'normal', 'wet']:
        df_avg_std.loc[f'avg{sc}', :] = onp.mean(dict_metrics_best[sc].values, axis=0)
        df_avg_std.loc[f'std{sc}', :] = onp.std(dict_metrics_best[sc].values, axis=0)
    file = base_path_figs / f"metrics_best_100_avg_std_optimized_with_KGE_multi.txt"
    df_avg_std.to_csv(file, header=True, index=True, sep="\t")

    # write average and standard deviation of best parameters to .txt
    df_avg_std = pd.DataFrame(index=['c1_mak', 'c2_mak', 'dmpv', 'lmpv', 'theta_eff', 'frac_lp', 'frac_fp', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks'], columns=['avg', 'std'])
    df_avg_std.loc[:, 'avg'] = onp.mean(df_params_metrics100.loc[:df_params_metrics100.index[99], ['c1_mak', 'c2_mak', 'dmpv', 'lmpv', 'theta_eff', 'frac_lp', 'frac_fp', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']].values, axis=0)
    df_avg_std.loc[:, 'std'] = onp.std(df_params_metrics100.loc[:df_params_metrics100.index[99], ['c1_mak', 'c2_mak', 'dmpv', 'lmpv', 'theta_eff', 'frac_lp', 'frac_fp', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']].values, axis=0)
    file = base_path_figs / f"params_best_100_avg_std_optimized_with_KGE_multi.txt"
    df_avg_std.to_csv(file, header=True, index=True, sep="\t")

    # diagnostic polar plots
    file = base_path / "svat_monte_carlo" / "results" / "params_metrics.txt"
    df_params_metrics = pd.read_csv(file, sep="\t")

    df_params_metrics100 = df_params_metrics.copy()
    df_params_metrics100.loc[:, "id"] = range(len(df_params_metrics100.index))
    df_params_metrics100 = df_params_metrics100.sort_values(by=[metric_for_opt], ascending=False)
    df_for_diag100 = df_params_metrics100.loc[: df_params_metrics100.index[99], :]
    vars_sim = ["aet", "q_ss"]
    for var_sim in vars_sim:
        fig = de.diag_polar_plot_multi(
            df_for_diag100.loc[:, f"brel_mean_{var_sim}"].values,
            df_for_diag100.loc[:, f"temp_cor_{var_sim}"].values,
            df_for_diag100.loc[:, f"DE_{var_sim}"].values,
            df_for_diag100.loc[:, f"b_dir_{var_sim}"].values,
            df_for_diag100.loc[:, f"phi_{var_sim}"].values,
            df_for_diag100.loc[:, f"b_hf_{var_sim}"].values,
            df_for_diag100.loc[:, f"b_lf_{var_sim}"].values,
            df_for_diag100.loc[:, f"b_tot_{var_sim}"].values,
            df_for_diag100.loc[:, f"err_hf_{var_sim}"].values,
            df_for_diag100.loc[:, f"err_lf_{var_sim}"].values,
            a0=df_for_diag100.loc[:, f"ioa0_{var_sim}"].values,
            share0=onp.round(onp.max(df_for_diag100.loc[:, f"p0_{var_sim}"]), 2),
        )
        file = f"diag_polar_plot_{var_sim}_100_optimized_with_KGE_multi.png"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)

    var_sim = "dS"
    fig = de.diag_polar_plot_multi(
        df_for_diag100.loc[:, f"brel_mean_{var_sim}"].values,
        df_for_diag100.loc[:, f"temp_cor_{var_sim}"].values,
        df_for_diag100.loc[:, f"DE_{var_sim}"].values,
        df_for_diag100.loc[:, f"b_dir_{var_sim}"].values,
        df_for_diag100.loc[:, f"phi_{var_sim}"].values,
        df_for_diag100.loc[:, f"b_hf_{var_sim}"].values,
        df_for_diag100.loc[:, f"b_lf_{var_sim}"].values,
        df_for_diag100.loc[:, f"b_tot_{var_sim}"].values,
        df_for_diag100.loc[:, f"err_hf_{var_sim}"].values,
        df_for_diag100.loc[:, f"err_lf_{var_sim}"].values,
    )
    file = f"diag_polar_plot_{var_sim}_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)

    df_params_metrics10 = df_params_metrics.copy()
    df_params_metrics10.loc[:, "id"] = range(len(df_params_metrics10.index))
    df_params_metrics10 = df_params_metrics10.sort_values(by=[metric_for_opt], ascending=False)
    df_for_diag10 = df_params_metrics10.loc[: df_params_metrics10.index[9], :]
    vars_sim = ["aet", "q_ss"]
    for var_sim in vars_sim:
        fig = de.diag_polar_plot_multi(
            df_for_diag10.loc[:, f"brel_mean_{var_sim}"].values,
            df_for_diag10.loc[:, f"temp_cor_{var_sim}"].values,
            df_for_diag10.loc[:, f"DE_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_dir_{var_sim}"].values,
            df_for_diag10.loc[:, f"phi_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_lf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_tot_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_lf_{var_sim}"].values,
            a0=df_for_diag10.loc[:, f"ioa0_{var_sim}"].values,
            share0=onp.round(onp.max(df_for_diag10.loc[:, f"p0_{var_sim}"]), 2),
        )
        file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi.png"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)
        file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi.pdf"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)

    var_sim = "dS"
    fig = de.diag_polar_plot_multi(
        df_for_diag10.loc[:, f"brel_mean_{var_sim}"].values,
        df_for_diag10.loc[:, f"temp_cor_{var_sim}"].values,
        df_for_diag10.loc[:, f"DE_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_dir_{var_sim}"].values,
        df_for_diag10.loc[:, f"phi_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_hf_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_lf_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_tot_{var_sim}"].values,
        df_for_diag10.loc[:, f"err_hf_{var_sim}"].values,
        df_for_diag10.loc[:, f"err_lf_{var_sim}"].values,
    )
    file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)
    file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)

    df_for_diag1 = df_params_metrics10.loc[: df_params_metrics10.index[0], :]
    vars_sim = ["aet", "q_ss"]
    for var_sim in vars_sim:
        fig = de.diag_polar_plot_multi(
            df_for_diag1.loc[:, f"brel_mean_{var_sim}"].values,
            df_for_diag1.loc[:, f"temp_cor_{var_sim}"].values,
            df_for_diag1.loc[:, f"DE_{var_sim}"].values,
            df_for_diag1.loc[:, f"b_dir_{var_sim}"].values,
            df_for_diag1.loc[:, f"phi_{var_sim}"].values,
            df_for_diag1.loc[:, f"b_hf_{var_sim}"].values,
            df_for_diag1.loc[:, f"b_lf_{var_sim}"].values,
            df_for_diag1.loc[:, f"b_tot_{var_sim}"].values,
            df_for_diag1.loc[:, f"err_hf_{var_sim}"].values,
            df_for_diag1.loc[:, f"err_lf_{var_sim}"].values,
            a0=df_for_diag1.loc[:, f"ioa0_{var_sim}"].values,
            share0=onp.round(onp.max(df_for_diag1.loc[:, f"p0_{var_sim}"]), 2),
        )
        file = f"diag_polar_plot_{var_sim}_1_optimized_with_KGE_multi.png"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)

    var_sim = "dS"
    fig = de.diag_polar_plot_multi(
        df_for_diag1.loc[:, f"brel_mean_{var_sim}"].values,
        df_for_diag1.loc[:, f"temp_cor_{var_sim}"].values,
        df_for_diag1.loc[:, f"DE_{var_sim}"].values,
        df_for_diag1.loc[:, f"b_dir_{var_sim}"].values,
        df_for_diag1.loc[:, f"phi_{var_sim}"].values,
        df_for_diag1.loc[:, f"b_hf_{var_sim}"].values,
        df_for_diag1.loc[:, f"b_lf_{var_sim}"].values,
        df_for_diag1.loc[:, f"b_tot_{var_sim}"].values,
        df_for_diag1.loc[:, f"err_hf_{var_sim}"].values,
        df_for_diag1.loc[:, f"err_lf_{var_sim}"].values,
    )
    file = f"diag_polar_plot_{var_sim}_1_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)

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
        # plot observed and simulated time series
        fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        file_str = "%s.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        # plot cumulated observed and simulated time series
        fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        file_str = "%s_cum.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        fig = eval_utils.plot_obs_sim_cum_year_facet(
            df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        )
        file_str = "%s_cum_year_facet.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
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
        # plot observed and simulated time series
        fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        file_str = "%s_best_10.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        # plot cumulated observed and simulated time series
        fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        file_str = "%s_cum_best_10.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        fig = eval_utils.plot_obs_sim_cum_year_facet(
            df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        )
        file_str = "%s_cum_year_facet_best_10.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
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
        # plot observed and simulated time series
        fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        file_str = "%s_best_100.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        # plot cumulated observed and simulated time series
        fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        file_str = "%s_cum_best_100.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        fig = eval_utils.plot_obs_sim_cum_year_facet(
            df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        )
        file_str = "%s_cum_year_facet_best_100.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
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
        # plot observed and simulated time series
        fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        file_str = "%s_best_for_tm.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        # plot cumulated observed and simulated time series
        fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        file_str = "%s_cum_best_for_tm.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        fig = eval_utils.plot_obs_sim_cum_year_facet(
            df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        )
        file_str = "%s_cum_year_facet_best_for_tm.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
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
        # plot observed and simulated time series
        fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        file_str = "%s.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        # plot cumulated observed and simulated time series
        fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
        file_str = "%s_cum.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
        fig = eval_utils.plot_obs_sim_cum_year_facet(
            df_eval, labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
        )
        file_str = "%s_cum_year_facet.pdf" % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
    plt.close("all")

    fig, axs = plt.subplots(1, 1, figsize=(6, 1.2))
    axs.plot(df_eval.index, df_eval.loc[:, "sim0"] - df_eval.loc[:, "obs"], "-", color="black", lw=1)
    axs.set_ylabel(r"[mm]")
    axs.set_xlabel("Time [year]")
    axs.set_xlim(df_eval.index[0], df_eval.index[-1])
    fig.tight_layout()
    file = base_path_figs / "prec_residuals.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    vars_obs = ["TA"]
    vars_sim = ["ta"]
    for var_obs, var_sim in zip(vars_obs, vars_sim):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        # plot observed time series
        fig = eval_utils.plot_sim(df_obs, labs._Y_LABS_DAILY[var_sim])
        file_str = "%s.pdf" % (var_obs)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=300)
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
            color="grey",
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
            color="grey",
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
            color="grey",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
        color="gray",
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
    file = f"prec_et_dS_perc_obs_sim_cumulated_optimized_with_KGE_multi_for_best_tm.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"prec_et_dS_perc_obs_sim_cumulated_optimized_with_KGE_multi_for_best_tm.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    # plot macropore infiltration
    df_params_mp_100 = df_params_metrics100.loc[: df_params_metrics100.index[99], ["lmpv", "dmpv", "ks"]]
    for x in range(100):
        inf_tot = (
            ds_sim_hm100["inf_mat_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_sc_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        )
        inf_mp = ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        df_params_mp_100.loc[df_params_mp_100.index[x], "inf_mp_share"] = (onp.sum(inf_mp) / onp.sum(inf_tot)) * 100

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(6, 2))
    axes[0].scatter(df_params_mp_100["lmpv"], df_params_mp_100["inf_mp_share"], s=2, color="black")
    axes[0].set_ylim(
        0,
    )
    axes[0].set_ylabel(r"$INF_{mp}$ [%]")
    axes[0].set_xlabel(r"$l_{mpv}$ [mm]")
    axes[1].scatter(df_params_mp_100["dmpv"], df_params_mp_100["inf_mp_share"], s=2, color="black")
    axes[1].set_ylim(
        0,
    )
    axes[1].set_xlabel(r"$\rho_{mpv}$ [$m^{-2}$]")
    axes[2].scatter(df_params_mp_100["ks"], df_params_mp_100["inf_mp_share"], s=2, color="black")
    axes[2].set_ylim(
        0,
    )
    axes[2].set_xlabel(r"$k_s$ [mm $h^{-1}$]")
    fig.tight_layout()
    file = "inf_mp_share_vs_mp_params1.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    res = sp.stats.linregress(df_params_mp_100["ks"].values, df_params_mp_100["inf_mp_share"].values)
    xpred = onp.arange(5, 150)

    cmap = cm.get_cmap("Oranges")
    norm_lmpv = Normalize(vmin=10, vmax=1500)
    norm_dmpv = Normalize(vmin=10, vmax=300)
    fig, axes = plt.subplots(1, 1, sharey=True, figsize=(4, 3))
    for x in range(100):
        lmpv = df_params_mp_100.loc[df_params_mp_100.index[x], "lmpv"]
        dmpv = df_params_mp_100.loc[df_params_mp_100.index[x], "dmpv"]
        axes.plot(
            df_params_mp_100.loc[df_params_mp_100.index[x], "ks"],
            df_params_mp_100.loc[df_params_mp_100.index[x], "inf_mp_share"],
            "o",
            markersize=1 + norm_dmpv(dmpv) * 7,
            color=cmap(norm_lmpv(lmpv)),
        )
    axes.plot(xpred, res.intercept + res.slope * xpred, "black")
    rlab = r"$R^2$: %s" % (f"{res.rvalue**2:.2f}")
    axes.text(
        0.12, 0.93, rlab, fontsize=9, horizontalalignment="center", verticalalignment="center", transform=axes.transAxes
    )
    axes.set_ylim(
        0,
    )
    axes.set_ylabel(r"$INF_{mp}$ [%]")
    axes.set_xlabel(r"$k_s$ [mm $h^{-1}$]")
    axl = fig.add_axes([0.8, 0.48, 0.02, 0.5])
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm_lmpv, orientation="vertical", ticks=[10, 500, 1000, 1500])
    cb1.set_label(r"$l_{mpv}$ [mm]")

    (l1,) = axes.plot([], [], "or", markersize=1)
    (l2,) = axes.plot([], [], "or", markersize=2)
    (l3,) = axes.plot([], [], "or", markersize=6)
    (l4,) = axes.plot([], [], "or", markersize=8)
    labels = ["10", "100", "200", "300"]
    axes.legend(
        [l1, l2, l3, l4],
        labels,
        ncol=1,
        frameon=False,
        handlelength=2,
        borderpad=1.8,
        handletextpad=1,
        title=r"$\rho_{mpv}$ [$m^{-2}$]",
        scatterpoints=1,
        loc="lower right",
        bbox_to_anchor=(1.4, -0.15),
    )
    fig.subplots_adjust(bottom=0.15, right=0.75, left=0.12, top=0.98)
    file = "inf_mp_share_vs_mp_params.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    axes.plot(ds_sim_hm100["Time"].values, prec, lw=1, color="blue", ls="-", label="Precipitation")
    for x in range(100):
        sim_vals = (
            ds_sim_hm100["inf_mat_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_sc_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        )
        axes.plot(ds_sim_hm100["Time"].values, sim_vals, lw=1, color="red", ls="-", alpha=0.8)
        sim_vals = ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        axes.plot(ds_sim_hm100["Time"].values, sim_vals, lw=1, color="red", ls=":", alpha=1)
    axes.set_ylabel(r"[mm/day]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    fig.tight_layout()
    file = f"inf_mp_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"inf_mp_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    axes.plot(ds_sim_hm100["Time"].values, prec, lw=1, color="blue", ls="-", label="Precipitation")
    for x in range(100):
        sim_vals = (
            ds_sim_hm100["inf_mat_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_sc_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        )
        sim_vals_cum = onp.cumsum(sim_vals)
        axes.plot(ds_sim_hm100["Time"].values, sim_vals_cum, lw=1, color="red", ls="-", alpha=0.8)
        sim_vals = ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        sim_vals_cum = onp.cumsum(sim_vals)
        axes.plot(ds_sim_hm100["Time"].values, sim_vals_cum, lw=1, color="red", ls=":", alpha=1)
    axes.plot(
        ds_sim_hm100["Time"].values, sim_vals_cum, lw=1, color="red", ls="-", alpha=0.8, label="Total infiltration"
    )
    axes.plot(
        ds_sim_hm100["Time"].values, sim_vals_cum, lw=1, color="red", ls=":", alpha=1, label="Macropore infiltration"
    )
    axes.set_ylabel(r"[mm]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    axes.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    file = f"inf_mp_cumulated_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"inf_mp_cumulated_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    axes.plot(ds_sim_hm100["Time"].values, prec, lw=1, color="blue", ls="-", label="Precipitation")
    sim_vals = (
        ds_sim_hm100["inf_mat_rz"].isel(y=0).values
        + ds_sim_hm100["inf_mp_rz"].isel(y=0).values
        + ds_sim_hm100["inf_sc_rz"].isel(y=0).values
        + ds_sim_hm100["inf_ss"].isel(y=0).values
    )
    sim_vals_5 = onp.nanquantile(sim_vals, 0.05, axis=0)
    sim_vals_50 = onp.nanmedian(sim_vals, axis=0)
    sim_vals_95 = onp.nanquantile(sim_vals, 0.95, axis=0)
    axes.plot(ds_sim_hm100["Time"].values, sim_vals_50, lw=1, color="red", ls="-", label="Total infiltration")
    axes.fill_between(ds_sim_hm100["Time"].values, sim_vals_5, sim_vals_95, color="red", edgecolor=None, alpha=0.2)
    sim_vals = ds_sim_hm100["inf_mp_rz"].isel(y=0).values + ds_sim_hm100["inf_ss"].isel(y=0).values
    sim_vals = onp.cumsum(sim_vals, axis=1)
    sim_vals_5 = onp.nanquantile(sim_vals, 0.05, axis=0)
    sim_vals_50 = onp.nanmedian(sim_vals, axis=0)
    sim_vals_95 = onp.nanquantile(sim_vals, 0.95, axis=0)
    axes.plot(ds_sim_hm100["Time"].values, sim_vals_50, lw=1, color="orange", ls="-", label="Macropore infiltration")
    axes.fill_between(ds_sim_hm100["Time"].values, sim_vals_5, sim_vals_95, color="orange", edgecolor=None, alpha=0.2)
    axes.set_ylabel(r"[mm/day]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    axes.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    file = f"inf_mp_conf_int_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"inf_mp_conf_int_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    prec_cum = onp.cumsum(prec)
    axes.plot(ds_sim_hm100["Time"].values, prec_cum, lw=1, color="blue", ls="-", label="Precipitation")
    sim_vals = (
        ds_sim_hm100["inf_mat_rz"].isel(y=0).values
        + ds_sim_hm100["inf_mp_rz"].isel(y=0).values
        + ds_sim_hm100["inf_sc_rz"].isel(y=0).values
        + ds_sim_hm100["inf_ss"].isel(y=0).values
    )
    sim_vals_cum = onp.cumsum(sim_vals, axis=1)
    sim_vals_cum_5 = onp.nanquantile(sim_vals_cum, 0.05, axis=0)
    sim_vals_cum_50 = onp.nanmedian(sim_vals_cum, axis=0)
    sim_vals_cum_95 = onp.nanquantile(sim_vals_cum, 0.95, axis=0)
    axes.plot(ds_sim_hm100["Time"].values, sim_vals_cum_50, lw=1, color="red", ls="-", label="Total infiltration")
    axes.fill_between(
        ds_sim_hm100["Time"].values, sim_vals_cum_5, sim_vals_cum_95, color="red", edgecolor=None, alpha=0.2
    )
    sim_vals = ds_sim_hm100["inf_mp_rz"].isel(y=0).values + ds_sim_hm100["inf_ss"].isel(y=0).values
    sim_vals_cum = onp.cumsum(sim_vals, axis=1)
    sim_vals_cum_5 = onp.nanquantile(sim_vals_cum, 0.05, axis=0)
    sim_vals_cum_50 = onp.nanmedian(sim_vals_cum, axis=0)
    sim_vals_cum_95 = onp.nanquantile(sim_vals_cum, 0.95, axis=0)
    axes.plot(
        ds_sim_hm100["Time"].values, sim_vals_cum_50, lw=1, color="orange", ls="-", label="Macropore infiltration"
    )
    axes.fill_between(
        ds_sim_hm100["Time"].values, sim_vals_cum_5, sim_vals_cum_95, color="orange", edgecolor=None, alpha=0.2
    )
    axes.set_ylabel(r"[mm]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    axes.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    file = f"inf_mp_cumulated_conf_int_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"inf_mp_cumulated_conf_int_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    prec_cum = onp.cumsum(prec)
    axes.plot(ds_sim_hm100["Time"].values, prec_cum, lw=1, color="blue", ls="-", label="Precipitation")
    sim_vals = (
        ds_sim_hm100["inf_mat_rz"].isel(y=0).values
        + ds_sim_hm100["inf_mp_rz"].isel(x=0, y=0).values
        + ds_sim_hm100["inf_sc_rz"].isel(x=0, y=0).values
        + ds_sim_hm100["inf_ss"].isel(x=0, y=0).values
    )
    sim_vals_cum = onp.cumsum(sim_vals, axis=1)
    sim_vals_cum_5 = onp.nanquantile(sim_vals_cum, 0.05, axis=0)
    sim_vals_cum_50 = onp.nanmedian(sim_vals_cum, axis=0)
    sim_vals_cum_95 = onp.nanquantile(sim_vals_cum, 0.95, axis=0)
    axes.plot(ds_sim_hm100["Time"].values, sim_vals_cum_50, lw=1, color="red", ls="-", label="Total infiltration")
    axes.fill_between(
        ds_sim_hm100["Time"].values, sim_vals_cum_5, sim_vals_cum_95, color="red", edgecolor=None, alpha=0.2
    )
    sim_vals = ds_sim_hm100["inf_ss"].isel(y=0).values
    sim_vals_cum = onp.cumsum(sim_vals, axis=1)
    sim_vals_cum_5 = onp.nanquantile(sim_vals_cum, 0.05, axis=0)
    sim_vals_cum_50 = onp.nanmedian(sim_vals_cum, axis=0)
    sim_vals_cum_95 = onp.nanquantile(sim_vals_cum, 0.95, axis=0)
    axes.plot(
        ds_sim_hm100["Time"].values, sim_vals_cum_50, lw=1, color="orange", ls="-", label="Macropore infiltration"
    )
    axes.fill_between(
        ds_sim_hm100["Time"].values, sim_vals_cum_5, sim_vals_cum_95, color="orange", edgecolor=None, alpha=0.2
    )
    axes.set_ylabel(r"[mm]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    axes.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    file = f"inf_mp_ss_cumulated_conf_int_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"inf_mp_ss_cumulated_conf_int_best_100_optimized_with_KGE_multi.pdf"
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
            / "results"
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm100"
            / f"params_metrics_{tms}.txt"
        )
        df_params_metrics = pd.read_csv(file, sep="\t")
        dict_params_metrics_tm_mc[tm_structure] = {}
        dict_params_metrics_tm_mc[tm_structure]["params_metrics"] = df_params_metrics

    # dotty plots of transport simulations
    fig, axes = plt.subplots(8, 4, sharey=True, figsize=(6, 8))
    for ncol, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        df_params = df_params_metrics.loc[
            :, ["c1_mak", "c2_mak", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks"]
        ]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=1, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((-1, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(8):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots_hm_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(2, 4, sharey=True, figsize=(6, 3))
    for ncol, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        df_params = df_params_metrics.loc[:, ["c1_mak", "c2_mak"]]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=2, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((-1, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(2):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots1_hm_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(3, 4, sharey=True, figsize=(6, 5))
    for ncol, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        df_params = df_params_metrics.loc[:, ["dmpv", "lmpv", "ks"]]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=2, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((-1, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(3):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots2_hm_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(3, 4, sharey=True, figsize=(6, 5))
    for ncol, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        df_params = df_params_metrics.loc[:, ["theta_ac", "theta_ufc", "theta_pwp"]]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=2, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((-1, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(3):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots3_hm_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(6, 2, figsize=(4, 8))
    for ncol, tm_structure in enumerate(tm_structures[2:]):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        if tm_structure == "advection-dispersion-power":
            df_params = df_params_metrics.loc[:, ["k_transp", "k_q_rz", "k_q_ss"]]
        elif tm_structure == "time-variant advection-dispersion-power":
            df_params = df_params_metrics.loc[:, ["k1_transp", "k2_transp", "k1_q_rz", "k2_q_rz", "k1_q_ss", "k2_q_ss"]]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=2, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((0.2, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        for nrow in range(6):
            if not axes[nrow, ncol].has_data():
                axes[nrow, ncol].set_axis_off()

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(3):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")
    for j in range(3, 6):
        axes[j, 1].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots_sas_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    # write evaluation metrics for different storage condtions to .txt
    df_kge_d18O_perc = pd.DataFrame(columns=["CM", "PI", "AD", "AD-TV"])
    for ncol, tm_structure in enumerate(tm_structures):
        for sc in ["", "dry", "normal", "wet"]:
            df_kge_d18O_perc.loc[f"{sc}", df_kge_d18O_perc.columns[ncol]] = onp.max(
                dict_params_metrics_tm_mc[tm_structure]["params_metrics"][f"KGE_C_iso_q_ss{sc}"]
            )
    file = base_path_figs / f"kge_d18O_perc_optimized_with_KGE_multi.txt"
    df_kge_d18O_perc.to_csv(file, header=True, index=True, sep="\t")

    # compare best model runs
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 6))
    df_obs = pd.DataFrame(index=date_obs)
    df_obs.loc[:, "d18O_prec"] = ds_obs["d18O_PREC"].isel(x=0, y=0).values
    ax.flatten()[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].fillna(method="bfill"), "-", color="blue")
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
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm100"
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
        ax.flatten()[i + 1].plot(ds_hydrus_18O["Time"].values, ds_hydrus_18O["d18O_perc"].values, color="grey", lw=1)
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
        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 6))
        df_obs = pd.DataFrame(index=date_obs)
        df_obs.loc[:, "d18O_prec"] = ds_obs["d18O_PREC"].isel(x=0, y=0).values
        df_obs = df_obs.loc[f"{year}":f"{year+2}", "d18O_prec"].to_frame()
        ax.flatten()[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].fillna(method="bfill"), "-", color="blue")
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
                / "deterministic"
                / "age_max_1500_days"
                / "optimized_with_KGE_multi_hm100"
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
                ds_hydrus_18O_year["Time"].values, ds_hydrus_18O_year["d18O_perc"].values, color="grey", lw=1
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
    ax.flatten()[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].fillna(method="bfill"), "-", color="blue")
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
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm100"
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
        ax.flatten()[i + 1].plot(ds_hydrus_18O["Time"].values, ds_hydrus_18O["d18O_perc"].values, color="grey", lw=1)
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
    ax.flatten()[0].plot(df_obs.index, df_obs.loc[:, "d18O_prec"].fillna(method="bfill"), "-", color="blue")
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
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm100"
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
        ax.flatten()[i + 1].plot(ds_hydrus_18O["Time"].values, ds_hydrus_18O["d18O_perc"].values, color="grey", lw=1)
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
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm100"
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
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm100"
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
        ax.flatten()[i].plot(ds_hydrus_18O["Time"].values, sim_vals_hydrus, color="grey", lw=1)
        ax.flatten()[i].scatter(date_obs, obs_vals, color="blue", s=1)
        ax.flatten()[i].set_title(_LABS_TM[tm_structure])
        ax[i].set_ylabel(r"$\delta^{18}$$O_{PERC}$ [‰]")
        ax.flatten()[i].set_ylim((-20, 0))
        ax.flatten()[i].set_xlim(ds_sim_tm["Time"].values[0], ds_sim_tm["Time"].values[-1])
    ax[-1].set_xlabel("Time [year]")
    fig.tight_layout()
    file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_conf_int_optimized_with_KGE_multi_extra.png"
    fig.savefig(file, dpi=300)

    # compare duration curve of 18O in percolation
    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(6, 1.8))
    for i, tm_structure in enumerate(tm_structures):
        idx_best = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]["KGE_C_iso_q_ss"].idxmax()
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        states_tm_file = (
            base_path
            / "svat_oxygen18_monte_carlo"
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm100"
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
        df_sim = pd.DataFrame(index=ds_sim_tm["Time"].values)
        df_sim.loc[:, "sim0"] = ds_sim_tm["d18O_perc_bs"].isel(x=idx_best, y=0).values
        df_sim.loc[df_sim.index[1] :, "sim1"] = ds_hydrus_18O["d18O_perc_bs"].values
        df_sim = df_sim.iloc[1:, :]
        df_eval = eval_utils.join_obs_on_sim(ds_sim_tm["Time"].values[1:], df_sim.values, df_obs)
        df_eval = df_eval.dropna()
        obs = df_eval.sort_values(by=["obs"], ascending=True)
        sim0 = df_eval.sort_values(by=["sim0"], ascending=True)
        sim1 = df_eval.sort_values(by=["sim1"], ascending=True)

        # calculate exceedence probability
        ranks_obs = sp.stats.rankdata(obs["obs"], method="ordinal")
        ranks_obs = ranks_obs[::-1]
        prob_obs = [(ranks_obs[i] / (len(obs["obs"]) + 1)) for i in range(len(obs["obs"]))]

        ranks_sim0 = sp.stats.rankdata(sim0["sim0"], method="ordinal")
        ranks_sim0 = ranks_sim0[::-1]
        prob_sim0 = [(ranks_sim0[i] / (len(sim0["sim0"]) + 1)) for i in range(len(sim0["sim0"]))]

        ranks_sim1 = sp.stats.rankdata(sim1["sim1"], method="ordinal")
        ranks_sim1 = ranks_sim1[::-1]
        prob_sim1 = [(ranks_sim1[i] / (len(sim1["sim1"]) + 1)) for i in range(len(sim1["sim1"]))]

        ax.flatten()[i].plot(prob_obs, obs["obs"], color="blue", lw=1)
        ax.flatten()[i].plot(prob_sim0, sim0["sim0"], color="red", lw=1, ls="-.", alpha=0.8)
        ax.flatten()[i].plot(prob_sim1, sim1["sim1"], color="grey", lw=1, ls="-.", alpha=0.8)
        ax.flatten()[i].set_xlim(0, 1)
        ax.flatten()[i].tick_params(axis="x", labelsize=8)
        ax.flatten()[i].tick_params(axis="y", labelsize=8)
        ax.flatten()[i].set_title(_LABS_TM[tm_structure], fontsize=9)

    ax[0].set_ylabel(r"$\delta^{18}$$O_{PERC}$ [‰]", fontsize=8)
    ax[0].set_xlabel("Exceedence probabilty [-]", fontsize=8)
    ax[1].set_xlabel("Exceedence probabilty [-]", fontsize=8)
    ax[2].set_xlabel("Exceedence probabilty [-]", fontsize=8)
    ax[3].set_xlabel("Exceedence probabilty [-]", fontsize=8)
    fig.tight_layout()
    file = base_path_figs / f"fdc_d18O_perc_sim_obs_tm_structures_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)

    # plot evapotranspiration, soil storage change, percolation and 18O in percolation
    years = onp.arange(1997, 2008).tolist()
    for year in years:
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 4))
        axes[0].plot(
            dict_obs_sim["AET"].loc[f"{year}", :].index,
            dict_obs_sim["AET"].loc[f"{year}", "sim"],
            lw=1,
            color="red",
            ls="-.",
            alpha=1,
        )
        axes[0].plot(
            dict_obs_sim["AET"].loc[f"{year}", :].index,
            dict_obs_sim["AET"].loc[f"{year}", "obs"],
            lw=1,
            color="blue",
            ls="-",
            alpha=0.5,
        )
        axes[0].set_ylabel("ET\n[mm/day]")
        axes[0].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[0].set_ylim(
            0,
        )
        axes[1].plot(
            dict_obs_sim["dWEIGHT"].loc[f"{year}", :].index,
            dict_obs_sim["dWEIGHT"].loc[f"{year}", "sim"],
            lw=1,
            color="red",
            ls="-.",
            alpha=1,
        )
        axes[1].plot(
            dict_obs_sim["dWEIGHT"].loc[f"{year}", :].index,
            dict_obs_sim["dWEIGHT"].loc[f"{year}", "obs"],
            lw=1,
            color="blue",
            ls="-",
            alpha=0.5,
        )
        axes[1].set_ylabel("$\Delta$S\n[mm/day]")
        axes[1].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[2].plot(
            dict_obs_sim["PERC"].loc[f"{year}", :].index,
            dict_obs_sim["PERC"].loc[f"{year}", "sim"],
            lw=1,
            color="red",
            ls="-.",
            alpha=1,
        )
        axes[2].plot(
            dict_obs_sim["PERC"].loc[f"{year}", :].index,
            dict_obs_sim["PERC"].loc[f"{year}", "obs"],
            lw=1,
            color="blue",
            ls="-",
            alpha=0.5,
        )
        axes[2].set_ylabel("PERC\n[mm/day]")
        axes[2].set_xlim((dict_obs["PREC"].loc[f"{year}", :].index[0], dict_obs["PREC"].loc[f"{year}", :].index[-1]))
        axes[2].set_ylim(
            0,
        )
        # load transport simulation
        tms = "advection-dispersion-power"
        idx_best = dict_params_metrics_tm_mc[tms]["params_metrics"]["KGE_C_iso_q_ss"].idxmax()
        states_tm_file = (
            base_path
            / "svat_oxygen18_monte_carlo"
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm100"
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
        ds_sim_tm_year = ds_sim_tm.sel(Time=slice(f"{year}-01-01", f"{year}-12-31"))
        # join observations on simulations
        obs_vals = ds_obs["d18O_PERC"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals
        df_obs = df_obs.loc[f"{year}":f"{year}", "obs"].to_frame()
        axes[3].plot(
            ds_sim_tm_year["Time"].values, ds_sim_tm_year["C_iso_q_ss"].isel(x=idx_best, y=0).values, color="red", lw=1
        )
        axes[3].scatter(df_obs.index, df_obs.iloc[:, 0], color="blue", s=1)
        axes[3].set_ylabel("$\delta^{18}$$O_{PERC}$\n[‰]")
        axes[3].set_ylim((-15, -7))
        axes[3].set_xlim(ds_sim_tm_year["Time"].values[0], ds_sim_tm_year["Time"].values[-1])
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
        axes[3].set_xlabel(r"Time [year-month]")
        fig.tight_layout()
        file = f"prec_et_dS_perc_d18O_obs_sim_{year}.png"
        path = base_path_figs / file
        fig.savefig(path, dpi=300)
        file = f"prec_et_dS_perc_d18O_obs_sim_{year}.pdf"
        path = base_path_figs / file
        fig.savefig(path, dpi=300)

    # bromide benchmark
    years = onp.arange(1997, 2007).tolist()
    cmap = cm.get_cmap("Reds")
    cmap_hydrus = cm.get_cmap("Greys")
    norm = Normalize(vmin=onp.min(years) - 2, vmax=onp.max(years))
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alphas = [0.8]
    for alpha in alphas:
        fig, axes = plt.subplots(5, 1, figsize=(6, 5), sharex=True)
        for i, tm_structure in enumerate(tm_structures):
            tms = tm_structure.replace(" ", "_")
            # add St. Gallen
            if alpha in [0.8]:
                if tm_structure in [
                    "complete-mixing",
                    "advection-dispersion-power",
                    "time-variant advection-dispersion-power",
                ]:
                    df_sim_br_conc = pd.DataFrame(index=df_obs_br.index)
                    states_br_file = (
                        base_path
                        / "svat_bromide_benchmark"
                        / "deterministic"
                        / f"states_{tms}_bromide_benchmark_stgallen.nc"
                    )
                    with xr.open_dataset(states_br_file, engine="h5netcdf", decode_times=False, group=f"1991") as ds:
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
                states_br_file = (
                    base_path / "svat_bromide_benchmark" / "deterministic" / f"states_{tms}_bromide_benchmark.nc"
                )
                with xr.open_dataset(states_br_file, engine="h5netcdf", decode_times=False, group=f"{year}") as ds:
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
            states_hydrus_br_file = base_path / "hydrus_benchmark" / "states_hydrus_bromide.nc"
            with xr.open_dataset(states_hydrus_br_file, engine="h5netcdf", decode_times=False, group=f"{year}") as ds:
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
        axes.flatten()[0].text(0.025, 0.89, "(a)", ha="center", va="center", transform=axes.flatten()[0].transAxes)
        axes.flatten()[1].text(0.025, 0.89, "(b)", ha="center", va="center", transform=axes.flatten()[1].transAxes)
        axes.flatten()[2].text(0.025, 0.89, "(c)", ha="center", va="center", transform=axes.flatten()[2].transAxes)
        axes.flatten()[3].text(0.025, 0.89, "(d)", ha="center", va="center", transform=axes.flatten()[3].transAxes)
        axes.flatten()[4].text(0.025, 0.89, "(e)", ha="center", va="center", transform=axes.flatten()[4].transAxes)
        lines1, labels1 = axes.flatten()[-2].get_legend_handles_labels()
        lines2, labels2 = axes.flatten()[-1].get_legend_handles_labels()
        fig.legend(lines1, labels1, loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(0.965, 0.87))
        fig.legend(lines2, labels2, loc="lower right", fontsize=8, frameon=False, bbox_to_anchor=(0.97, 0.07))
        fig.subplots_adjust(bottom=0.1, right=0.8, top=1.0, hspace=0.2)
        file = base_path_figs / f"bromide_benchmark_alpha_{alpha}.png"
        fig.savefig(file, dpi=300)
        file = base_path_figs / f"bromide_benchmark_alpha_{alpha}.pdf"
        fig.savefig(file, dpi=300)

    # travel time benchmark
    # compare backward travel time distributions
    fig, axes = plt.subplots(2, 5, sharey=True, figsize=(6, 3))
    for i, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        states_tm_file = (
            base_path
            / "svat_oxygen18"
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm1"
            / f"states_{tms}.nc"
        )
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
        states_tm_file = (
            base_path
            / "svat_oxygen18"
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm1"
            / f"states_{tms}.nc"
        )
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
        for j, age_metric in enumerate("ttavg_transp", "ttavg_q_ss"):
            df_age.iloc[j, i] = onp.nanmean(ds_sim_tm[age_metric].isel(x=idx_best, y=0).values)

    TT = onp.where(ds_hydrus_tt["bTT_perc"].values <= 0, onp.nan, ds_hydrus_tt["bTT_perc"].values)
    skipt = 1000
    df_age.iloc["MTT", "HYDRUS-1D"] = onp.nanmean(TT[skipt:, :], axis=0)

    # write evaluation metrics for different storage condtions to .txt
    df_kge_tm = pd.DataFrame(columns=["CM", "PI", "AD", "AD-TV"], index=["", "dry", "normal", "wet"])
    for i, tm_structure in enumerate(tm_structures):
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        for sc in ["", "dry", "normal", "wet"]:
            idx_best = df_params_metrics[f"KGE_C_iso_q_ss{sc}"].idxmax()
            df_kge_tm.loc[sc, df_kge_tm.columns[i]] = df_params_metrics.loc[
                df_params_metrics.index[idx_best], f"KGE_C_iso_q_ss{sc}"
            ]
    file = base_path_figs / "kge_d18O_perc.txt"
    df_kge_tm.to_csv(file, header=True, index=True, sep="\t")

    # diagnostic polar plots for transport models
    for i, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics10 = dict_params_metrics_tm_mc[tm_structure]["params_metrics"].copy()
        df_params_metrics10.loc[:, "id"] = range(len(df_params_metrics10.index))
        df_params_metrics10 = df_params_metrics10.sort_values(by=["KGE_C_iso_q_ss"], ascending=False)
        df_for_diag10 = df_params_metrics10.loc[: df_params_metrics10.index[9], :]

        var_sim = "C_iso_q_ss"
        fig = de.diag_polar_plot_multi(
            df_for_diag10.loc[:, f"brel_mean_{var_sim}"].values,
            df_for_diag10.loc[:, f"temp_cor_{var_sim}"].values,
            df_for_diag10.loc[:, f"DE_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_dir_{var_sim}"].values,
            df_for_diag10.loc[:, f"phi_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_lf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_tot_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_lf_{var_sim}"].values,
        )
        file = f"diag_polar_plot_{var_sim}_{tms}.pdf"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)

    # perform sensitivity analysis
    dict_params_metrics_tm_sa = {}
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        file = (
            base_path
            / "svat_oxygen18_sensitivity"
            / "results"
            / "deterministic"
            / "age_max_1500_days"
            / f"params_metrics_{tms}.txt"
        )
        df_params_metrics = pd.read_csv(file, sep="\t")
        dict_params_metrics_tm_sa[tm_structure] = {}
        dict_params_metrics_tm_sa[tm_structure]["params_metrics"] = df_params_metrics

    # sampled parameter space
    file_path = base_path / "svat_oxygen18_sensitivity" / "param_bounds.yml"
    with open(file_path, "r") as file:
        bounds = yaml.safe_load(file)

    _LABS_TITLE = {
        "KGE_C_iso_q_ss": "$KGE_{\delta^{18}O}$",
        "ttavg_transp": r"$\overline{TT_{transp}}$",
        "tt25_transp": r"$TT_{25-transp}$",
        "tt50_transp": r"$TT_{50-transp}$",
        "tt75_transp": r"$TT_{75-transp}$",
        "ttavg_q_ss": r"$\overline{TT_{perc_{ss}}}$",
        "tt25_q_ss": r"$TT_{25-perc_{ss}}$",
        "tt50_q_ss": r"$TT_{50-perc_{ss}}$",
        "tt75_q_ss": r"$TT_{75-perc_{ss}}$",
        "rtavg_s": r"$\overline{RT}$",
        "rt25_s": r"$RT_{25}$",
        "rt50_s": r"$RT_{50}$",
        "rt75_s": r"$RT_{75}$",
    }
    metrics_tt = ["ttavg", "tt25", "tt50", "tt75"]
    for metric_tt in metrics_tt:
        metrics_sa = [f"{metric_tt}_transp", f"{metric_tt}_q_ss", "KGE_C_iso_q_ss"]
        ncol = len(metrics_sa)
        nrow = len(tm_structures)
        cmap = cm.get_cmap("Reds")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(6, 6))
        for j, tm_structure in enumerate(tm_structures):
            tms = tm_structure.replace(" ", "_")
            df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
            dict_si = {}
            for name in metrics_sa:
                Y = df_params_metrics.loc[:, name].values
                Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
                Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
                Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
                dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

            # plot sobol indices
            xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][:8]
            colors = cmap(norm([0.5, 1.5]))
            for i, name in enumerate(metrics_sa):
                indices = dict_si[name][["S1", "ST"]].iloc[:8, :]
                err = dict_si[name][["S1_conf", "ST_conf"]].iloc[:8, :]
                indices.plot.bar(yerr=err.values.T, ax=ax[j, i], color=colors, width=1.0)
                ax[j, i].set_xticklabels(xaxis_labels)
                ax[0, i].set_title(_LABS_TITLE[name])
                ax[j, i].legend(["First-order", "Total"], frameon=False)
                ax[j, i].legend().set_visible(False)
        ax[-1, -1].legend().set_visible(True)
        ax[-1, -1].legend(["First-order", "Total"], frameon=False, loc="upper left", fontsize=8)
        ax[0, 0].set_ylabel("CM\nSobol index [-]")
        ax[1, 0].set_ylabel("PI\nSobol index [-]")
        ax[2, 0].set_ylabel("AD\nSobol index [-]")
        ax[3, 0].set_ylabel("AD-TV\nSobol index [-]")
        # fig.subplots_adjust(bottom=0.1, right=0.95, hspace=0.65)
        fig.tight_layout()
        file = base_path_figs / f"sobol_indices_{metric_tt}_hm.png"
        fig.savefig(file, dpi=300)

        metrics_sa = [f"{metric_tt}_transp", f"{metric_tt}_q_ss", "KGE_C_iso_q_ss"]
        ncol = len(metrics_sa)
        nrow = 2
        cmap = cm.get_cmap("Reds")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(6, 3))
        for j, tm_structure in enumerate(["advection-dispersion-power", "time-variant advection-dispersion-power"]):
            tms = tm_structure.replace(" ", "_")
            df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
            dict_si = {}
            for name in metrics_sa:
                Y = df_params_metrics.loc[:, name].values
                Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
                Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
                dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

            # plot sobol indices
            xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][8:]
            colors = cmap(norm([0.5, 1.5]))
            for i, name in enumerate(metrics_sa):
                indices = dict_si[name][["S1", "ST"]].iloc[8:, :]
                err = dict_si[name][["S1_conf", "ST_conf"]].iloc[8:, :]
                indices.plot.bar(yerr=err.values.T, ax=ax[j, i], color=colors)
                ax[j, i].set_xticklabels(xaxis_labels)
                ax[j, i].tick_params(axis="x", rotation=33)
                ax[0, i].set_title(_LABS_TITLE[name])
                ax[j, i].legend(["First-order", "Total"], frameon=False)
                ax[j, i].legend().set_visible(False)
        ax[-1, -2].legend().set_visible(True)
        ax[-1, -2].legend(["First-order", "Total"], frameon=False, loc="upper left", fontsize=8)
        ax[0, 0].set_ylabel("AD\nSobol index [-]")
        ax[1, 0].set_ylabel("AD-TV\nSobol index [-]")
        fig.tight_layout()
        file = base_path_figs / f"sobol_indices_{metric_tt}_tm.png"
        fig.savefig(file, dpi=300)

    metrics_sa = ["rtavg_s", "rt25_s", "rt50_s", "rt75_s"]
    ncol = len(metrics_sa)
    nrow = len(tm_structures)
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=0, vmax=2)
    colors = cmap(norm([0.5, 1.5]))
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(6, 6))
    for j, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
        dict_si = {}
        for name in metrics_sa:
            Y = df_params_metrics.loc[:, name].values
            Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

        # plot sobol indices
        xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][:8]
        colors = cmap(norm([0.5, 1.5]))
        for i, name in enumerate(metrics_sa):
            indices = dict_si[name][["S1", "ST"]].iloc[:8, :]
            err = dict_si[name][["S1_conf", "ST_conf"]].iloc[:8, :]
            indices.plot.bar(yerr=err.values.T, ax=ax[j, i], color=colors, width=1)
            ax[j, i].set_xticklabels(xaxis_labels)
            ax[0, i].set_title(_LABS_TITLE[name])
            ax[j, i].legend(["First-order", "Total"], frameon=False)
            ax[j, i].legend().set_visible(False)
    ax[-1, -1].legend().set_visible(True)
    ax[-1, -1].legend(["First-order", "Total"], frameon=False, loc="upper right", fontsize=8, bbox_to_anchor=(1.8, 1.1))
    ax[0, 0].set_ylabel("CM\nSobol index [-]")
    ax[1, 0].set_ylabel("PI\nSobol index [-]")
    ax[2, 0].set_ylabel("AD\nSobol index [-]")
    ax[3, 0].set_ylabel("AD-TV\nSobol index [-]")
    fig.subplots_adjust(bottom=0.2, right=0.85, hspace=0.65)
    file = base_path_figs / "sobol_indices_rt_hm.png"
    fig.savefig(file, dpi=300)

    ncol = len(metrics_sa)
    nrow = 2
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=0, vmax=2)
    colors = cmap(norm([0.5, 1.5]))
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(6, 3))
    for j, tm_structure in enumerate(["advection-dispersion-power", "time-variant advection-dispersion-power"]):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
        dict_si = {}
        for name in metrics_sa:
            Y = df_params_metrics.loc[:, name].values
            Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

        # plot sobol indices
        xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][8:]
        colors = cmap(norm([0.5, 1.5]))
        for i, name in enumerate(metrics_sa):
            indices = dict_si[name][["S1", "ST"]].iloc[8:, :]
            err = dict_si[name][["S1_conf", "ST_conf"]].iloc[8:, :]
            indices.plot.bar(yerr=err.values.T, ax=ax[j, i], color=colors)
            ax[j, i].set_xticklabels(xaxis_labels)
            ax[j, i].tick_params(axis="x", rotation=33)
            ax[0, i].set_title(_LABS_TITLE[name])
            ax[j, i].legend(["First-order", "Total"], frameon=False)
            ax[j, i].legend().set_visible(False)
    ax[-1, -1].legend().set_visible(True)
    ax[-1, -1].legend(["First-order", "Total"], frameon=False, loc="upper right", fontsize=8, bbox_to_anchor=(1.8, 1.1))
    ax[0, 0].set_ylabel("AD\nSobol index [-]")
    ax[1, 0].set_ylabel("AD-TV\nSobol index [-]")
    fig.subplots_adjust(bottom=0.2, right=0.85, hspace=0.65)
    file = base_path_figs / "sobol_indices_rt_tm.png"
    fig.savefig(file, dpi=300)

    metrics_sa = ["KGE_C_iso_q_ss"]
    nrow = len(metrics_sa)
    ncol = len(tm_structures)
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=0, vmax=2)
    colors = cmap(norm([0.5, 1.5]))
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(6, 1.2))
    for j, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
        dict_si = {}
        for name in metrics_sa:
            Y = df_params_metrics.loc[:, name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

        # plot sobol indices
        xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][:8]
        colors = cmap(norm([0.5, 1.5]))
        for i, name in enumerate(metrics_sa):
            indices = dict_si[name][["S1", "ST"]].iloc[:8, :]
            err = dict_si[name][["S1_conf", "ST_conf"]].iloc[:8, :]
            indices.plot.bar(yerr=err.values.T, ax=ax[j], color=colors, width=1)
            ax[j].set_xticklabels(xaxis_labels)
            ax[j].legend(["First-order", "Total"], frameon=False)
            ax[j].legend().set_visible(False)
    ax[-1].legend().set_visible(True)
    ax[-1].legend(["First-order", "Total"], frameon=False, loc="upper left", fontsize=8)
    ax[0].set_title("CM")
    ax[1].set_title("PI")
    ax[2].set_title("AD")
    ax[3].set_title("AD-TV")
    ax[0].set_ylabel("Sobol index [-]")
    fig.tight_layout()
    file = base_path_figs / "sobol_indices_kge_18O_perc_hm.png"
    fig.savefig(file, dpi=300)

    nrow = len(metrics_sa)
    ncol = 2
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=0, vmax=2)
    colors = cmap(norm([0.5, 1.5]))
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(4, 1.2))
    for j, tm_structure in enumerate(["advection-dispersion-power", "time-variant advection-dispersion-power"]):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
        dict_si = {}
        for name in metrics_sa:
            Y = df_params_metrics.loc[:, name].values
            Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

        # plot sobol indices
        xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][8:]
        colors = cmap(norm([0.5, 1.5]))
        for i, name in enumerate(metrics_sa):
            indices = dict_si[name][["S1", "ST"]].iloc[8:, :]
            err = dict_si[name][["S1_conf", "ST_conf"]].iloc[8:, :]
            indices.plot.bar(yerr=err.values.T, ax=ax[j], color=colors, width=1)
            ax[j].set_xticklabels(xaxis_labels)
            ax[j].tick_params(axis="x", rotation=33)
            ax[j].legend(["First-order", "Total"], frameon=False)
            ax[j].legend().set_visible(False)
    ax[0].legend().set_visible(True)
    ax[0].legend(["First-order", "Total"], frameon=False, loc="upper left", fontsize=8)
    ax[0].set_title("AD")
    ax[1].set_title("AD-TV")
    ax[0].set_ylabel("Sobol index [-]")
    fig.tight_layout()
    file = base_path_figs / "sobol_indices_kge_18O_perc_tm.png"
    fig.savefig(file, dpi=300)

    # dotty plots of HYDRUS-1D monte carlo simulations
    file = base_path / "hydrus_benchmark" / "params_metrics.txt"
    df_params_metrics_hydrus = pd.read_csv(file, sep="\t")
    df_params_metrics_hydrus.loc[:, "ks"] = df_params_metrics_hydrus.loc[:, "ks"] * (10 / 24)
    df_metrics_hydrus = df_params_metrics_hydrus.loc[
        :, ["kge_aet", "kge_theta", "kge_perc", "kge_d18O_perc_bs", "kge_multi"]
    ]
    df_params_hydrus = df_params_metrics_hydrus.loc[
        :, ["theta_sat_m", "alpha", "n", "ks", "theta_sat_im", "omega", "D_l"]
    ]
    nrow = len(df_metrics_hydrus.columns)
    ncol = len(df_params_hydrus.columns)
    idx_best = df_metrics_hydrus["kge_multi"].idxmax()
    fig, ax = plt.subplots(nrow, ncol, sharey=True, sharex="col", figsize=(6, 3))
    for i in range(nrow):
        for j in range(ncol):
            y = df_metrics_hydrus.iloc[:, i]
            x = df_params_hydrus.iloc[:, j]
            ax[i, j].scatter(x, y, s=1, c="grey", alpha=0.5)
            ax[i, j].set_xlabel("")
            ax[i, j].set_ylabel("")
            ax[i, j].set_ylim(0, 1)
            # best parameter set for multi-objective criteria
            y_best = df_metrics_hydrus.iloc[idx_best, i]
            x_best = df_params_hydrus.iloc[idx_best, j]
            ax[i, j].scatter(x_best, y_best, s=2, c="red", alpha=1)

    for j in range(ncol):
        xlabel = _LABS_HYDRUS[df_params_hydrus.columns[j]]
        ax[-1, j].set_xlabel(xlabel)

    ax[0, 0].set_ylabel("$KGE_{ET}$\n [-]")
    ylab_kge_theta = r"""$KGE_{\theta}$
    [-]"""
    ax[1, 0].set_ylabel(ylab_kge_theta)
    ax[2, 0].set_ylabel("$KGE_{PERC}$\n [-]", labelpad=7.5)
    ax[3, 0].set_ylabel("$KGE_{\delta^{18}O_{perc}}$\n [-]")
    ax[4, 0].set_ylabel("$KGE_{multi}$\n [-]", labelpad=0.5)

    fig.subplots_adjust(bottom=0.15, left=0.1, top=0.98, right=0.98, wspace=0.2, hspace=0.3)
    file = base_path_figs / "dotty_plots_hydrus.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "dotty_plots_hydrus.pdf"
    fig.savefig(file, dpi=300)

    # plot mean residence time along soil depth
    cmap = copy.copy(plt.cm.get_cmap("Blues_r"))
    norm = mpl.colors.Normalize(vmin=0, vmax=500)

    fig, axes = plt.subplots(1, 1, figsize=(6, 1.5))
    sns.heatmap(
        ds_hydrus_tt["mrt_s"].values,
        xticklabels=366,
        yticklabels=int(50 / 2),
        cmap="Blues_r",
        vmax=500,
        vmin=0,
        cbar=False,
        ax=axes,
    )
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_yticklabels([0, 0.5, 1, 1.5, 2])
    axes.set_xticklabels(list(range(1997, 2008)))
    axes.set_ylabel("Soil depth [m]")
    axes.set_xlabel("Time [year]")

    axl = fig.add_axes([0.88, 0.3, 0.02, 0.58])
    cb1 = mpl.colorbar.ColorbarBase(
        axl, cmap=cmap, norm=norm, orientation="vertical", ticks=[0, 100, 200, 300, 400, 500]
    )
    cb1.ax.invert_yaxis()
    cb1.set_label(r"age [days]")
    fig.subplots_adjust(bottom=0.3, right=0.85)
    file = "mean_residence_time_soil.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    # plot soil bromide concentrations
    years = onp.arange(1997, 2007).tolist()
    cmap = copy.copy(plt.cm.get_cmap("Oranges"))
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    fig, axes = plt.subplots(5, 2, figsize=(6, 10))
    for i, year in enumerate(years):
        states_hydrus_br_file = base_path / "hydrus_benchmark" / "states_hydrus_bromide.nc"
        with xr.open_dataset(states_hydrus_br_file, engine="h5netcdf", decode_times=False, group=f"{year}") as ds:
            sns.heatmap(
                ds["Br_soil"].values,
                xticklabels=100,
                yticklabels=int(50 / 2),
                cmap="Oranges",
                vmax=100,
                vmin=0,
                cbar=False,
                ax=axes.flatten()[i],
            )
        axes.flatten()[i].set_title(r"$12^{th}$ Nov %s" % (year))
        axes.flatten()[i].set_yticks([0, 25, 50, 75, 100])
        axes.flatten()[i].set_yticklabels([0, 0.5, 1, 1.5, 2])

    axes[0, 0].set_ylabel("Soil depth [m]")
    axes[1, 0].set_ylabel("Soil depth [m]")
    axes[2, 0].set_ylabel("Soil depth [m]")
    axes[3, 0].set_ylabel("Soil depth [m]")
    axes[4, 0].set_ylabel("Soil depth [m]")
    axes[4, 0].set_xlabel("Time [days since injection]")
    axes[4, 1].set_xlabel("Time [days since injection]")
    axl = fig.add_axes([0.88, 0.38, 0.02, 0.2])
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation="vertical", ticks=[0, 50, 100])
    cb1.ax.set_yticklabels(["0", "50", ">100"])
    cb1.set_label("Bromide [mg/l]", labelpad=-1)
    fig.subplots_adjust(left=0.1, bottom=0.05, top=0.98, right=0.85, hspace=0.7)
    file = "bromide_conc_soil.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    # plot soil bromide mass
    years = onp.arange(1997, 2007).tolist()
    cmap = copy.copy(plt.cm.get_cmap("Oranges"))
    norm = mpl.colors.Normalize(vmin=0, vmax=100)
    fig, axes = plt.subplots(5, 2, figsize=(6, 10))
    for i, year in enumerate(years):
        states_hydrus_br_file = base_path / "hydrus_benchmark" / "states_hydrus_bromide.nc"
        with xr.open_dataset(states_hydrus_br_file, engine="h5netcdf", decode_times=False, group=f"{year}") as ds:
            sns.heatmap(
                ds["Br_soil"].values * ds["swc"].values * 20,
                xticklabels=100,
                yticklabels=int(50 / 2),
                cmap="Oranges",
                vmax=100,
                vmin=0,
                cbar=False,
                ax=axes.flatten()[i],
            )
        axes.flatten()[i].set_title(r"$12^{th}$ Nov %s" % (year))
        axes.flatten()[i].set_yticks([0, 25, 50, 75, 100])
        axes.flatten()[i].set_yticklabels([0, 0.5, 1, 1.5, 2])

    axes[0, 0].set_ylabel("Soil depth [m]")
    axes[1, 0].set_ylabel("Soil depth [m]")
    axes[2, 0].set_ylabel("Soil depth [m]")
    axes[3, 0].set_ylabel("Soil depth [m]")
    axes[4, 0].set_ylabel("Soil depth [m]")
    axes[4, 0].set_xlabel("Time [days since injection]")
    axes[4, 1].set_xlabel("Time [days since injection]")
    axl = fig.add_axes([0.88, 0.38, 0.02, 0.2])
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation="vertical", ticks=[0, 50, 100])
    cb1.ax.set_yticklabels(["0", "50", ">100"])
    cb1.set_label("Bromide [mg]", labelpad=-1)
    fig.subplots_adjust(left=0.1, bottom=0.05, top=0.98, right=0.85, hspace=0.7)
    file = "bromide_mass_soil.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    # plot isotope ratios of precipitation, soil and percolation
    cmap = copy.copy(plt.cm.get_cmap("YlGnBu_r"))
    norm = mpl.colors.Normalize(vmin=-20, vmax=0)

    fig, axes = plt.subplots(3, 1, sharex=False, figsize=(6, 3))
    axes[0].bar(
        ds_hydrus_18O["Time"].values,
        ds_hydrus_18O["prec"].values,
        width=-1,
        edgecolor=cmap(norm(ds_hydrus_18O["d18O_prec"].values)),
        align="edge",
    )
    axes[0].set_ylabel("Precipitation\n[mm $day^{-1}$]")
    axes[0].set_xlim(ds_hydrus_18O["Time"].values[0], ds_hydrus_18O["Time"].values[-1])
    axes[0].set_xticklabels([])
    sns.heatmap(
        ds_hydrus_18O["d18O_soil"].values,
        xticklabels=366,
        yticklabels=int(50 / 2),
        cmap="YlGnBu_r",
        vmax=0,
        vmin=-20,
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_yticks([0, 25, 50, 75, 100])
    axes[1].set_yticklabels([0, 0.5, 1, 1.5, 2])
    axes[1].set_xticklabels([])
    axes[1].set_ylabel("Soil depth\n[m]")

    axes[2].bar(
        ds_hydrus_18O["Time"].values,
        ds_hydrus_18O["perc"].values,
        width=-1,
        edgecolor=cmap(norm(ds_hydrus_18O["d18O_perc"].values)),
        align="edge",
    )
    axes[2].set_xlim(ds_hydrus_18O["Time"].values[0], ds_hydrus_18O["Time"].values[-1])
    axes[2].set_ylabel("Percolation\n[mm $day^{-1}$]")
    axes[2].set_ylim(
        0,
    )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Time [year]")

    axl = fig.add_axes([0.87, 0.34, 0.02, 0.3])
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation="vertical", ticks=[0, -5, -10, -15, -20])
    cb1.set_label(r"$\delta^{18}$O [‰]")
    fig.subplots_adjust(bottom=0.15, right=0.85)
    file = "hydrus_d18O_prec_soil_perc.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    # plot precipitation, soil water content and percolation
    cmap = copy.copy(plt.cm.get_cmap("YlGnBu"))
    norm = mpl.colors.Normalize(vmin=0.2, vmax=0.4)

    fig, axes = plt.subplots(3, 1, sharex=False, figsize=(6, 3))
    axes[0].bar(ds_hydrus_18O["Time"].values, ds_hydrus_18O["prec"].values, width=-1, edgecolor="blue", align="edge")
    axes[0].set_ylabel("Precipitation\n[mm $day^{-1}$]")
    axes[0].set_xlim(ds_hydrus_18O["Time"].values[0], ds_hydrus_18O["Time"].values[-1])
    axes[0].set_xticklabels([])
    sns.heatmap(
        ds_hydrus_18O["swc"].values,
        xticklabels=366,
        yticklabels=int(50 / 2),
        cmap="YlGnBu",
        vmax=0.4,
        vmin=0.2,
        cbar=False,
        ax=axes[1],
    )
    axes[1].set_yticks([0, 25, 50, 75, 100])
    axes[1].set_yticklabels([0, 0.5, 1, 1.5, 2])
    axes[1].set_xticklabels([])
    axes[1].set_ylabel("Soil depth\n[m]")

    axes[2].bar(ds_hydrus_18O["Time"].values, ds_hydrus_18O["perc"].values, width=-1, edgecolor="grey", align="edge")
    axes[2].set_xlim(ds_hydrus_18O["Time"].values[0], ds_hydrus_18O["Time"].values[-1])
    axes[2].set_ylabel("Percolation\n[mm $day^{-1}$]")
    axes[2].set_ylim(
        0,
    )
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Time [year]")

    axl = fig.add_axes([0.87, 0.34, 0.02, 0.3])
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm, orientation="vertical", ticks=[0.2, 0.3, 0.4])
    cb1.set_label(r"$\theta$ [-]")
    fig.subplots_adjust(bottom=0.15, right=0.85)
    file = "hydrus_prec_theta_perc.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    plt.close("all")

    # figures for talk
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
            alpha=0.7,
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
    fig.tight_layout()
    file = f"prec_et_dS_perc_obs_sim_cumulated_best_100_optimized_with_KGE_multi_for_talk.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"prec_et_dS_perc_obs_sim_cumulated_best_100_optimized_with_KGE_multi_for_talk.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

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
        states_tm_file = (
            base_path
            / "svat_oxygen18_monte_carlo"
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm100"
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
                states_br_file = (
                    base_path
                    / "svat_bromide_benchmark"
                    / "deterministic"
                    / f"states_{tms}_bromide_benchmark_stgallen.nc"
                )
                with xr.open_dataset(states_br_file, engine="h5netcdf", decode_times=False, group=f"1991") as ds:
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
                states_br_file = (
                    base_path / "svat_bromide_benchmark" / "deterministic" / f"states_{tms}_bromide_benchmark.nc"
                )
                with xr.open_dataset(states_br_file, engine="h5netcdf", decode_times=False, group=f"{year}") as ds:
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
            states_hydrus_br_file = base_path / "hydrus_benchmark" / "states_hydrus_bromide.nc"
            with xr.open_dataset(states_hydrus_br_file, engine="h5netcdf", decode_times=False, group=f"{year}") as ds:
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
        states_tm_file = (
            base_path
            / "svat_oxygen18"
            / "deterministic"
            / "age_max_1500_days"
            / "optimized_with_KGE_multi_hm1"
            / f"states_{tms}.nc"
        )
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
