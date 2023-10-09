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

    # load data from bromide experiment
    path_obs_br = Path(__file__).parent / "observations" / "bromide_breakthrough.csv"
    df_obs_br = pd.read_csv(path_obs_br, skiprows=1, sep=";", na_values="")

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
    plt.close("all")
    return


if __name__ == "__main__":
    main()
