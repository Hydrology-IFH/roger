from pathlib import Path
import os
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.dates as mdates
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

    fig, axs = plt.subplots(1, 1, figsize=(6, 2))
    axs.plot(
        df_lys_obs_nonan.loc["2006":, :].index,
        df_lys_obs_nonan.loc["2006":, "dS_flux_corr"] - df_lys_obs_nonan.loc["2006":, "dS_weight"],
        "-",
        color="black",
        lw=0.8,
    )
    axs.set_ylabel(r"[mm]")
    axs.set_xlabel("Time [year]")
    axs.set_xlim(df_lys_obs.loc["2006":, :].index[0], df_lys_obs.loc["2006":, :].index[-1])
    fig.tight_layout()
    file = base_path_figs / "dS_weight_vs_dS_flux_residuals_2006_2007.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(6, 2))
    axs.plot(
        df_lys_obs_nonan.loc["2006":, :].index,
        df_lys_obs_nonan.loc["2006":, "dS_flux_corr"],
        "-",
        color="blue",
        lw=1,
        label=r"dS from fluxes",
    )
    axs.plot(
        df_lys_obs_nonan.loc["2006":, :].index,
        df_lys_obs_nonan.loc["2006":, "dS_weight"],
        "-",
        color="red",
        lw=0.8,
        label="dS from lysimeter weight",
    )
    axs.set_ylabel(r"[mm]")
    axs.set_xlabel("Time [year]")
    axs.legend(frameon=False, loc="upper right", fontsize=6)
    axs.set_xlim(df_lys_obs.loc["2006":, :].index[0], df_lys_obs.loc["2006":, :].index[-1])
    fig.tight_layout()
    file = base_path_figs / "dS_weight_vs_dS_flux_2006_2007.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(6, 2))
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
    plt.close("all")
    return


if __name__ == "__main__":
    main()
