from pathlib import Path
import os
import numpy as onp
import pandas as pd
import click
import matplotlib as mpl
import seaborn as sns
import datetime

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["legend.title_fontsize"] = 11
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


@click.command("main")
def main():
    base_path = Path(__file__).parent

    Ta_path = base_path / "input" / "TA.txt"
    PREC_path = base_path / "input" / "PREC.txt"
    PET_path = base_path / "input" / "PET.txt"

    df_precip_10mins = pd.read_csv(
        PREC_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_precip_10mins.index = pd.to_datetime(dict(year=df_precip_10mins.YYYY, month=df_precip_10mins.MM, day=df_precip_10mins.DD, hour=df_precip_10mins.hh, minute=df_precip_10mins.mm))
    df_precip_10mins = df_precip_10mins.loc[:, ["PREC"]]
    df_precip_10mins.index = df_precip_10mins.index.rename("Index")
    df_precip_10mins.columns = ["PRECIP"]

    df_pet_daily = pd.read_csv(
        PET_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_pet_daily.index = pd.to_datetime(dict(year=df_pet_daily.YYYY, month=df_pet_daily.MM, day=df_pet_daily.DD, hour=df_pet_daily.hh, minute=df_pet_daily.mm))
    df_pet_daily = df_pet_daily.loc[:, ["PET"]]
    df_pet_daily.index = df_pet_daily.index.rename("Index")

    df_ta_daily = pd.read_csv(
        Ta_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_ta_daily.index = pd.to_datetime(dict(year=df_ta_daily.YYYY, month=df_ta_daily.MM, day=df_ta_daily.DD, hour=df_ta_daily.hh, minute=df_ta_daily.mm))
    df_ta_daily = df_ta_daily.loc[:, "TA":]
    df_ta_daily.index = df_ta_daily.index.rename("Index")


    # Resample the data to daily frequency
    df_precip_daily = df_precip_10mins.resample("D").sum()

    # Resample the data to monthly frequency
    df_precip_monthly = df_precip_daily.resample("ME").sum()
    df_pet_monthly = df_pet_daily.resample("ME").sum()
    df_ta_monthly = df_ta_daily.resample("ME").mean()

    cond_april = (df_precip_monthly.index.month == 4)
    cond_may = (df_precip_monthly.index.month == 5)
    cond_june = (df_precip_monthly.index.month == 6)
    cond_july = (df_precip_monthly.index.month == 7)
    cond_april_may = (df_precip_monthly.index.month == 4) & (df_precip_monthly.index.month == 5)
    cond_june_july = (df_precip_monthly.index.month == 6) & (df_precip_monthly.index.month == 7)

    df_precip_april_may = df_precip_monthly.loc[cond_april_may, "PRECIP"]
    df_pet_april_may = df_pet_monthly.loc[cond_april_may, "PET"]
    df_ta_april_may = df_ta_monthly.loc[cond_april_may, ["TA_max", "TA", "TA_min"]]

    df_precip_june_july = df_precip_monthly.loc[cond_june_july, "PRECIP"]
    df_pet_june_july = df_pet_monthly.loc[cond_june_july, "PET"]
    df_ta_june_july = df_ta_monthly.loc[cond_june_july, ["TA_max", "TA", "TA_min"]]

    # Resample to annual frequency
    df_precip_annual = df_precip_daily.resample("YE").sum()
    df_pet_annual = df_pet_daily.resample("YE").sum()
    df_ta_annual = df_ta_daily.resample("YE").mean()

    # plot annual ddata
    years = df_ta_annual.index.year.values.tolist()
    idx = onp.asarray([i for i in range(len(years))])
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6, 4))
    axs[0].bar(idx, df_precip_annual["PRECIP"], color="blue", width=1)
    axs[0].set_xticks(idx)
    axs[0].set_ylim(0, 1500)
    axs[0].set_ylabel('[mm/Year]')
    axs[0].set_xlabel('')
    axs[1].plot(idx, df_ta_annual["TA_max"], color="orange", marker="^", ls="--")
    axs[1].plot(idx, df_ta_annual["TA"], color="orange", marker=".")
    axs[1].plot(idx, df_ta_annual["TA_min"], color="orange", marker="v", ls="--")
    axs[1].set_xticks(idx)
    axs[1].set_ylim(0, 25)
    axs[1].set_ylabel('[degC]')
    axs[1].set_xlabel('')
    axs[2].bar(idx, df_pet_annual["PET"], color="green", width=1)
    axs[2].set_ylim(0, 1500)
    axs[2].set_ylabel('[mm/Year]')
    axs[2].set_xlabel('')
    axs[2].set_xticks(idx)
    axs[2].set_xticklabels(years, rotation=65)
    axs[2].set_xlabel('[Year]')
    fig.tight_layout()
    file_str = "meteo_annual.png"
    path_fig = base_path / "figures" / file_str
    fig.savefig(path_fig, dpi=300)
    plt.close("all")


    return


if __name__ == "__main__":
    main()
    