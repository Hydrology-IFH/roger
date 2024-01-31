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

    crops_lys = {"lys2": crops_lys2_lys3_lys8,
                 "lys3": crops_lys2_lys3_lys8,
                 "lys4": crops_lys4_lys9,
                 "lys8": crops_lys2_lys3_lys8,
                 "lys9": crops_lys4_lys9}

    fert_lys2_lys3_lys8 = {"lys2": "130% N-fertilized",
                           "lys3": "100% N-fertilized",
                           "lys8": "70% N-fertilized"
    }

    fert_lys4_lys9 = {"lys4": "Organic",
                      "lys9": "PEP-intensive"
    }

    fert_lys = {"lys2": "130% N-fertilized",
                "lys3": "100% N-fertilized",
                "lys8": "70% N-fertilized",
                "lys4": "Organic",
                "lys9": "PEP-intensive"}

    _lys = {"lys2": "Lys 2",
            "lys3": "Lys 3",
            "lys4": "Lys 4",
            "lys8": "Lys 8",
            "lys9": "Lys 9",
    }

    _y_labels_cumulated = {"prec": "PRECIP [mm]",
                           "pet": "PET [mm]",
                           "perc": "PERC [mm]",
                           "weight": "WEIGHT [mm]",
                           "theta": r"$\theta$ [-]"
                           }
    
    _y_labels_daily = {"prec": "PRECIP [mm/day]",
                       "pet": "PET [mm/day]",
                       "perc": "PERC [mm/day]",
                       "weight": "WEIGHT [mm]",
                       "theta": r"$\theta$ [-]"
                       }
                           
    dict_obs = {}
    dict_obs_nonan = {}
    # load observations (measured data)
    lys_experiments = ["lys2", "lys3", "lys4", "lys8", "lys9"]
    for lys_experiment in lys_experiments:
        path_obs = Path(__file__).parent / "observations" / "reckenholz_lysimeter.nc"
        ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)
        # assign date
        days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_obs = num2date(
            days_obs,
            units=f"days since {ds_obs['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_obs = ds_obs.assign_coords(Time=("Time", date_obs))

        # check water balance of lysimeter
        df_lys_obs = pd.DataFrame(index=date_obs)
        df_lys_obs.loc[:, "ta"] = ds_obs["TA"].isel(x=0, y=0).values
        df_lys_obs.loc[:, "ta_min"] = ds_obs["TA_MIN"].isel(x=0, y=0).values
        df_lys_obs.loc[:, "ta_max"] = ds_obs["TA_MAX"].isel(x=0, y=0).values
        df_lys_obs.loc[:, "prec"] = ds_obs["PREC"].isel(x=0, y=0).values
        if lys_experiment in ["lys2"]:
            df_lys_obs.loc[:, "prec_corr"] = ds_obs["PREC"].isel(x=0, y=0).values
        else:
            df_lys_obs.loc[:, "prec_corr"] = ds_obs["PREC_CORR"].isel(x=0, y=0).values
        df_lys_obs.loc[:, "pet"] = ds_obs["PET"].isel(x=0, y=0).values
        df_lys_obs.loc[:, "perc"] = ds_obs["PERC"].isel(x=0, y=0).values
        df_lys_obs.loc[:, "theta"] = onp.nanmean(ds_obs["THETA"].isel(x=0, y=0).values.T, axis=-1)
        df_lys_obs.loc[:, "weight"] = ds_obs["WEIGHT"].isel(x=0, y=0).values
        df_lys_obs.loc[:, "perc_pet_ratio"] = df_lys_obs.loc[:, "perc"] / (df_lys_obs.loc[:, "perc"] + df_lys_obs.loc[:, "pet"])
        file = base_path_figs / f"{lys_experiment}_fluxes_weight.csv"
        df_lys_obs.to_csv(file, header=True, index=True, sep=";")
        dict_obs[lys_experiment] = df_lys_obs
        # data with bad quality flags have been removed
        df_lys_obs_nonan = df_lys_obs.loc[:, ["ta", "prec", "prec_corr", "pet", "perc", "theta", "weight", "perc_pet_ratio"]].dropna().copy()
        df_lys_obs_nonan.loc[:, "pet + perc"] = df_lys_obs_nonan.loc[:, "pet"] + df_lys_obs_nonan.loc[:, "perc"]
        df_lys_obs_nonan.loc[:, "dS_flux"] = (
            df_lys_obs_nonan.loc[:, "prec"] - df_lys_obs_nonan.loc[:, "pet"] - df_lys_obs_nonan.loc[:, "perc"]
        )
        df_lys_obs_nonan.loc[:, "dS_flux_corr"] = (
            df_lys_obs_nonan.loc[:, "prec_corr"] - df_lys_obs_nonan.loc[:, "pet"] - df_lys_obs_nonan.loc[:, "perc"]
        )
        dict_obs_nonan[lys_experiment] = df_lys_obs_nonan

        fig, axs = plt.subplots(1, 1, figsize=(6, 1.2))
        axs.plot(
            df_lys_obs_nonan.loc[:, :].index,
            df_lys_obs_nonan.loc[:, "dS_flux"].cumsum(),
            "-",
            color="#6a51a3",
            lw=0.8,
            label=r"dS from fluxes",
        )
        axs.plot(
            df_lys_obs_nonan.loc[:, :].index,
            df_lys_obs_nonan.loc[:, "dS_flux_corr"].cumsum(),
            "-",
            color="#9e9ac8",
            lw=0.5,
            label="dS from fluxes\n(with corrected PREC)",
        )
        axs.set_ylabel(r"[mm]")
        axs.set_xlabel("Time [year]")
        axs.legend(frameon=False, loc="upper left", fontsize=5)
        axs.set_xlim(df_lys_obs.loc[:, :].index[0], df_lys_obs.loc[:, :].index[-1])
        fig.tight_layout()
        file = base_path_figs / f"{lys_experiment}_dS_flux_cumulated.png"
        fig.savefig(file, dpi=300)
        plt.close(fig=fig)

        years = onp.arange(2010, 2018).tolist()
        fig, axes = plt.subplots(4, 2, figsize=(6, 6))
        axs = axes.flatten()
        for i, year in enumerate(years):
            axs[i].plot(
                df_lys_obs_nonan.loc[f"{year}", :].index,
                df_lys_obs_nonan.loc[f"{year}", "prec"].cumsum(),
                "-",
                color="#034e7b",
                lw=1,
                label=r"PRECIP",
            )
            # axs[i].plot(
            #     df_lys_obs_nonan.loc[f"{year}", :].index,
            #     df_lys_obs_nonan.loc[f"{year}", "prec_corr"].cumsum(),
            #     "-",
            #     color="#0570b0",
            #     lw=1,
            #     label=r"PREC (corr.)",
            # )
            axs[i].plot(
                df_lys_obs_nonan.loc[f"{year}", :].index,
                df_lys_obs_nonan.loc[f"{year}", "pet + perc"].cumsum(),
                "-",
                color="#74a9cf",
                lw=1,
                label=r"PET + PERC",
            )
            axs[i].plot(
                df_lys_obs_nonan.loc[f"{year}", :].index,
                df_lys_obs_nonan.loc[f"{year}", "dS_flux"].cumsum(),
                "-",
                color="#9e9ac8",
                lw=1,
                label=r"dS from fluxes",
            )
            # axs[i].plot(
            #     df_lys_obs_nonan.loc[f"{year}", :].index,
            #     df_lys_obs_nonan.loc[f"{year}", "dS_flux_corr"].cumsum(),
            #     "-",
            #     color="#9e9ac8",
            #     lw=1,
            #     label=r"dS from fluxes (corr.)",
            # )
            lys_weight = df_lys_obs_nonan.loc[f"{year}", "weight"].values - df_lys_obs_nonan.loc[f"{year}", "weight"].values[0]
            axs[i].plot(
                df_lys_obs_nonan.loc[f"{year}", :].index,
                lys_weight,
                "-",
                color="black",
                lw=1,
                label=r"WEIGHT",
            )
    
            axs[i].set_xlim(
                df_lys_obs_nonan.loc[f"{year}", :].index[0],
                df_lys_obs_nonan.loc[f"{year}", :].index[-1],
            )

            axs[i].tick_params(axis="x", labelrotation=90)
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%m"))
            axs[i].set_title(f"{year}: {crops_lys[lys_experiment][year]} ({fert_lys[lys_experiment]})")
            dS_diff = onp.round(df_lys_obs_nonan.loc[f"{year}", "dS_flux"].cumsum().values[-1] - lys_weight[-1], 2)
            dS_ratio = onp.round(dS_diff / df_lys_obs_nonan.loc[f"{year}", "pet"].cumsum().values[-1], 2)
            axs[i].text(0.85,
            0.4,
            f"{dS_diff} mm",
            fontsize=8,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[i].transAxes)
            axs[i].text(0.85,
            0.5,
            f"{dS_ratio}",
            fontsize=8,
            horizontalalignment="center",
            verticalalignment="center",
            transform=axs[i].transAxes)

        axes[0, 0].set_ylabel(r"[mm]")
        axes[1, 0].set_ylabel(r"[mm]")
        axes[2, 0].set_ylabel(r"[mm]")
        axes[3, 0].set_ylabel(r"[mm]")
        axes[-1, -1].set_xlabel("Time [month]")
        axes[-1, 0].set_xlabel("Time [month]")
        lines, labels = axes[-1, 1].get_legend_handles_labels()
        fig.legend(lines[:2], labels[:2], loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.1, 0.96))
        fig.legend(lines[2:], labels[2:], loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.1, 0.715))
        fig.tight_layout()
        file = base_path_figs / f"{lys_experiment}_cumulated_annually.png"
        fig.savefig(file, dpi=300)
        plt.close(fig=fig)

        fig, axs = plt.subplots(1, 1, figsize=(6, 2))
        axs.plot(
            df_lys_obs_nonan.loc[:, :].index,
            df_lys_obs_nonan.loc[:, "prec"].cumsum(),
            "-",
            color="#034e7b",
            lw=1,
            label=r"PRECIP",
        )
        # axs.plot(
        #     df_lys_obs_nonan.loc[:, :].index,
        #     df_lys_obs_nonan.loc[:, "prec_corr"].cumsum(),
        #     "-",
        #     color="#0570b0",
        #     lw=1,
        #     label=r"PREC (corr.)",
        # )
        axs.plot(
            df_lys_obs_nonan.loc[:, :].index,
            df_lys_obs_nonan.loc[:, "pet + perc"].cumsum(),
            "-",
            color="#74a9cf",
            lw=1,
            label=r"PET + PERC",
        )
        axs.plot(
            df_lys_obs_nonan.loc[:, :].index,
            df_lys_obs_nonan.loc[:, "dS_flux"].cumsum(),
            "-",
            color="#9e9ac8",
            lw=1,
            label=r"dS from fluxes",
        )
        # axs.plot(
        #     df_lys_obs_nonan.loc[:, :].index,
        #     df_lys_obs_nonan.loc[:, "dS_flux_corr"].cumsum(),
        #     "-",
        #     color="#9e9ac8",
        #     lw=1,
        #     label=r"dS from fluxes (corr.)",
        # )
        lys_weight = df_lys_obs_nonan.loc[:, "weight"].values - df_lys_obs_nonan.loc[:, "weight"].values[0]
        axs.plot(
            df_lys_obs_nonan.loc[:, :].index,
            lys_weight,
            "-",
            color="black",
            lw=1,
            label=r"WEIGHT",
        )
        dS_diff = onp.round(df_lys_obs_nonan.loc[:, "dS_flux"].cumsum().values[-1] - lys_weight[-1], 2)
        dS_ratio = onp.round(dS_diff / df_lys_obs_nonan.loc[:, "pet"].cumsum().values[-1], 2)
        axs.text(0.85,
        0.4,
        f"{dS_diff} mm",
        fontsize=8,
        horizontalalignment="left",
        verticalalignment="center",
        transform=axs.transAxes)
        axs.text(0.85,
        0.5,
        f"{dS_ratio}",
        fontsize=8,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axs.transAxes)

        axs.tick_params(axis="x", rotation=45)
        axs.set_ylabel(r"[mm]")
        axs.set_xlabel("Time [year]")
        axs.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)
        axs.set_xlim(df_lys_obs_nonan.loc[:, :].index[0], df_lys_obs_nonan.loc[:, :].index[-1])
        fig.tight_layout()
        file = base_path_figs / f"{lys_experiment}_cumulated.png"
        fig.savefig(file, dpi=300)
        plt.close(fig=fig)

        df_lys_obs_nonan.loc[:, "year"] = df_lys_obs_nonan.index.year
        df_lys_ann = df_lys_obs_nonan.loc[:, :].groupby("year").sum()
        df_lys_ann = df_lys_ann.loc[:, ["prec", "prec_corr", "pet + perc"]]
        df_lys_ann.columns = ["PRECIP", "PREC (corr.)", "PET + PERC"]
        df_lys_ann.loc[:, "year"] = df_lys_ann.index
        df_lys_ann.loc[:, "dS from fluxes"] = df_lys_ann.loc[:, "PRECIP"] - df_lys_ann.loc[:, "PET + PERC"]
        df_lys_ann.loc[:, "dS from fluxes (corr.)"] = (
            df_lys_ann.loc[:, "PREC (corr.)"] - df_lys_ann.loc[:, "PET + PERC"]
        )
        df_lys_ann = pd.melt(
            df_lys_ann,
            id_vars=["year"],
            value_vars=[
                "PRECIP",
                "PREC (corr.)",
                "PET + PERC",
                "dS from fluxes",
                "dS from fluxes (corr.)",
            ],
        )
        fig, axs = plt.subplots(1, 1, figsize=(6, 1.5))
        g = sns.barplot(df_lys_ann, x="year", hue="variable", y="value", ax=axs, palette="PuBu_r")
        axs.set_ylabel(r"[mm]")
        axs.set_xlabel("Time [year]")
        g.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.5, 1.1))
        fig.subplots_adjust(bottom=0.3, right=0.68)
        file = base_path_figs / f"{lys_experiment}_balance_annual.png"
        fig.savefig(file, dpi=300)
        plt.close(fig=fig)
        plt.close("all")

        fig, axes = plt.subplots(4, 2, figsize=(6, 6))
        df = df_lys_obs_nonan.copy()
        cond = df.loc[:, "perc_pet_ratio"] < 0.5
        df.loc[cond, "weight"] = onp.nan
        df.loc[cond, "perc"] = onp.nan
        axs = axes.flatten()
        for i, year in enumerate(years):
            axs[i].plot(
                df.loc[f"{year}", :].index[1:],
                df.loc[f"{year}", "weight"].values[1:] - df.loc[f"{year}", "weight"].values[:-1],
                "-",
                color="#034e7b",
                lw=1,
                label=r"dS",
            )
            axs[i].plot(
                df.loc[f"{year}", :].index,
                -df.loc[f"{year}", "perc"],
                "-",
                color="#74a9cf",
                lw=1,
                label=r"PERC",
            )
            axs[i].set_xlim(
                df_lys_obs_nonan.loc[f"{year}", :].index[0],
                df_lys_obs_nonan.loc[f"{year}", :].index[-1],
            )

            axs[i].tick_params(axis="x", labelrotation=90)
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%m"))
            axs[i].set_title(f"{year}: {crops_lys[lys_experiment][year]} ({fert_lys[lys_experiment]})")
        axes[0, 0].set_ylabel(r"[mm/day]")
        axes[1, 0].set_ylabel(r"[mm/day]")
        axes[2, 0].set_ylabel(r"[mm/day]")
        axes[3, 0].set_ylabel(r"[mm/day]")
        axes[-1, -1].set_xlabel("Time [month]")
        axes[-1, 0].set_xlabel("Time [month]")
        lines, labels = axes[-1, 1].get_legend_handles_labels()
        fig.legend(lines[:2], labels[:2], loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.1, 0.96))
        fig.legend(lines[2:], labels[2:], loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.1, 0.715))
        fig.tight_layout()
        file = base_path_figs / f"{lys_experiment}_dS_perc.png"
        fig.savefig(file, dpi=300)
        plt.close(fig=fig)


        fig, axes = plt.subplots(4, 2, figsize=(6, 6))
        df = df_lys_obs_nonan.copy()
        df.loc[:, "dS"] = onp.nan
        df.loc[df.index[1]:, "dS"] = df.loc[:, "weight"].values[1:] - df.loc[:, "weight"].values[:-1]
        cond = (df.loc[:, "perc_pet_ratio"] < 0.5) | (df.loc[:, "dS"] > df.loc[:, "prec"])
        df.loc[cond, "dS"] = onp.nan
        df.loc[cond, "perc"] = onp.nan
        axs = axes.flatten()
        for i, year in enumerate(years):
            axs[i].plot(
                df.loc[f"{year}", :].index,
                df.loc[f"{year}", "dS"].values,
                "-",
                color="#034e7b",
                lw=1,
                label=r"dS",
            )
            axs[i].plot(
                df.loc[f"{year}", :].index,
                -df.loc[f"{year}", "perc"].values,
                "-",
                color="#74a9cf",
                lw=1,
                label=r"PERC",
            )
            axs[i].set_xlim(
                df_lys_obs_nonan.loc[f"{year}", :].index[0],
                df_lys_obs_nonan.loc[f"{year}", :].index[-1],
            )

            axs[i].tick_params(axis="x", labelrotation=90)
            axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%m"))
            axs[i].set_title(f"{year}: {crops_lys[lys_experiment][year]} ({fert_lys[lys_experiment]})")
        axes[0, 0].set_ylabel(r"[mm/day]")
        axes[1, 0].set_ylabel(r"[mm/day]")
        axes[2, 0].set_ylabel(r"[mm/day]")
        axes[3, 0].set_ylabel(r"[mm/day]")
        axes[-1, -1].set_xlabel("Time [month]")
        axes[-1, 0].set_xlabel("Time [month]")
        lines, labels = axes[-1, 1].get_legend_handles_labels()
        fig.legend(lines[:2], labels[:2], loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.1, 0.96))
        fig.legend(lines[2:], labels[2:], loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.1, 0.715))
        fig.tight_layout()
        file = base_path_figs / f"{lys_experiment}_dSneg_perc.png"
        fig.savefig(file, dpi=300)
        plt.close(fig=fig)


    for lys_experiment in lys_experiments:
        for var_obs in ["perc", "pet", "prec"]:
            fig, axs = plt.subplots(1, 1, figsize=(6, 2))
            df = pd.DataFrame(index=pd.date_range(start="2010-01-01", end="2017-12-31", freq="D"))
            df1 = dict_obs["lys8"].loc[:, f"{var_obs}"].to_frame()
            df1.columns = ["lys8"]
            df2 = dict_obs["lys3"].loc[:, f"{var_obs}"].to_frame()
            df2.columns = ["lys3"]
            df3 = dict_obs["lys2"].loc[:, f"{var_obs}"].to_frame()
            df3.columns = ["lys2"]
            df = df.join([df1, df2, df3]).dropna()
            axs.plot(
                df.index,
                df.loc[:, "lys8"].cumsum(),
                "-",
                color="#d7b5d8",
                lw=0.7,
                label=f"{fert_lys['lys8']}",
            )
            axs.plot(
                df.index,
                df.loc[:, "lys3"].cumsum(),
                "-",
                color="#df65b0",
                lw=1,
                label=f"{fert_lys['lys3']}",
            )
            axs.plot(
                df.index,
                df.loc[:, "lys2"].cumsum(),
                "-",
                color="#ce1256",
                lw=1.3,
                label=f"{fert_lys['lys2']}",
            )

            axs.tick_params(axis="x", rotation=45)
            axs.set_ylabel(f"{_y_labels_cumulated[var_obs]}")
            axs.set_xlabel("Time [year]")
            axs.legend(frameon=False, loc="upper left", fontsize=8)
            axs.set_xlim(df_lys_obs_nonan.loc[:, :].index[0], df_lys_obs_nonan.loc[:, :].index[-1])
            fig.tight_layout()
            file = base_path_figs / f"{var_obs}_cumulated.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

        for var_obs in ["theta", "weight"]:
            fig, axs = plt.subplots(1, 1, figsize=(6, 2))
            df = pd.DataFrame(index=pd.date_range(start="2010-01-01", end="2017-12-31", freq="D"))
            df1 = dict_obs["lys8"].loc[:, f"{var_obs}"].to_frame()
            df1.columns = ["lys8"]
            df2 = dict_obs["lys3"].loc[:, f"{var_obs}"].to_frame()
            df2.columns = ["lys3"]
            df3 = dict_obs["lys2"].loc[:, f"{var_obs}"].to_frame()
            df3.columns = ["lys2"]
            df = df.join([df1, df2, df3]).dropna()
            axs.plot(
                df.index,
                df.loc[:, "lys8"],
                "-",
                color="#d7b5d8",
                lw=0.7,
                label=f"{fert_lys['lys8']}",
            )
            axs.plot(
                df.index,
                df.loc[:, "lys3"],
                "-",
                color="#df65b0",
                lw=1,
                label=f"{fert_lys['lys3']}",
            )
            axs.plot(
                df.index,
                df.loc[:, "lys2"],
                "-",
                color="#ce1256",
                lw=1.3,
                label=f"{fert_lys['lys2']}",
            )

            axs.tick_params(axis="x", rotation=45)
            axs.set_ylabel(f"{_y_labels_daily[var_obs]}")
            axs.set_xlabel("Time [year]")
            axs.legend(frameon=False, loc="lower left", fontsize=8, ncol=3)
            axs.set_xlim(df_lys_obs_nonan.loc[:, :].index[0], df_lys_obs_nonan.loc[:, :].index[-1])
            fig.tight_layout()
            file = base_path_figs / f"{var_obs}.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

        for var_obs in ["prec", "pet"]:
            years = onp.arange(2010, 2018).tolist()
            fig, axes = plt.subplots(4, 2, figsize=(6, 6), sharey=True)
            axs = axes.flatten()
            for i, year in enumerate(years):
                axs[i].plot(
                    dict_obs["lys8"].loc[f"{year}", :].index,
                    dict_obs["lys8"].loc[f"{year}", f"{var_obs}"].cumsum(),
                    "-",
                    color="#d7b5d8",
                    lw=0.7,
                    label=f"{fert_lys['lys8']}",
                )
                axs[i].plot(
                    dict_obs["lys3"].loc[f"{year}", :].index,
                    dict_obs["lys3"].loc[f"{year}", f"{var_obs}"].cumsum(),
                    "-",
                    color="#df65b0",
                    lw=1,
                    label=f"{fert_lys['lys3']}",
                )
                axs[i].plot(
                    dict_obs["lys2"].loc[f"{year}", :].index,
                    dict_obs["lys2"].loc[f"{year}", f"{var_obs}"].cumsum(),
                    "-",
                    color="#ce1256",
                    lw=1.3,
                    label=f"{fert_lys['lys2']}",
                )
                axs[i].set_xlim(
                    df_lys_obs_nonan.loc[f"{year}", :].index[0],
                    df_lys_obs_nonan.loc[f"{year}", :].index[-1],
                )
                axs[i].tick_params(axis="x", labelrotation=90)
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%m"))
                axs[i].set_title(f"{year}: {crops_lys['lys8'][year]} ")

            axes[0, 0].set_ylabel(f"{_y_labels_cumulated[var_obs]}")
            axes[1, 0].set_ylabel(f"{_y_labels_cumulated[var_obs]}")
            axes[2, 0].set_ylabel(f"{_y_labels_cumulated[var_obs]}")
            axes[3, 0].set_ylabel(f"{_y_labels_cumulated[var_obs]}")
            axes[-1, -1].set_xlabel("Time [month]")
            axes[-1, 0].set_xlabel("Time [month]")
            lines, labels = axes[-1, 1].get_legend_handles_labels()
            fig.legend(lines, labels, loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.1, 0.96))
            fig.tight_layout()
            file = base_path_figs / f"{var_obs}_cumulated_annually.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

        for var_obs in ["perc"]:
            years = onp.arange(2010, 2018).tolist()
            df = pd.DataFrame(index=pd.date_range(start="2010-01-01", end="2017-12-31", freq="D"))
            df1 = dict_obs["lys8"].loc[:, f"{var_obs}"].to_frame()
            df1.columns = ["lys8"]
            df2 = dict_obs["lys3"].loc[:, f"{var_obs}"].to_frame()
            df2.columns = ["lys3"]
            df3 = dict_obs["lys2"].loc[:, f"{var_obs}"].to_frame()
            df3.columns = ["lys2"]
            df = df.join([df1, df2, df3]).dropna()
            fig, axes = plt.subplots(4, 2, figsize=(6, 6), sharey=True)
            axs = axes.flatten()
            for i, year in enumerate(years):
                axs[i].plot(
                    df.loc[f"{year}", :].index,
                    df.loc[f"{year}", "lys8"].cumsum(),
                    "-",
                    color="#d7b5d8",
                    lw=0.7,
                    label=f"{fert_lys['lys8']}",
                )
                axs[i].plot(
                    df.loc[f"{year}", :].index,
                    df.loc[f"{year}", "lys3"].cumsum(),
                    "-",
                    color="#df65b0",
                    lw=1,
                    label=f"{fert_lys['lys3']}",
                )
                axs[i].plot(
                    df.loc[f"{year}", :].index,
                    df.loc[f"{year}", "lys2"].cumsum(),
                    "-",
                    color="#ce1256",
                    lw=1.3,
                    label=f"{fert_lys['lys2']}",
                )
                axs[i].set_xlim(
                    df_lys_obs_nonan.loc[f"{year}", :].index[0],
                    df_lys_obs_nonan.loc[f"{year}", :].index[-1],
                )
                axs[i].tick_params(axis="x", labelrotation=90)
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%m"))
                axs[i].set_title(f"{year}: {crops_lys[lys_experiment][year]} ")

            axes[0, 0].set_ylabel(f"{_y_labels_cumulated[var_obs]}")
            axes[1, 0].set_ylabel(f"{_y_labels_cumulated[var_obs]}")
            axes[2, 0].set_ylabel(f"{_y_labels_cumulated[var_obs]}")
            axes[3, 0].set_ylabel(f"{_y_labels_cumulated[var_obs]}")
            axes[-1, -1].set_xlabel("Time [month]")
            axes[-1, 0].set_xlabel("Time [month]")
            lines, labels = axes[-1, 1].get_legend_handles_labels()
            fig.legend(lines, labels, loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.1, 0.96))
            fig.tight_layout()
            file = base_path_figs / f"{var_obs}_cumulated_annually.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

        for var_obs in ["theta", "weight"]:
            years = onp.arange(2010, 2018).tolist()
            df = pd.DataFrame(index=pd.date_range(start="2010-01-01", end="2017-12-31", freq="D"))
            df1 = dict_obs["lys8"].loc[:, f"{var_obs}"].to_frame()
            df1.columns = ["lys8"]
            df2 = dict_obs["lys3"].loc[:, f"{var_obs}"].to_frame()
            df2.columns = ["lys3"]
            df3 = dict_obs["lys2"].loc[:, f"{var_obs}"].to_frame()
            df3.columns = ["lys2"]
            df = df.join([df1, df2, df3]).dropna()
            fig, axes = plt.subplots(4, 2, figsize=(6, 6), sharey=True)
            axs = axes.flatten()
            for i, year in enumerate(years):
                axs[i].plot(
                    df.loc[f"{year}", :].index,
                    df.loc[f"{year}", "lys8"],
                    "-",
                    color="#d7b5d8",
                    lw=0.7,
                    label=f"{fert_lys['lys8']}",
                )
                axs[i].plot(
                    df.loc[f"{year}", :].index,
                    df.loc[f"{year}", "lys3"],
                    "-",
                    color="#df65b0",
                    lw=1,
                    label=f"{fert_lys['lys3']}",
                )
                axs[i].plot(
                    df.loc[f"{year}", :].index,
                    df.loc[f"{year}", "lys2"],
                    "-",
                    color="#ce1256",
                    lw=1.3,
                    label=f"{fert_lys['lys2']}",
                )
                axs[i].set_xlim(
                    df_lys_obs_nonan.loc[f"{year}", :].index[0],
                    df_lys_obs_nonan.loc[f"{year}", :].index[-1],
                )
                axs[i].tick_params(axis="x", labelrotation=90)
                axs[i].xaxis.set_major_formatter(mdates.DateFormatter("%m"))
                axs[i].set_title(f"{year}: {crops_lys[lys_experiment][year]} ")

            axes[0, 0].set_ylabel(f"{_y_labels_daily[var_obs]}")
            axes[1, 0].set_ylabel(f"{_y_labels_daily[var_obs]}")
            axes[2, 0].set_ylabel(f"{_y_labels_daily[var_obs]}")
            axes[3, 0].set_ylabel(f"{_y_labels_daily[var_obs]}")
            axes[-1, -1].set_xlabel("Time [month]")
            axes[-1, 0].set_xlabel("Time [month]")
            lines, labels = axes[-1, 1].get_legend_handles_labels()
            fig.legend(lines, labels, loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.1, 0.96))
            fig.tight_layout()
            file = base_path_figs / f"{var_obs}_annually.png"
            fig.savefig(file, dpi=300)
            plt.close(fig=fig)

    return


if __name__ == "__main__":
    main()
