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
    base_path_output = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")
    # directory of figures
    base_path_figs = base_path.parent / "figures" / "svat_crop_nitrate_monte_carlo_nitrogen"
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

    _lys = {"lys2": "Lys 2 (130% N-fertilized)",
            "lys3": "Lys 3 (100% N-fertilized)",
            "lys4": "Lys 4",
            "lys8": "Lys 8 (70% N-fertilized)",
            "lys9": "Lys 9",
    }

    __lys = {"lys2": "130%",
            "lys3": "100%",
            "lys4": "Lys 4",
            "lys8": "70%",
            "lys9": "Lys 9",
    }

    _y_labels_unit = {"N_fert": r"$N_{fert}$ [kg N/ha]",
                 "N_uptake": r"$N_{up}$ [kg N/ha]",
                 "N_perc": r"$N_{leach}$ [kg N/ha]",
                 "dN": r"$\Delta N$ [kg N/ha]",
                 }
    _y_labels = {"N_fert": r"$N_{fertilisation}$",
                 "N_uptake": r"$N_{uptake}$",
                 "N_perc": r"$N_{leaching}$",
                 "dN": r"$\Delta N$",
                 }
                           
    dict_data = {}
    # load observations (measured data)
    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / "output" / "svat_crop_nitrate_monte_carlo_nitrogen" / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
        df_params_metrics["E_multi"] = df_params_metrics["KGE_NO3_perc_bs_2011-2015"]
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        idx_best = df_params_metrics.loc[:df_params_metrics.index[0], "id"].values[0]
        # load simulation
        sim_file = base_path_output / "output" / "svat_crop_nitrate_monte_carlo_nitrogen" / f"SVATCROPNITRATE_advection-dispersion-power_{lys_experiment}.nc"
        ds_sim = xr.open_dataset(sim_file, engine="h5netcdf")
        # assign date
        days_sim = ds_sim["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim = num2date(
            days_sim,
            units=f"days since {ds_sim['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim = ds_sim.assign_coords(Time=("Time", date_sim))


        # check nitrogen balance of lysimeter
        df_N_fert = pd.DataFrame(index=date_sim)
        df_N_fert.loc[:, "N_fert"] = ds_sim["Nfert_org"].isel(x=idx_best, y=0).values * 0.01  # convert to kg N/ha

        df_N_perc = pd.DataFrame(index=date_sim)
        df_N_perc.loc[:, "N_perc"] = ds_sim["M_q_ss"].isel(x=idx_best, y=0).values * 0.01  # convert to kg N/ha

        df_N_up = pd.DataFrame(index=date_sim)
        # df_N_up.loc[:, "N_up"] = ds_sim["nh4_up"].isel(x=idx_best, y=0).values * 0.01  # convert to kg N/ha
        df_N_up.loc[:, "N_up"] = ds_sim["M_transp"].isel(x=idx_best, y=0).values * 0.01 + ds_sim["nh4_up"].isel(x=idx_best, y=0).values * 0.01  # convert to kg N/ha

        # aggregate to annual values
        df_N_fert_annual = df_N_fert.resample("YE").sum()
        df_N_perc_annual = df_N_perc.resample("YE").sum()
        df_N_up_annual = df_N_up.resample("YE").sum()

        df_lys_sim = pd.DataFrame(index=df_N_fert_annual.index)
        df_lys_sim.loc[:, "N_fert"] = df_N_fert_annual.loc[:, "N_fert"].values
        df_lys_sim.loc[:, "N_uptake"] = df_N_up_annual.loc[:, "N_up"].values
        df_lys_sim.loc[:, "N_perc"] = df_N_perc_annual.loc[:, "N_perc"].values
        df_lys_sim.loc[:, "dN"] = df_lys_sim.loc[:, "N_fert"] - df_lys_sim.loc[:, "N_uptake"] - df_lys_sim.loc[:, "N_perc"]
        df_lys_sim = df_lys_sim.loc["2011":"2015", :]

        ll_df_year = []
        for year in onp.arange(2011, 2016):
            df_lys_sim_year = df_lys_sim.loc[f"{year}", :]
            df_lys_sim_year.loc[:, "year"] = year
            df_lys_sim_year.loc[:, "lys"] = lys_experiment
            ll_df_year.append(df_lys_sim_year)
        df_lys_sim_year = pd.DataFrame(index=["total"], columns=df_lys_sim.columns)
        df_lys_sim_year.loc["total", "N_fert"] = df_lys_sim.loc[:, "N_fert"].mean()
        df_lys_sim_year.loc["total", "N_uptake"] = df_lys_sim.loc[:, "N_uptake"].mean()
        df_lys_sim_year.loc["total", "N_perc"] = df_lys_sim.loc[:, "N_perc"].mean()
        df_lys_sim_year.loc["total", "dN"] = df_lys_sim_year.loc["total", "N_fert"] - df_lys_sim_year.loc["total", "N_uptake"] - df_lys_sim_year.loc["total", "N_perc"]
        df_lys_sim_year.loc[:, "year"] = "total"
        df_lys_sim_year.loc[:, "lys"] = lys_experiment
        ll_df_year.append(df_lys_sim_year)
        df_lys = pd.concat(ll_df_year)
        df_long = pd.melt(df_lys, id_vars=["year", "lys"], value_vars=["N_fert", "N_uptake", "N_perc", "dN"])
        dict_data[lys_experiment] = df_long

    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharey=True)
    lys_experiments = ["lys2", "lys3", "lys8"]
    for i, lys_experiment in enumerate(lys_experiments):
        data = dict_data[lys_experiment]
        g = sns.barplot(
            ax=axs[i],
            data=data,
            x="year",
            y="value",
            hue="variable",
            palette=["#c994c7", "#addd8e", "#2b8cbe", "#feb24c"],
            errorbar=None,
        )
        axs[i].set_ylabel("[kg N/ha]")
        axs[i].set_xlabel("")
        axs[i].set_title(_lys[lys_experiment])
        axs[i].legend_.remove()
    handels, _labels = axs[-1].get_legend_handles_labels()
    labels = [_y_labels[label] for label in _labels]
    axs[-1].legend(handels, labels, loc="upper center", fontsize=8, ncol=4, frameon=False)
    fig.tight_layout()
    file = base_path_figs / "annual_nitrogen_balance.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharey=True)
    lys_experiments = ["lys2", "lys3", "lys8"]
    for i, lys_experiment in enumerate(lys_experiments):
        data = dict_data[lys_experiment]
        data = data.loc[data["year"] != "total", :]
        g = sns.barplot(
            ax=axs[i],
            data=data,
            x="year",
            y="value",
            hue="variable",
            palette=["#c994c7", "#addd8e", "#2b8cbe", "#feb24c"],
            errorbar=None,
        )
        axs[i].set_ylabel("[kg N/ha]")
        axs[i].set_xlabel("")
        axs[i].set_title(_lys[lys_experiment])
        axs[i].legend_.remove()
    handels, _labels = axs[-1].get_legend_handles_labels()
    labels = [_y_labels[label] for label in _labels]
    axs[-1].legend(handels, labels, loc="upper center", fontsize=8, ncol=4, frameon=False)
    fig.tight_layout()
    file = base_path_figs / "annual_nitrogen_balance_.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    ll_data = [dict_data[lys_experiment] for lys_experiment in lys_experiments]
    data = pd.concat(ll_data)

    fig, axs = plt.subplots(1, 1, figsize=(6, 2))
    data_ = data.loc[data["variable"] == "N_perc", :]
    g = sns.barplot(
        ax=axs,
        data=data_,
        x="year",
        y="value",
        hue="lys",
        palette=["#dd1c77", "#c994c7", "#e7e1ef"],
        errorbar=None,
    )
    axs.set_ylabel("[kg N/ha]")
    axs.set_xlabel("")
    handels, _labels = axs.get_legend_handles_labels()
    labels = [__lys[label] for label in _labels]
    axs.legend(handels, labels, loc="upper right", fontsize=8, ncol=3, frameon=False)    
    fig.tight_layout()
    file = base_path_figs / "annual_nitrogen_leaching.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(6, 2))
    data_ = data.loc[data["variable"] == "N_fert", :]
    g = sns.barplot(
        ax=axs,
        data=data_,
        x="year",
        y="value",
        hue="lys",
        palette=["#dd1c77", "#c994c7", "#e7e1ef"],
        errorbar=None,
    )
    axs.set_ylabel("[kg N/ha]")
    axs.set_xlabel("")
    handels, _labels = axs.get_legend_handles_labels()
    labels = [__lys[label] for label in _labels]
    axs.legend(handels, labels, loc="upper right", fontsize=8, ncol=3, frameon=False)
    fig.tight_layout()
    file = base_path_figs / "annual_nitrogen_fertilization.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(6, 2))
    data_ = data.loc[data["variable"] == "N_fert", :]
    g = sns.barplot(
        ax=axs,
        data=data_,
        x="year",
        y="value",
        hue="lys",
        palette=["#dd1c77", "#c994c7", "#e7e1ef"],
        errorbar=None,
    )
    axs.set_ylabel("[kg N/ha]")
    axs.set_xlabel("")
    handels, _labels = axs.get_legend_handles_labels()
    labels = [__lys[label] for label in _labels]    
    axs.legend(handels, labels, loc="upper right", fontsize=8, ncol=3, frameon=False)
    fig.tight_layout()
    file = base_path_figs / "annual_nitrogen_uptake.png"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)
    return


if __name__ == "__main__":
    main()
