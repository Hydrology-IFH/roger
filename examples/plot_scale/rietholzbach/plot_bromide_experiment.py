from pathlib import Path
import os
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.cm as cm
import yaml
import click
import copy
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
    # load data from bromide experiment
    path_obs_br = Path(__file__).parent / "observations" / "bromide_breakthrough.csv"
    df_obs_br = pd.read_csv(path_obs_br, skiprows=1, sep=";", na_values="")

    # plot virtual bromide exoeriment
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
                        base_path / "svat_bromide_benchmark" / "output" / f"states_{tms}_bromide_benchmark_stgallen.nc"
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
                states_br_file = base_path / "svat_bromide_benchmark" / "output" / f"states_{tms}_bromide_benchmark.nc"
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

    # plot soil bromide concentrations simulated with HYDRUS-1D
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

    # plot soil bromide mass simulated with HYDRUS-1D
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
    plt.close("all")
    return


if __name__ == "__main__":
    main()
