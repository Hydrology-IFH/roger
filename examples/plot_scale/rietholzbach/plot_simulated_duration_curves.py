from pathlib import Path
import os
import scipy as sp
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
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

    # compare duration curve of 18O in percolation
    fig, ax = plt.subplots(1, 4, sharey=True, figsize=(6, 1.8))
    for i, tm_structure in enumerate(tm_structures):
        idx_best = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]["KGE_C_iso_q_ss"].idxmax()
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        states_tm_file = base_path / "svat_oxygen18_monte_carlo" / "output" / f"states_{tms}_monte_carlo.nc"
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
    plt.close("all")
    return


if __name__ == "__main__":
    main()
