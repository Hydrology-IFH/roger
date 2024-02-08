import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import roger.tools.labels as labs
import matplotlib as mpl
import matplotlib.dates as mdates
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


def nanmeanweighted(y, w, axis=None):
    w1 = w / onp.nansum(w, axis=axis)
    w2 = onp.where(onp.isnan(w), 0, w1)
    w3 = onp.where(onp.isnan(y), 0, w2)
    y1 = onp.where(onp.isnan(y), 0, y)
    wavg = onp.sum(y1 * w3, axis=axis) / onp.sum(w3, axis=axis)

    return wavg


base_path = Path(__file__).parent
# directory of results
base_path_output = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output"
if not os.path.exists(base_path_output):
    os.mkdir(base_path_output)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# identifiers for simulations
locations = [
    "freiburg",
]
crop_rotation_scenarios = ["summer-wheat_clover_winter-wheat", "summer-wheat_winter-wheat", 
                            "summer-wheat_winter-wheat_corn", "summer-wheat_winter-wheat_winter-rape", 
                            "winter-wheat_clover", "winter-wheat_clover_corn", "winter-wheat_corn", 
                            "winter-wheat_sugar-beet_corn", "winter-wheat_winter-rape",
                            "winter-wheat_winter-grain-pea_winter-rape", "summer-wheat_winter-wheat_yellow-mustard", 
                            "summer-wheat_winter-wheat_corn_yellow-mustard", "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                            "winter-wheat_corn_yellow-mustard", "winter-wheat_sugar-beet_corn_yellow-mustard",
                            "summer-wheat_winter-wheat_winter-rape_yellow-mustard"]

_lab_unit1 = {
    "q_ss": "PERC [mm/day]",
    "q_hof": "$Q_{HOF}$ [mm/day]",
    "transp": "TRANSP [mm/day]",
    "evap_soil": "$EVAP_{soil}$ [mm/day]",
    "theta": r"$\theta$ [-]",
    "tt10_transp": "$TT_{10-TRANSP}$ [days]",
    "tt50_transp": "$TT_{50-TRANSP}$ [days]",
    "tt90_transp": "$TT_{90-TRANSP}$ [days]",
    "tt10_q_ss": "$TT_{10-PERC}$ [days]",
    "tt50_q_ss": "$TT_{50-PERC}$ [days]",
    "tt90_q_ss": "$TT_{90-PERC}$ [days]",
    "rt10_s": "$RT_{10}$ [days]",
    "rt50_s": "$RT_{50}$ [days]",
    "rt90_s": "$RT_{90}$ [days]",
    "M_transp": "$M_{TRANSP}$ [mg]",
    "M_q_ss": "$M_{PERC}$ [mg]",
    "theta_ac": r"$\theta_{ac}$ [-]",
    "theta_ufc": r"$\theta_{ufc}$ [-]",
    "theta_pwp": r"$\theta_{pwp}$ [-]",
    "ks": "$k_s$ [mm/day]",
    "ground_cover": "GC [-]",
}

_lab_unit2 = {
    "q_ss": "PERC [mm]",
    "q_hof": "$Q_{HOF}$ [mm]",
    "transp": "TRANSP [mm]",
    "evap_soil": "$EVAP_{soil}$ [mm]",
    "ground_cover": "GC [-]",
}

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)

# load simulated fluxes and states
dict_fluxes_states = {}
for location in locations:
    dict_fluxes_states[location] = {}
    for crop_rotation_scenario in crop_rotation_scenarios:
        dict_fluxes_states[location][crop_rotation_scenario] = {}
        output_hm_file = (
            base_path_output
            / "svat_crop"
            / f"SVATCROP_{location}_{crop_rotation_scenario}.nc"
        )
        ds_fluxes_states = xr.open_dataset(output_hm_file, engine="h5netcdf")
        # assign date
        days = ds_fluxes_states["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date = num2date(
            days,
            units=f"days since {ds_fluxes_states['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_fluxes_states = ds_fluxes_states.assign_coords(Time=("Time", date))
        dict_fluxes_states[location][crop_rotation_scenario] = ds_fluxes_states


vars_sim = ["theta", "ground_cover", "q_hof"]
for var_sim in vars_sim:
    fig, axes = plt.subplots(figsize=(6, 1.5))
    for i, location in enumerate(locations):
        for j, crop_rotation_scenario in enumerate(crop_rotation_scenarios):
            fig, axes = plt.subplots(figsize=(6, 2))
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds[var_sim].isel(y=0).values
            sim_vals_avg = onp.nanmean(sim_vals, axis=0)
            sim_vals_5 = onp.nanquantile(sim_vals, 0.05, axis=0)
            sim_vals_50 = onp.nanmedian(sim_vals, axis=0)
            sim_vals_95 = onp.nanquantile(sim_vals, 0.95, axis=0)
            df = pd.DataFrame(
                index= ds["Time"].values,
                columns=["avg", "p5", "p50", "p95"],
                data=onp.stack(
                    [sim_vals_avg, sim_vals_5, sim_vals_50, sim_vals_95], axis=1
                ),
            )
            df.iloc[0, :] = onp.nan
            df.iloc[-1, :] = onp.nan

            axes.plot(df.index, df["avg"], ls="--", color="red", lw=1)
            axes.plot(df.index, df["p50"], ls="-", color="red", lw=1)
            axes.fill_between(
                df.index, df["p5"], df["p95"], color="red", edgecolor=None, alpha=0.2
            )
            axes.set_xlim(df.index[0], df.index[-1])
            axes.set_xlabel("Time [year]")
            axes.set_ylabel("%s" % (labs._Y_LABS_DAILY[var_sim]))
            fig.autofmt_xdate()
            fig.tight_layout()
            file = base_path_figs / f"{var_sim}_{location}_{crop_rotation_scenario}.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

vars_sim = ["transp", "evap_soil", "q_ss"]
for location in locations:
    for crop_rotation_scenario in crop_rotation_scenarios:
        fig, axes = plt.subplots(3, 1, sharex="row", sharey=True, figsize=(6, 4.5))
        for i, var_sim in enumerate(vars_sim):
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals =  ds[var_sim].isel(y=0).values
            sim_vals_avg = onp.nanmean(onp.cumsum(sim_vals, axis=1), axis=0)
            sim_vals_5 = onp.nanquantile(onp.cumsum(sim_vals, axis=1), 0.05, axis=0)
            sim_vals_50 = onp.nanmedian(onp.cumsum(sim_vals, axis=1), axis=0)
            sim_vals_95 = onp.nanquantile(onp.cumsum(sim_vals, axis=1), 0.95, axis=0)
            df = pd.DataFrame(
                index= ds["Time"].values,
                columns=["avg", "p5", "p50", "p95"],
                data=onp.stack(
                    [sim_vals_avg, sim_vals_5, sim_vals_50, sim_vals_95], axis=1
                ),
            )

            axes[i].plot(df.index, df["avg"], ls="--", color="red", lw=1)
            axes[i].plot(df.index, df["p50"], ls="-", color="red", lw=1)
            axes[i].plot(df.index, df["p5"], ls="-", color="red", lw=1, alpha=0.2)
            axes[i].plot(df.index, df["p95"], ls="-", color="red", lw=1, alpha=0.2)
            axes[i].fill_between(
                df.index, df["p5"], df["p95"], color="red", edgecolor=None, alpha=0.2
            )

            axes[i].set_xlim(df.index[0], df.index[-1])
            # axes[i].xaxis.set_major_locator(mpl.dates.YearLocator(5, month=1, day=1))
            # axes[i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
            axes[i].set_ylabel("%s" % (_lab_unit2[var_sim]))
        upper_ylim = axes[-1].get_ylim()[-1]
        axes[-1].set_ylim(0, upper_ylim)
        axes[-1].set_xlabel("Time [year]")
        fig.tight_layout()
        file = base_path_figs / f"et_perc_{location}_{crop_rotation_scenario}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")
