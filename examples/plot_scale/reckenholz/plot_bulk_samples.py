import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import matplotlib as mpl

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


# directory of figures
base_path_figs = Path(__file__).parent / "figures"
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

fert_lys2_lys3_lys8 = {"lys2": "130% N-fertilized",
                        "lys3": "100% N-fertilized",
                        "lys8": "70% N-fertilized"
}



_Y_LABS_DAILY = {
    "C_q_ss_bs": r"${NO_3^-N}$ [mg/l]",
    "M_q_ss_bs": r"${NO_3^-N}$ [mg]",
    "q_ss_bs": r"PERC [mm]",
}

fig, axs = plt.subplots(3, 2, sharey='col', sharex=True, figsize=(6, 5))
lys_experiments = ["lys2", "lys3", "lys8"]
for i, lys_experiment in enumerate(lys_experiments):
    # load observations (measured data)
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
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    if lys_experiment == "lys2":
        col = "#dd1c77"
    elif lys_experiment == "lys3":
        col = "#c994c7"
    elif lys_experiment == "lys8":
        col = "#e7e1ef"

    obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    axs[i, 0].scatter(df_obs.index, df_obs["obs"], color=col, s=5, zorder=1)
    axs[i, 0].set_ylabel('PERC [mm/14 days]')
    axs[i, 0].set_ylim(0, 140)

    obs_vals = ds_obs["NO3_PERC"].isel(x=0, y=0).values * 4.43 # convert nitrate-nitrogen to nitrate
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    axs[i, 1].scatter(df_obs.index, df_obs["obs"], color=col, s=5, zorder=1)
    axs[i, 1].set_ylabel('$NO_3$ [mg/l]')
    axs[i, 1].set_ylim(0, 60)
    # rotate tick labels
    axs[i, 0].tick_params(axis='x', rotation=33)
    axs[i, 1].tick_params(axis='x', rotation=33)

axs[-1, 0].set_xlabel('Time [year]')
axs[-1, 1].set_xlabel('Time [year]')
fig.tight_layout()
file_str = "bulk_samples_PERC_NO3_conc.png"
path_fig = base_path_figs / file_str
fig.savefig(path_fig, dpi=300)
plt.close("all")

fig, axs = plt.subplots(1, 2, sharey='col', sharex=True, figsize=(6, 2))
lys_experiments = ["lys2", "lys3", "lys8"]
for i, lys_experiment in enumerate(lys_experiments):
    # load observations (measured data)
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
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    df_obs = df_obs.dropna()
    if lys_experiment == "lys2":
        col = "#dd1c77"
    elif lys_experiment == "lys3":
        col = "#c994c7"
    elif lys_experiment == "lys8":
        col = "#e7e1ef"
    axs[0].plot(df_obs.index, df_obs["obs"], color=col, marker='.')
    axs[0].set_ylabel('PERC [mm/14 days]')
    axs[0].set_ylim(0, 140)

    obs_vals = ds_obs["NO3_PERC"].isel(x=0, y=0).values * 4.43 # convert nitrate-nitrogen to nitrate
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    df_obs = df_obs.dropna()
    axs[1].plot(df_obs.index, df_obs["obs"], color=col, marker='.')
    axs[1].set_ylabel('$NO_3$ [mg/l]')
    axs[1].set_ylim(0, 60)
    # rotate tick labels
    axs[0].tick_params(axis='x', rotation=33)
    axs[1].tick_params(axis='x', rotation=33)

axs[0].set_xlabel('Time [year]')
axs[1].set_xlabel('Time [year]')
fig.tight_layout()
file_str = "bulk_samples_PERC_NO3_conc_3.png"
path_fig = base_path_figs / file_str
fig.savefig(path_fig, dpi=300)
plt.close("all")

fig, axs = plt.subplots(1, 2, sharey='col', sharex=True, figsize=(6, 2))
lys_experiments = ["lys8"]
for i, lys_experiment in enumerate(lys_experiments):
    # load observations (measured data)
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
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    df_obs = df_obs.dropna()
    if lys_experiment == "lys2":
        col = "#dd1c77"
    elif lys_experiment == "lys3":
        col = "#c994c7"
    elif lys_experiment == "lys8":
        col = "#e7e1ef"
    axs[0].plot(df_obs.index, df_obs["obs"], color=col, marker='.')
    axs[0].set_ylabel('PERC [mm/14 days]')
    axs[0].set_ylim(0, 140)

    obs_vals = ds_obs["NO3_PERC"].isel(x=0, y=0).values * 4.43 # convert nitrate-nitrogen to nitrate
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    df_obs = df_obs.dropna()
    axs[1].plot(df_obs.index, df_obs["obs"], color=col, marker='.')
    axs[1].set_ylabel('$NO_3$ [mg/l]')
    axs[1].set_ylim(0, 60)
    # rotate tick labels
    axs[0].tick_params(axis='x', rotation=33)
    axs[1].tick_params(axis='x', rotation=33)

axs[0].set_xlabel('Time [year]')
axs[1].set_xlabel('Time [year]')
fig.tight_layout()
file_str = "bulk_samples_PERC_NO3_conc_1.png"
path_fig = base_path_figs / file_str
fig.savefig(path_fig, dpi=300)
plt.close("all")

fig, axs = plt.subplots(1, 2, sharey='col', sharex=True, figsize=(6, 2))
lys_experiments = ["lys8", "lys3"]
for i, lys_experiment in enumerate(lys_experiments):
    # load observations (measured data)
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
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    df_obs = df_obs.dropna()
    if lys_experiment == "lys2":
        col = "#dd1c77"
    elif lys_experiment == "lys3":
        col = "#c994c7"
    elif lys_experiment == "lys8":
        col = "#e7e1ef"
    axs[0].plot(df_obs.index, df_obs["obs"], color=col, marker='.')
    axs[0].set_ylabel('PERC [mm/14 days]')
    axs[0].set_ylim(0, 140)

    obs_vals = ds_obs["NO3_PERC"].isel(x=0, y=0).values * 4.43 # convert nitrate-nitrogen to nitrate
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    df_obs = df_obs.dropna()
    axs[1].plot(df_obs.index, df_obs["obs"], color=col, marker='.')
    axs[1].set_ylabel('$NO_3$ [mg/l]')
    axs[1].set_ylim(0, 60)
    # rotate tick labels
    axs[0].tick_params(axis='x', rotation=33)
    axs[1].tick_params(axis='x', rotation=33)

axs[0].set_xlabel('Time [year]')
axs[1].set_xlabel('Time [year]')
fig.tight_layout()
file_str = "bulk_samples_PERC_NO3_conc_2.png"
path_fig = base_path_figs / file_str
fig.savefig(path_fig, dpi=300)
plt.close("all")




fig, axs = plt.subplots(3, 2, sharey='col', sharex=True, figsize=(6, 5))
lys_experiments = ["lys2", "lys3", "lys8"]
for i, lys_experiment in enumerate(lys_experiments):
    # load observations (measured data)
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
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    axs[i, 0].scatter(df_obs.index, df_obs["obs"], color="black", s=5, zorder=1)
    axs[i, 0].set_ylabel('PERC [mm/14 days]')
    axs[i, 0].set_ylim(0, )

    obs_vals = ds_obs["NO3_PERC_MASS"].isel(x=0, y=0).values * 4.43 # convert nitrate-nitrogen to nitrate
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    axs[i, 1].scatter(df_obs.index, df_obs["obs"], color="black", s=5, zorder=1)
    axs[i, 1].set_ylabel('$NO_3$ [mg]')
    axs[i, 1].set_ylim(0, )

axs[-1, 0].set_xlabel('Time [year]')
axs[-1, 1].set_xlabel('Time [year]')
fig.tight_layout()
file_str = "bulk_samples_PERC_NO3_load.png"
path_fig = base_path_figs / file_str
fig.savefig(path_fig, dpi=300)
plt.close("all")

fig, axs = plt.subplots(1, 2, sharey='col', sharex=True, figsize=(6, 2))
lys_experiments = ["lys2", "lys3", "lys8"]
for i, lys_experiment in enumerate(lys_experiments):
    # load observations (measured data)
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
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    df_obs = df_obs.dropna()
    if lys_experiment == "lys2":
        col = "#dd1c77"
    elif lys_experiment == "lys3":
        col = "#c994c7"
    elif lys_experiment == "lys8":
        col = "#e7e1ef"
    axs[0].plot(df_obs.index, df_obs["obs"], color=col, marker='.')
    axs[0].set_ylabel('PERC [mm/14 days]')
    axs[0].set_ylim(0, 140)

    obs_vals = ds_obs["NO3_PERC_MASS"].isel(x=0, y=0).values * 4.43 # convert nitrate-nitrogen to nitrate
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    df_obs = df_obs.dropna()
    axs[1].plot(df_obs.index, df_obs["obs"], color=col, marker='.')
    axs[1].set_ylabel('$NO_3$ [mg]')
    axs[1].set_ylim(0, 60)
    # rotate tick labels
    axs[0].tick_params(axis='x', rotation=33)
    axs[1].tick_params(axis='x', rotation=33)

axs[0].set_xlabel('Time [year]')
axs[1].set_xlabel('Time [year]')
fig.tight_layout()
file_str = "bulk_samples_PERC_NO3_load_3.png"
path_fig = base_path_figs / file_str
fig.savefig(path_fig, dpi=300)
plt.close("all")



fig, axs = plt.subplots(3, 2, sharey='col', sharex=True, figsize=(6, 5))
lys_experiments = ["lys2", "lys3", "lys8"]
for i, lys_experiment in enumerate(lys_experiments):
    # load observations (measured data)
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
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    obs_vals = ds_obs["PERC_bs"].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    axs[i, 0].scatter(df_obs.index, df_obs["obs"].cumsum(), color="black", s=5, zorder=1)
    axs[i, 0].set_ylabel('PERC [mm]')
    axs[i, 0].set_ylim(0, )

    obs_vals = ds_obs["NO3_PERC_MASS"].isel(x=0, y=0).values * 4.43 # convert nitrate-nitrogen to nitrate
    df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
    df_obs.loc[:, "obs"] = obs_vals
    df_obs = df_obs.loc['2011':'2015']
    axs[i, 1].scatter(df_obs.index, df_obs["obs"].cumsum(), color="black", s=5, zorder=1)
    axs[i, 1].set_ylabel('$NO_3$ [mg]')
    axs[i, 1].set_ylim(0, )

axs[-1, 0].set_xlabel('Time [year]')
axs[-1, 1].set_xlabel('Time [year]')
fig.tight_layout()
file_str = "bulk_samples_PERC_NO3_load_cumulated.png"
path_fig = base_path_figs / file_str
fig.savefig(path_fig, dpi=300)
plt.close("all")