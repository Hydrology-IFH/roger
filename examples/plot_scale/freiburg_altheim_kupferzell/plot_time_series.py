import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import datetime
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
base_path_output = Path("/Volumes/LaCie/roger/examples/plot_scale/freiburg_altheim_kupferzell") / "output"
if not os.path.exists(base_path_output):
    os.mkdir(base_path_output)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# identifiers for simulations
locations = ["freiburg", "altheim", "kupferzell"]
Locations = ["Freiburg", "Altheim", "Kupferzell"]
land_cover_scenarios = ["grass", "corn", "corn_catch_crop", "crop_rotation"]
Land_cover_scenarios = ["Grass", "Corn", "Corn & Catch crop", "Crop rotation"]
climate_scenarios = ["CCCma-CanESM2_CCLM4-8-17", "MPI-M-MPI-ESM-LR_RCA4"]
periods = ["1985-2014", "2030-2059", "2070-2099"]
start_dates = [datetime.date(1985, 1, 1), datetime.date(2030, 1, 1), datetime.date(2070, 1, 1)]
end_dates = [datetime.date(2014, 12, 31), datetime.date(2059, 12, 31), datetime.date(2099, 12, 31)]

_lab_unit1 = {
    "q_ss": "PERC [mm/day]",
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
}

_lab_unit2 = {
    "q_ss": "PERC [mm]",
    "transp": "TRANSP [mm]",
    "evap_soil": "$EVAP_{soil}$ [mm]",
}

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
cond_soil_depth_300 = df_params.loc[:, "z_soil"].values == 300
cond_soil_depth_600 = df_params.loc[:, "z_soil"].values == 600
cond_soil_depth_900 = df_params.loc[:, "z_soil"].values == 900
cond_soil_depth = onp.copy(cond_soil_depth_300)
cond_soil_depth[:] = True
soil_depths = ["all", "shallow", "medium", "deep"]
_soil_depths = {
    "all": cond_soil_depth,
    "shallow": cond_soil_depth_300,
    "medium": cond_soil_depth_600,
    "deep": cond_soil_depth_900,
}

# load simulated fluxes and states
dict_fluxes_states = {}
for location in locations:
    dict_fluxes_states[location] = {}
    for land_cover_scenario in land_cover_scenarios:
        dict_fluxes_states[location][land_cover_scenario] = {}
        for period in periods:
            dict_fluxes_states[location][land_cover_scenario][period] = {}
            for climate_scenario in climate_scenarios:
                try:
                    output_hm_file = (
                        base_path_output
                        / "svat"
                        / f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc"
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
                    if period == "1985-2014":
                        nn = len(pd.DataFrame(index=date).loc[:"2014", :].index)
                        dict_fluxes_states[location][land_cover_scenario][period][
                            climate_scenario
                        ] = ds_fluxes_states.sel(Time=date[1:nn])
                    elif period == "2030-2059":
                        nn = len(pd.DataFrame(index=date).loc[:"2059", :].index)
                        dict_fluxes_states[location][land_cover_scenario][period][
                            climate_scenario
                        ] = ds_fluxes_states.sel(Time=date[1:nn])
                    elif period == "2070-2099":
                        nn = len(pd.DataFrame(index=date).loc[:"2099", :].index)
                        dict_fluxes_states[location][land_cover_scenario][period][
                            climate_scenario
                        ] = ds_fluxes_states.sel(Time=date[1:nn])

                except KeyError:
                    print(f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc")

# load simulated tracer concentrations and water ages
dict_conc_ages = {}
for location in locations:
    dict_conc_ages[location] = {}
    for land_cover_scenario in land_cover_scenarios:
        dict_conc_ages[location][land_cover_scenario] = {}
        for period in periods:
            dict_conc_ages[location][land_cover_scenario][period] = {}
            for climate_scenario in climate_scenarios:
                output_tm_file = (
                    base_path_output
                    / "svat_transport"
                    / f"SVATTRANSPORT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc"
                )
                ds_conc_ages = xr.open_dataset(output_tm_file, engine="h5netcdf", decode_times=False)
                # assign date
                date = num2date(
                    ds_conc_ages["Time"].values,
                    units=f"days since {ds_conc_ages['Time'].attrs['time_origin']}",
                    calendar="standard",
                    only_use_cftime_datetimes=False,
                )
                ds_conc_ages = ds_conc_ages.assign_coords(Time=("Time", date))
                if period == "1985-2014":
                    nn = len(pd.DataFrame(index=date).loc[:"2014", :].index)
                    dict_conc_ages[location][land_cover_scenario][period][climate_scenario] = ds_conc_ages.sel(
                        Time=date[1:nn]
                    )
                elif period == "2030-2059":
                    nn = len(pd.DataFrame(index=date).loc[:"2059", :].index)
                    dict_conc_ages[location][land_cover_scenario][period][climate_scenario] = ds_conc_ages.sel(
                        Time=date[1:nn]
                    )
                elif period == "2070-2099":
                    nn = len(pd.DataFrame(index=date).loc[:"2099", :].index)
                    dict_conc_ages[location][land_cover_scenario][period][climate_scenario] = ds_conc_ages.sel(
                        Time=date[1:nn]
                    )

# vars_sim = ["theta"]
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex=True, figsize=(6, 4.5))
#     for i, location in enumerate(locations):
#         for j, land_cover_scenario in enumerate(land_cover_scenarios):
#             dfs_CanESM = []
#             dfs_MPIM = []
#             for period in periods:
#                 ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]["CCCma-CanESM2_CCLM4-8-17"]
#                 sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                 sim_vals_avg_CanESM = onp.nanmean(sim_vals_CanESM, axis=0)
#                 sim_vals_5_CanESM = onp.nanquantile(sim_vals_CanESM, 0.05, axis=0)
#                 sim_vals_50_CanESM = onp.nanmedian(sim_vals_CanESM, axis=0)
#                 sim_vals_95_CanESM = onp.nanquantile(sim_vals_CanESM, 0.95, axis=0)
#                 df_CanESM = pd.DataFrame(
#                     index=ds_CanESM["Time"].values,
#                     columns=["avg", "p5", "p50", "p95"],
#                     data=onp.stack(
#                         [sim_vals_avg_CanESM, sim_vals_5_CanESM, sim_vals_50_CanESM, sim_vals_95_CanESM], axis=1
#                     ),
#                 )
#                 df_CanESM.iloc[0, :] = onp.nan
#                 df_CanESM.iloc[-1, :] = onp.nan
#                 dfs_CanESM.append(df_CanESM)

#                 ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]["MPI-M-MPI-ESM-LR_RCA4"]
#                 sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                 sim_vals_avg_MPIM = onp.nanmean(sim_vals_MPIM, axis=0)
#                 sim_vals_5_MPIM = onp.nanquantile(sim_vals_MPIM, 0.05, axis=0)
#                 sim_vals_50_MPIM = onp.nanmedian(sim_vals_MPIM, axis=0)
#                 sim_vals_95_MPIM = onp.nanquantile(sim_vals_MPIM, 0.95, axis=0)
#                 df_MPIM = pd.DataFrame(
#                     index=ds_MPIM["Time"].values,
#                     columns=["avg", "p5", "p50", "p95"],
#                     data=onp.stack([sim_vals_avg_MPIM, sim_vals_5_MPIM, sim_vals_50_MPIM, sim_vals_95_MPIM], axis=1),
#                 )
#                 df_MPIM.iloc[0, :] = onp.nan
#                 df_MPIM.iloc[-1, :] = onp.nan
#                 dfs_MPIM.append(df_MPIM)

#             df_CanESM = pd.concat(dfs_CanESM)
#             df_MPIM = pd.concat(dfs_MPIM)
#             dates = df_CanESM.index
#             x1 = len(df_CanESM.loc[:"1994", :].index) + 1
#             x2 = len(df_CanESM.loc[:"2049", :].index) + 1
#             x3 = len(df_CanESM.loc[:"2089", :].index) + 1
#             x2040 = len(df_CanESM.loc[:"2039", :].index) + 1
#             x2080 = len(df_CanESM.loc[:"2079", :].index) + 1
#             df_CanESM.index = range(len(df_CanESM.index))
#             df_MPIM.index = range(len(df_MPIM.index))

#             axes[i, j].plot(df_CanESM.index, df_CanESM["avg"], ls="--", color="red", lw=1)
#             axes[i, j].plot(df_CanESM.index, df_CanESM["p50"], ls="-", color="red", lw=1)
#             axes[i, j].fill_between(
#                 df_CanESM.index, df_CanESM["p5"], df_CanESM["p95"], color="red", edgecolor=None, alpha=0.2
#             )

#             axes[i, j].plot(df_MPIM.index, df_MPIM["avg"], ls="--", color="blue", lw=1)
#             axes[i, j].plot(df_MPIM.index, df_MPIM["p50"], ls="-", color="blue", lw=1)
#             axes[i, j].fill_between(
#                 df_MPIM.index, df_MPIM["p5"], df_MPIM["p95"], color="blue", edgecolor=None, alpha=0.2
#             )
#             axes[i, j].axvline(x=x2040, color="grey")
#             axes[i, j].axvline(x=x2080, color="grey")
#             axes[i, j].set_xticks([x1, x2, x3], labels=["1985 - 2014", "2030 - 2059", "2070 - 2099"])
#             axes[i, j].set_xlim(df_CanESM.index[0], df_CanESM.index[-1])
#             axes[-1, j].set_xlabel("Time [year]")
#             axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
#         axes[i, 0].set_ylabel("%s\n%s" % (Locations[i], labs._Y_LABS_DAILY[var_sim]))
#     fig.autofmt_xdate()
#     fig.tight_layout()
#     file = base_path_figs / "time_series" / f"{var_sim}.png"
#     fig.savefig(file, dpi=300)
#     plt.close("all")

#     for location in locations:
#         for land_cover_scenario in land_cover_scenarios:
#             fig, axes = plt.subplots(3, 1, sharex="row", sharey=True, figsize=(6, 4.5))
#             for i, period in enumerate(periods):
#                 ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]["CCCma-CanESM2_CCLM4-8-17"]
#                 sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                 sim_vals_avg_CanESM = onp.nanmean(sim_vals_CanESM, axis=0)
#                 sim_vals_5_CanESM = onp.nanquantile(sim_vals_CanESM, 0.05, axis=0)
#                 sim_vals_50_CanESM = onp.nanmedian(sim_vals_CanESM, axis=0)
#                 sim_vals_95_CanESM = onp.nanquantile(sim_vals_CanESM, 0.95, axis=0)
#                 df_CanESM = pd.DataFrame(
#                     index=ds_CanESM["Time"].values,
#                     columns=["avg", "p5", "p50", "p95"],
#                     data=onp.stack(
#                         [sim_vals_avg_CanESM, sim_vals_5_CanESM, sim_vals_50_CanESM, sim_vals_95_CanESM], axis=1
#                     ),
#                 )

#                 ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]["MPI-M-MPI-ESM-LR_RCA4"]
#                 sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                 sim_vals_avg_MPIM = onp.nanmean(sim_vals_MPIM, axis=0)
#                 sim_vals_5_MPIM = onp.nanquantile(sim_vals_MPIM, 0.05, axis=0)
#                 sim_vals_50_MPIM = onp.nanmedian(sim_vals_MPIM, axis=0)
#                 sim_vals_95_MPIM = onp.nanquantile(sim_vals_MPIM, 0.95, axis=0)
#                 df_MPIM = pd.DataFrame(
#                     index=ds_MPIM["Time"].values,
#                     columns=["avg", "p5", "p50", "p95"],
#                     data=onp.stack([sim_vals_avg_MPIM, sim_vals_5_MPIM, sim_vals_50_MPIM, sim_vals_95_MPIM], axis=1),
#                 )

#                 axes[i].plot(df_CanESM.index, df_CanESM["avg"], ls="--", color="red", lw=1)
#                 axes[i].plot(df_CanESM.index, df_CanESM["p50"], ls="-", color="red", lw=1)
#                 axes[i].fill_between(
#                     df_CanESM.index, df_CanESM["p5"], df_CanESM["p95"], color="red", edgecolor=None, alpha=0.2
#                 )

#                 axes[i].plot(df_MPIM.index, df_MPIM["avg"], ls="--", color="blue", lw=1)
#                 axes[i].plot(df_MPIM.index, df_MPIM["p50"], ls="-", color="blue", lw=1)
#                 axes[i].fill_between(
#                     df_MPIM.index, df_MPIM["p5"], df_MPIM["p95"], color="blue", edgecolor=None, alpha=0.2
#                 )
#                 axes[i].set_xlim(df_CanESM.index[0], df_CanESM.index[-1])
#                 axes[i].xaxis.set_major_locator(mpl.dates.YearLocator(5, month=1, day=1))
#                 axes[i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
#                 axes[i].set_ylabel("%s" % (labs._Y_LABS_DAILY[var_sim]))
#             axes[-1].set_xlabel("Time [year]")
#             fig.tight_layout()
#             file = base_path_figs / "time_series" / f"{var_sim}_{location}_{land_cover_scenario}.png"
#             fig.savefig(file, dpi=300)
#             plt.close("all")

vars_sim = ["rt50_s", "tt50_q_ss", "tt50_transp"]
for var_sim in vars_sim:
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            fig, axes = plt.subplots(3, 1, sharex="row", sharey=True, figsize=(6, 4.5))
            for i, period in enumerate(periods):
                ds_CanESM = dict_conc_ages[location][land_cover_scenario][period]["CCCma-CanESM2_CCLM4-8-17"]
                sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
                sim_vals_avg_CanESM = onp.nanmean(sim_vals_CanESM, axis=0)
                sim_vals_5_CanESM = onp.nanquantile(sim_vals_CanESM, 0.05, axis=0)
                sim_vals_50_CanESM = onp.nanmedian(sim_vals_CanESM, axis=0)
                sim_vals_95_CanESM = onp.nanquantile(sim_vals_CanESM, 0.95, axis=0)
                df_CanESM = pd.DataFrame(
                    index=ds_CanESM["Time"].values,
                    columns=["avg", "p5", "p50", "p95"],
                    data=onp.stack(
                        [sim_vals_avg_CanESM, sim_vals_5_CanESM, sim_vals_50_CanESM, sim_vals_95_CanESM], axis=1
                    ),
                )

                ds_MPIM = dict_conc_ages[location][land_cover_scenario][period]["MPI-M-MPI-ESM-LR_RCA4"]
                sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
                sim_vals_avg_MPIM = onp.nanmean(sim_vals_MPIM, axis=0)
                sim_vals_5_MPIM = onp.nanquantile(sim_vals_MPIM, 0.05, axis=0)
                sim_vals_50_MPIM = onp.nanmedian(sim_vals_MPIM, axis=0)
                sim_vals_95_MPIM = onp.nanquantile(sim_vals_MPIM, 0.95, axis=0)
                df_MPIM = pd.DataFrame(
                    index=ds_MPIM["Time"].values,
                    columns=["avg", "p5", "p50", "p95"],
                    data=onp.stack([sim_vals_avg_MPIM, sim_vals_5_MPIM, sim_vals_50_MPIM, sim_vals_95_MPIM], axis=1),
                )

                axes[i].plot(df_CanESM.index, df_CanESM["avg"], ls="--", color="red", lw=1)
                axes[i].plot(df_CanESM.index, df_CanESM["p50"], ls="-", color="red", lw=1)
                axes[i].fill_between(
                    df_CanESM.index, df_CanESM["p5"], df_CanESM["p95"], color="red", edgecolor=None, alpha=0.2
                )

                axes[i].plot(df_MPIM.index, df_MPIM["avg"], ls="--", color="blue", lw=1)
                axes[i].plot(df_MPIM.index, df_MPIM["p50"], ls="-", color="blue", lw=1)
                axes[i].fill_between(
                    df_MPIM.index, df_MPIM["p5"], df_MPIM["p95"], color="blue", edgecolor=None, alpha=0.2
                )
                axes[i].set_xlim(df_CanESM.index[0], df_CanESM.index[-1])
                axes[i].xaxis.set_major_locator(mpl.dates.YearLocator(5, month=1, day=1))
                axes[i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
                axes[i].set_ylabel("%s" % (_lab_unit1[var_sim]))
                if var_sim == "tt50_q_ss":
                    axes[i].set_ylim(0, 600)
                else:
                    axes[i].set_ylim(
                        0,
                    )
            axes[-1].set_xlabel("Time [year]")
            fig.tight_layout()
            file = base_path_figs / "time_series" / f"{var_sim}_{location}_{land_cover_scenario}.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

vars_sim = ["rt50_s", "tt50_q_ss", "tt50_transp"]
for var_sim in vars_sim:
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            fig, axes = plt.subplots(3, 1, sharex="row", sharey=True, figsize=(6, 4.5))
            for i, period in enumerate(periods):
                ds_CanESM = dict_conc_ages[location][land_cover_scenario][period]["CCCma-CanESM2_CCLM4-8-17"]
                sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
                sim_vals_avg_CanESM = onp.cumsum(onp.nanmean(sim_vals_CanESM, axis=0))
                sim_vals_5_CanESM = onp.cumsum(onp.nanquantile(sim_vals_CanESM, 0.05, axis=0))
                sim_vals_50_CanESM = onp.cumsum(onp.nanmedian(sim_vals_CanESM, axis=0))
                sim_vals_95_CanESM = onp.cumsum(onp.nanquantile(sim_vals_CanESM, 0.95, axis=0))
                df_CanESM = pd.DataFrame(
                    index=ds_CanESM["Time"].values,
                    columns=["avg", "p5", "p50", "p95"],
                    data=onp.stack(
                        [sim_vals_avg_CanESM, sim_vals_5_CanESM, sim_vals_50_CanESM, sim_vals_95_CanESM], axis=1
                    ),
                )

                ds_MPIM = dict_conc_ages[location][land_cover_scenario][period]["MPI-M-MPI-ESM-LR_RCA4"]
                sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
                sim_vals_avg_MPIM = onp.cumsum(onp.nanmean(sim_vals_MPIM, axis=0))
                sim_vals_5_MPIM = onp.cumsum(onp.nanquantile(sim_vals_MPIM, 0.05, axis=0))
                sim_vals_50_MPIM = onp.cumsum(onp.nanmedian(sim_vals_MPIM, axis=0))
                sim_vals_95_MPIM = onp.cumsum(onp.nanquantile(sim_vals_MPIM, 0.95, axis=0))
                df_MPIM = pd.DataFrame(
                    index=ds_MPIM["Time"].values,
                    columns=["avg", "p5", "p50", "p95"],
                    data=onp.stack([sim_vals_avg_MPIM, sim_vals_5_MPIM, sim_vals_50_MPIM, sim_vals_95_MPIM], axis=1),
                )

                axes[i].plot(df_CanESM.index, df_CanESM["avg"], ls="--", color="red", lw=1)
                axes[i].plot(df_CanESM.index, df_CanESM["p50"], ls="-", color="red", lw=1)
                axes[i].plot(df_CanESM.index, df_CanESM["p5"], ls="-", color="red", lw=1, alpha=0.2)
                axes[i].plot(df_CanESM.index, df_CanESM["p95"], ls="-", color="red", lw=1, alpha=0.2)
                axes[i].fill_between(
                    df_CanESM.index, df_CanESM["p5"], df_CanESM["p95"], color="red", edgecolor=None, alpha=0.2
                )

                axes[i].plot(df_MPIM.index, df_MPIM["avg"], ls="--", color="blue", lw=1)
                axes[i].plot(df_MPIM.index, df_MPIM["p50"], ls="-", color="blue", lw=1)
                axes[i].plot(df_MPIM.index, df_MPIM["p5"], ls="-", color="blue", lw=1, alpha=0.2)
                axes[i].plot(df_MPIM.index, df_MPIM["p95"], ls="-", color="blue", lw=1, alpha=0.2)
                axes[i].fill_between(
                    df_MPIM.index, df_MPIM["p5"], df_MPIM["p95"], color="blue", edgecolor=None, alpha=0.2
                )
                axes[i].set_xlim(df_CanESM.index[0], df_CanESM.index[-1])
                axes[i].xaxis.set_major_locator(mpl.dates.YearLocator(5, month=1, day=1))
                axes[i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
                axes[i].set_ylabel("%s" % (_lab_unit1[var_sim]))
            upper_ylim = axes[-1].get_ylim()[-1]
            axes[-1].set_ylim(0, upper_ylim)
            axes[-1].set_xlabel("Time [year]")
            fig.tight_layout()
            file = base_path_figs / "time_series" / f"{var_sim}_{location}_{land_cover_scenario}_cumulated.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

vars_sim = ["transp", "evap_soil", "q_ss"]
for var_sim in vars_sim:
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            fig, axes = plt.subplots(3, 1, sharex="row", sharey=True, figsize=(6, 4.5))
            for i, period in enumerate(periods):
                ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]["CCCma-CanESM2_CCLM4-8-17"]
                sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
                sim_vals_avg_CanESM = onp.nanmean(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
                sim_vals_5_CanESM = onp.nanquantile(onp.cumsum(sim_vals_CanESM, axis=1), 0.05, axis=0)
                sim_vals_50_CanESM = onp.nanmedian(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
                sim_vals_95_CanESM = onp.nanquantile(onp.cumsum(sim_vals_CanESM, axis=1), 0.95, axis=0)
                df_CanESM = pd.DataFrame(
                    index=ds_CanESM["Time"].values,
                    columns=["avg", "p5", "p50", "p95"],
                    data=onp.stack(
                        [sim_vals_avg_CanESM, sim_vals_5_CanESM, sim_vals_50_CanESM, sim_vals_95_CanESM], axis=1
                    ),
                )

                ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]["MPI-M-MPI-ESM-LR_RCA4"]
                sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
                sim_vals_avg_MPIM = onp.nanmean(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
                sim_vals_5_MPIM = onp.nanquantile(onp.cumsum(sim_vals_MPIM, axis=1), 0.05, axis=0)
                sim_vals_50_MPIM = onp.nanmedian(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
                sim_vals_95_MPIM = onp.nanquantile(onp.cumsum(sim_vals_MPIM, axis=1), 0.95, axis=0)
                df_MPIM = pd.DataFrame(
                    index=ds_MPIM["Time"].values,
                    columns=["avg", "p5", "p50", "p95"],
                    data=onp.stack([sim_vals_avg_MPIM, sim_vals_5_MPIM, sim_vals_50_MPIM, sim_vals_95_MPIM], axis=1),
                )

                axes[i].plot(df_CanESM.index, df_CanESM["avg"], ls="--", color="red", lw=1)
                axes[i].plot(df_CanESM.index, df_CanESM["p50"], ls="-", color="red", lw=1)
                axes[i].plot(df_CanESM.index, df_CanESM["p5"], ls="-", color="red", lw=1, alpha=0.2)
                axes[i].plot(df_CanESM.index, df_CanESM["p95"], ls="-", color="red", lw=1, alpha=0.2)
                axes[i].fill_between(
                    df_CanESM.index, df_CanESM["p5"], df_CanESM["p95"], color="red", edgecolor=None, alpha=0.2
                )

                axes[i].plot(df_MPIM.index, df_MPIM["avg"], ls="--", color="blue", lw=1)
                axes[i].plot(df_MPIM.index, df_MPIM["p50"], ls="-", color="blue", lw=1)
                axes[i].plot(df_MPIM.index, df_MPIM["p5"], ls="-", color="blue", lw=1, alpha=0.2)
                axes[i].plot(df_MPIM.index, df_MPIM["p95"], ls="-", color="blue", lw=1, alpha=0.2)
                axes[i].fill_between(
                    df_MPIM.index, df_MPIM["p5"], df_MPIM["p95"], color="blue", edgecolor=None, alpha=0.2
                )
                axes[i].set_xlim(df_CanESM.index[0], df_CanESM.index[-1])
                axes[i].xaxis.set_major_locator(mpl.dates.YearLocator(5, month=1, day=1))
                axes[i].xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
                axes[i].set_ylabel("%s" % (_lab_unit2[var_sim]))
            upper_ylim = axes[-1].get_ylim()[-1]
            axes[-1].set_ylim(0, upper_ylim)
            axes[-1].set_xlabel("Time [year]")
            fig.tight_layout()
            file = base_path_figs / "time_series" / f"{var_sim}_{location}_{land_cover_scenario}.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

# plot cumulated precipitation
# vars_sim = ['prec']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(len(locations), len(periods), sharex='col', figsize=(6, 4.5))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#             sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_CanESM = onp.cumsum(sim_vals_CanESM)

#             ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#             sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_MPIM = onp.cumsum(sim_vals_MPIM)

#             axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
#             axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='blue', lw=1)
#             axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#             axes[i, j].set_ylim(0,)
#             axes[-1, j].set_xlabel('Time [year]')
#         axes[i, 0].set_ylabel('PRECIP [mm]')
#     fig.tight_layout()
#     file = base_path_figs / f"{var_sim}_cumulated.png"
#     fig.savefig(file, dpi=300)
#     plt.close('all')

# vars_sim = ['prec']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(len(locations), len(periods), sharex='col', figsize=(6, 4.5))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#             sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_CanESM = onp.cumsum(sim_vals_CanESM)

#             ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#             sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_MPIM = onp.cumsum(sim_vals_MPIM)

#             axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='-', color='red', lw=1)
#             axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='-', color='blue', lw=1)
#             axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#             axes[i, j].set_ylim(0,)
#             axes[-1, j].set_xlabel('Time [year]')
#         axes[i, 0].set_ylabel('%s\nPRECIP [mm]' % (Locations[i]))
#     fig.tight_layout()
#     file = base_path_figs / f"{var_sim}_cumulated.png"
#     fig.savefig(file, dpi=300)
#     plt.close('all')
