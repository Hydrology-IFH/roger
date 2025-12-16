import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import datetime
import matplotlib as mpl
import seaborn as sns
import pickle

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

_lab = {
    "q_ss": "PERC",
    "transp": "TRANSP",
    "evap_soil": "$EVAP_{soil}$",
    "theta": r"$\theta$",
    "tt10_transp": "$TT_{10-TRANSP}$",
    "tt50_transp": "$TT_{50-TRANSP}$",
    "tt90_transp": "$TT_{90-TRANSP}$",
    "tt10_q_ss": "$TT_{10-PERC}$",
    "tt50_q_ss": "$TT_{50-PERC}$",
    "tt90_q_ss": "$TT_{90-PERC}$",
    "rt10_s": "$RT_{10}$",
    "rt50_s": "$RT_{50}$",
    "rt90_s": "$RT_{90}$",
    "M_transp": "$M_{TRANSP}$",
    "M_q_ss": "$M_{PERC}$",
    "dAvg": r"$\overline{\Delta}$",
    "dIPR": r"$\Delta IPR$",
    "dSum": r"$\Delta\sum$",
    "CCCma-CanESM2_CCLM4-8-17": "CCC",
    "MPI-M-MPI-ESM-LR_RCA4": "MPI",
    "nf": "NF",
    "ff": "FF",
    "RMSE": "Root mean squared error",
    "MAE": "Mean absolute error",
    "MEAE": "Median absolute error",
    "AIC": "AIC",
    "AICC": "AICc",
    "theta_pwp:theta_ufc": r"$\theta_{pwp}$ x $\theta_{ufc}$",
    "theta_pwp:theta_ac": r"$\theta_{pwp}$ x $\theta_{ac}$",
    "theta_ufc:theta_ac": r"$\theta_{ufc}$ x $\theta_{ac}$",
    "theta_pwp:theta_ufc:theta_ac": r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$",
}

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

# Load delta changes of precipitation and air temperature
with open(base_path_figs / "delta_changes_climate.pkl", "rb") as handle:
    deltas_climate = pickle.load(handle)

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
                        nn = len(pd.DataFrame(index=date).loc[:"2004", :].index)
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
                    nn = len(pd.DataFrame(index=date).loc[:"2004", :].index)
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

# calculate mean and percentiles
if not os.path.exists(base_path_figs / "delta_changes.pkl"):
    dict_statistics = {}
    dict_deltas = {}
    vars_sim = ["transp", "q_ss"]
    for location in locations:
        dict_statistics[location] = {}
        for land_cover_scenario in land_cover_scenarios:
            dict_statistics[location][land_cover_scenario] = {}
            for climate_scenario in climate_scenarios:
                dict_statistics[location][land_cover_scenario][climate_scenario] = {}
                for period in periods:
                    dict_statistics[location][land_cover_scenario][climate_scenario][period] = {}
                    ds = dict_fluxes_states[location][land_cover_scenario][period][climate_scenario]
                    for var_sim in vars_sim:
                        dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = {}
                        sim_vals = ds[var_sim].isel(y=0).values
                        df = pd.DataFrame(
                            index=range(sim_vals.shape[0]), columns=["Avg", "Sum", "p10", "p50", "p90", "IPR"]
                        )
                        df.loc[:, "Avg"] = onp.nanmean(sim_vals, axis=-1)
                        df.loc[:, "Sum"] = onp.nansum(sim_vals, axis=-1)
                        # df.loc[:, 'p10'] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.1, axis=-1)
                        # df.loc[:, 'p50'] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.5, axis=-1)
                        # df.loc[:, 'p90'] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.9, axis=-1)
                        df.loc[:, "p10"] = onp.quantile(sim_vals, 0.1, axis=-1)
                        df.loc[:, "p50"] = onp.quantile(sim_vals, 0.5, axis=-1)
                        df.loc[:, "p90"] = onp.quantile(sim_vals, 0.9, axis=-1)
                        df.loc[:, "IPR"] = df.loc[:, "p90"] - df.loc[:, "p10"]
                        dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = df

    vars_sim = ["theta"]
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            for climate_scenario in climate_scenarios:
                for period in periods:
                    ds = dict_fluxes_states[location][land_cover_scenario][period][climate_scenario]
                    for var_sim in vars_sim:
                        dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = {}
                        sim_vals = ds[var_sim].isel(y=0).values
                        df = pd.DataFrame(index=range(sim_vals.shape[0]), columns=["Avg", "p10", "p50", "p90", "IPR"])
                        df.loc[:, "Avg"] = onp.nanmean(sim_vals, axis=-1)
                        df.loc[:, "p10"] = onp.quantile(sim_vals, 0.1, axis=-1)
                        df.loc[:, "p50"] = onp.quantile(sim_vals, 0.5, axis=-1)
                        df.loc[:, "p90"] = onp.quantile(sim_vals, 0.9, axis=-1)
                        df.loc[:, "IPR"] = df.loc[:, "p90"] - df.loc[:, "p10"]
                        dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = df

    vars_sim = [
        "rt10_s",
        "rt50_s",
        "rt90_s",
        "tt10_transp",
        "tt50_transp",
        "tt90_transp",
        "tt10_q_ss",
        "tt50_q_ss",
        "tt90_q_ss",
    ]
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            for climate_scenario in climate_scenarios:
                for period in periods:
                    ds = dict_conc_ages[location][land_cover_scenario][period][climate_scenario]
                    for var_sim in vars_sim:
                        dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = {}
                        sim_vals = ds[var_sim].isel(y=0).values
                        df = pd.DataFrame(index=range(sim_vals.shape[0]), columns=["Avg", "p10", "p50", "p90", "IPR"])
                        df.loc[:, "Avg"] = onp.nanmean(sim_vals, axis=-1)
                        df.loc[:, "p10"] = onp.nanquantile(sim_vals, 0.1, axis=-1)
                        df.loc[:, "p50"] = onp.nanquantile(sim_vals, 0.5, axis=-1)
                        df.loc[:, "p90"] = onp.nanquantile(sim_vals, 0.9, axis=-1)
                        df.loc[:, "IPR"] = df.loc[:, "p90"] - df.loc[:, "p10"]
                        dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = df

    vars_sim = ["M_q_ss", "M_transp"]
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            for climate_scenario in climate_scenarios:
                for period in periods:
                    ds = dict_conc_ages[location][land_cover_scenario][period][climate_scenario]
                    for var_sim in vars_sim:
                        dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = {}
                        sim_vals = ds[var_sim].isel(y=0).values
                        df = pd.DataFrame(
                            index=range(sim_vals.shape[0]), columns=["Avg", "Sum", "p10", "p50", "p90", "IPR"]
                        )
                        df.loc[:, "Avg"] = onp.nanmean(sim_vals, axis=-1)
                        df.loc[:, "Sum"] = onp.nansum(sim_vals, axis=-1)
                        df.loc[:, "p10"] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.1, axis=-1)
                        df.loc[:, "p50"] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.5, axis=-1)
                        df.loc[:, "p90"] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.9, axis=-1)
                        df.loc[:, "IPR"] = df.loc[:, "p90"] - df.loc[:, "p10"]
                        dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = df

    # calculate deltas of mean and interpercentile range for near future and far future
    vars_sim = ["transp", "q_ss", "M_q_ss", "M_transp"]
    for location in locations:
        dict_deltas[location] = {}
        for land_cover_scenario in land_cover_scenarios:
            dict_deltas[location][land_cover_scenario] = {}
            for climate_scenario in climate_scenarios:
                dict_deltas[location][land_cover_scenario][climate_scenario] = {}
                for var_sim in vars_sim:
                    dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = {}
                    df_statistics_ref = dict_statistics[location][land_cover_scenario][climate_scenario]["1985-2014"][
                        var_sim
                    ]
                    df_statistics_nf = dict_statistics[location][land_cover_scenario][climate_scenario]["2030-2059"][
                        var_sim
                    ]
                    df_statistics_ff = dict_statistics[location][land_cover_scenario][climate_scenario]["2070-2099"][
                        var_sim
                    ]
                    df_deltas = pd.DataFrame(
                        index=df_statistics_ref.index,
                        columns=["dAvg_nf", "dSum_nf", "dIPR_nf", "dAvg_ff", "dSum_ff", "dIPR_ff"],
                    )
                    df_deltas.loc[:, "dAvg_nf"] = (
                        (df_statistics_nf["Avg"].values - df_statistics_ref["Avg"].values)
                        / df_statistics_ref["Avg"].values
                    ) * 100
                    df_deltas.loc[:, "dAvg_ff"] = (
                        (df_statistics_ff["Avg"].values - df_statistics_ref["Avg"].values)
                        / df_statistics_ref["Avg"].values
                    ) * 100
                    df_deltas.loc[:, "dSum_nf"] = (
                        (df_statistics_nf["Sum"].values - df_statistics_ref["Sum"].values)
                        / df_statistics_ref["Sum"].values
                    ) * 100
                    df_deltas.loc[:, "dSum_ff"] = (
                        (df_statistics_ff["Sum"].values - df_statistics_ref["Sum"].values)
                        / df_statistics_ref["Sum"].values
                    ) * 100
                    df_deltas.loc[:, "dIPR_nf"] = (
                        (df_statistics_nf["IPR"].values - df_statistics_ref["IPR"].values)
                        / df_statistics_ref["IPR"].values
                    ) * 100
                    df_deltas.loc[:, "dIPR_ff"] = (
                        (df_statistics_ff["IPR"].values - df_statistics_ref["IPR"].values)
                        / df_statistics_ref["IPR"].values
                    ) * 100
                    dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = df_deltas

    vars_sim = ["theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            for climate_scenario in climate_scenarios:
                for var_sim in vars_sim:
                    dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = {}
                    df_statistics_ref = dict_statistics[location][land_cover_scenario][climate_scenario]["1985-2014"][
                        var_sim
                    ]
                    df_statistics_nf = dict_statistics[location][land_cover_scenario][climate_scenario]["2030-2059"][
                        var_sim
                    ]
                    df_statistics_ff = dict_statistics[location][land_cover_scenario][climate_scenario]["2070-2099"][
                        var_sim
                    ]
                    df_deltas = pd.DataFrame(
                        index=df_statistics_ref.index, columns=["dAvg_nf", "dIPR_nf", "dAvg_ff", "dIPR_ff"]
                    )
                    df_deltas.loc[:, "dAvg_nf"] = (
                        (df_statistics_nf["Avg"].values - df_statistics_ref["Avg"].values)
                        / df_statistics_ref["Avg"].values
                    ) * 100
                    df_deltas.loc[:, "dAvg_ff"] = (
                        (df_statistics_ff["Avg"].values - df_statistics_ref["Avg"].values)
                        / df_statistics_ref["Avg"].values
                    ) * 100
                    df_deltas.loc[:, "dIPR_nf"] = (
                        (df_statistics_nf["IPR"].values - df_statistics_ref["IPR"].values)
                        / df_statistics_ref["IPR"].values
                    ) * 100
                    df_deltas.loc[:, "dIPR_ff"] = (
                        (df_statistics_ff["IPR"].values - df_statistics_ref["IPR"].values)
                        / df_statistics_ref["IPR"].values
                    ) * 100
                    dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = df_deltas

    vars_sim = ["prec", "ta"]
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            for climate_scenario in climate_scenarios:
                for var_sim in vars_sim:
                    dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = {}
                    dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = deltas_climate[location][
                        climate_scenario
                    ][var_sim]

    # Store data (serialize)
    file = base_path_figs / "delta_changes.pkl"
    with open(file, "wb") as handle:
        pickle.dump(dict_deltas, handle, protocol=pickle.HIGHEST_PROTOCOL)

else:
    # Load the data (deserialize)
    with open(base_path_figs / "delta_changes.pkl", "rb") as handle:
        dict_deltas = pickle.load(handle)

vars_sim = ["transp", "q_ss", "M_q_ss", "M_transp"]
deltas = ["dSum", "dIPR"]
for var_sim in vars_sim:
    for delta in deltas:
        for soil_depth in soil_depths:
            cond = _soil_depths[soil_depth]
            fig, axes = plt.subplots(
                len(locations), len(land_cover_scenarios), sharex="col", sharey="row", figsize=(6, 4.5)
            )
            for i, location in enumerate(locations):
                for j, land_cover_scenario in enumerate(land_cover_scenarios):
                    df_deltas_canesm_nf = dict_deltas[location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][
                        var_sim
                    ].loc[:, ["dSum_nf", "dIPR_nf"]]
                    df_deltas_canesm_nf.columns = deltas
                    df_deltas_canesm_nf.loc[:, "Period"] = "NF/Ref"
                    df_deltas_canesm_nf.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"
                    df_deltas_canesm_ff = dict_deltas[location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][
                        var_sim
                    ].loc[:, ["dSum_ff", "dIPR_ff"]]
                    df_deltas_canesm_ff.columns = deltas
                    df_deltas_canesm_ff.loc[:, "Period"] = "FF/Ref"
                    df_deltas_canesm_ff.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"

                    df_deltas_mpiesm_nf = dict_deltas[location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][
                        var_sim
                    ].loc[:, ["dSum_nf", "dIPR_nf"]]
                    df_deltas_mpiesm_nf.columns = deltas
                    df_deltas_mpiesm_nf.loc[:, "Period"] = "NF/Ref"
                    df_deltas_mpiesm_nf.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"
                    df_deltas_mpiesm_ff = dict_deltas[location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][
                        var_sim
                    ].loc[:, ["dSum_ff", "dIPR_ff"]]
                    df_deltas_mpiesm_ff.columns = deltas
                    df_deltas_mpiesm_ff.loc[:, "Period"] = "FF/Ref"
                    df_deltas_mpiesm_ff.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"
                    df_deltas = pd.concat(
                        [
                            df_deltas_canesm_nf.loc[cond, :],
                            df_deltas_canesm_ff.loc[cond, :],
                            df_deltas_mpiesm_nf.loc[cond, :],
                            df_deltas_mpiesm_ff.loc[cond, :],
                        ],
                        ignore_index=True,
                    )
                    df_deltas_long = pd.melt(
                        df_deltas, id_vars=["Period", "Climate model"], value_vars=[delta], ignore_index=False
                    )
                    sns.boxplot(
                        x="Period",
                        y="value",
                        hue="Climate model",
                        palette=["red", "blue"],
                        data=df_deltas_long,
                        ax=axes[i, j],
                        showfliers=False,
                    )
                    axes[i, j].legend([], [], frameon=False)
                    axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
                    axes[i, j].set_ylabel("")
                    axes[i, j].set_xlabel("")
                    axes[i, j].tick_params(axis="x", rotation=33)
                axes[i, 0].set_ylabel(f"{Locations[i]}\n{_lab[delta]} {_lab[var_sim]} [%]")
            fig.tight_layout()
            file = base_path_figs / "distributions" / f"{var_sim}_{delta}_{soil_depth}_boxplot.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp", "M_q_ss", "M_transp"]
deltas = ["dAvg", "dIPR"]
for var_sim in vars_sim:
    for delta in deltas:
        for soil_depth in soil_depths:
            cond = _soil_depths[soil_depth]
            fig, axes = plt.subplots(
                len(locations), len(land_cover_scenarios), sharex="col", sharey=True, figsize=(6, 4.5)
            )
            for i, location in enumerate(locations):
                for j, land_cover_scenario in enumerate(land_cover_scenarios):
                    df_deltas_canesm_nf = dict_deltas[location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][
                        var_sim
                    ].loc[:, ["dAvg_nf", "dIPR_nf"]]
                    df_deltas_canesm_nf.columns = deltas
                    df_deltas_canesm_nf.loc[:, "Period"] = "NF/Ref"
                    df_deltas_canesm_nf.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"
                    df_deltas_canesm_ff = dict_deltas[location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][
                        var_sim
                    ].loc[:, ["dAvg_ff", "dIPR_ff"]]
                    df_deltas_canesm_ff.columns = deltas
                    df_deltas_canesm_ff.loc[:, "Period"] = "FF/Ref"
                    df_deltas_canesm_ff.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"

                    df_deltas_mpiesm_nf = dict_deltas[location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][
                        var_sim
                    ].loc[:, ["dAvg_nf", "dIPR_nf"]]
                    df_deltas_mpiesm_nf.columns = deltas
                    df_deltas_mpiesm_nf.loc[:, "Period"] = "NF/Ref"
                    df_deltas_mpiesm_nf.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"
                    df_deltas_mpiesm_ff = dict_deltas[location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][
                        var_sim
                    ].loc[:, ["dAvg_ff", "dIPR_ff"]]
                    df_deltas_mpiesm_ff.columns = deltas
                    df_deltas_mpiesm_ff.loc[:, "Period"] = "FF/Ref"
                    df_deltas_mpiesm_ff.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"
                    df_deltas = pd.concat(
                        [
                            df_deltas_canesm_nf.loc[cond, :],
                            df_deltas_canesm_ff.loc[cond, :],
                            df_deltas_mpiesm_nf.loc[cond, :],
                            df_deltas_mpiesm_ff.loc[cond, :],
                        ],
                        ignore_index=True,
                    )
                    df_deltas_long = pd.melt(
                        df_deltas, id_vars=["Period", "Climate model"], value_vars=[delta], ignore_index=False
                    )
                    sns.boxplot(
                        x="Period",
                        y="value",
                        hue="Climate model",
                        palette=["red", "blue"],
                        data=df_deltas_long,
                        ax=axes[i, j],
                        showfliers=False,
                    )
                    axes[i, j].legend([], [], frameon=False)
                    axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
                    axes[i, j].set_ylabel("")
                    axes[i, j].set_xlabel("")
                    axes[i, j].tick_params(axis="x", rotation=33)
                axes[i, 0].set_ylabel(f"{Locations[i]}\n{_lab[delta]} {_lab[var_sim]} [%]")
            fig.tight_layout()
            file = base_path_figs / "distributions" / f"{var_sim}_{delta}_{soil_depth}_boxplot.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

# plot distributions of single sites
vars_sim = ["q_ss", "transp"]
for var_sim in vars_sim:
    for climate_scenario in climate_scenarios:
        for x in [0, 226, 451]:
            fig, axes = plt.subplots(
                len(locations), len(land_cover_scenarios), sharex=True, sharey=True, figsize=(6, 4.5)
            )
            for i, location in enumerate(locations):
                for j, land_cover_scenario in enumerate(land_cover_scenarios):
                    ds_ref = dict_fluxes_states[location][land_cover_scenario]["1985-2014"][climate_scenario]
                    ds_nf = dict_fluxes_states[location][land_cover_scenario]["2030-2059"][climate_scenario]
                    ds_ff = dict_fluxes_states[location][land_cover_scenario]["2070-2099"][climate_scenario]
                    sim_vals_ref = ds_ref[var_sim].isel(x=x, y=0).values.flatten()
                    sim_vals_nf = ds_nf[var_sim].isel(x=x, y=0).values.flatten()
                    sim_vals_ff = ds_ff[var_sim].isel(x=x, y=0).values.flatten()
                    sns.kdeplot(data=sim_vals_ref, ax=axes[i, j], fill=False, color="grey", lw=2, clip=(0, 25))
                    sns.kdeplot(
                        data=sim_vals_nf, ax=axes[i, j], fill=False, color="#9e9ac8", lw=1.5, ls="-.", clip=(0, 25)
                    )
                    sns.kdeplot(
                        data=sim_vals_ff, ax=axes[i, j], fill=False, color="#3f007d", lw=1, ls="--", clip=(0, 25)
                    )
                    axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
                    axes[i, j].set_xlim(
                        0,
                    )
                    axes[i, j].set_ylabel("")
                    axes[i, j].set_xlabel("")
                    axes[i, j].tick_params(axis="x", rotation=33)
                    axes[-1, j].set_xlabel(f"{_lab_unit2[var_sim]}")
                axes[i, 0].set_ylabel("Density [-]")
            fig.tight_layout()
            file = base_path_figs / "distributions" / f"{var_sim}_kde_{climate_scenario}_{x}.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

vars_sim = ["theta"]
for var_sim in vars_sim:
    for climate_scenario in climate_scenarios:
        for x in [0, 226, 451]:
            fig, axes = plt.subplots(
                len(locations), len(land_cover_scenarios), sharex=True, sharey=True, figsize=(6, 4.5)
            )
            for i, location in enumerate(locations):
                for j, land_cover_scenario in enumerate(land_cover_scenarios):
                    ds_ref = dict_fluxes_states[location][land_cover_scenario]["1985-2014"][climate_scenario]
                    ds_nf = dict_fluxes_states[location][land_cover_scenario]["2030-2059"][climate_scenario]
                    ds_ff = dict_fluxes_states[location][land_cover_scenario]["2070-2099"][climate_scenario]
                    sim_vals_ref = ds_ref[var_sim].isel(x=x, y=0).values.flatten()
                    sim_vals_nf = ds_nf[var_sim].isel(x=x, y=0).values.flatten()
                    sim_vals_ff = ds_ff[var_sim].isel(x=x, y=0).values.flatten()
                    sns.kdeplot(data=sim_vals_ref, ax=axes[i, j], fill=False, color="grey", lw=2, clip=(0.1, 0.7))
                    sns.kdeplot(
                        data=sim_vals_nf, ax=axes[i, j], fill=False, color="#9e9ac8", lw=1.5, ls="-.", clip=(0.1, 0.7)
                    )
                    sns.kdeplot(
                        data=sim_vals_ff, ax=axes[i, j], fill=False, color="#3f007d", lw=1, ls="--", clip=(0.1, 0.7)
                    )
                    axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
                    axes[i, j].set_ylabel("")
                    axes[i, j].set_xlabel("")
                    axes[i, j].tick_params(axis="x", rotation=33)
                    axes[-1, j].set_xlabel(f"{_lab_unit2[var_sim]}")
                axes[i, 0].set_ylabel("Density [-]")
            fig.tight_layout()
            file = base_path_figs / "distributions" / f"{var_sim}_kde_{climate_scenario}_{x}.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

# plot distributions of land cover scenarios
vars_sim = ["transp", "theta", "q_ss"]
deltas = ["dAvg", "dIPR"]
for delta in deltas:
    for soil_depth in soil_depths:
        cond = _soil_depths[soil_depth]
        fig, axes = plt.subplots(len(vars_sim), len(land_cover_scenarios), sharex="col", sharey=True, figsize=(6, 4.5))
        for i, var_sim in enumerate(vars_sim):
            for j, land_cover_scenario in enumerate(land_cover_scenarios):
                ll_dfs = []
                for location in locations:
                    df_deltas_canesm_nf = dict_deltas[location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][
                        var_sim
                    ].loc[:, ["dAvg_nf", "dIPR_nf"]]
                    df_deltas_canesm_nf.columns = deltas
                    df_deltas_canesm_nf.loc[:, "Period"] = "NF/Ref"
                    df_deltas_canesm_nf.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"
                    df_deltas_canesm_ff = dict_deltas[location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][
                        var_sim
                    ].loc[:, ["dAvg_ff", "dIPR_ff"]]
                    df_deltas_canesm_ff.columns = deltas
                    df_deltas_canesm_ff.loc[:, "Period"] = "FF/Ref"
                    df_deltas_canesm_ff.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"

                    df_deltas_mpiesm_nf = dict_deltas[location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][
                        var_sim
                    ].loc[:, ["dAvg_nf", "dIPR_nf"]]
                    df_deltas_mpiesm_nf.columns = deltas
                    df_deltas_mpiesm_nf.loc[:, "Period"] = "NF/Ref"
                    df_deltas_mpiesm_nf.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"
                    df_deltas_mpiesm_ff = dict_deltas[location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][
                        var_sim
                    ].loc[:, ["dAvg_ff", "dIPR_ff"]]
                    df_deltas_mpiesm_ff.columns = deltas
                    df_deltas_mpiesm_ff.loc[:, "Period"] = "FF/Ref"
                    df_deltas_mpiesm_ff.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"
                    df_deltas = pd.concat(
                        [
                            df_deltas_canesm_nf.loc[cond, :],
                            df_deltas_canesm_ff.loc[cond, :],
                            df_deltas_mpiesm_nf.loc[cond, :],
                            df_deltas_mpiesm_ff.loc[cond, :],
                        ],
                        ignore_index=True,
                    )
                    df_deltas_long = pd.melt(
                        df_deltas, id_vars=["Period", "Climate model"], value_vars=[delta], ignore_index=False
                    )
                    ll_dfs.append(df_deltas_long)
                df_deltas = pd.concat(ll_dfs, ignore_index=True)
                df_deltas.astype({"value": "float64"}).dtypes
                sns.boxplot(
                    x="Period",
                    y="value",
                    hue="Climate model",
                    palette=["red", "blue"],
                    data=df_deltas,
                    ax=axes[i, j],
                    showfliers=False,
                )
                axes[i, j].legend([], [], frameon=False)
                axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
                axes[i, j].set_ylabel("")
                axes[i, j].set_xlabel("")
                axes[i, j].tick_params(axis="x", rotation=33)
            axes[i, 0].set_ylabel(f"{_lab[delta]} {_lab[var_sim]} [%]")
        # axes[-1, -1].legend(frameon=False)
        fig.tight_layout()
        file = base_path_figs / "distributions" / f"transp_theta_perc_{delta}_{soil_depth}_boxplot.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

vars_sim = ["tt50_transp", "rt50_s", "tt50_q_ss"]
deltas = ["dAvg", "dIPR"]
for delta in deltas:
    for soil_depth in soil_depths:
        cond = _soil_depths[soil_depth]
        fig, axes = plt.subplots(len(vars_sim), len(land_cover_scenarios), sharex="col", sharey=True, figsize=(6, 4.5))
        for i, var_sim in enumerate(vars_sim):
            for j, land_cover_scenario in enumerate(land_cover_scenarios):
                ll_dfs = []
                for location in locations:
                    df_deltas_canesm_nf = dict_deltas[location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][
                        var_sim
                    ].loc[:, ["dAvg_nf", "dIPR_nf"]]
                    df_deltas_canesm_nf.columns = deltas
                    df_deltas_canesm_nf.loc[:, "Period"] = "NF/Ref"
                    df_deltas_canesm_nf.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"
                    df_deltas_canesm_ff = dict_deltas[location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][
                        var_sim
                    ].loc[:, ["dAvg_ff", "dIPR_ff"]]
                    df_deltas_canesm_ff.columns = deltas
                    df_deltas_canesm_ff.loc[:, "Period"] = "FF/Ref"
                    df_deltas_canesm_ff.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"

                    df_deltas_mpiesm_nf = dict_deltas[location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][
                        var_sim
                    ].loc[:, ["dAvg_nf", "dIPR_nf"]]
                    df_deltas_mpiesm_nf.columns = deltas
                    df_deltas_mpiesm_nf.loc[:, "Period"] = "NF/Ref"
                    df_deltas_mpiesm_nf.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"
                    df_deltas_mpiesm_ff = dict_deltas[location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][
                        var_sim
                    ].loc[:, ["dAvg_ff", "dIPR_ff"]]
                    df_deltas_mpiesm_ff.columns = deltas
                    df_deltas_mpiesm_ff.loc[:, "Period"] = "FF/Ref"
                    df_deltas_mpiesm_ff.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"
                    df_deltas = pd.concat(
                        [
                            df_deltas_canesm_nf.loc[cond, :],
                            df_deltas_canesm_ff.loc[cond, :],
                            df_deltas_mpiesm_nf.loc[cond, :],
                            df_deltas_mpiesm_ff.loc[cond, :],
                        ],
                        ignore_index=True,
                    )
                    df_deltas_long = pd.melt(
                        df_deltas, id_vars=["Period", "Climate model"], value_vars=[delta], ignore_index=False
                    )
                    ll_dfs.append(df_deltas_long)
                df_deltas = pd.concat(ll_dfs, ignore_index=True)
                df_deltas.astype({"value": "float64"}).dtypes
                sns.boxplot(
                    x="Period",
                    y="value",
                    hue="Climate model",
                    palette=["red", "blue"],
                    data=df_deltas,
                    ax=axes[i, j],
                    showfliers=False,
                )
                axes[i, j].legend([], [], frameon=False)
                axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
                axes[i, j].set_ylabel("")
                axes[i, j].set_xlabel("")
                axes[i, j].tick_params(axis="x", rotation=33)
            axes[i, 0].set_ylabel(f"{_lab[delta]} {_lab[var_sim]} [%]")
        # axes[-1, -1].legend(frameon=False)
        fig.tight_layout()
        file = base_path_figs / "distributions" / f"tt_rt_{delta}_{soil_depth}_boxplot.png"
        fig.savefig(file, dpi=300)
        plt.close("all")
