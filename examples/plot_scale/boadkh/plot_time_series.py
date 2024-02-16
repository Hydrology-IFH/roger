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
locations = ["freiburg", "lahr", "muellheim", 
             "stockach", "gottmadingen", "weingarten",
             "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
             "ehingen-kirchen", "merklingen", "hayingen",
             "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
locations = [
    "freiburg",
]

crop_rotation_scenarios = ["winter-wheat_clover",
                           "winter-wheat_silage-corn",
                           "summer-wheat_winter-wheat",
                           "summer-wheat_clover_winter-wheat",
                           "winter-wheat_clover_silage-corn",
                           "winter-wheat_sugar-beet_silage-corn",
                           "summer-wheat_winter-wheat_silage-corn",
                           "summer-wheat_winter-wheat_winter-rape",
                           "winter-wheat_winter-rape",
                           "winter-wheat_soybean_winter-rape",
                           "sugar-beet_winter-wheat_winter-barley", 
                           "grain-corn_winter-wheat_winter-rape", 
                           "grain-corn_winter-wheat_winter-barley",
                           "grain-corn_winter-wheat_clover",
                           "winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_yellow-mustard",
                           "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                           "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-barley_yellow-mustard"]

# crop_rotation_scenarios = ["winter-wheat_winter-rape"]

fertilization_intensities = ["low", "medium", "high"]


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
    "M_q_ss": "$NO_3$${}_{PERC}$ [kg N/day/ha]",
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
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()

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


vars_sim = ["ground_cover", "q_hof"]
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for var_sim in vars_sim:
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
            cond1 = (df_params["CLUST_flag"] == 1)
            df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
            vals = df.values.T
            fig, ax = plt.subplots(1, 1, figsize=(6, 2))
            median_vals = onp.median(vals, axis=0)
            min_vals = onp.min(vals, axis=0)
            max_vals = onp.max(vals, axis=0)
            p5_vals = onp.nanquantile(vals, 0.05, axis=0)
            p25_vals = onp.nanquantile(vals, 0.25, axis=0)
            p75_vals = onp.nanquantile(vals, 0.75, axis=0)
            p95_vals = onp.nanquantile(vals, 0.95, axis=0)
            ax.fill_between(
                df.index,
                min_vals,
                max_vals,
                edgecolor='black',
                facecolor='black',
                alpha=0.33,
                label="Min-Max interval",
            )
            ax.fill_between(
                df.index,
                p5_vals,
                p95_vals,
                edgecolor='black',
                facecolor='black',
                alpha=0.66,
                label="95% interval",
            )
            ax.fill_between(
                df.index,
                p25_vals,
                p75_vals,
                edgecolor='black',
                facecolor='black',
                alpha=1,
                label="75% interval",
            )
            ax.plot(df.index, median_vals, color="black", label="Median", linewidth=1)
            ax.legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.93, 1.19))
            ax.set_xlabel("Time [Year]")
            ax.set_ylabel(_lab_unit1[var_sim])
            ax.set_xlim(df.index[0], df.index[-1])
            fig.tight_layout()
            file = base_path_figs / f"trace_{var_sim}_{location}_{crop_rotation_scenario}.png"
            fig.savefig(file, dpi=250)
            plt.close(fig)


crop_rotation_scenarios = ["winter-wheat_clover",
                           "winter-wheat_silage-corn",
                           "summer-wheat_winter-wheat",
                           "summer-wheat_clover_winter-wheat",
                           "winter-wheat_clover_silage-corn",
                           "winter-wheat_sugar-beet_silage-corn",
                           "summer-wheat_winter-wheat_silage-corn",
                           "summer-wheat_winter-wheat_winter-rape",
                           "winter-wheat_winter-rape",
                           "winter-wheat_soybean_winter-rape",
                           "sugar-beet_winter-wheat_winter-barley", 
                           "grain-corn_winter-wheat_winter-rape", 
                           "grain-corn_winter-wheat_winter-barley",
                           "grain-corn_winter-wheat_clover",
                           "winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_yellow-mustard",
                           "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                           "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-barley_yellow-mustard"]

# # load nitrogen loads and concentrations
# dict_nitrate = {}
# for location in locations:
#     dict_nitrate[location] = {}
#     for crop_rotation_scenario in crop_rotation_scenarios:
#         dict_nitrate[location][crop_rotation_scenario] = {}
#         for fertilization_intensity in fertilization_intensities:
#             dict_nitrate[location][crop_rotation_scenario][fertilization_intensity] = {}
#             output_nitrate_file = (
#                 base_path_output
#                 / "svat_crop_nitrate"
#                 / f"SVATCROPNITRATE_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.nc"
#             )
#             ds_nitrate = xr.open_dataset(output_nitrate_file, engine="h5netcdf")
#             # assign date
#             days = ds_nitrate["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
#             date = num2date(
#                 days,
#                 units=f"days since {ds_nitrate['Time'].attrs['time_origin']}",
#                 calendar="standard",
#                 only_use_cftime_datetimes=False,
#             )
#             ds_nitrate = ds_nitrate.assign_coords(Time=("Time", date))
#             dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] = ds_nitrate


vars_sim = ["M_q_ss"]
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for var_sim in vars_sim:
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
            cond1 = (df_params["CLUST_flag"] == 1)
            df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
            vals = df.values.T * 0.01
            fig, ax = plt.subplots(1, 1, figsize=(6, 2))
            median_vals = onp.median(vals, axis=0)
            min_vals = onp.min(vals, axis=0)
            max_vals = onp.max(vals, axis=0)
            p5_vals = onp.nanquantile(vals, 0.05, axis=0)
            p25_vals = onp.nanquantile(vals, 0.25, axis=0)
            p75_vals = onp.nanquantile(vals, 0.75, axis=0)
            p95_vals = onp.nanquantile(vals, 0.95, axis=0)
            ax.fill_between(
                df.index,
                min_vals,
                max_vals,
                edgecolor='black',
                facecolor='black',
                alpha=0.33,
                label="Min-Max interval",
            )
            ax.fill_between(
                df.index,
                p5_vals,
                p95_vals,
                edgecolor='black',
                facecolor='black',
                alpha=0.66,
                label="95% interval",
            )
            ax.fill_between(
                df.index,
                p25_vals,
                p75_vals,
                edgecolor='black',
                facecolor='black',
                alpha=1,
                label="75% interval",
            )
            ax.plot(df.index, median_vals, color="black", label="Median", linewidth=1)
            ax.legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.93, 1.19))
            ax.set_xlabel("Time [Year]")
            ax.set_ylabel(labs._Y_LABS_DAILY[var_sim])
            ax.set_xlim(df.index[0], df.index[-1])
            fig.tight_layout()
            file = base_path_figs / f"trace_{var_sim}_{location}_{crop_rotation_scenario}.png"
            fig.savefig(file, dpi=250)
            plt.close(fig)
