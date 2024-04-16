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

# identifiers for crop rotation scenarios
_dict_ffid = {"winter-wheat_clover": "1_0",
              "winter-wheat_silage-corn": "2_0",
              "summer-wheat_winter-wheat": "3_0",
              "summer-wheat_clover_winter-wheat": "4_0",
              "winter-wheat_clover_silage-corn": "5_0",
              "winter-wheat_sugar-beet_silage-corn": "6_0",
              "summer-wheat_winter-wheat_silage-corn": "7_0",
              "summer-wheat_winter-wheat_winter-rape": "8_0",
              "winter-wheat_winter-rape": "9_0",
              "winter-wheat_soybean_winter-rape": "10_0",
              "sugar-beet_winter-wheat_winter-barley": "11_0", 
              "grain-corn_winter-wheat_winter-rape": "12_0", 
              "grain-corn_winter-wheat_winter-barley": "13_0",
              "grain-corn_winter-wheat_clover": "14_0",
              "miscanthus": "15_0",
              "bare-grass": "16_0",
              "winter-wheat_silage-corn_yellow-mustard": "2_1",
              "summer-wheat_winter-wheat_yellow-mustard": "3_1",
              "winter-wheat_sugar-beet_silage-corn_yellow-mustard": "6_1",
              "summer-wheat_winter-wheat_silage-corn_yellow-mustard": "7_1",
              "summer-wheat_winter-wheat_winter-rape_yellow-mustard": "8_1",
              "sugar-beet_winter-wheat_winter-barley_yellow-mustard": "11_1", 
              "grain-corn_winter-wheat_winter-rape_yellow-mustard": "12_1", 
              "grain-corn_winter-wheat_winter-barley_yellow-mustard": "13_1", 
}

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
                           "grain-corn_winter-wheat_winter-barley_yellow-mustard",
                           "miscanthus",
                           "bare-grass"]

_lab_unit_daily = {
    "q_hof": "$Q_{HOF}$ [mm/day]",
    "ground_cover": "GC [-]",
    "transp": "$TRANSP$ [mm/day]",
    "evap_soil": "$EVAP_{soil}$ [mm/day]",
    "aet": "$AET$ [mm/day]",
    "q_ss": "$PERC$ [mm/day]",
    "theta": "$\theta$ [-]",
}

_lab_unit_annual = {
    "q_hof": "$Q_{HOF}$ [mm/year]",
    "ground_cover": "GC [-]",
    "transp": "$TRANSP$ [mm/year]",
    "evap_soil": "$EVAP_{soil}$ [mm/year]",
    "aet": "$AET$ [mm/year]",
    "q_ss": "$PERC$ [mm/year]",
    "theta": "$\theta$ [-]",
}

_lab_unit_total = {
    "q_hof": "$Q_{HOF}$ [mm]",
    "ground_cover": "GC [-]",
    "transp": "$TRANSP$ [mm]",
    "evap_soil": "$EVAP_{soil}$ [mm]",
    "aet": "$AET$ [mm]",
    "q_ss": "$PERC$ [mm]",
    "theta": "$\theta$ [-]",
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
            cond1 = (df_params["CLUST_flag"] == 2)
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
                edgecolor='grey',
                facecolor='grey',
                alpha=0.33,
                label="Min-Max interval",
            )
            ax.fill_between(
                df.index,
                p5_vals,
                p95_vals,
                edgecolor='grey',
                facecolor='grey',
                alpha=0.66,
                label="95% interval",
            )
            ax.fill_between(
                df.index,
                p25_vals,
                p75_vals,
                edgecolor='grey',
                facecolor='grey',
                alpha=1,
                label="75% interval",
            )
            ax.plot(df.index, median_vals, color="black", label="Median", linewidth=1)
            ax.legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.93, 1.19))
            ax.set_xlabel("Time [Year]")
            ax.set_ylabel(_lab_unit_daily[var_sim])
            ax.set_xlim(df.index[0], df.index[-1])
            fig.tight_layout()
            file = base_path_figs / f"trace_{var_sim}_{location}_{crop_rotation_scenario}.png"
            fig.savefig(file, dpi=250)
            plt.close(fig)

# plot average annual sums/means
vars_sim = ["q_hof", "q_ss", "transp", "evap_soil", "aet", "ground_cover", "theta"]
for var_sim in vars_sim:
    for location in locations:
        fig, axes = plt.subplots(1, 1, figsize=(6, 2), sharex=True, sharey=True)
        ll_df = []
        for crop_rotation_scenario in crop_rotation_scenarios:
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
            cond1 = (df_params["CLUST_flag"] == 2)
            df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
            if var_sim in ["q_hof", "q_ss", "transp", "evap_soil", "aet"]:
                # calculate average of annual sums
                df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
                df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                color = "grey"
            elif var_sim in ["ground_cover", "theta"]:
                # calculate average of annual means
                df_ann_avg = df.resample("YE").mean().iloc[:-1, :].mean(axis=0).to_frame()
                df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                color = "green"
            ll_df.append(df_ann_avg)
        df_ann_avg = pd.concat(ll_df)
        df_ann_avg_long = df_ann_avg.melt(id_vars="crop_rotation", value_name="vals").loc[:, ["crop_rotation", "vals"]]        
        sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", ax=axes, whis=(5, 95), showfliers=False, color=color)
        axes.set_ylabel(_lab_unit_total[var_sim])
        axes.set_xlabel("Crop rotation")
        plt.xticks(rotation=33)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_{location}.png"
        fig.savefig(file, dpi=250)
        plt.close(fig)


# plot annual sums/means
# vars_sim = ["q_hof", "q_ss", "transp", "evap_soil", "aet", "ground_cover", "theta"]
# for var_sim in vars_sim:
#     for location in locations:
#         for crop_rotation_scenario in crop_rotation_scenarios:
#             fig, axes = plt.subplots(1, 1, figsize=(6, 2), sharex=True, sharey=True)
#             ds = dict_fluxes_states[location][crop_rotation_scenario]
#             sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
#             cond1 = (df_params["CLUST_flag"] == 2)
#             df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
#             if var_sim in ["q_hof", "q_ss", "transp", "evap_soil", "aet"]:
#                 # calculate average of annual sums
#                 df_ann = df.resample("YE").sum().iloc[:-1, :]
#                 df_ann.loc[:, "year"] = df_ann.index.year
#                 color = "grey"
#                 ylim = (0, 800)
#             elif var_sim in ["ground_cover", "theta"]:
#                 # calculate average of annual means
#                 df_ann = df.resample("YE").mean().iloc[:-1, :]
#                 df_ann.loc[:, "year"] = df_ann.index.year
#                 color = "green"
#                 ylim = (0, 1)
#             df_ann_long = df_ann.melt(id_vars="year", value_name="vals")     
#             sns.boxplot(data=df_ann_long, x="year", y="vals", ax=axes, whis=(5, 95), showfliers=False, color=color)
#             axes.set_ylabel(_lab_unit_total[var_sim])
#             axes.set_xlabel("Time [Year]")
#             axes.set_ylim(ylim)
#             plt.xticks(rotation=33)
#             fig.tight_layout()
#             file = base_path_figs / f"boxplot_annual_{var_sim}_{location}_{crop_rotation_scenario}.png"
#             fig.savefig(file, dpi=250)
#             plt.close(fig)


# vars_sim = ["q_hof", "q_ss", "transp", "evap_soil", "aet"]
# for var_sim in vars_sim:
#     for location in locations:
#         for crop_rotation_scenario in crop_rotation_scenarios:
#             ds = dict_fluxes_states[location][crop_rotation_scenario]
#             sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
#             cond1 = (df_params["CLUST_flag"] == 2)
#             df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
#             # calculate average of annual sums
#             df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
#             maxid = df_ann_avg.idxmax().values[0]
#             maxval = int(df_ann_avg.max().values[0])
#             print(f'{location} {crop_rotation_scenario}: {maxval} -> {maxid}')

# ds = dict_fluxes_states[location][crop_rotation_scenario]
# sim_vals = ds['prec'].isel(y=0).values[:, 1:]
# df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T)
# meanval = int(df.resample("YE").sum().iloc[:, 0].mean())
# print(f'PREC: {meanval}')