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

fertilization_intensities = ["low", "medium", "high"]


_lab_unit_annual = {
    "Nfert": "FERT-$N$\n [kg N/year/ha]",
    "nh4_up": "TRANSP-$N$\n [kg N/year/ha]",
    "nit_s": "NIT\n [kg N/year/ha]",
    "denit_s": "DENIT\n [kg N/year/ha]",
    "min_s": "MIN\n [kg N/year/ha]",
    "nfix_s": "NFIX\n [kg N/year/ha]",
    "M_in": "FERT-$NO_3$\n [kg $NO_3$-N/year/ha]",
    "M_transp": "TRANSP-$NO_3$\n [kg $NO_3$-N/year/ha]",
    "M_s": "SOIL-$NO_3$\n [kg $NO_3$-N/ha]",
    "Nmin_s": "SOIL-$NMIN$\n [kg $N/ha]",
    "M_q_ss": "PERC-$NO_3$\n [kg $NO_3$-N/year/ha]",
}

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()


crop_rotation_scenarios = ["winter-wheat_winter-rape"]

# load nitrogen loads and concentrations
dict_nitrate = {}
for location in locations:
    dict_nitrate[location] = {}
    for crop_rotation_scenario in crop_rotation_scenarios:
        dict_nitrate[location][crop_rotation_scenario] = {}
        for fertilization_intensity in fertilization_intensities:
            dict_nitrate[location][crop_rotation_scenario][fertilization_intensity] = {}
            output_nitrate_file = (
                base_path_output
                / "svat_crop_nitrate"
                / f"SVATCROPNITRATE_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.nc"
            )
            ds_nitrate = xr.open_dataset(output_nitrate_file, engine="h5netcdf")
            # assign date
            days = ds_nitrate["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
            date = num2date(
                days,
                units=f"days since {ds_nitrate['Time'].attrs['time_origin']}",
                calendar="standard",
                only_use_cftime_datetimes=False,
            )
            ds_nitrate = ds_nitrate.assign_coords(Time=("Time", date))
            dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] = ds_nitrate


# plot annual nitrate-nitrogen balance
vars_sim = ["Nfert", "M_transp", "M_s", "M_q_ss"]
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for fertilization_intensity in fertilization_intensities:
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert']
            fig, axes = plt.subplots(4, 1, figsize=(6, 6)) 
            for i, var_sim in enumerate(vars_sim):
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:] * 0.01 # convert from mg/m2 to kg/ha
                cond1 = (df_params["CLUST_flag"] == 2)
                df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
                if var_sim in ["M_s"]:
                    # calculate annual sum
                    df_ann = df.resample("YE").sum().iloc[:-1, :]
                    df_ann.loc[:, "year"] = df_ann.index.year
                    df_ann_long = df_ann.melt(id_vars="year", value_name="vals", var_name='Nfert')
                    df_ann_long.loc[:, "Nfert"] = f'{fertilization_intensity}'
                else:
                    # calculate annual average
                    df_ann = df.resample("YE").mean().iloc[:-1, :]
                    df_ann.loc[:, "year"] = df_ann.index.year
                    df_ann_long = df_ann.melt(id_vars="year", value_name="vals", var_name='Nfert')
                    df_ann_long.loc[:, "Nfert"] = f'{fertilization_intensity}'
    
                sns.boxplot(data=df_ann_long, x="year", y="vals", ax=axes[i], color="red", whis=(5, 95), showfliers=False)
                axes[i].set_xlabel("Time [Year]")
                axes[i].set_ylabel(_lab_unit_annual[var_sim])
                axes[i].set_ylim(0, )
                fig.tight_layout()
                file = base_path_figs / f"boxplot_{var_sim}_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.png"
                fig.savefig(file, dpi=250)
                plt.close(fig)

# plot annual nitrate-nitrogen balance
vars_sim = ["M_in", "M_transp", "M_s", "M_q_ss"]
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        fig, axes = plt.subplots(4, 1, figsize=(6, 6)) 
        for i, var_sim in enumerate(vars_sim):
            ll_df = []
            for fertilization_intensity in fertilization_intensities:
                ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert']
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:] * 0.01 # convert from mg/m2 to kg/ha
                cond1 = (df_params["CLUST_flag"] == 2)
                df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
                if var_sim in ["M_s"]:
                    # calculate annual sum
                    df_ann = df.resample("YE").sum().iloc[:-1, :]
                    df_ann.loc[:, "year"] = df_ann.index.year
                    df_ann_long = df_ann.melt(id_vars="year", value_name="vals", var_name='Nfert')
                    df_ann_long.loc[:, "Nfert"] = f'{fertilization_intensity}'
                else:
                    # calculate annual average
                    df_ann = df.resample("YE").mean().iloc[:-1, :]
                    df_ann.loc[:, "year"] = df_ann.index.year
                    df_ann_long = df_ann.melt(id_vars="year", value_name="vals", var_name='Nfert')
                    df_ann_long.loc[:, "Nfert"] = f'{fertilization_intensity}'
    
            sns.boxplot(data=df_ann_long, x="year", y="vals", ax=axes[i], color="red", whis=(5, 95), showfliers=False)
            axes[i].set_xlabel("Time [Year]")
            axes[i].set_ylabel(_lab_unit_annual[var_sim])
            axes[i].set_ylim(0, )
        fig.tight_layout()
        file = base_path_figs / f"boxplot_{var_sim}_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.png"
        fig.savefig(file, dpi=250)
        plt.close(fig)


vars_sim = ["M_q_ss"]
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for var_sim in vars_sim:
            ll_df = []
            for fertilization_intensity in fertilization_intensities:
                ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:] * 0.01 # convert from mg/m2 to kg/ha
                cond1 = (df_params["CLUST_flag"] == 2)
                df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
                # calculate annual sum
                df_ann = df.resample("YE").sum().iloc[:-1, :]
                df_ann.loc[:, "year"] = df_ann.index.year
                df_ann_long = df_ann.melt(id_vars="year", value_name="vals", var_name='Nfert')
                df_ann_long.loc[:, "Nfert"] = f'{fertilization_intensity}'
                ll_df.append(df_ann_long)
        df_ann_long = pd.concat(ll_df)
    
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        sns.boxplot(data=df_ann_long, x="year", y="vals", hue="Nfert", ax=ax, whis=(5, 95), palette="magma_r", showfliers=False)
        ax.set_xlabel("Time [Year]")
        ax.set_ylabel(_lab_unit_annual[var_sim])
        ax.set_ylim(0, )
        plt.legend(title="N-Fertilization intensity", frameon=False)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_{var_sim}_{location}_{crop_rotation_scenario}.png"
        fig.savefig(file, dpi=250)
        plt.close(fig)
