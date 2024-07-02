from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

# mpl.rcParams["font.size"] = 8
# mpl.rcParams["axes.titlesize"] = 8
# mpl.rcParams["axes.labelsize"] = 9
# mpl.rcParams["xtick.labelsize"] = 8
# mpl.rcParams["ytick.labelsize"] = 8
# mpl.rcParams["legend.fontsize"] = 8
# mpl.rcParams["legend.title_fontsize"] = 9

mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["legend.title_fontsize"] = 11
sns.set_style("ticks")
# sns.plotting_context(
#     "paper",
#     font_scale=1,
#     rc={
#         "font.size": 8.0,
#         "axes.labelsize": 9.0,
#         "axes.titlesize": 8.0,
#         "xtick.labelsize": 8.0,
#         "ytick.labelsize": 8.0,
#         "legend.fontsize": 8.0,
#         "legend.title_fontsize": 9.0,
#     },
# )

sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "legend.title_fontsize": 10.0,
    },
)

_LABS_UNIT = {
    "z_soil": r"$z_{soil}$ [mm]",
    "lmpv": r"$l_{mpv}$ [mm]",
    "dmpv": r"$\rho_{mpv}$ [1/$m^2$]",
    "theta_ac": r"$\theta_{ac}$ [-]",
    "theta_ufc": r"$\theta_{ufc}$ [-]",
    "theta_pwp": r"$\theta_{pwp}$ [-]",
    "ks": r"$k_{s}$ [mm/hour]",
    "kf": r"$k_{f}$ [mm/hour]",
    "soil_fertility": r"soil fertility",
    "clay": r"clay [-]",
}

_LABS = {
    "z_soil": r"$z_{soil}$",
    "lmpv": r"$l_{mpv}$",
    "dmpv": r"$\rho_{mpv}$",
    "theta_ac": r"$\theta_{ac}$",
    "theta_ufc": r"$\theta_{ufc}$",
    "theta_pwp": r"$\theta_{pwp}$",
    "ks": r"$k_{s}$",
    "kf": r"$k_{f}$",
    "soil_fertility": r"soil fertility",
    "clay": r"clay",
}

_bins = {
    "z_soil": [250, 500, 750, 1000, 1250, 1500],
    "dmpv": [0, 50, 100, 150, 200],
    "lmpv": [25, 26, 27, 28, 29, 30],
    "theta_ac": [0.1, 0.15, 0.2, 0.25, 0.3],
    "theta_ufc": [0.1, 0.15, 0.2, 0.25, 0.3],
    "theta_pwp": [0.1, 0.15, 0.2, 0.25, 0.3],
    "ks": [0.1, 1, 5, 20, 40],
    "clay": [0, 0.1, 0.2, 0.3],
    "soil_fertility": [1, 2, 3, 4],
}

_bins_labels = {"z_soil": ["250-500", "500-750", "750-1000", "1000-1250", "1250-1500"],
                "dmpv": ["0-50", "50-100", "100-150", "150-200"],
                "lmpv": ["26", "27", "28", "29", "30"],
                "theta_ac": ["0.1-0.15", "0.15-0.2", "0.2-0.25", "0.25-0.3"],
                "theta_ufc": ["0.1-0.15", "0.15-0.2", "0.2-0.25", "0.25-0.3"],
                "theta_pwp": ["0.1-0.15", "0.15-0.2", "0.2-0.25", "0.25-0.3"],
                "ks": ["0.1-1", "1-5", "5-20", "20-40"],
                "clay": ["0-0.1", "0.1-0.2", "0.2-0.3"],
                "soil_fertility": ["1-2", "2-3", "3-4"]
}


base_path = Path(__file__).parent


locations = ["freiburg", "lahr", "muellheim", 
             "stockach", "gottmadingen", "weingarten",
             "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
             "ehingen-kirchen", "merklingen", "hayingen",
             "kupferzell", "oehringen", "vellberg-kleinaltdorf"]

regions = ["upper rhine valley", "lake constance", "kraichgau", "alb-danube", "hohenlohe"]


# load linkage between BK50 and cropland clusters
file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "link_shp_clust_acker.h5"
df_link_bk50_cluster_cropland = pd.read_hdf(file)

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
cond = (df_params["CLUST_flag"] == 1)
df_params = df_params.loc[cond, :]
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()

ll_params = ['z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'soil_fertility', 'clay']

file = base_path / "output" / "areas.csv"
if not file.exists():
    ll_regions = []
    ll_locations = []
    ll_clust_ids = []
    ll_areas = []
    ll_z_soil = []
    ll_dmpv = []
    ll_lmpv = []
    ll_ac = []
    ll_ufc = []
    ll_pwp = []    
    ll_ks = []
    ll_soil_fertility = []
    ll_clay = []
    for location in locations:
        file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "BK50_NBiomasseBW_for_assignment.gpkg"
        gdf1 = gpd.read_file(file)
        mask = (gdf1['stationsna'] == location)
        gdf = gdf1.loc[mask, :]
        if location in ["freiburg", "lahr", "muellheim"]:
            region = "upper rhine valley"
        elif location in ["stockach", "gottmadingen", "weingarten"]:
            region = "lake constance"
        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
            region = "kraichgau"
        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
            region = "alb-danube"
        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
            region = "hohenlohe"
        # assign aggregated values to polygons
        for clust_id in clust_ids:
            cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
            shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
            cond2 = gdf["SHP_ID"].isin(shp_ids)
            if cond2.any():
                ll_regions.append(region)
                ll_locations.append(location)
                ll_clust_ids.append(clust_id)
                cond = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 1)
                ll_areas.append(gdf.loc[cond2, "area"].sum())
                ll_z_soil.append(df_params.loc[cond, "z_soil"].values[0])
                ll_dmpv.append(df_params.loc[cond, "dmpv"].values[0])
                ll_lmpv.append(df_params.loc[cond, "lmpv"].values[0])
                ll_ac.append(df_params.loc[cond, "theta_ac"].values[0])
                ll_ufc.append(df_params.loc[cond, "theta_ufc"].values[0])
                ll_pwp.append(df_params.loc[cond, "theta_pwp"].values[0])
                ll_ks.append(df_params.loc[cond, "ks"].values[0])
                ll_soil_fertility.append(df_params.loc[cond, "soil_fertility"].values[0])
                ll_clay.append(df_params.loc[cond, "clay"].values[0])

    df = pd.DataFrame({"region": ll_regions, "location": ll_locations, "clust_id": ll_clust_ids,
                    "area": ll_areas, "z_soil": ll_z_soil, "dmpv": ll_dmpv, "lmpv": ll_lmpv, 
                    "theta_ac": ll_ac, "theta_ufc": ll_ufc, "theta_pwp": ll_pwp, "ks": ll_ks, "soil_fertility": ll_soil_fertility, "clay": ll_clay})
    df.loc[:, "area_share"] = 0.0
    for location in locations:
        cond = (df["location"] == location)
        df_location = df.loc[cond, :]
        area = df_location["area"].sum()
        df.loc[cond, "area_share"] = df_location["area"] / area
    file = base_path / "output" / "areas.csv"
    df.to_csv(file, sep=";", index=False)

df = pd.read_csv(base_path / "output" / "areas.csv", sep=";")

columns = ['region', 'clust_id', 'z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks',
       'soil_fertility', 'clay', 'area_share']
df_region = df.loc[:, columns].groupby(["region", "clust_id"]).mean()
columns = ['region', 'clust_id', 'area_share']
df_region.loc[:, "area_share"] = df.loc[:, columns].groupby(["region", "clust_id"]).sum()
df_region.loc[:, "area_share"] = (df_region.loc[:, "area_share"].values / 3) * 100
df_region["region1"] = df_region.index.get_level_values(0)

colors = sns.color_palette("YlGnBu", n_colors=5)
legend_elements= []
for j, region in enumerate(regions):
    element = Line2D([0], [0], color=colors[j], lw=2, label=region)
    legend_elements.append(element)
ll_params = ['z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'clay', 'soil_fertility']
fig, axs = plt.subplots(3, 3, figsize=(6, 6), sharey=True)
for i, param in enumerate(ll_params):
    axes = axs.flatten()[i]
    g = sns.lineplot(data=df_region, x=param, y="area_share", hue="region", ax=axes, errorbar=None, palette="YlGnBu")
    axes.set_ylim(0, 50)
    axes.set_ylabel("area [%]")
    axes.set_xlabel(_LABS_UNIT[param])
    g.legend().set_visible(False)
fig.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=False)
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.8, wspace=0.15)
file = base_path / "figures" / "plot_parameters_cluster2.png"
fig.savefig(file, dpi=300)
plt.close(fig)

colors = sns.color_palette("YlGnBu", n_colors=5)
legend_elements= []
for j, region in enumerate(regions):
    element = Patch(facecolor=colors[j], edgecolor='k',
                          label=region)
    legend_elements.append(element)

ll_params = ['z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'clay', 'soil_fertility']
fig, axs = plt.subplots(3, 3, figsize=(6, 6), sharey=True)
for i, param in enumerate(ll_params):
    print(param)
    df_region['category'] = pd.cut(df_region[param], bins=_bins[param], labels=_bins_labels[param])
    columns = ['region1', 'category', 'area_share']
    df1 = df_region.loc[:, columns].groupby(["region1", "category"]).sum()
    axes = axs.flatten()[i]
    g = sns.barplot(data=df1, x="category", y="area_share", hue="region1", ax=axes, errorbar=None, palette="YlGnBu")
    axes.set_ylim(0, 100)
    axes.set_ylabel("area [%]")
    axes.set_xlabel(_LABS_UNIT[param])
    g.legend().set_visible(False)
    axes.tick_params(axis='x', labelrotation=25)
fig.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=False)
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.8, wspace=0.15)
file = base_path / "figures" / "barplot_parameters_cluster2.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# regions = ["upper rhine valley"]
# colors = sns.color_palette("YlGnBu", n_colors=5)
# ll_params = ['z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'clay', 'soil_fertility']
# fig, axs = plt.subplots(3, 3, figsize=(6, 6), sharey=True)
# for i, param in enumerate(ll_params):
#     axes = axs.flatten()[i]
#     for j, region in enumerate(regions):
#         cond = (df_region.index.get_level_values(0) == region)
#         df = df_region.loc[cond, :]
#         g = sns.barplot(data=df, x=param, y="area_share", color=colors[j], ax=axes, errorbar=None, legend=False)
#     axes.set_ylim(0, 50)
#     axes.set_ylabel("area [%]")
#     axes.set_xlabel(_LABS_UNIT[param])
# fig.tight_layout()
# file = base_path / "figures" / "histplot_parameters_cluster2.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)


# df_params_locations_wavg = pd.DataFrame(index=locations, columns=ll_params)
# for i, location in enumerate(locations):
#     cond = (df["location"] == location)
#     df_location = df.loc[cond, :]
#     for param in ll_params:
#         val = np.sum(df_location[param].values * df_location["area_share"].values)
#         df_params_locations_wavg.loc[df_params_locations_wavg.index[i], param] = val
# file = base_path / "output" / "parameters_locations_weighted_avg.csv"
# df_params_locations_wavg.to_csv(file, sep=";", index=True)

# df_params_regions_wavg = pd.DataFrame(index=regions, columns=ll_params)
# for i, region in enumerate(regions):
#     cond = (df["region"] == region)
#     df_region = df.loc[cond, :]
#     df_region.loc[:, "area_share"] = df_region["area_share"].values / np.sum(df_region["area_share"].values)
#     for param in ll_params:
#         val = np.sum(df_region[param].values * df_region["area_share"].values)
#         df_params_regions_wavg.loc[df_params_regions_wavg.index[i], param] = val
# file = base_path / "output" / "parameters_regions_weighted_avg.csv"
# df_params_regions_wavg.to_csv(file, sep=";", index=True)

# df_params_locations_avg = pd.DataFrame(index=locations, columns=ll_params)
# for i, location in enumerate(locations):
#     cond = (df["location"] == location)
#     df_location = df.loc[cond, :]
#     for param in ll_params:
#         val = np.mean(df_location[param].values)
#         df_params_locations_avg.loc[df_params_locations_avg.index[i], param] = val
# file = base_path / "output" / "parameters_locations_avg.csv"
# df_params_locations_avg.to_csv(file, sep=";", index=True)

# df_params_regions_avg = pd.DataFrame(index=regions, columns=ll_params)
# for i, region in enumerate(regions):
#     cond = (df["region"] == region)
#     df_region = df.loc[cond, :]
#     df_region.loc[:, "area_share"] = df_region["area_share"].values / np.sum(df_region["area_share"].values)
#     for param in ll_params:
#         val = np.mean(df_region[param].values)
#         df_params_regions_avg.loc[df_params_regions_avg.index[i], param] = val
# file = base_path / "output" / "parameters_regions_avg.csv"
# df_params_regions_avg.to_csv(file, sep=";", index=True)


# df_params_regions_wavg = pd.read_csv(base_path / "output" / "parameters_regions_weighted_avg.csv", sep=";", index_col=0)
# df_params_locations_wavg = pd.read_csv(base_path / "output" / "parameters_locations_weighted_avg.csv", sep=";", index_col=0)

# # plot parameters per region
# # 1=upper rhine valley 
# # 2=lake constance 
# # 3=kraichgau 
# # 4=alb-danube 
# # 5=hohenlohe
# fig, axes = plt.subplots(3, 3, figsize=(6, 5), sharex=True, sharey=False)
# for i, param in enumerate(ll_params):
#     ax = axes.flatten()[i]
#     cond = df_params_regions_wavg.loc[:, param].to_frame()
#     df = df_params_regions_wavg
#     ax.scatter(df.index, df[param], color="black", s=10)
#     # cond = df_params_regions_avg.loc[:, param].to_frame()
#     # df = df_params_regions_avg
#     # ax.scatter(df.index, df[param], color="purple", s=10)
#     ax.set_ylabel(_LABS_UNIT[param])
#     ax.set_xlabel("")
#     ax.set_xticks([0, 1, 2, 3, 4])
#     ax.set_xlim(-0.5, 4.5)
#     ax.set_xticklabels(['1', '2', '3', '4', '5'])
# fig.tight_layout()
# file = base_path / "figures" / "parameters_regions.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# # plot parameters per location
# # 1=freiburg
# # 2=lahr
# # 3=muellheim 
# # 4=stockach
# # 5=gottmadingen 
# # 6=weingarten
# # 7=eppingen-elsenz 
# # 8=bruchsal-heidelsheim 
# # 9=bretten
# # 10=ehingen-kirchen 
# # 11=merklingen 
# # 12=hayingen
# # 13=kupferzell 
# # 14=oehringen 
# # 15=vellberg-kleinaltdorf
# fig, axes = plt.subplots(3, 3, figsize=(6, 5), sharex=True, sharey=False)
# for i, param in enumerate(ll_params):
#     ax = axes.flatten()[i]
#     cond = df_params_locations_wavg.loc[:, param].to_frame()
#     df = df_params_locations_wavg
#     ax.scatter(df.index, df[param], color="black", s=10)
#     ax.set_ylabel(_LABS_UNIT[param])
#     ax.set_xlabel("")
#     ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
#     ax.set_xlim(-0.5, 14.5)
#     ax.set_xticklabels(['1', '', '3', '', '5', '', '7', '', '9', '', '11', '', '13', '', '15'], rotation=90, fontsize=9)
# fig.tight_layout()
# file = base_path / "figures" / "parameters_locations.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# # load model parameters
# csv_file = base_path / "parameters.csv"
# df_params = pd.read_csv(csv_file, sep=";", skiprows=1)

# fig, axs = plt.subplots(2, 5, figsize=(6, 4))
# for i, param in enumerate(ll_params):
#     cond = (df_params["CLUST_flag"] == 1)
#     df = df_params.loc[cond, param].to_frame()
#     df_long = pd.melt(df, value_vars=df.columns)
#     axes = axs.flatten()[i]
#     sns.boxplot(data=df_long, x="variable", y="value", ax=axes, whis=(0, 100), showfliers=False, color="grey")
#     axes.set_xlabel("")
#     axes.set_ylabel(_LABS_UNIT[param])
#     axes.set_xticklabels([])
# axs[-1, -1].set_axis_off()
# fig.tight_layout()
# file = base_path / "figures" / "boxplot_parameters_cluster1.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# ll_params = ['z_soil', 'dmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'lmpv', 'ks', 'clay', 'soil_fertility']
# fig, axs = plt.subplots(2, 5, figsize=(6, 4))
# for i, param in enumerate(ll_params):
#     cond = (df_params["CLUST_flag"] == 2)
#     df = df_params.loc[cond, param].to_frame()
#     df_long = pd.melt(df, value_vars=df.columns)
#     axes = axs.flatten()[i]
#     sns.boxplot(data=df_long, x="variable", y="value", ax=axes, whis=(0, 100), showfliers=False, color="grey")
#     axes.set_xlabel("")
#     axes.set_ylabel(_LABS_UNIT[param])
#     axes.set_xticklabels([])
# axs[-1, -1].set_axis_off()
# fig.tight_layout()
# file = base_path / "figures" / "boxplot_parameters_cluster2.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# ll_params = ['z_soil', 'dmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'lmpv', 'ks', 'clay', 'soil_fertility']
# fig, axs = plt.subplots(2, 5, figsize=(6, 4), sharex=True)
# for i, param in enumerate(ll_params):
#     cond = (df_params["CLUST_flag"] == 2)
#     df = df_params.loc[cond, param].to_frame()
#     df_long = pd.melt(df, value_vars=df.columns)
#     axes = axs.flatten()[i]
#     sns.histplot(data=df_long, y="value", ax=axes, color="grey")
#     axes.set_xlabel("Count")
#     axes.set_ylabel(_LABS_UNIT[param])
# axs[-1, -1].set_axis_off()
# fig.tight_layout()
# file = base_path / "figures" / "histplot_parameters_cluster2.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)
