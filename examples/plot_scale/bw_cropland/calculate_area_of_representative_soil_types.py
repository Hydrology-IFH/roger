from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns
import yaml
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["legend.title_fontsize"] = 11
sns.set_style("ticks")
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
    "z_soil": r"$z_{Boden}$ [mm]",
    "lmpv": r"$l_{mpv}$ [mm]",
    "dmpv": r"$\rho_{mpv}$ [1/$m^2$]",
    "theta_ac": r"$\theta_{lk}$ [-]",
    "theta_ufc": r"$\theta_{nfk}$ [-]",
    "theta_pwp": r"$\theta_{pwp}$ [-]",
    "ks": r"$k_{s}$ [mm/h]",
    "kf": r"$k_{f}$ [mm/h]",
    "soil_fertility": "Bodenfrucht-\nbarkeit",
    "clay": r"Tongehalt [-]",
    "area_share": r"Fläche [%]",
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

# load the configuration file
with open(base_path / "config.yml", "r") as file:
    config = yaml.safe_load(file)

# identifiers of the simulations
locations = config["locations"]


# load linkage between BK50 and cropland clusters
file = base_path / "input" / "link_cluster_geometries_cropland.h5"
df_link_bk50_cluster_cropland = pd.read_hdf(file)

# load model parameters
csv_file = base_path / "representative_agricultural_soil_types_parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
cond = (df_params["CLUST_flag"] == 1)
df_params = df_params.loc[cond, :]
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()

ll_params = ['z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'soil_fertility', 'clay']

file = base_path / "output" / "areas.csv"
if not file.exists():
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
        file = base_path / "input" / "BK50_cropland_areas.gpkg"
        gdf1 = gpd.read_file(file)
        mask = (gdf1['stationsna'].str.lower() == location)
        gdf = gdf1.loc[mask, :]
        # assign aggregated values to polygons
        for clust_id in clust_ids:
            cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
            shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
            cond2 = gdf["SHP_ID"].isin(shp_ids)
            if cond2.any():
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

    df = pd.DataFrame({"location": ll_locations, "clust_id": ll_clust_ids,
                    "area": ll_areas, "z_soil": ll_z_soil, "dmpv": ll_dmpv, "lmpv": ll_lmpv, 
                    "theta_ac": ll_ac, "theta_ufc": ll_ufc, "theta_pwp": ll_pwp, "ks": ll_ks, "soil_fertility": ll_soil_fertility, "clay": ll_clay}, index=range(len(ll_locations)))
    df.loc[:, "area_share"] = 0.0
    for location in locations:
        cond = (df["location"] == location)
        df_location = df.loc[cond, :]
        area = df_location["area"].sum()
        df.loc[cond, "area_share"] = df_location["area"] / area
    file = base_path / "output" / "areas.csv"
    df.to_csv(file, sep=";", index=False)

df = pd.read_csv(base_path / "output" / "areas.csv", sep=";")
df["clust_id"] = [int(x.split("-")[-1]) for x in df["clust_id"].values]

df_soil_types = df.iloc[:, 2:]
df_soil_types["clust_id"] = df.iloc[:, 1]
df_soil_types.index = df_soil_types["clust_id"]
df_soil_types = df_soil_types.sort_index(inplace=False)
df_soil_types["area"] = df.groupby(["clust_id"]).sum()["area"]
df_soil_types["area_share"] = (df_soil_types["area"]/df_soil_types["area"].sum()) * 100


columns = ['location', 'clust_id', 'z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks',
       'soil_fertility', 'clay', 'area_share']
df_region = df.loc[:, columns].groupby(["location", "clust_id"]).mean()
columns = ['location', 'clust_id', 'area_share']
df_region.loc[:, "area_share"] = df.loc[:, columns].groupby(["location", "clust_id"]).sum()
df_region.loc[:, "area_share"] = (df_region.loc[:, "area_share"].values / 3) * 100
df_region["location1"] = df_region.index.get_level_values(0)

colors = sns.color_palette("RdYlBu", n_colors=len(locations))
legend_elements= []
for j, location in enumerate(locations):
    element = Line2D([0], [0], color=colors[j], lw=2, label=location)
    legend_elements.append(element)
ll_params = ['z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'clay', 'soil_fertility']
fig, axs = plt.subplots(3, 3, figsize=(6, 6), sharey=True)
for i, param in enumerate(ll_params):
    axes = axs.flatten()[i]
    g = sns.lineplot(data=df_region, x=param, y="area_share", hue="location", ax=axes, errorbar=None, palette="RdYlBu")
    axes.set_ylim(0, 50)
    axes.set_ylabel("area [%]")
    axes.set_xlabel(_LABS_UNIT[param])
    g.legend().set_visible(False)
fig.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=False)
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.8, wspace=0.15)
file = base_path / "figures" / "plot_parameters_cluster2.png"
fig.savefig(file, dpi=300)
plt.close(fig)

colors = sns.color_palette("RdYlBu", n_colors=len(locations))
legend_elements= []
for j, location in enumerate(locations):
    element = Patch(facecolor=colors[j], edgecolor='k',
                          label=location)
    legend_elements.append(element)

ll_params = ['z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'clay', 'soil_fertility']
fig, axs = plt.subplots(3, 3, figsize=(6, 6), sharey=True)
for i, param in enumerate(ll_params):
    print(param)
    df_region['category'] = pd.cut(df_region[param], bins=_bins[param], labels=_bins_labels[param])
    columns = ['location1', 'category', 'area_share']
    df1 = df_region.loc[:, columns].groupby(["location1", "category"]).sum()
    axes = axs.flatten()[i]
    g = sns.barplot(data=df1, x="category", y="area_share", hue="location1", ax=axes, errorbar=None, palette="RdYlBu")
    axes.set_ylim(0, 100)
    axes.set_ylabel("Fläche [%]")
    axes.set_xlabel(_LABS_UNIT[param])
    g.legend().set_visible(False)
    axes.tick_params(axis='x', labelrotation=25)
fig.legend(handles=legend_elements, loc='upper center', ncol=3, frameon=False)
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.8, wspace=0.15)
file = base_path / "figures" / "barplot_parameters_cluster2.png"
fig.savefig(file, dpi=300)
plt.close(fig)

ll_params = ['', 'area_share', '', 'z_soil', 'dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'clay', 'soil_fertility']
fig, axs = plt.subplots(4, 3, figsize=(6, 5), sharex=True, sharey=False)
for i, param in enumerate(ll_params):
    axes = axs.flatten()[i]
    if i == 0 or i == 2:
        axes.set_axis_off()
    else:
        g = sns.barplot(data=df_soil_types, x="clust_id", y=param, hue="clust_id", ax=axes, errorbar=None, palette="Oranges")
        axes.set_ylabel(_LABS_UNIT[param])
        g.legend().set_visible(False)
        axes.tick_params(axis='x', labelrotation=90)
axs.flatten()[-3].set_xlabel("")
axs.flatten()[-1].set_xlabel("")
axs.flatten()[-2].set_xlabel("# repräsentativer Bodentyp")
axs.flatten()[-1].set_xticklabels(['1', '', '3', '', '5', '', '7', '', '9', '', '11', '', '13', '', '15', '', '17', '', '19'], rotation=90, fontsize=9)
axs.flatten()[-2].set_xticklabels(['1', '', '3', '', '5', '', '7', '', '9', '', '11', '', '13', '', '15', '', '17', '', '19'], rotation=90, fontsize=9)
axs.flatten()[-3].set_xticklabels(['1', '', '3', '', '5', '', '7', '', '9', '', '11', '', '13', '', '15', '', '17', '', '19'], rotation=90, fontsize=9)
fig.tight_layout()
file = base_path / "figures" / "barplot_parameters_cluster.png"
fig.savefig(file, dpi=300)
plt.close(fig)