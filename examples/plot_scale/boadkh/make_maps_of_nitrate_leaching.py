
from pathlib import Path
import pandas as pd
import geopandas as gpd
import contextily as ctx
import numpy as onp
import matplotlib as mpl
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar
import adjustText as aT
import click
import warnings
warnings.filterwarnings('ignore')

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

_c = 5
mpl.rcParams["font.family"] = "monospace"
mpl.rcParams["font.size"] = 9 * _c
mpl.rcParams["axes.titlesize"] = 8 * _c
mpl.rcParams["axes.labelsize"] = 9 * _c
mpl.rcParams["xtick.labelsize"] = 8 * _c
mpl.rcParams["ytick.labelsize"] = 8 * _c
mpl.rcParams["legend.fontsize"] = 9 * _c
mpl.rcParams["legend.title_fontsize"] = 9 * _c
mpl.rcParams["figure.dpi"] = 1440  # set high dpi to render polygons
mpl.rcParams["axes.linewidth"] = 4
mpl.rcParams["xtick.major.size"] = 12
mpl.rcParams["ytick.major.size"] = 12
mpl.rcParams["xtick.major.width"] = 3
mpl.rcParams["ytick.major.width"] = 3


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

_dict_fert = {"low": 1,
              "medium": 2,
              "high": 3,
}

_dict_crop_id = {"winter-wheat": 115,
                 "clover": 422,
                 "silage-corn": 411,
                 "summer-wheat": 116,
                 "sugar-beet": 603,
                 "winter-rape": 311,
                 "soybean": 330,
                 "grain-corn": 171,
                 "winter-barley": 131,
                 "grass": 591,
                 "miscanthus": 852,
                }



_dict_var_names = {"q_hof": "QSUR",
                   "ground_cover": "GC",
                   "M_q_ss": "MPERC",
                   "C_q_ss": "CPERC",
                   "q_ss": "PERC",
}


@click.option("-td", "--tmp-dir", type=str, default=Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output")
@click.command("main")
def main(tmp_dir):
    file = Path(tmp_dir) / "data_nitrate_leaching" / "nitrate_leaching_geometries.gpkg"
    gdf_ = gpd.read_file(file)
    gdf_["SID"] = gdf_["SID"].astype(int) - 209

    file = Path(tmp_dir) / "data_nitrate_leaching" / "nitrate_leaching_values.csv"
    df_ = pd.read_csv(file, sep=";")

    file = Path(tmp_dir).parent / "DWD-stations_WeatherDB.shp"
    shp = gpd.read_file(file)
    Locations = ['Müllheim', 'Freiburg', 'Lahr',
                 'Bretten, Kreis Karlsruhe', 'Bruchsal-Heidelsheim', 'Eppingen-Elsenz',
                 'Vellberg-Kleinaltdorf', 'Kupferzell-Rechbach', 'Öhringen',
                 'Merklingen', 'Ehingen-Kirchen', 'Hayingen',
                 'Gottmadingen', 'Stockach', 'Weingarten, Kr. Ravensburg']
    
    cond = shp["stationsna"].isin(Locations)
    meteo_stations = shp.loc[cond, :].to_crs(gdf_.crs)
    meteo_stations["stationsna"] = ['Eppingen', 'Öhringen', 'Lahr', 'Vellberg', 'Bruchsal', 'Ehingen', 'Gottmadingen', 'Kupferzell', 'Merklingen', 'Müllheim', 'Weingarten', 'Stockach', 'Bretten', 'Freiburg', 'Hayingen'] 


    # get the unique object identifiers
    oids = onp.unique(gdf_["OID"].values.astype(int)).tolist()

    regions = ['Oberrhein',
               'Kraichgau',
               'Hohenlohe',
               'Donau-Alb',
               'Bodensee']
    locations = ['muellheim', 'freiburg', 'lahr',
                 'bretten', 'bruchsal-heidelsheim', 'eppingen-elsenz',
                 'vellberg-kleinaltdorf', 'kupferzell', 'oehringen',
                 'merklingen', 'ehingen-kirchen', 'hayingen',
                 'gottmadingen', 'stockach', 'weingarten']

    gdf = gdf_.copy()
    gdf["value"] = onp.nan

    for i, region in enumerate(regions):
        mask = (gdf["agr_region"].values.astype(str) == region)
        gdf.loc[mask, "value"] = i + 0.5

    # plot map of representative soil types

    cmap = mpl.colormaps["Oranges"]  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = onp.linspace(1, 20, 20)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    gdf.plot(column="SID", ax=ax, cmap=cmap, norm=norm, legend=True, legend_kwds={"label": "# repräsentativer Bodentyp", "shrink": 0.62})
    meteo_stations.plot(ax=ax, color="black", markersize=100, zorder=2)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
    north_arrow(
    ax, scale=1.25, location="upper left", rotation={"crs": gdf.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": gdf.crs, "height": 0.3}, text = {"fontfamily": "monospace", "fontsize": 24})
    ax.text(
    0.11,
    0.025,
    "EPSG: 25832",
    fontsize=28,
    horizontalalignment="center",
    verticalalignment="center",
    transform=ax.transAxes,
    )
    texts = []
    for x, y, label in zip(meteo_stations.geometry.x, meteo_stations.geometry.y, meteo_stations["stationsna"]):
        texts.append(ax.text(x, y, label, fontsize=32, weight="bold"))
    aT.adjust_text(texts, force_static=(0.05, 0.05), force_text=(0.1, 0.1), expand=(1, 1), time_lim=5, pull_threshold=5, force_pull=(0.06, 0.06),
                   arrowprops=dict(arrowstyle="-", color='grey', lw=4), ax=ax)
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / "map_soil_type.png"
    fig.savefig(file, dpi=300)

    # # plot the map (increase the dpi and figsize to make polygons visible)
    # fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    # gdf.plot(column="value", ax=ax, cmap="RdYlBu", vmin=0, vmax=5, legend=False, zorder=1)
    # meteo_stations.plot(ax=ax, color="black", markersize=100, zorder=2)
    # ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
    # north_arrow(
    # ax, scale=1.25, location="upper left", rotation={"crs": gdf.crs, "reference": "center"}
    # )
    # scale_bar(ax, location="lower right", style="boxes", bar={"projection": gdf.crs, "height": 0.3}, text = {"fontfamily": "monospace", "fontsize": 24})
    # ax.text(
    # 0.09,
    # 0.025,
    # "EPSG: 25832",
    # fontsize=28,
    # horizontalalignment="center",
    # verticalalignment="center",
    # transform=ax.transAxes,
    # )
    # texts = []
    # for x, y, label in zip(meteo_stations.geometry.x, meteo_stations.geometry.y, meteo_stations["stationsna"]):
    #     texts.append(ax.text(x, y, label, fontsize=32, weight="bold"))
    # aT.adjust_text(texts, force_static=(0.05, 0.05), force_text=(0.1, 0.1), expand=(1, 1), time_lim=5, pull_threshold=5, force_pull=(0.06, 0.06),
    #                arrowprops=dict(arrowstyle="-", color='grey', lw=4), ax=ax)
    # fig.tight_layout()
    # file = Path(__file__).parent / "figures" / "map_regions.png"
    # fig.savefig(file, dpi=300)

    # gdf = gdf_.copy()
    # gdf["value"] = onp.nan

    # color_ids = [0.1, 0.4, 0.7, 1.3, 1.5, 1.7, 2.5, 2.6, 2.7, 3.1, 3.4, 3.7, 4.5, 4.75, 5.0]

    # for i, location in enumerate(locations):
    #     mask = (gdf["stationsna"].values.astype(str) == location)
    #     gdf.loc[mask, "value"] = color_ids[i]

    # # plot the map (increase the dpi and figsize to make polygons visible)
    # fig, ax = plt.subplots(1, 1, figsize=(20, 24))
    # gdf.plot(column="value", ax=ax, cmap="RdYlBu", vmin=0, vmax=5, legend=False, zorder=1)
    # meteo_stations.plot(ax=ax, color="black", markersize=100, zorder=2)
    # ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
    # north_arrow(
    # ax, scale=1.25, location="upper left", rotation={"crs": gdf.crs, "reference": "center"},
    # )
    # scale_bar(ax, location="lower right", style="boxes", bar={"projection": gdf.crs, "height": 0.3}, text = {"fontfamily": "monospace", "fontsize": 24})
    # ax.text(
    # 0.09,
    # 0.025,
    # "EPSG: 25832",
    # fontsize=28,
    # horizontalalignment="center",
    # verticalalignment="center",
    # transform=ax.transAxes,
    # )
    # fig.tight_layout()
    # file = Path(__file__).parent / "figures" / "map_locations.png"
    # fig.savefig(file, dpi=300)


    bounds = [10, 25, 37.5, 50, 75]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("", ["#feb24c", "#800026"])
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    vars_sim = ["C_q_ss"]
    crop_rotations = ["winter-wheat_silage-corn", "winter-wheat_silage-corn_yellow-mustard", "winter-wheat_soybean_winter-rape"]
    fertilisation_intensities = ["low", "medium", "high"]

    for var_sim in vars_sim:
        for crop_rotation in crop_rotations:
            for fertilisation_intensity in fertilisation_intensities:
                gdf = gdf_.copy()
                gdf["value"] = onp.nan
                # assign the values to the geometries
                for oid in oids:
                    mask = (gdf["OID"].values.astype(int) == oid)
                    ffid = _dict_ffid[crop_rotation]
                    variable = f"{_dict_var_names[var_sim]}_N{_dict_fert[fertilisation_intensity]}"
                    cond = (df_["OID"] == oid) & (df_["FFID"] == ffid) & (df_["CID"] == 400)
                    value = df_.loc[cond, variable].values[0]
                    gdf.loc[mask, "value"] = value

                # plot the map (increase the dpi and figsize to make polygons visible)
                fig, ax = plt.subplots(1, 1, figsize=(20, 24))
                gdf.plot(column="value", ax=ax, cmap="Purples", vmin=0, vmax=70, legend=True, legend_kwds={"label": "$NO_3$ [mg/l]", "shrink": 0.62})
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
                north_arrow(
                ax, scale=1.25, location="upper left", rotation={"crs": gdf.crs, "reference": "center"}
                )
                scale_bar(ax, location="lower right", style="boxes", bar={"projection": gdf.crs, "height": 0.3}, text = {"fontfamily": "monospace", "fontsize": 24})
                ax.text(
                0.11,
                0.025,
                "EPSG: 25832",
                fontsize=28,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                )
                fig.tight_layout()
                file = Path(__file__).parent / "figures" / f"map_{crop_rotation}_{fertilisation_intensity}Nfert_{var_sim}.png"
                fig.savefig(file, dpi=300)


    for var_sim in vars_sim:
        for crop_rotation in crop_rotations:
            for fertilisation_intensity in fertilisation_intensities:
                gdf = gdf_.copy()
                gdf["value"] = onp.nan
                # assign the values to the geometries
                for oid in oids:
                    mask = (gdf["OID"].values.astype(int) == oid)
                    ffid = _dict_ffid[crop_rotation]
                    variable = f"{_dict_var_names[var_sim]}_N{_dict_fert[fertilisation_intensity]}"
                    cond = (df_["OID"] == oid) & (df_["FFID"] == ffid) & (df_["CID"] == 400)
                    value = df_.loc[cond, variable].values[0]
                    gdf.loc[mask, "value"] = value

                # plot the map (increase the dpi and figsize to make polygons visible)
                fig, ax = plt.subplots(1, 1, figsize=(20, 24))
                gdf.plot(column="value", ax=ax, cmap=cmap, norm=norm, legend=True, legend_kwds={"label": "$NO_3$ [mg/l]", "shrink": 0.5, "ticks": [10, 25, 37.5, 50, 75], "format": mpl.ticker.FixedFormatter(['<10', '25', '37.5', '50', '>75'])})
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
                north_arrow(
                ax, scale=1.25, location="upper left", rotation={"crs": gdf.crs, "reference": "center"}
                )
                scale_bar(ax, location="lower right", style="boxes", bar={"projection": gdf.crs, "height": 0.3}, text = {"fontfamily": "monospace", "fontsize": 24})
                ax.text(
                0.11,
                0.025,
                "EPSG: 25832",
                fontsize=28,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                )
                ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                fig.tight_layout()
                file = Path(__file__).parent / "figures" / f"map_{crop_rotation}_{fertilisation_intensity}Nfert_{var_sim}_.png"
                fig.savefig(file, dpi=300)

    vars_sim = ["C_q_ss"]
    crops = ["summer-wheat", "winter-wheat", "silage-corn", "grain-corn", "sugar-beet", "winter-rape"]
    fertilisation_intensities = ["low", "medium", "high"]

    for var_sim in vars_sim:
        for crop in crops:
            for fertilisation_intensity in fertilisation_intensities:
                gdf = gdf_.copy()
                gdf["value"] = onp.nan
                # assign the values to the geometries
                for oid in oids:
                    mask = (gdf["OID"].values.astype(int) == oid)
                    cid = _dict_crop_id[crop]
                    variable = f"{_dict_var_names[var_sim]}_N{_dict_fert[fertilisation_intensity]}"
                    cond = (df_["OID"] == oid) & (df_["CID"] == cid)
                    value = onp.nanmean(df_.loc[cond, variable].values)
                    gdf.loc[mask, "value"] = value

                # plot the map (increase the dpi and figsize to make polygons visible)
                fig, ax = plt.subplots(1, 1, figsize=(20, 24))
                gdf.plot(column="value", ax=ax, cmap="Purples", vmin=0, vmax=70, legend=True, legend_kwds={"label": "$NO_3$ [mg/l]", "shrink": 0.62})
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs, alpha=0.5)
                north_arrow(
                ax, scale=1.25, location="upper left", rotation={"crs": gdf.crs, "reference": "center"}
                )
                scale_bar(ax, location="lower right", style="boxes", bar={"projection": gdf.crs, "height": 0.3}, text = {"fontfamily": "monospace", "fontsize": 24})
                ax.text(
                0.11,
                0.025,
                "EPSG: 25832",
                fontsize=28,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                )
                fig.tight_layout()
                file = Path(__file__).parent / "figures" / f"map_{crop}_{fertilisation_intensity}Nfert_{var_sim}.png"
                fig.savefig(file, dpi=300)

    for var_sim in vars_sim:
        for crop in crops:
            for fertilisation_intensity in fertilisation_intensities:
                gdf = gdf_.copy()
                gdf["value"] = onp.nan
                # assign the values to the geometries
                for oid in oids:
                    mask = (gdf["OID"].values.astype(int) == oid)
                    cid = _dict_crop_id[crop]
                    variable = f"{_dict_var_names[var_sim]}_N{_dict_fert[fertilisation_intensity]}"
                    cond = (df_["OID"] == oid) & (df_["CID"] == cid)
                    value = onp.nanmean(df_.loc[cond, variable].values)
                    gdf.loc[mask, "value"] = value

                # plot the map (increase the dpi and figsize to make polygons visible)
                fig, ax = plt.subplots(1, 1, figsize=(20, 24))
                gdf.plot(column="value", ax=ax, cmap=cmap, norm=norm, legend=True, legend_kwds={"label": "$NO_3$ [mg/l]", "shrink": 0.5, "ticks": [10, 25, 37.5, 50, 75], "format": mpl.ticker.FixedFormatter(['<10', '25', '37.5', '50', '>75'])})
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs, alpha=0.9)
                north_arrow(
                ax, scale=1.25, location="upper left", rotation={"crs": gdf.crs, "reference": "center"}
                )
                scale_bar(ax, location="lower right", style="boxes", bar={"projection": gdf.crs, "height": 0.3}, text = {"fontfamily": "monospace", "fontsize": 24})
                ax.text(
                0.11,
                0.025,
                "EPSG: 25832",
                fontsize=28,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                )
                fig.tight_layout()
                file = Path(__file__).parent / "figures" / f"map_{crop}_{fertilisation_intensity}Nfert_{var_sim}_.png"
                fig.savefig(file, dpi=300)


if __name__ == "__main__":
    main()