
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import geopandas as gpd
import contextily as ctx
import numpy as onp
import matplotlib as mpl
import seaborn as sns
import click
import warnings
warnings.filterwarnings('ignore')

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

_c = 5
mpl.rcParams["font.size"] = 8 * _c
mpl.rcParams["axes.titlesize"] = 8 * _c
mpl.rcParams["axes.labelsize"] = 9 * _c
mpl.rcParams["xtick.labelsize"] = 8 * _c
mpl.rcParams["ytick.labelsize"] = 8 * _c
mpl.rcParams["legend.fontsize"] = 8 * _c
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
                 "silage-corn": 411,
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
    file = Path(tmp_dir) / "data_for_nitrate_leaching" / "nitrate_leaching_geometries.gpkg"
    gdf_ = gpd.read_file(file)

    file = Path(tmp_dir) / "data_for_nitrate_leaching" / "nitrate_leaching_values.csv"
    df_ = pd.read_csv(file, sep=";")

    # get the unique object identifiers
    oids = onp.unique(gdf_["OID"].values.astype(int)).tolist()

    bounds = [10, 25, 37.5, 50, 75]
    norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["OrRd"].N)

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
                gdf.plot(column="value", ax=ax, cmap="OrRd", vmin=10, vmax=120, legend=True, legend_kwds={"label": "$NO_3$ [mg/l]", "shrink": 0.68})
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
                ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
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
                gdf.plot(column="value", ax=ax, cmap="OrRd", norm=norm, legend=True, legend_kwds={"label": "$NO_3$ [mg/l]", "shrink": 0.5, "ticks": [10, 25, 37.5, 50, 75], "format": mpl.ticker.FixedFormatter(['<10', '25', '37.5', '50', '>75'])})
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
                ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                fig.tight_layout()
                file = Path(__file__).parent / "figures" / f"map_{crop_rotation}_{fertilisation_intensity}Nfert_{var_sim}_.png"
                fig.savefig(file, dpi=300)

    vars_sim = ["C_q_ss"]
    crops = ["winter-wheat", "silage-corn"]
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
                gdf.plot(column="value", ax=ax, cmap="OrRd", vmin=10, vmax=120, legend=True, legend_kwds={"label": "$NO_3$ [mg/l]", "shrink": 0.68})
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
                ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
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
                gdf.plot(column="value", ax=ax, cmap="OrRd", norm=norm, legend=True, legend_kwds={"label": "$NO_3$ [mg/l]", "shrink": 0.5, "ticks": [10, 25, 37.5, 50, 75], "format": mpl.ticker.FixedFormatter(['<10', '25', '37.5', '50', '>75'])})
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
                ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
                fig.tight_layout()
                file = Path(__file__).parent / "figures" / f"map_{crop}_{fertilisation_intensity}Nfert_{var_sim}_.png"
                fig.savefig(file, dpi=300)


if __name__ == "__main__":
    main()