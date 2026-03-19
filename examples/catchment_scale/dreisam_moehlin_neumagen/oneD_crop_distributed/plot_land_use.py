import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import numpy as onp
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import click
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.title_fontsize'] = 9
sns.set_style("ticks")
sns.plotting_context("paper", font_scale=1, rc={'font.size': 8.0,
                                                'axes.labelsize': 9.0,
                                                'axes.titlesize': 8.0,
                                                'xtick.labelsize': 8.0,
                                                'ytick.labelsize': 8.0,
                                                'legend.fontsize': 8.0,
                                                'legend.title_fontsize': 9.0})


_dict_lu_names = {
    0: "Siedlungsfläche",
    5: "Landwirtschaft",
    6: "Landwirtschaft",
    7: "Landwirtschaft",
    8: "Grünland",
    9: "Sonstiges",
    10: "Wald",
    11: "Wald",
    12: "Wald",
    13: "Feuchtgebiet",
    14: "Oberflaechengewaesser",
    20: "Oberflaechengewaesser",
}

_dict_crop_names = {
    81: "ext. + int.\nGrünland",
    82: "ext. + int.\nGrünland",
    510: "Sonderkulturen",
    513: "Sonderkulturen",
    525: "Mais",
    539: "Mais",
    541: "Soja",
    542: "Sommergetreide",
    550: "Sonderkulturen",
    556: "Wintergetreide",
    557: "Wintergetreide",
    559: "Winterraps",
    563: "Zuckerrübe",
    580: "Kleegras",
    589: "Miscanthus",

}

@click.command("main")
def main():
    base_path = Path(__file__).parent
    # directory of output
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load the model parameters
    param_file = base_path / "parameters_roger.nc"
    ds_param = xr.open_dataset(param_file)
    lu_ids = ds_param["lanu"].values
    maskCatch = (ds_param["maskCatch"].values == 1)
    lu_ids[~maskCatch] = -9999
    mask_crops = (lu_ids == 5)

    # load mask of porous aquifer
    file = base_path / "input" / "mask_porous_25m.tif"
    with rasterio.open(file) as src:
        _mask_porous = src.read(1)
    mask_porous = _mask_porous == 1

    # get unique land use types and their counts
    unique_lu_ids, counts = onp.unique(lu_ids[mask_porous], return_counts=True)
    # make data frame with land use type and count
    land_use_df = pd.DataFrame({"Land Use ID": unique_lu_ids, "Count": counts})
    # map land use IDs to names
    land_use_df["Land Use Type"] = land_use_df["Land Use ID"].map(_dict_lu_names)
    # sort by land use type    
    land_use_df = land_use_df.sort_values("Land Use Type")
    # group by land use type and sum counts
    land_use_df = land_use_df.groupby("Land Use Type", as_index=False).sum()
    # get percentage of each land use type
    land_use_df["Percentage"] = land_use_df["Count"] / land_use_df["Count"].sum() * 100
    
    # plot bar chart for Siedlungsflaeche, Landwirtschaft, Gruenland und Wald
    plt.figure(figsize=(5, 2))
    sns.barplot(x="Land Use Type", y="Percentage", data=land_use_df, order=["Landwirtschaft", "Siedlungsfläche", "Wald", "Grünland"], hue="Percentage", palette="PuBuGn", dodge=False)
    plt.ylabel("Flächenanteil [%]", fontsize=11)
    plt.xlabel("", fontsize=11)
    # remove legend    
    plt.legend().remove()
    plt.tight_layout()
    file = base_path_figs / "land_use_distribution.png"
    plt.savefig(file, dpi=300, bbox_inches="tight")

    file = base_path / "input" / "crops_2018-2022.nc"
    ds_cr_2018_2022 = xr.open_dataset(file)
    crops = ds_cr_2018_2022["Nutzcode"].values
    for i in range(crops.shape[0]):
        crops[i, ~mask_crops] = onp.nan
        crops[i, ~mask_porous] = onp.nan

    # get unique land use types and their counts
    unique_crop_ids, counts = onp.unique(crops, return_counts=True)
    # make data frame with land use type and count
    crop_df = pd.DataFrame({"Crop ID": unique_crop_ids, "Count": counts})
    # remove rows with crop ID 599 and nan
    crop_df = crop_df[(crop_df["Crop ID"] != 599) & (crop_df["Crop ID"].notna())]
    # map land use IDs to names
    crop_df["Crop Type"] = crop_df["Crop ID"].map(_dict_crop_names)
    # sort by land use type    
    crop_df = crop_df.sort_values("Crop Type")
    # group by land use type and sum counts
    crop_df = crop_df.groupby("Crop Type", as_index=False).sum()
    # get percentage of each land use type
    crop_df["Percentage"] = crop_df["Count"] / crop_df["Count"].sum() * 100

    # plot bar chart for Grünland, Mais, Sommergetreide, Wintergetreide, Winterraps, Zuckerruebe, Kleegras und Sonderkulturen
    plt.figure(figsize=(5, 2))
    sns.barplot(x="Crop Type", y="Percentage", data=crop_df, order=["Mais", "ext. + int.\nGrünland", "Wintergetreide", "Sonderkulturen", "Sommergetreide",], hue="Percentage", palette="Greens", dodge=False)
    plt.ylabel("Flächenanteil\n an landw. Fläche [%]", fontsize=11)
    plt.xlabel("", fontsize=11)
    # remove legend    
    plt.legend().remove()
    plt.tight_layout()
    file = base_path_figs / "crop_distribution.png"
    plt.savefig(file, dpi=300, bbox_inches="tight")

    return


if __name__ == "__main__":
    main()