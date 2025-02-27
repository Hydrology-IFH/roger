import os
from pathlib import Path
import xarray as xr
import numpy as onp
import matplotlib.pyplot as plt
import contextily as ctx
import yaml
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

_lab_unit2 = {
    "theta": r"$\theta$ [-]",
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

    # load configuration file
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        config = yaml.safe_load(file)

    # load the model parameters
    param_file = base_path / "parameters.nc"
    ds_param = xr.open_dataset(param_file)
    lu_id = ds_param["lanu"].values
    lu_id = onp.where(lu_id < 0, onp.nan, lu_id)
    lu_id = onp.where(lu_id == 5, 0.5, 1.5)

    soil_depth = ds_param["GRUND"].values
    soil_depth = onp.where(ds_param["lanu"].values == 5, soil_depth/100, onp.nan)

    ufc = ds_param["NFK"].values
    ufc = onp.where(ds_param["lanu"].values == 5, ufc/100, onp.nan)

    gw_depth = ds_param["gwfa_gew"].values
    gw_depth = onp.where(ds_param["lanu"].values == 5, gw_depth/100, onp.nan)

    bounds = [0, 1, 2]
    norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
    fig, ax = plt.subplots(figsize=(6,4))
    extent = (0, config["ny"]*25/1000, 0, config["nx"]*25/1000)
    im = ax.imshow(lu_id, cmap='Dark2', aspect='equal', norm=norm, zorder=2, extent=extent)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_ticks(ticks=[0.5, 1.5], labels=['crops', 'no crops'])
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.grid(zorder=-1)
    plt.tight_layout()
    file = base_path_figs / "crop_map.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    extent = (0, config["ny"]*25/1000, 0, config["nx"]*25/1000)
    im = ax.imshow(ds_param["dgm_gew"].values, cmap='terrain', aspect='equal', zorder=2, extent=extent)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='[m a.s.l.]')
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.grid(zorder=-1)
    plt.tight_layout()
    file = base_path_figs / "topography.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(6,4))
    extent = (0, config["ny"]*25/1000, 0, config["nx"]*25/1000)
    im = ax.imshow(soil_depth, cmap='Blues', aspect='equal', zorder=2, extent=extent)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='soil depth [m]')
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.grid(zorder=-1)
    plt.tight_layout()
    file = base_path_figs / "soil_depth_cropland.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    extent = (0, config["ny"]*25/1000, 0, config["nx"]*25/1000)
    im = ax.imshow(gw_depth, cmap='Blues_r', aspect='equal', zorder=2, extent=extent, vmin=0, vmax=20)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label='groundwater depth [m]')
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.grid(zorder=-1)
    plt.tight_layout()
    file = base_path_figs / "gw_depth_cropland.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6,4))
    # extent = (396331.5, 396331.5 + config["ny"]*25, 5325918.5, 5325918.5 - config["nx"]*25)
    extent = (396331.5, 396331.5 + config["ny"]*25, 5325918.5 - config["nx"]*25, 5325918.5)
    im = ax.imshow(ufc, cmap='Blues', aspect='equal', zorder=2, extent=extent)
    # ctx.add_basemap(ax, crs=ds_param["spatial_ref"].crs_wkt, zoom=19)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, label=r'$\theta_{ufc}$ [-]')
    plt.xlabel('')
    plt.ylabel('')
    plt.grid(zorder=-1)
    plt.tight_layout()
    file = base_path_figs / "ufc_cropland.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    return


if __name__ == "__main__":
    main()