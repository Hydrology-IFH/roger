import os
from pathlib import Path
import xarray as xr
import numpy as onp
import matplotlib.pyplot as plt
import yaml
import click
from PIL import Image
import imageio
import copy
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

    # load hydrologic simulations
    states_hm_file = base_path_output / f"{config['identifier']}.nc"
    ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf")
    # load catchment 
    params_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_file, engine="h5netcdf")
    mask = onp.isfinite(ds_params["dgm"].values)

    # assign date
    days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

    # make GIF for Online-Documentation
    frames = []
    for t in range(1, 364):
        fig, axes = plt.subplots(3, 1, figsize=(4, 6))
        axes[0].imshow(onp.where(mask, ds_sim['prec'].isel(Time=t).values.T, onp.nan), origin="lower", cmap='viridis_r', vmin=0, vmax=20)
        axes[0].set_xticks(onp.arange(-config['dx']/2, config['nx'], config['dx']))
        axes[0].set_yticks(onp.arange(-config['dy']/2, config['ny'], config['dy']))
        axes[0].set_xticklabels(onp.arange(0, config['nx'], config['dx']) * config['dx'])
        axes[0].set_yticklabels(onp.arange(0, config['ny'], config['dy']) * config['dy'])
        axes[0].set_xlabel('')
        axes[0].set_ylabel('Distance in y-direction [m]')
        cmap = copy.copy(plt.cm.get_cmap('viridis_r'))
        norm = mpl.colors.Normalize(vmin=0, vmax=20)
        axl1 = fig.add_axes([0.81, 0.68, 0.02, 0.2])
        cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                        orientation='vertical',
                                        ticks=[0, 10, 20])
        cb1.ax.set_yticklabels(['0', '10', '20'])
        cb1.set_label(r'Precipitation [mm/day]')

        axes[1].imshow(onp.where(mask, ds_sim['theta'].isel(Time=t).values.T, onp.nan), origin="lower", cmap='viridis_r', vmin=0.15, vmax=0.35)
        axes[1].set_xticks(onp.arange(-config['dx']/2, config['nx'], config['dx']))
        axes[1].set_yticks(onp.arange(-config['dy']/2, config['ny'], config['dy']))
        axes[1].set_xticklabels(onp.arange(0, config['nx'], config['dx']) * config['dx'])
        axes[1].set_yticklabels(onp.arange(0, config['ny'], config['dy']) * config['dy'])
        axes[1].set_xlabel('')
        axes[1].set_ylabel('Distance in y-direction [m]')
        cmap = copy.copy(plt.cm.get_cmap('viridis_r'))
        norm = mpl.colors.Normalize(vmin=0.15, vmax=0.35)
        axl1 = fig.add_axes([0.81, 0.4, 0.02, 0.2])
        cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                        orientation='vertical',
                                        ticks=[0.15, 0.25, 0.35])
        cb1.ax.set_yticklabels(['<0.15', '0.25', '>0.35'])
        cb1.set_label(r'Soil water content [-]')

        axes[2].imshow(onp.where(mask, ds_sim['q_ss'].isel(Time=t).values.T, onp.nan), origin="lower", cmap='viridis_r', vmin=0, vmax=5)
        axes[2].set_xticks(onp.arange(-config['dx']/2, config['nx'], config['dx']))
        axes[2].set_yticks(onp.arange(-config['dy']/2, config['ny'], config['dy']))
        axes[2].set_xticklabels(onp.arange(0, config['nx'], config['dx']) * config['dx'])
        axes[2].set_yticklabels(onp.arange(0, config['ny'], config['dy']) * config['dy'])
        axes[2].set_xlabel('Distance in x-direction [m]')
        axes[2].set_ylabel('Distance in y-direction [m]')
        cmap = copy.copy(plt.cm.get_cmap('viridis_r'))
        norm = mpl.colors.Normalize(vmin=0, vmax=10)
        axl1 = fig.add_axes([0.81, 0.12, 0.02, 0.2])
        cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                        orientation='vertical',
                                        ticks=[0, 2.5, 5])
        cb1.ax.set_yticklabels(['0', '2.5', '>5'])
        cb1.set_label(r'Percolation [mm/day]')
        fig.suptitle(f't = {t}d', fontsize=9)
        fig.subplots_adjust(wspace=0, hspace=0.2, left=0.2, right=0.8, bottom=0.1, top=0.9)
        file = base_path_figs / f"t{t}.png"
        fig.savefig(file, dpi=250)
        plt.close('all')
        img = imageio.v2.imread(file)
        frames.append(img)
        
    file = base_path_figs / "prec_theta_perc.gif"
    imageio.mimsave(file,
                    frames,
                    fps = 7)
    return


if __name__ == "__main__":
    main()