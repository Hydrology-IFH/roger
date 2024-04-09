import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import numpy as onp
import copy
import imageio
import matplotlib as mpl
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

nrows = 53
ncols = 80

base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# load hydrologic simulations
states_hm_file = base_path / "svat_distributed" / "output" / "SVAT.nc"
ds_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
# assign date
days_hm = ds_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
date_hm = num2date(
    days_hm,
    units=f"days since {ds_hm['Time'].attrs['time_origin']}",
    calendar="standard",
    only_use_cftime_datetimes=False,
)
ds_hm = ds_hm.assign_coords(Time=("Time", date_hm))

# load hydrologic model parameters
params_hm_file = base_path / "svat_distributed" / "parameters.nc"
ds_params_hm = xr.open_dataset(params_hm_file, engine="h5netcdf")

# load transport simulations
states_tm_file = base_path / "svat_oxygen18_distributed" / "output" / "SVAT18O.nc"
ds_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
# assign date
days_tm = ds_tm["Time"].values
date_tm = num2date(days_tm, units="days since 2019-10-31", calendar="standard", only_use_cftime_datetimes=False)
ds_tm = ds_tm.assign_coords(Time=("Time", date_tm))

# make GIF for Online-Documentation
for t in range(1, 1097):
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    axes[0, 0].set_axis_off()
    axes[0, 2].set_axis_off()

    axes[0, 1].imshow(onp.where(ds_hm["prec"].isel(Time=t).values <= -9999, onp.nan, ds_hm["prec"].isel(Time=t).values).T, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=0, vmax=20)
    axes[0, 1].grid(zorder=0)
    axes[0, 1].set_xlabel("")
    axes[0, 1].set_ylabel("[m]")
    axes[0, 1].set_title(str(ds_hm["Time"].values[t]).split("T")[0], weight='bold')
    cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
    norm = mpl.colors.Normalize(vmin=0, vmax=20)
    axl1 = fig.add_axes([0.62, 0.73, 0.01, 0.15])
    cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="vertical", ticks=[0, 10, 20])
    cb1.ax.set_yticklabels(["0", "10", ">20"])
    cb1.set_label("PRECIP [mm/day]")

    axes[1, 0].imshow(onp.where(ds_hm["transp"].isel(Time=t).values <= -9999, onp.nan, ds_hm["transp"].isel(Time=t).values).T, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=0, vmax=5)
    axes[1, 0].grid(zorder=0)
    axes[1, 0].set_xlabel("")
    axes[1, 0].set_ylabel("[m]")
    cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
    norm = mpl.colors.Normalize(vmin=0, vmax=5)
    axl1 = fig.add_axes([0.1, 0.66, 0.2, 0.01])
    cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="horizontal", ticks=[0, 2.5, 5])
    cb1.ax.set_xticklabels(["0", "2.5", ">5"])
    cb1.set_label("TRANSP [mm/day]")

    axes[1, 1].imshow(onp.where(ds_hm["theta"].isel(Time=t).values <= -9999, onp.nan, ds_hm["theta"].isel(Time=t).values).T, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=0.1, vmax=0.3)
    axes[1, 1].grid(zorder=0)
    axes[1, 1].set_xlabel("")
    axes[1, 1].set_ylabel("")
    axes[1, 1].set_yticklabels([])
    cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
    norm = mpl.colors.Normalize(vmin=0.1, vmax=0.4)
    axl1 = fig.add_axes([0.38, 0.66, 0.2, 0.01])
    cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="horizontal", ticks=[0.1, 0.2, 0.3, 0.4])
    cb1.ax.set_xticklabels(["<0.1", "0.2", "0.3", ">0.4"])
    cb1.set_label(r"$\theta$ [-]")

    axes[1, 2].imshow(onp.where(ds_hm["q_ss"].isel(Time=t).values <= -9999, onp.nan, ds_hm["q_ss"].isel(Time=t).values).T, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=0, vmax=20)
    axes[1, 2].grid(zorder=0)
    axes[1, 2].set_xlabel("")
    axes[1, 2].set_ylabel("")
    axes[1, 2].set_yticklabels([])
    cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
    norm = mpl.colors.Normalize(vmin=0, vmax=20)
    axl1 = fig.add_axes([0.65, 0.66, 0.2, 0.01])
    cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="horizontal", ticks=[0, 10, 20])
    cb1.ax.set_xticklabels(["0", "10", ">20"])
    cb1.set_label("PERC [mm/day]")

    axes[2, 0].imshow(ds_tm["ttavg_transp"].isel(Time=t).values.T, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=1, vmax=200)
    axes[2, 0].grid(zorder=0)
    axes[2, 0].set_xlabel("[m]")
    axes[2, 0].set_ylabel("[m]")
    cmap = copy.copy(mpl.colormaps.get_cmap("viridis"))
    norm = mpl.colors.Normalize(vmin=1, vmax=200)
    axl2 = fig.add_axes([0.1, 0.38, 0.2, 0.01])
    cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm, orientation="horizontal", ticks=[1, 100, 200])
    cb2.ax.set_xticklabels(["1", "100", ">200"])
    cb2.ax.invert_yaxis()
    cb2.set_label(r"$\overline{TT}_{TRANSP}$ [days]")

    axes[2, 1].imshow(ds_tm["rtavg_s"].isel(Time=t).values.T, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=1, vmax=200)
    axes[2, 1].grid(zorder=0)
    axes[2, 1].set_xlabel("[m]")
    axes[2, 1].set_yticklabels([])
    cmap = copy.copy(mpl.colormaps.get_cmap("viridis"))
    norm = mpl.colors.Normalize(vmin=1, vmax=300)
    axl2 = fig.add_axes([0.38, 0.38, 0.2, 0.01])
    cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm, orientation="horizontal", ticks=[1, 100, 200, 300])
    cb2.ax.set_xticklabels(["1", "100", "200", ">300"])
    cb2.ax.invert_yaxis()
    cb2.set_label(r"$\overline{RT}_{\theta}$ [days]")

    axes[2, 2].imshow(ds_tm["ttavg_q_ss"].isel(Time=t).values.T, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=1, vmax=200)
    axes[2, 2].grid(zorder=0)
    axes[2, 2].set_xlabel("[m]")
    axes[2, 2].set_yticklabels([])
    cmap = copy.copy(mpl.colormaps.get_cmap("viridis"))
    norm = mpl.colors.Normalize(vmin=1, vmax=400)
    axl2 = fig.add_axes([0.65, 0.38, 0.2, 0.01])
    cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm, orientation="horizontal", ticks=[1, 100, 200, 300, 400])
    cb2.ax.set_xticklabels(["1", "100", "200", "300", ">400"])
    cb2.ax.invert_yaxis()
    cb2.set_label(r"$\overline{TT}_{PERC}$ [days]")
    fig.subplots_adjust(left=0.1, bottom=0.1, top=0.94, right=0.85, wspace=0.35, hspace=0.1)
    file = base_path_figs / f"fluxes_theta_and_tt_rt_{t}.png"
    fig.savefig(file, dpi=250)
    plt.close("all")

images_data = []
#load images
for t in range(1, 1097):
    data = imageio.v2.imread(base_path_figs / f"fluxes_theta_and_tt_rt_{t}.png")
    images_data.append(data)

file = base_path_figs / "fluxes_theta_and_tt_rt.gif"
imageio.mimwrite(file, images_data, format='.gif', fps=10)

plt.close("all")
