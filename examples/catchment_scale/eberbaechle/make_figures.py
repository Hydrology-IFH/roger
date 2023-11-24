import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import matplotlib.dates as mdates
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
x1 = 76
y1 = 43
t_dry = 1079
t_wet = 10
t_drywet = 217
t_wetdry = 234

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
# lu_id = ds_params_hm["lu_id"].values
# ds_params_hm["lu_id"].isel(x=x1, y=y1)
# ds_params_hm["theta_pwp"].isel(x=x1, y=y1)
# ds_params_hm["z_soil"].isel(x=x1, y=y1)

# import pandas as pd
# df = pd.DataFrame(index=date_hm, columns=["theta", "transp"])
# for t in range(len(date_hm)):
#     vals = onp.where(ds_hm["theta"].isel(Time=t).values <= -9999, onp.nan, ds_hm["theta"].isel(Time=t).values)
#     df.iloc[t, 0] = onp.nanmean(vals)
#     vals = onp.where(ds_hm["transp"].isel(Time=t).values <= -9999, onp.nan, ds_hm["transp"].isel(Time=t).values)
#     df.iloc[t, 1] = onp.nanmax(vals)
# df.to_csv(base_path_figs / "theta_transp.csv", sep=";", index=True)

# load transport simulations
states_tm_file = base_path / "svat_oxygen18_distributed" / "output" / "SVAT18O.nc"
ds_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
# assign date
days_tm = ds_tm["Time"].values
date_tm = num2date(days_tm, units="days since 2019-10-31", calendar="standard", only_use_cftime_datetimes=False)
ds_tm = ds_tm.assign_coords(Time=("Time", date_tm))

# plot spatially distributed soil moisture and median travel time of percolation at different dates
fig, axes = plt.subplots(2, 2, figsize=(6, 4))
axes[0, 0].imshow(
    onp.where(ds_hm["theta"].isel(Time=t_dry).values <= -9999, onp.nan, ds_hm["theta"].isel(Time=t_dry).values),
    extent=(0, 80 * 25, 0, 53 * 25),
    cmap="viridis_r",
    vmin=0.15,
    vmax=0.3,
)
axes[0, 0].grid(zorder=0)
axes[0, 0].set_xlabel("[m]")
axes[0, 0].set_ylabel("[m]")
axes[0, 0].set_title(str(ds_hm["Time"].values[t_dry]).split("T")[0])
axes[0, 1].imshow(
    onp.where(ds_hm["theta"].isel(Time=t_wet).values <= -9999, onp.nan, ds_hm["theta"].isel(Time=t_wet).values),
    extent=(0, 80 * 25, 0, 53 * 25),
    cmap="viridis_r",
    vmin=0.15,
    vmax=0.3,
)
axes[0, 1].grid(zorder=0)
axes[0, 1].set_xlabel("[m]")
axes[0, 1].set_xlabel("[m]")
axes[0, 1].set_title(str(ds_hm["Time"].values[t_wet]).split("T")[0])
cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
norm = mpl.colors.Normalize(vmin=0.15, vmax=0.3)
axl1 = fig.add_axes([0.85, 0.6, 0.02, 0.3])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="vertical", ticks=[0.15, 0.2, 0.25, 0.3])
cb1.ax.set_yticklabels(["<0.15", "0.2", "0.25", ">0.3"])
cb1.set_label(r"$\theta$ [-]")

axes[1, 0].imshow(
    ds_tm["ttavg_q_ss"].isel(Time=t_dry).values, extent=(0, 80 * 25, 0, 53 * 25), cmap="viridis", vmin=1, vmax=1000
)
axes[1, 0].grid(zorder=0)
axes[1, 0].set_xlabel("[m]")
axes[1, 0].set_ylabel("[m]")
axes[1, 0].set_title(str(ds_tm["Time"].values[t_dry]).split("T")[0])
axes[1, 1].imshow(
    ds_tm["ttavg_q_ss"].isel(Time=t_wet).values, extent=(0, 80 * 25, 0, 53 * 25), cmap="viridis", vmin=1, vmax=1000
)
axes[1, 1].grid(zorder=0)
axes[1, 1].set_xlabel("[m]")
axes[1, 1].set_title(str(ds_tm["Time"].values[t_wet]).split("T")[0])
cmap = copy.copy(mpl.colormaps.get_cmap("viridis"))
norm = mpl.colors.Normalize(vmin=1, vmax=1000)
axl2 = fig.add_axes([0.85, 0.13, 0.02, 0.3])
cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm, orientation="vertical", ticks=[1, 200, 400, 600, 800, 1000])
cb2.ax.set_yticklabels(["1", "200", "400", "600", "800", ">1000"])
cb2.ax.invert_yaxis()
cb2.set_label(r"$\overline{TT}_{PERC}$ [days]")
fig.subplots_adjust(left=0.1, bottom=0.1, top=0.94, right=0.8, wspace=0.3, hspace=0.3)
file = base_path_figs / "theta_and_ttavg.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "theta_and_ttavg.pdf"
fig.savefig(file, dpi=300)

# plot fluxes and isotopic signals of a single grid cell
fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].bar(date_hm, ds_hm["prec"].isel(x=x1, y=y1).values, width=0.1, color="blue", align="edge", edgecolor="blue")
axes[0, 0].set_ylabel(r"PRECIP [mm/day]")
axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[0, 0].tick_params(axis="x", labelrotation=60)

axes[0, 1].plot(date_tm, ds_tm["C_iso_in"].isel(x=x1, y=y1).values, "-", color="blue", lw=1)
axes[0, 1].set_xlim(date_tm[0], date_tm[-1])
axes[0, 1].set_ylabel(r"$\delta^{18}$O [‰]")
axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[0, 1].tick_params(axis="x", labelrotation=60)

axes[1, 0].plot(date_hm, ds_hm["transp"].isel(x=x1, y=y1).values, "-", color="#31a354", lw=1)
axes[1, 0].plot(date_hm, ds_hm["evap_soil"].isel(x=x1, y=y1).values, "--", color="#c2e699", lw=0.8)
axes[1, 0].set_xlim(date_hm[0], date_hm[-1])
axes[1, 0].set_ylabel(r"ET [mm/day]")
axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[1, 0].tick_params(axis="x", labelrotation=60)

axes[2, 0].axvline(date_hm[t_dry], color="red", alpha=0.5)
axes[2, 0].axvline(date_hm[t_wetdry], color="red", alpha=0.5)
axes[2, 0].axvline(date_hm[t_drywet], color="red", alpha=0.5)
axes[2, 0].axvline(date_hm[t_wet], color="red", alpha=0.5)
axes[2, 0].plot(date_hm, ds_hm["theta"].isel(x=x1, y=y1).values, "-", color="brown", lw=1)
axes[2, 0].set_xlim(date_hm[0], date_hm[-1])
axes[2, 0].set_ylabel(r"$\theta$ [-]")
axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[2, 0].tick_params(axis="x", labelrotation=60)

axes[3, 0].axvline(date_hm[t_dry], color="red", alpha=0.5)
axes[3, 0].axvline(date_hm[t_wetdry], color="red", alpha=0.5)
axes[3, 0].axvline(date_hm[t_drywet], color="red", alpha=0.5)
axes[3, 0].axvline(date_hm[t_wet], color="red", alpha=0.5)
axes[3, 0].plot(date_hm, ds_hm["q_ss"].isel(x=x1, y=y1).values, "-", color="grey", lw=1)
axes[3, 0].set_xlim(date_hm[0], date_hm[-1])
axes[3, 0].set_ylabel(r"PERC [mm/day]")
axes[3, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[3, 0].set_xlabel(r"Time [year-month]")
axes[3, 0].tick_params(axis="x", labelrotation=60)

axes[1, 1].plot(date_tm, ds_tm["C_iso_transp"].isel(x=x1, y=y1).values, "-", color="#31a354", lw=1)
axes[1, 1].plot(date_tm, ds_tm["C_iso_evap_soil"].isel(x=x1, y=y1).values, "--", color="#c2e699", lw=0.8)
axes[1, 1].set_xlim(date_tm[0], date_tm[-1])
axes[1, 1].set_ylim(-15, -5)
axes[1, 1].set_ylabel(r"$\delta^{18}$O [‰]")
axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[1, 1].tick_params(axis="x", labelrotation=60)

axes[2, 1].axvline(date_hm[t_dry], color="red", alpha=0.5)
axes[2, 1].axvline(date_hm[t_wetdry], color="red", alpha=0.5)
axes[2, 1].axvline(date_hm[t_drywet], color="red", alpha=0.5)
axes[2, 1].axvline(date_hm[t_wet], color="red", alpha=0.5)
axes[2, 1].plot(date_tm, ds_tm["C_iso_s"].isel(x=x1, y=y1).values, "-", color="brown", lw=1)
axes[2, 1].set_xlim(date_tm[0], date_tm[-1])
axes[2, 1].set_ylim(-15, -5)
axes[2, 1].set_ylabel(r"$\delta^{18}$O [‰]")
axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[2, 1].tick_params(axis="x", labelrotation=60)

axes[3, 1].axvline(date_hm[t_dry], color="red", alpha=0.5)
axes[3, 1].axvline(date_hm[t_wetdry], color="red", alpha=0.5)
axes[3, 1].axvline(date_hm[t_drywet], color="red", alpha=0.5)
axes[3, 1].axvline(date_hm[t_wet], color="red", alpha=0.5)
axes[3, 1].plot(date_tm, ds_tm["C_iso_q_ss"].isel(x=x1, y=y1).values, "-", color="grey", lw=1)
axes[3, 1].set_xlim(date_tm[0], date_tm[-1])
axes[3, 1].set_ylim(-15, -5)
axes[3, 1].set_ylabel(r"$\delta^{18}$O [‰]")
axes[3, 1].set_xlabel(r"Time [year-month]")
axes[3, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[3, 1].tick_params(axis="x", labelrotation=60)

axes[0, 2].set_axis_off()

axes[1, 2].fill_between(
    date_tm,
    ds_tm["tt10_transp"].isel(x=x1, y=y1).values,
    ds_tm["tt90_transp"].isel(x=x1, y=y1).values,
    color="#31a354",
    edgecolor=None,
    alpha=0.2,
)
axes[1, 2].plot(date_tm, ds_tm["ttavg_transp"].isel(x=x1, y=y1).values, "-", color="#31a354", lw=1)
axes[1, 2].set_xlim(date_tm[0], date_tm[-1])
axes[1, 2].set_ylabel(r"age [days]")
axes[1, 2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[1, 2].tick_params(axis="x", labelrotation=60)

axes[2, 2].axvline(date_hm[t_dry], color="red", alpha=0.5)
axes[2, 2].axvline(date_hm[t_wetdry], color="red", alpha=0.5)
axes[2, 2].axvline(date_hm[t_drywet], color="red", alpha=0.5)
axes[2, 2].axvline(date_hm[t_wet], color="red", alpha=0.5)
axes[2, 2].fill_between(
    date_tm,
    ds_tm["rt10_s"].isel(x=x1, y=y1).values,
    ds_tm["rt90_s"].isel(x=x1, y=y1).values,
    color="brown",
    edgecolor=None,
    alpha=0.2,
)
axes[2, 2].plot(date_tm, ds_tm["rtavg_s"].isel(x=x1, y=y1).values, "-", color="brown", lw=1)
axes[2, 2].set_xlim(date_tm[0], date_tm[-1])
axes[2, 2].set_ylabel(r"age [days]")
axes[2, 2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[2, 2].tick_params(axis="x", labelrotation=60)

axes[3, 2].axvline(date_hm[t_dry], color="red", alpha=0.5)
axes[3, 2].axvline(date_hm[t_wetdry], color="red", alpha=0.5)
axes[3, 2].axvline(date_hm[t_drywet], color="red", alpha=0.5)
axes[3, 2].axvline(date_hm[t_wet], color="red", alpha=0.5)
axes[3, 2].fill_between(
    date_tm,
    ds_tm["tt10_q_ss"].isel(x=x1, y=y1).values,
    ds_tm["tt90_q_ss"].isel(x=x1, y=y1).values,
    color="grey",
    edgecolor=None,
    alpha=0.2,
)
axes[3, 2].plot(date_tm, ds_tm["ttavg_q_ss"].isel(x=x1, y=y1).values, "-", color="grey", lw=1)
axes[3, 2].set_xlim(date_tm[0], date_tm[-1])
axes[3, 2].set_ylabel(r"age [days]")
axes[3, 2].set_xlabel(r"Time [year-month]")
axes[3, 2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[3, 2].tick_params(axis="x", labelrotation=60)

fig.subplots_adjust(left=0.1, bottom=0.13, top=0.95, right=0.98, hspace=0.6, wspace=0.42)
file = base_path_figs / "ts_single_grid_cell.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "ts_single_grid_cell.pdf"
fig.savefig(file, dpi=300)

# plot flux distributions, isotopic distributions and age distributions of all grid cells at wet and dry conditions
fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].hist(
    ds_hm["evap_soil"].isel(Time=t_wet).values.flatten(), color="#c2e699", bins=10, range=(0, 1), align="mid"
)
axes[0, 0].set_xlabel(r"$EVAP_{soil}$ [mm/day]")
axes[0, 0].set_ylabel("# grid cells")

axes[1, 0].hist(ds_hm["transp"].isel(Time=t_wet).values.flatten(), color="#31a354", bins=40, range=(0, 4), align="mid")
axes[1, 0].set_xlabel(r"$TRANSP$ [mm/day]")
axes[1, 0].set_ylabel("# grid cells")

axes[2, 0].hist(ds_hm["theta"].isel(Time=t_wet).values.flatten(), color="brown", bins=20, range=(0.2, 0.4), align="mid")
axes[2, 0].set_xlabel(r"$\theta$ [-]")
axes[2, 0].set_ylabel("# grid cells")

axes[3, 0].hist(ds_hm["q_ss"].isel(Time=t_wet).values.flatten(), color="grey", bins=30, range=(0, 15), align="mid")
axes[3, 0].set_xlabel(r"$PERC$ [mm/day]")
axes[3, 0].set_ylabel("# grid cells")

axes[0, 1].hist(
    ds_tm["C_iso_evap_soil"].isel(Time=t_wet).values.flatten(), color="#c2e699", bins=24, range=(-12, -6), align="mid"
)
axes[0, 1].set_xlabel(r"$\delta^{18}$O [‰]")

axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_wet).values.flatten(), color="#31a354", bins=24, range=(-12, -6), align="mid"
)
axes[1, 1].set_xlabel(r"$\delta^{18}$O [‰]")

axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_wet).values.flatten(), color="brown", bins=24, range=(-12, -6), align="mid"
)
axes[2, 1].set_xlabel(r"$\delta^{18}$O [‰]")

axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_wet).values.flatten(), color="grey", bins=24, range=(-12, -6), align="mid"
)
axes[3, 1].set_xlabel(r"$\delta^{18}$O [‰]")

axes[0, 2].set_axis_off()

axes[1, 2].hist(
    ds_tm["tt50_transp"].isel(Time=t_wet).values.flatten(), color="#31a354", bins=50, range=(0, 600), align="mid"
)
axes[1, 2].set_xlabel(r"$\overline{TT_{TRANSP}}$ [days]")

axes[2, 2].hist(ds_tm["rt50_s"].isel(Time=t_wet).values.flatten(), color="brown", bins=50, range=(0, 600), align="mid")
axes[2, 2].set_xlabel(r"$\overline{RT_{\theta}}$ [days]")

axes[3, 2].hist(
    ds_tm["tt50_q_ss"].isel(Time=t_wet).values.flatten(), color="grey", bins=50, range=(0, 600), align="mid"
)
axes[3, 2].set_xlabel(r"$\overline{TT_{PERC}}$ [days]")

fig.tight_layout()
file = base_path_figs / "dist_states_wet.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "dist_states_wet.pdf"
fig.savefig(file, dpi=300)

fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].hist(ds_hm["evap_soil"].isel(Time=t_dry).values.flatten(), color="#c2e699", bins=10, range=(0, 1))
axes[0, 0].set_xlabel(r"$EVAP_{soil}$ [mm/day]")
axes[0, 0].set_ylabel("# grid cells")

axes[1, 0].hist(ds_hm["transp"].isel(Time=t_dry).values.flatten(), color="#31a354", bins=40, range=(0, 4))
axes[1, 0].set_xlabel(r"$TRANSP$ [mm/day]")
axes[1, 0].set_ylabel("# grid cells")

axes[2, 0].hist(ds_hm["theta"].isel(Time=t_dry).values.flatten(), color="brown", bins=20, range=(0.2, 0.4))
axes[2, 0].set_xlabel(r"$\theta$ [-]")
axes[2, 0].set_ylabel("# grid cells")

axes[3, 0].hist(ds_hm["q_ss"].isel(Time=t_dry).values.flatten(), color="grey", bins=30, range=(0, 15))
axes[3, 0].set_xlabel(r"$PERC$ [mm/day]")
axes[3, 0].set_ylabel("# grid cells")

axes[0, 1].hist(ds_tm["C_iso_evap_soil"].isel(Time=t_dry).values.flatten(), color="#c2e699", bins=24, range=(-12, -6))
axes[0, 1].set_xlabel(r"$\delta^{18}$O [‰]")

axes[1, 1].hist(ds_tm["C_iso_transp"].isel(Time=t_dry).values.flatten(), color="#31a354", bins=24, range=(-12, -6))
axes[1, 1].set_xlabel(r"$\delta^{18}$O [‰]")

axes[2, 1].hist(ds_tm["C_iso_s"].isel(Time=t_dry).values.flatten(), color="brown", bins=24, range=(-12, -6))
axes[2, 1].set_xlabel(r"$\delta^{18}$O [‰]")

axes[3, 1].hist(ds_tm["C_iso_q_ss"].isel(Time=t_dry).values.flatten(), color="grey", bins=24, range=(-12, -6))
axes[3, 1].set_xlabel(r"$\delta^{18}$O [‰]")

axes[0, 2].set_axis_off()

axes[1, 2].hist(ds_tm["tt50_transp"].isel(Time=t_dry).values.flatten(), color="#31a354", bins=50, range=(0, 600))
axes[1, 2].set_xlabel(r"$\overline{TT_{TRANSP}}$ [days]")

axes[2, 2].hist(ds_tm["rt50_s"].isel(Time=t_dry).values.flatten(), color="brown", bins=50, range=(0, 600))
axes[2, 2].set_xlabel(r"$\overline{RT_{\theta}}$ [days]")

axes[3, 2].hist(ds_tm["ttavg_q_ss"].isel(Time=t_dry).values.flatten(), color="grey", bins=50, range=(0, 600))
axes[3, 2].set_xlabel(r"$\overline{TT_{PERC}}$ [days]")

fig.tight_layout()
file = base_path_figs / "dist_states_dry.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "dist_states_dry.pdf"
fig.savefig(file, dpi=300)

fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].hist(
    ds_hm["evap_soil"].isel(Time=t_wetdry).values.flatten(),
    color="#6baed6",
    bins=10,
    range=(0, 1),
    align="mid",
    alpha=0.5,
)
axes[0, 0].hist(
    ds_hm["evap_soil"].isel(Time=t_wetdry).values.flatten(),
    color="#08306b",
    bins=10,
    range=(0, 1),
    align="mid",
    alpha=0.5,
)
axes[0, 0].set_xlabel(r"$EVAP_{soil}$ [mm/day]")
axes[0, 0].set_ylabel("# grid cells")
axes[0, 0].text(
    0.95,
    1.12,
    "(a)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[0, 0].transAxes,
)

axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_wetdry).values.flatten(), color="#6baed6", bins=40, range=(0, 4), align="mid", alpha=0.5
)
axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_wet).values.flatten(), color="#08306b", bins=40, range=(0, 4), align="mid", alpha=0.5
)
axes[1, 0].set_xlabel(r"$TRANSP$ [mm/day]")
axes[1, 0].set_ylabel("# grid cells")
axes[1, 0].text(
    0.95,
    1.12,
    "(b)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1, 0].transAxes,
)

axes[2, 0].hist(
    ds_hm["theta"].isel(Time=t_wetdry).values.flatten(),
    color="#6baed6",
    bins=20,
    range=(0.2, 0.4),
    align="mid",
    alpha=0.5,
)
axes[2, 0].hist(
    ds_hm["theta"].isel(Time=t_wet).values.flatten(), color="#08306b", bins=20, range=(0.2, 0.4), align="mid", alpha=0.5
)
axes[2, 0].set_xlabel(r"$\theta$ [-]")
axes[2, 0].set_ylabel("# grid cells")
axes[2, 0].text(
    0.95,
    1.12,
    "(c)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2, 0].transAxes,
)

axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_wetdry).values.flatten(), color="#6baed6", bins=30, range=(0, 15), align="mid", alpha=0.5
)
axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_wet).values.flatten(), color="#08306b", bins=30, range=(0, 15), align="mid", alpha=0.5
)
axes[3, 0].set_xlabel(r"$PERC$ [mm/day]")
axes[3, 0].set_ylabel("# grid cells")
axes[3, 0].text(
    0.95,
    1.12,
    "(d)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[3, 0].transAxes,
)

axes[0, 1].hist(
    ds_tm["C_iso_evap_soil"].isel(Time=t_wetdry).values.flatten(),
    color="#6baed6",
    bins=24,
    range=(-12, -6),
    align="mid",
    alpha=0.5,
)
axes[0, 1].hist(
    ds_tm["C_iso_evap_soil"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=24,
    range=(-12, -6),
    align="mid",
    alpha=0.5,
)
axes[0, 1].set_xlabel(r"$\delta^{18}$$O_{EVAP_{soil}}$ [‰]")
axes[0, 1].text(
    0.95,
    1.12,
    "(e)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[0, 1].transAxes,
)

axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_wetdry).values.flatten(),
    color="#6baed6",
    bins=24,
    range=(-12, -6),
    align="mid",
    alpha=0.5,
)
axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=24,
    range=(-12, -6),
    align="mid",
    alpha=0.5,
)
axes[1, 1].set_xlabel(r"$\delta^{18}$$O_{TRANSP}$ [‰]")
axes[1, 1].text(
    0.95,
    1.12,
    "(f)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1, 1].transAxes,
)

axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_wetdry).values.flatten(),
    color="#6baed6",
    bins=24,
    range=(-12, -6),
    align="mid",
    alpha=0.5,
)
axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=24,
    range=(-12, -6),
    align="mid",
    alpha=0.5,
)
axes[2, 1].set_xlabel(r"$\delta^{18}$$O_{\theta}$ [‰]")
axes[2, 1].text(
    0.95,
    1.12,
    "(g)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2, 1].transAxes,
)

axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#6baed6",
    bins=24,
    range=(-12, -6),
    align="mid",
    alpha=0.5,
)
axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=24,
    range=(-12, -6),
    align="mid",
    alpha=0.5,
)
axes[3, 1].set_xlabel(r"$\delta^{18}$$O_{PERC}$ [‰]")
axes[3, 1].text(
    0.95,
    1.12,
    "(h)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[3, 1].transAxes,
)

axes[0, 2].set_axis_off()

axes[1, 2].hist(
    ds_tm["ttavg_transp"].isel(Time=t_wetdry).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(0, 600),
    align="mid",
    alpha=0.5,
    label=r"wet condtions",
)
axes[1, 2].hist(
    ds_tm["ttavg_transp"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(0, 600),
    align="mid",
    alpha=0.5,
    label=r"dry condtions",
)
axes[1, 2].set_xlabel(r"$\overline{TT_{TRANSP}}$ [days]")
axes[1, 2].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.6))
axes[1, 2].text(
    0.95,
    1.12,
    "(i)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1, 2].transAxes,
)

axes[2, 2].hist(
    ds_tm["rtavg_s"].isel(Time=t_wetdry).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(0, 600),
    align="mid",
    alpha=0.5,
)
axes[2, 2].hist(
    ds_tm["rtavg_s"].isel(Time=t_wet).values.flatten(), color="#08306b", bins=50, range=(0, 600), align="mid", alpha=0.5
)
axes[2, 2].set_xlabel(r"$\overline{RT_{\theta}}$ [days]")
axes[2, 2].text(
    0.95,
    1.12,
    "(j)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2, 2].transAxes,
)

axes[3, 2].hist(
    ds_tm["ttavg_q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(0, 600),
    align="mid",
    alpha=0.5,
)
axes[3, 2].hist(
    ds_tm["ttavg_q_ss"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(0, 600),
    align="mid",
    alpha=0.5,
)
axes[3, 2].set_xlabel(r"$\overline{TT_{PERC}}$ [days]")
axes[3, 2].text(
    0.95,
    1.12,
    "(k)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[3, 2].transAxes,
)

fig.tight_layout()
file = base_path_figs / "dist_states_wet_dry.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "dist_states_wet_dry.pdf"
fig.savefig(file, dpi=300)

# plot cumulated distributions of fluxes, isotopic signal and median age of all grid cells at dry, normal and wet conditions
fig, axes = plt.subplots(4, 3, figsize=(6, 5), sharey=True)
axes[0, 0].hist(
    ds_hm["evap_soil"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(0, 2),
    histtype="step",
    cumulative=True,
    lw=2.0,
    ls="--",
)
axes[0, 0].hist(
    ds_hm["evap_soil"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(0, 2),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[0, 0].hist(
    ds_hm["evap_soil"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(0, 2),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[0, 0].set_xlim(0, 2)
axes[0, 0].set_xlabel(r"$EVAP_{soil}$ [mm/day]")
axes[0, 0].set_ylabel("# grid cells")
axes[0, 0].text(
    0.95,
    1.12,
    "(a)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[0, 0].transAxes,
)

axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=40,
    range=(0, 4),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=40,
    range=(0, 4),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=40,
    range=(0, 4),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=40,
    range=(0, 4),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[1, 0].set_xlim(0, 4)
axes[1, 0].set_xlabel(r"$TRANSP$ [mm/day]")
axes[1, 0].set_ylabel("# grid cells")
axes[1, 0].text(
    0.95,
    1.12,
    "(b)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1, 0].transAxes,
)
axes[2, 0].hist(
    ds_hm["theta"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(0.1, 0.5),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[2, 0].hist(
    ds_hm["theta"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(0.1, 0.5),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[2, 0].hist(
    ds_hm["theta"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(0.1, 0.5),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[2, 0].hist(
    ds_hm["theta"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=50,
    range=(0.1, 0.5),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[2, 0].set_xlim(0.15, 0.45)
axes[2, 0].set_xlabel(r"$\theta$ [-]")
axes[2, 0].set_ylabel("# grid cells")
axes[2, 0].text(
    0.95,
    1.12,
    "(c)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2, 0].transAxes,
)
axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(0, 10),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(0, 10),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(0, 10),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=50,
    range=(0, 10),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[3, 0].set_xlim(0, 10)
axes[3, 0].set_xlabel(r"$PERC$ [mm/day]")
axes[3, 0].set_ylabel("# grid cells")
axes[3, 0].text(
    0.95,
    1.12,
    "(d)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[3, 0].transAxes,
)

axes[3, 0].fill([0, 0.4, 0.4, 0], [0, 0, 288, 288], color="grey", alpha=0.5)

axes[0, 1].hist(
    ds_tm["C_iso_evap_soil"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=2.0,
    ls="--",
)
axes[0, 1].hist(
    ds_tm["C_iso_evap_soil"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[0, 1].hist(
    ds_tm["C_iso_evap_soil"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[0, 1].set_xlim(-15, -5)
axes[0, 1].set_xlabel(r"$\delta^{18}$$O_{EVAP_{soil}}$ [‰]")
axes[0, 1].text(
    0.95,
    1.12,
    "(e)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[0, 1].transAxes,
)
axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=1.5,
    ls="-.",
)
axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=1.0,
)
axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[1, 1].set_xlim(-15, -5)
axes[1, 1].set_xlabel(r"$\delta^{18}$$O_{TRANSP}$ [‰]")
axes[1, 1].text(
    0.95,
    1.12,
    "(f)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1, 1].transAxes,
)
axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[2, 1].set_xlim(-15, -5)
axes[2, 1].set_xlabel(r"$\delta^{18}$$O_{\theta}$ [‰]")
axes[2, 1].text(
    0.95,
    1.12,
    "(g)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2, 1].transAxes,
)
axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=50,
    range=(-15, -5),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[3, 1].set_xlim(-15, -5)
axes[3, 1].set_xlabel(r"$\delta^{18}$$O_{PERC}$ [‰]")
axes[3, 1].text(
    0.95,
    1.12,
    "(h)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[3, 1].transAxes,
)

axes[0, 2].set_axis_off()

axes[1, 2].hist(
    ds_tm["ttavg_transp"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[1, 2].hist(
    ds_tm["ttavg_transp"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[1, 2].hist(
    ds_tm["ttavg_transp"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[1, 2].hist(
    ds_tm["ttavg_transp"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[1, 2].set_xlim(0, 1000)
axes[1, 2].set_xlabel(r"$\overline{TT_{TRANSP}}$ [days]")
axes[1, 2].text(
    0.95,
    1.12,
    "(i)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1, 2].transAxes,
)
axes[1, 2].plot([], [], color="#fed976", label="dry condtions", lw=0.5, ls="--")
axes[1, 2].plot([], [], color="#feb24c", label="transition to dry condtions", lw=1.0, ls="-.")
axes[1, 2].plot([], [], color="#6baed6", label="transition to wet condtions", lw=1.5)
axes[1, 2].plot([], [], color="#08306b", label="wet condtions", lw=2.0)
lines, labels = axes[1, 2].get_legend_handles_labels()
fig.legend(lines, labels, loc="upper right", frameon=False, bbox_to_anchor=(0.98, 1.005))

axes[2, 2].hist(
    ds_tm["rtavg_s"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[2, 2].hist(
    ds_tm["rtavg_s"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[2, 2].hist(
    ds_tm["rtavg_s"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[2, 2].hist(
    ds_tm["rtavg_s"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[2, 2].set_xlim(0, 1000)
axes[2, 2].set_xlabel(r"$\overline{RT}$ [days]")
axes[2, 2].text(
    0.95,
    1.12,
    "(j)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2, 2].transAxes,
)
axes[3, 2].hist(
    ds_tm["ttavg_q_ss"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[3, 2].hist(
    ds_tm["ttavg_q_ss"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[3, 2].hist(
    ds_tm["ttavg_q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[3, 2].hist(
    ds_tm["ttavg_q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=1000,
    range=(0, 1000),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[3, 2].set_xlim(0, 1000)
axes[3, 2].set_xlabel(r"$\overline{TT_{PERC}}$ [days]")
axes[3, 2].text(
    0.95,
    1.12,
    "(k)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[3, 2].transAxes,
)

inset = fig.add_axes([0.225, 0.15, 0.11, 0.08])
inset.hist(
    ds_hm["q_ss"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=40,
    range=(0, 0.4),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
inset.hist(
    ds_hm["q_ss"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=40,
    range=(0, 0.4),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
inset.hist(
    ds_hm["q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=40,
    range=(0, 0.4),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
inset.hist(
    ds_hm["q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=40,
    range=(0, 0.4),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
inset.set_xlim(0, 0.4)

fig.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.98, hspace=0.6, wspace=0.25)
file = base_path_figs / "cumulated_dist_states_dry_normal_wet.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "cumulated_dist_states_dry_normal_wet.pdf"
fig.savefig(file, dpi=300)

fig, axes = plt.subplots(figsize=(3, 2))
axes.hist(
    ds_hm["q_ss"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=50,
    range=(0, 0.5),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes.hist(
    ds_hm["q_ss"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(0, 0.5),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes.hist(
    ds_hm["q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(0, 0.5),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes.hist(
    ds_hm["q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=50,
    range=(0, 0.5),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes.set_xlim(0, 0.5)
axes.set_xlabel(r"$PERC$ [mm/day]")
axes.set_ylabel("# grid cells")
fig.tight_layout()
file = base_path_figs / "cumulated_dist_perc_inset.png"
fig.savefig(file, dpi=300)


# # make GIF for Online-Documentation
# for t in range(1, 1000):
#     fig, axes = plt.subplots(3, 3, figsize=(6, 6))

#     axes[0, 0].set_axis_off()
#     axes[0, 2].set_axis_off()

#     axes[0, 1].imshow(onp.where(ds_hm["prec"].isel(Time=t).values <= -9999, onp.nan, ds_hm["prec"].isel(Time=t).values), extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=0, vmax=20)
#     axes[0, 1].grid(zorder=0)
#     axes[0, 1].set_xlabel("")
#     axes[0, 1].set_ylabel("[m]")
#     axes[0, 1].set_title(str(ds_hm["Time"].values[t]).split("T")[0], weight='bold')
#     cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
#     norm = mpl.colors.Normalize(vmin=0, vmax=20)
#     axl1 = fig.add_axes([0.62, 0.73, 0.01, 0.15])
#     cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="vertical", ticks=[0, 10, 20])
#     cb1.ax.set_yticklabels(["0", "10", ">20"])
#     cb1.set_label("PRECIP [mm/day]")

#     axes[1, 0].imshow(onp.where(ds_hm["transp"].isel(Time=t).values <= -9999, onp.nan, ds_hm["transp"].isel(Time=t).values), extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=0, vmax=5)
#     axes[1, 0].grid(zorder=0)
#     axes[1, 0].set_xlabel("")
#     axes[1, 0].set_ylabel("[m]")
#     cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
#     norm = mpl.colors.Normalize(vmin=0, vmax=5)
#     axl1 = fig.add_axes([0.1, 0.66, 0.2, 0.01])
#     cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="horizontal", ticks=[0, 2.5, 5])
#     cb1.ax.set_xticklabels(["0", "2.5", ">5"])
#     cb1.set_label("TRANSP [mm/day]")

#     axes[1, 1].imshow(onp.where(ds_hm["theta"].isel(Time=t).values <= -9999, onp.nan, ds_hm["theta"].isel(Time=t).values), extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=0.15, vmax=0.3)
#     axes[1, 1].grid(zorder=0)
#     axes[1, 1].set_xlabel("")
#     axes[1, 1].set_ylabel("")
#     axes[1, 1].set_yticklabels([])
#     cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
#     norm = mpl.colors.Normalize(vmin=0.15, vmax=0.3)
#     axl1 = fig.add_axes([0.38, 0.66, 0.2, 0.01])
#     cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="horizontal", ticks=[0.15, 0.2, 0.25, 0.3])
#     cb1.ax.set_xticklabels(["<0.15", "0.2", "0.25", ">0.3"])
#     cb1.set_label(r"$\theta$ [-]")

#     axes[1, 2].imshow(onp.where(ds_hm["q_ss"].isel(Time=t).values <= -9999, onp.nan, ds_hm["q_ss"].isel(Time=t).values), extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=0, vmax=20)
#     axes[1, 2].grid(zorder=0)
#     axes[1, 2].set_xlabel("")
#     axes[1, 2].set_ylabel("")
#     axes[1, 2].set_yticklabels([])
#     cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
#     norm = mpl.colors.Normalize(vmin=0, vmax=20)
#     axl1 = fig.add_axes([0.65, 0.66, 0.2, 0.01])
#     cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="horizontal", ticks=[0, 10, 20])
#     cb1.ax.set_xticklabels(["0", "10", ">20"])
#     cb1.set_label("PERC [mm/day]")

#     axes[2, 0].imshow(ds_tm["ttavg_transp"].isel(Time=t).values, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=1, vmax=1000)
#     axes[2, 0].grid(zorder=0)
#     axes[2, 0].set_xlabel("[m]")
#     axes[2, 0].set_ylabel("[m]")
#     cmap = copy.copy(mpl.colormaps.get_cmap("viridis"))
#     norm = mpl.colors.Normalize(vmin=1, vmax=1000)
#     axl2 = fig.add_axes([0.1, 0.38, 0.2, 0.01])
#     cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm, orientation="horizontal", ticks=[1, 500, 1000])
#     cb2.ax.set_xticklabels(["1", "500", ">1000"])
#     cb2.ax.invert_yaxis()
#     cb2.set_label(r"$\overline{TT}_{TRANSP}$ [days]")

#     axes[2, 1].imshow(ds_tm["rtavg_s"].isel(Time=t).values, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=1, vmax=1000)
#     axes[2, 1].grid(zorder=0)
#     axes[2, 1].set_xlabel("[m]")
#     axes[2, 1].set_yticklabels([])
#     cmap = copy.copy(mpl.colormaps.get_cmap("viridis"))
#     norm = mpl.colors.Normalize(vmin=1, vmax=1000)
#     axl2 = fig.add_axes([0.38, 0.38, 0.2, 0.01])
#     cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm, orientation="horizontal", ticks=[1, 500, 1000])
#     cb2.ax.set_xticklabels(["1", "500", ">1000"])
#     cb2.ax.invert_yaxis()
#     cb2.set_label(r"$\overline{RT}_{SOIL}$ [days]")

#     axes[2, 2].imshow(ds_tm["ttavg_q_ss"].isel(Time=t).values, extent=(0, 80*25, 0, 53*25), cmap="viridis_r", vmin=1, vmax=1000)
#     axes[2, 2].grid(zorder=0)
#     axes[2, 2].set_xlabel("[m]")
#     axes[2, 2].set_yticklabels([])
#     cmap = copy.copy(mpl.colormaps.get_cmap("viridis"))
#     norm = mpl.colors.Normalize(vmin=1, vmax=1000)
#     axl2 = fig.add_axes([0.65, 0.38, 0.2, 0.01])
#     cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm, orientation="horizontal", ticks=[1, 500, 1000])
#     cb2.ax.set_xticklabels(["1", "500", ">1000"])
#     cb2.ax.invert_yaxis()
#     cb2.set_label(r"$\overline{TT}_{PERC}$ [days]")
#     fig.subplots_adjust(left=0.1, bottom=0.1, top=0.94, right=0.85, wspace=0.35, hspace=0.1)
#     file = base_path_figs / f"fluxes_theta_and_tt_rt_{t}.png"
#     fig.savefig(file, dpi=300)
#     plt.close("all")

# images_data = []
# #load 10 images
# for t in range(1, 1000):
#     data = imageio.v2.imread(base_path_figs / f"fluxes_theta_and_tt_rt_{t}.png")
#     images_data.append(data)

# file = base_path_figs / "fluxes_theta_and_tt_rt.gif"
# imageio.mimwrite(file, images_data, format='.gif', fps=12)

# plt.close("all")
