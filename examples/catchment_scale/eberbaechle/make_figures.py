import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import matplotlib.dates as mdates
from datetime import datetime
import numpy as onp
import pandas as pd
import copy
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

# mpl.rcParams["font.size"] = 10
# mpl.rcParams["axes.titlesize"] = 10
# mpl.rcParams["axes.labelsize"] = 11
# mpl.rcParams["xtick.labelsize"] = 10
# mpl.rcParams["ytick.labelsize"] = 10
# mpl.rcParams["legend.fontsize"] = 10
# mpl.rcParams["legend.title_fontsize"] = 11
# sns.set_style("ticks")
# sns.plotting_context(
#     "paper",
#     font_scale=1,
#     rc={
#         "font.size": 10.0,
#         "axes.labelsize": 11.0,
#         "axes.titlesize": 10.0,
#         "xtick.labelsize": 10.0,
#         "ytick.labelsize": 10.0,
#         "legend.fontsize": 10.0,
#         "legend.title_fontsize": 11.0,
#     },
# )

nrows = 53
ncols = 80
x1 = 12
y1 = 7
t_dry = 298 # 24.08.2020
t_wet = 127 # 06.03.2020
t_drywet = 352 # 11.03.2021
t_wetdry = 849 # 26.02.2022

date_dry = "$24^{th}$ Aug 2020"
date_wet = "$6^{th}$ Mar 2020"
date_drywet = "$11^{th}$ Mar 2021"
date_wetdry = "$26^{th}$ Feb 2022"

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

print(ds_params_hm["lu_id"].isel(x=x1, y=y1).values)
print(ds_params_hm["z_soil"].isel(x=x1, y=y1).values)

# df = pd.DataFrame(
#     index=date_hm,
#     columns=[
#         "theta_min",
#         "theta_50",
#         "theta_avg",
#         "theta_max",
#         "evap",
#         "transp",
#         "perc_50",
#         "perc_avg",
#         "perc_max",
#         "day",
#         "rank",
#         "prob",
#     ],
# )
# for t in range(len(date_hm)):
#     vals = onp.where(ds_hm["theta"].isel(Time=t).values <= -9999, onp.nan, ds_hm["theta"].isel(Time=t).values)
#     df.iloc[t, 0] = onp.nanmin(vals)
#     df.iloc[t, 1] = onp.nanmedian(vals)
#     df.iloc[t, 2] = onp.nanmean(vals)
#     df.iloc[t, 3] = onp.nanmax(vals)
#     vals = onp.where(ds_hm["evap_soil"].isel(Time=t).values <= -9999, onp.nan, ds_hm["evap_soil"].isel(Time=t).values)
#     df.iloc[t, 4] = onp.nanmax(vals)
#     vals = onp.where(ds_hm["transp"].isel(Time=t).values <= -9999, onp.nan, ds_hm["transp"].isel(Time=t).values)
#     df.iloc[t, 5] = onp.nanmax(vals)
#     vals = onp.where(ds_hm["q_ss"].isel(Time=t).values <= -9999, onp.nan, ds_hm["q_ss"].isel(Time=t).values)
#     df.iloc[t, 6] = onp.nanmedian(vals)
#     df.iloc[t, 7] = onp.nanmean(vals)
#     df.iloc[t, 8] = onp.nanmax(vals)
# df.loc[:, "day"] = range(len(date_hm))
# df = df.sort_values(by=["theta_50"])
# df.loc[:, "rank"] = range(len(date_hm))
# df.loc[:, "prob"] = df.loc[:, "rank"] / len(date_hm)
# df = df.sort_values(by=["day"])
# df.to_csv(base_path_figs / f"theta_et_perc.csv", sep=";", index=True)

# load transport simulations
states_tm_file = base_path / "svat_oxygen18_distributed" / "output" / "SVAT18O.nc"
ds_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
# assign date
days_tm = ds_tm["Time"].values
date_tm = num2date(days_tm, units="days since 2019-10-31", calendar="standard", only_use_cftime_datetimes=False)
ds_tm = ds_tm.assign_coords(Time=("Time", date_tm))

# plot spatially distributed soil moisture and average travel time of percolation at different dates
fig, axes = plt.subplots(2, 2, figsize=(6, 4))
axes[0, 0].imshow(
    onp.where(ds_hm["theta"].isel(Time=t_dry).values <= -9999, onp.nan, ds_hm["theta"].isel(Time=t_dry).values).T,
    extent=(0, 80 * 25, 0, 53 * 25),
    cmap="viridis_r",
    vmin=0.1,
    vmax=0.4,
)
axes[0, 0].grid(zorder=0)
axes[0, 0].set_xlabel("[m]")
axes[0, 0].set_ylabel("[m]")
axes[0, 0].set_title(str(ds_hm["Time"].values[t_dry]).split("T")[0])
axes[0, 1].imshow(
    onp.where(ds_hm["theta"].isel(Time=t_wet).values <= -9999, onp.nan, ds_hm["theta"].isel(Time=t_wet).values).T,
    extent=(0, 80 * 25, 0, 53 * 25),
    cmap="viridis_r",
    vmin=0.1,
    vmax=0.4,
)
axes[0, 1].grid(zorder=0)
axes[0, 1].set_xlabel("[m]")
axes[0, 1].set_xlabel("[m]")
axes[0, 1].set_title(str(ds_hm["Time"].values[t_wet]).split("T")[0])
cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
norm = mpl.colors.Normalize(vmin=0.1, vmax=0.4)
axl1 = fig.add_axes([0.85, 0.6, 0.02, 0.3])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="vertical", ticks=[0.1, 0.2, 0.3, 0.4])
cb1.ax.set_yticklabels(["<0.1", "0.2", "0.3", ">0.4"])
cb1.set_label(r"$\theta$ [-]")

axes[1, 0].imshow(
    ds_tm["ttavg_q_ss"].isel(Time=t_dry).values.T, extent=(0, 80 * 25, 0, 53 * 25), cmap="viridis", vmin=1, vmax=400
)
axes[1, 0].grid(zorder=0)
axes[1, 0].set_xlabel("[m]")
axes[1, 0].set_ylabel("[m]")
axes[1, 0].set_title(str(ds_tm["Time"].values[t_dry]).split("T")[0])
axes[1, 1].imshow(
    ds_tm["ttavg_q_ss"].isel(Time=t_wet).values.T, extent=(0, 80 * 25, 0, 53 * 25), cmap="viridis", vmin=1, vmax=400
)
axes[1, 1].grid(zorder=0)
axes[1, 1].set_xlabel("[m]")
axes[1, 1].set_title(str(ds_tm["Time"].values[t_wet]).split("T")[0])
cmap = copy.copy(mpl.colormaps.get_cmap("viridis"))
norm = mpl.colors.Normalize(vmin=1, vmax=400)
axl2 = fig.add_axes([0.85, 0.13, 0.02, 0.3])
cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm, orientation="vertical", ticks=[1, 100, 200, 300, 400])
cb2.ax.set_yticklabels(["1", "100", "200", "300", ">400"])
cb2.ax.invert_yaxis()
cb2.set_label(r"$\overline{TT}_{PERC}$ [days]")
fig.subplots_adjust(left=0.1, bottom=0.1, top=0.94, right=0.8, wspace=0.3, hspace=0.3)
file = base_path_figs / "theta_and_ttavg.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "theta_and_ttavg.pdf"
fig.savefig(file, dpi=300)

# plot spatially distributed percolation and average travel time of percolation at different dates
fig, axes = plt.subplots(2, 2, figsize=(6, 4))
axes[0, 0].imshow(
    onp.where(ds_hm["q_ss"].isel(Time=t_dry).values <= -9999, onp.nan, ds_hm["q_ss"].isel(Time=t_dry).values),
    extent=(0, 80 * 25, 0, 53 * 25),
    cmap="viridis_r",
    vmin=0,
    vmax=50,
)
axes[0, 0].grid(zorder=0)
axes[0, 0].set_xlabel("[m]")
axes[0, 0].set_ylabel("[m]")
axes[0, 0].set_title(str(ds_hm["Time"].values[t_dry]).split("T")[0])
axes[0, 1].imshow(
    onp.where(ds_hm["q_ss"].isel(Time=t_wet).values <= -9999, onp.nan, ds_hm["q_ss"].isel(Time=t_wet).values),
    extent=(0, 80 * 25, 0, 53 * 25),
    cmap="viridis_r",
    vmin=0,
    vmax=50,
)
axes[0, 1].grid(zorder=0)
axes[0, 1].set_xlabel("[m]")
axes[0, 1].set_xlabel("[m]")
axes[0, 1].set_title(str(ds_hm["Time"].values[t_wet]).split("T")[0])
cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
norm = mpl.colors.Normalize(vmin=0, vmax=50)
axl1 = fig.add_axes([0.85, 0.6, 0.02, 0.3])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="vertical", ticks=[0, 25, 50])
cb1.ax.set_yticklabels(["0", "25", ">50"])
cb1.set_label(r"$PERC$ [mm/day]")

axes[1, 0].imshow(
    ds_tm["ttavg_q_ss"].isel(Time=t_dry).values, extent=(0, 80 * 25, 0, 53 * 25), cmap="viridis", vmin=1, vmax=200
)
axes[1, 0].grid(zorder=0)
axes[1, 0].set_xlabel("[m]")
axes[1, 0].set_ylabel("[m]")
axes[1, 0].set_title(str(ds_tm["Time"].values[t_dry]).split("T")[0])
axes[1, 1].imshow(
    ds_tm["ttavg_q_ss"].isel(Time=t_wet).values, extent=(0, 80 * 25, 0, 53 * 25), cmap="viridis", vmin=1, vmax=200
)
axes[1, 1].grid(zorder=0)
axes[1, 1].set_xlabel("[m]")
axes[1, 1].set_title(str(ds_tm["Time"].values[t_wet]).split("T")[0])
cmap = copy.copy(mpl.colormaps.get_cmap("viridis"))
norm = mpl.colors.Normalize(vmin=1, vmax=200)
axl2 = fig.add_axes([0.85, 0.13, 0.02, 0.3])
cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm, orientation="vertical", ticks=[1, 100, 200])
cb2.ax.set_yticklabels(["1", "100", ">200"])
cb2.ax.invert_yaxis()
cb2.set_label(r"$\overline{TT}_{PERC}$ [days]")
fig.subplots_adjust(left=0.1, bottom=0.1, top=0.94, right=0.8, wspace=0.3, hspace=0.3)
file = base_path_figs / "perc_and_ttavg.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "perc_and_ttavg.pdf"
fig.savefig(file, dpi=300)

fig, axes = plt.subplots(2, 2, figsize=(6, 4))
axes[0, 0].imshow(
    onp.where(ds_hm["q_ss"].isel(Time=t_dry).values <= -9999, onp.nan, ds_hm["q_ss"].isel(Time=t_dry).values),
    extent=(0, 80 * 25, 0, 53 * 25),
    cmap="viridis_r",
    vmin=0,
    vmax=0.1,
)
axes[0, 0].grid(zorder=0)
axes[0, 0].set_xlabel("[m]")
axes[0, 0].set_ylabel("[m]")
axes[0, 0].set_title(str(ds_hm["Time"].values[t_dry]).split("T")[0])
axes[0, 1].imshow(
    onp.where(ds_hm["q_ss"].isel(Time=t_wet).values <= -9999, onp.nan, ds_hm["q_ss"].isel(Time=t_wet).values),
    extent=(0, 80 * 25, 0, 53 * 25),
    cmap="viridis_r",
    vmin=0,
    vmax=0.1,
)
axes[0, 1].grid(zorder=0)
axes[0, 1].set_xlabel("[m]")
axes[0, 1].set_xlabel("[m]")
axes[0, 1].set_title(str(ds_hm["Time"].values[t_wet]).split("T")[0])
cmap = copy.copy(mpl.colormaps.get_cmap("viridis_r"))
norm = mpl.colors.Normalize(vmin=0, vmax=0.1)
axl1 = fig.add_axes([0.85, 0.6, 0.02, 0.3])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm, orientation="vertical", ticks=[0, 0.05, 0.1])
cb1.ax.set_yticklabels(["0", "0.05", ">0.1"])
cb1.set_label(r"$PERC$ [mm/day]")

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
file = base_path_figs / "perc1_and_ttavg.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "perc1_and_ttavg.pdf"
fig.savefig(file, dpi=300)


# plot fluxes and isotopic signals of a single grid cell
fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].bar(date_hm, ds_hm["prec"].isel(x=x1, y=y1).values, width=0.1, color="blue", align="edge", edgecolor="blue")
axes[0, 0].set_xlim(date_hm[0], date_hm[-1])
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
axes[1, 0].set_ylim(
    0,
)
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
axes[3, 0].set_ylim(
    0,
)
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
axes[1, 2].plot(date_tm, ds_tm["tt50_transp"].isel(x=x1, y=y1).values, "-", color="#31a354", lw=1)
axes[1, 2].set_xlim(date_tm[0], date_tm[-1])
axes[1, 2].set_ylabel(r"age [days]")
axes[1, 2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[1, 2].tick_params(axis="x", labelrotation=60)
axes[1, 2].set_ylim(
    0, 300
)

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
axes[2, 2].plot(date_tm, ds_tm["rt50_s"].isel(x=x1, y=y1).values, "-", color="brown", lw=1)
axes[2, 2].set_xlim(date_tm[0], date_tm[-1])
axes[2, 2].set_ylabel(r"age [days]")
axes[2, 2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[2, 2].tick_params(axis="x", labelrotation=60)
axes[2, 2].set_ylim(
    0, 1000
)
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
axes[3, 2].plot(date_tm, ds_tm["tt50_q_ss"].isel(x=x1, y=y1).values, "-", color="grey", lw=1)
axes[3, 2].set_xlim(date_tm[0], date_tm[-1])
axes[3, 2].set_ylabel(r"age [days]")
axes[3, 2].set_xlabel(r"Time [year-month]")
axes[3, 2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes[3, 2].tick_params(axis="x", labelrotation=60)
axes[3, 2].set_ylim(0, 1000)

fig.subplots_adjust(left=0.1, bottom=0.13, top=0.95, right=0.98, hspace=0.6, wspace=0.42)
file = base_path_figs / "ts_single_grid_cell.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "ts_single_grid_cell.pdf"
fig.savefig(file, dpi=300)


# plot percolation of a single grid cell
fig, axes = plt.subplots(1, 1, figsize=(6, 2))
axes.plot(date_hm, ds_hm["q_ss"].isel(x=x1, y=y1).values, "-", color="grey", lw=1)
axes.set_xlim(date_hm[0], date_hm[-1])
axes.set_ylim(
    0, 5
)
axes.set_ylabel(r"PERC [mm/day]")
axes.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
axes.set_xlabel(r"Time [year-month]")
axes.tick_params(axis="x", labelrotation=60)

fig.tight_layout()
file = base_path_figs / "ts_perc_single_grid_cell.png"
fig.savefig(file, dpi=300)
file = base_path_figs / "ts_perc_single_grid_cell.pdf"
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
    ds_hm["evap_soil"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=10,
    range=(0, 1),
    align="mid",
    alpha=0.5,
)
axes[0, 0].hist(
    ds_hm["evap_soil"].isel(Time=t_wet).values.flatten(),
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
    1.14,
    "(a)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[0, 0].transAxes,
)

axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_dry).values.flatten(), color="#fed976", bins=40, range=(0, 4), align="mid", alpha=0.5
)
axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_wet).values.flatten(), color="#08306b", bins=40, range=(0, 4), align="mid", alpha=0.5
)
axes[1, 0].set_xlabel(r"$TRANSP$ [mm/day]")
axes[1, 0].set_ylabel("# grid cells")
axes[1, 0].text(
    0.95,
    1.14,
    "(b)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1, 0].transAxes,
)

axes[2, 0].hist(
    ds_hm["theta"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=30,
    range=(0.15, 0.45),
    align="mid",
    alpha=0.5,
)
axes[2, 0].hist(
    ds_hm["theta"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=30,
    range=(0.15, 0.45),
    align="mid",
    alpha=0.5,
)
axes[2, 0].set_xlabel(r"$\theta$ [-]")
axes[2, 0].set_ylabel("# grid cells")
axes[2, 0].text(
    0.95,
    1.14,
    "(c)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2, 0].transAxes,
)

axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_dry).values.flatten(), color="#fed976", bins=30, range=(0, 15), align="mid", alpha=0.5
)
axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_wet).values.flatten(), color="#08306b", bins=30, range=(0, 15), align="mid", alpha=0.5
)
axes[3, 0].set_xlabel(r"$PERC$ [mm/day]")
axes[3, 0].set_ylabel("# grid cells")
axes[3, 0].text(
    0.95,
    1.14,
    "(d)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[3, 0].transAxes,
)

axes[0, 1].hist(
    ds_tm["C_iso_evap_soil"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
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
    1.14,
    "(e)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[0, 1].transAxes,
)

axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
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
    1.14,
    "(f)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1, 1].transAxes,
)

axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
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
    1.14,
    "(g)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2, 1].transAxes,
)

axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
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
    1.14,
    "(h)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[3, 1].transAxes,
)

axes[0, 2].set_axis_off()

axes[1, 2].hist(
    ds_tm["ttavg_transp"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
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
axes[1, 2].legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.8))
axes[1, 2].text(
    0.95,
    1.14,
    "(i)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1, 2].transAxes,
)

axes[2, 2].hist(
    ds_tm["rtavg_s"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
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
    1.14,
    "(j)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2, 2].transAxes,
)

axes[3, 2].hist(
    ds_tm["ttavg_q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
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
    1.14,
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
    bins=55,
    range=(0, 1.1),
    histtype="step",
    cumulative=True,
    lw=2.0,
    ls="--",
)
axes[0, 0].hist(
    ds_hm["evap_soil"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=55,
    range=(0, 1.1),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[0, 0].hist(
    ds_hm["evap_soil"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=55,
    range=(0, 1.1),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[0, 0].set_xlim(0, 1.0)
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
    bins=41,
    range=(0, 4.1),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=41,
    range=(0, 4.1),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=41,
    range=(0, 4.1),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[1, 0].hist(
    ds_hm["transp"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=41,
    range=(0, 4.1),
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
    bins=101,
    range=(0, 101),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=101,
    range=(0, 101),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=101,
    range=(0, 101),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[3, 0].hist(
    ds_hm["q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=101,
    range=(0, 101),
    histtype="step",
    cumulative=True,
    lw=0.5,
    ls="--",
)
axes[3, 0].set_xlim(0, 100)
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
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=2.0,
    ls="--",
)
axes[0, 1].hist(
    ds_tm["C_iso_evap_soil"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[0, 1].hist(
    ds_tm["C_iso_evap_soil"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(-15, -4.9),
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
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=1.5,
    ls="-.",
)
axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=1.0,
)
axes[1, 1].hist(
    ds_tm["C_iso_transp"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=50,
    range=(-15, -4.9),
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
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[2, 1].hist(
    ds_tm["C_iso_s"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=50,
    range=(-15, -4.9),
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
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=50,
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=50,
    range=(-15, -4.9),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
axes[3, 1].hist(
    ds_tm["C_iso_q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=50,
    range=(-15, -4.9),
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
axes[1, 2].set_xlim(0, 400)
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
axes[1, 2].plot([], [], color="#fed976", label="dry condtions\n (%s)" % (date_dry), lw=0.5, ls="--")
axes[1, 2].plot([], [], color="#feb24c", label="transition to dry condtions\n (%s)" % (date_wetdry), lw=1.0, ls="-.")
axes[1, 2].plot([], [], color="#6baed6", label="transition to wet condtions\n (%s)" % (date_drywet), lw=1.5)
axes[1, 2].plot([], [], color="#08306b", label="wet condtions\n (%s)" % (date_wet), lw=2.0)
lines, labels = axes[1, 2].get_legend_handles_labels()
fig.legend(lines, labels, loc="upper right", frameon=False, bbox_to_anchor=(1.01, 1.01))

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
axes[2, 2].set_xlim(0, 400)
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
axes[3, 2].set_xlim(0, 400)
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

inset = fig.add_axes([0.225, 0.14, 0.11, 0.075])
inset.hist(
    ds_hm["q_ss"].isel(Time=t_wet).values.flatten(),
    color="#08306b",
    bins=1000,
    range=(0, 100),
    histtype="step",
    cumulative=True,
    lw=2.0,
)
inset.hist(
    ds_hm["q_ss"].isel(Time=t_drywet).values.flatten(),
    color="#6baed6",
    bins=1000,
    range=(0, 100),
    histtype="step",
    cumulative=True,
    lw=1.5,
)
inset.hist(
    ds_hm["q_ss"].isel(Time=t_wetdry).values.flatten(),
    color="#feb24c",
    bins=1000,
    range=(0, 100),
    histtype="step",
    cumulative=True,
    lw=1.0,
    ls="-.",
)
inset.hist(
    ds_hm["q_ss"].isel(Time=t_dry).values.flatten(),
    color="#fed976",
    bins=1000,
    range=(0, 100),
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

cell_width = 25
x, y = 412636.5, 5312188.5
grid_extent = (0, 80*25, 0, 53*25)

Ta_path = base_path / "svat_distributed" / "input" / "TA.txt"
PREC_path = base_path / "svat_distributed" / "input" / "PREC.txt"
PET_path = base_path / "svat_distributed" / "input" / "PET.txt"

df_PREC = pd.read_csv(
    PREC_path,
    sep=r"\s+",
    skiprows=0,
    header=0,
    na_values=-9999,
)
df_PREC.index = [pd.to_datetime(f"{df_PREC.iloc[i, 0]} {df_PREC.iloc[i, 1]} {df_PREC.iloc[i, 2]} {df_PREC.iloc[i, 3]} {df_PREC.iloc[i, 4]}", format="%Y %m %d %H %M") for i in range(len(df_PREC.index))]
df_PREC = df_PREC.loc[:, ["PREC"]]
df_PREC.index = df_PREC.index.rename("Index")

if os.path.exists(PET_path):
    df_pet = pd.read_csv(
        PET_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_pet.index = [pd.to_datetime(f"{df_pet.iloc[i, 0]} {df_pet.iloc[i, 1]} {df_pet.iloc[i, 2]} {df_pet.iloc[i, 3]} {df_pet.iloc[i, 4]}", format="%Y %m %d %H %M") for i in range(len(df_pet.index))]
    df_pet = df_pet.loc[:, ["PET"]]
    df_pet.index = df_pet.index.rename("Index")

else:
    df_pet = None

df_ta = pd.read_csv(
    Ta_path,
    sep=r"\s+",
    skiprows=0,
    header=0,
    na_values=-9999,
)
df_ta.index = [pd.to_datetime(f"{df_ta.iloc[i, 0]} {df_ta.iloc[i, 1]} {df_ta.iloc[i, 2]} {df_ta.iloc[i, 3]} {df_ta.iloc[i, 4]}", format="%Y %m %d %H %M") for i in range(len(df_ta.index))]
df_ta = df_ta.loc[:, "TA":]
df_ta.index = df_ta.index.rename("Index")

# reset index of precipitation time series
# time series starts on first day at 00:00 and ends on last day at 23:50
prec_ind = df_PREC.index
new_prec_ind = pd.date_range(
    start=datetime(prec_ind[0].year, prec_ind[0].month, prec_ind[0].day, 0, 0),
    end=datetime(prec_ind[-1].year, prec_ind[-1].month, prec_ind[-1].day, 23, 50),
    freq="10min",
)
prec_10mins = pd.DataFrame(index=new_prec_ind)
prec_10mins["PREC"] = 0.
prec_10mins.loc[df_PREC.index, "PREC"] = df_PREC["PREC"].values.astype(float)

ta_avg = df_ta.mean()[0]
pet_avg = df_pet.sum()[0] / 3
prec_avg = prec_10mins.resample("D").sum().sum()[0] / 3

mask = (ds_params_hm["prec_weight"].values >= 0)

fig, ax = plt.subplots(figsize=(6,4))
fig.patch.set_alpha(0)
plt.imshow(onp.where(mask, ds_params_hm["prec_weight"].values * prec_avg, onp.nan), extent=grid_extent, cmap='viridis_r', zorder=1)
plt.colorbar(label='Precipitation\n [mm/year]', shrink=0.7)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "annual_average_PRECIP.png"
fig.savefig(file, dpi=300)

fig, ax = plt.subplots(figsize=(6,4))
fig.patch.set_alpha(0)
plt.imshow(onp.where(mask, ds_params_hm["pet_weight"].values * pet_avg, onp.nan), extent=grid_extent, cmap='viridis', zorder=1)
plt.colorbar(label='Potential evapotranspiration\n [mm/year]', shrink=0.7)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "annual_average_PET.png"
fig.savefig(file, dpi=300)

fig, ax = plt.subplots(figsize=(6,4))
fig.patch.set_alpha(0)
plt.imshow(onp.where(mask, ds_params_hm["ta_offset"].values + ta_avg, onp.nan), extent=grid_extent, cmap='viridis', zorder=1)
plt.colorbar(label='Air temperature\n [°C]', shrink=0.7)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "annual_average_TA.png"
fig.savefig(file, dpi=300)

plt.close("all")
