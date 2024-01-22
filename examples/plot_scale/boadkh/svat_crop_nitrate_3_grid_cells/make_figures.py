import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import matplotlib.dates as mdates
import numpy as onp
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import click

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

def _make_df_with_annual_sums(date, vals):
    df = pd.DataFrame(index=date)
    df.loc[:, "vals"] = vals
    df.loc[:, "vals"] = vals
    df = df.loc["2013":, :]

    # annual sums
    df_annual = df.groupby(df.index.year).sum()
    df_annual.index = pd.to_datetime(df_annual.index, format="%Y")

    return df_annual

def _make_df_with_annual_avg(date, vals):
    df = pd.DataFrame(index=date)
    df.loc[:, "vals"] = vals
    df = df.loc["2013":, :]

    # annual averages
    df_annual = df.groupby(df.index.year).mean()
    df_annual.index = pd.to_datetime(df_annual.index, format="%Y")

    return df_annual

def _make_df_with_annual_sums3(date, vals1, vals2, vals3):
    df = pd.DataFrame(index=date)
    df.loc[:, "low"] = vals1
    df.loc[:, "medium"] = vals2
    df.loc[:, "high"] = vals3

    # annual sums
    df_annual = df.groupby(df.index.year).sum()
    df_annual.index = pd.to_datetime(df_annual.index, format="%Y")

    return df_annual

@click.command("main")
def main():
    base_path = Path(__file__).parent
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    id = "5-8_2090295_1"
    location = "freiburg"
    crop_rotation_scenario = "summer-wheat_winter-wheat_corn"

    # load hydrologic simulations
    states_hm_file = base_path.parent / "output" / "svat_crop" / f"SVATCROP_{id}_{location}_{crop_rotation_scenario}_3_grid_cells.nc"
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

    # load transport simulations
    states_tm_file = base_path.parent / "output" / "svat_crop_nitrate_3_grid_cells" / f"SVATCROPNITRATE_{id}_{location}_{crop_rotation_scenario}.nc"
    ds_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
    # assign date
    days_tm = ds_tm["Time"].values
    date_tm = num2date(days_tm, units="days since 2012-12-31", calendar="standard", only_use_cftime_datetimes=False)
    ds_tm = ds_tm.assign_coords(Time=("Time", date_tm))

    # plot fluxes, nitrogen mass and water ages
    fig, axes = plt.subplots(4, 3, figsize=(6, 5))
    axes[0, 0].bar(date_hm, ds_hm["prec"].isel(x=0, y=0).values, width=0.1, color="blue", align="edge", edgecolor="blue")
    axes[0, 0].set_ylabel(r"PRECIP [mm/day]")
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[0, 0].tick_params(axis="x", labelrotation=60)
    axes[0, 0].set_xlim(date_hm[0], date_hm[-1])

    axes[0, 1].plot(date_tm, ds_tm["M_in"].isel(x=0, y=0).values * 0.01, "-", color="blue", lw=1)
    axes[0, 1].set_xlim(date_tm[0], date_tm[-1])
    axes[0, 1].set_ylabel(r"[kg N/ha/day]")
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[0, 1].tick_params(axis="x", labelrotation=60)
    axes[0, 1].set_ylim(
        0,
    )

    axes[1, 0].plot(date_hm, ds_hm["transp"].isel(x=0, y=0).values, "-", color="#31a354", lw=1)
    axes[1, 0].set_xlim(date_hm[0], date_hm[-1])
    axes[1, 0].set_ylabel(r"TRANSP [mm/day]")
    axes[1, 0].set_ylim(
        0,
    )
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[1, 0].tick_params(axis="x", labelrotation=60)

    axes[2, 0].plot(date_hm, ds_hm["theta"].isel(x=0, y=0).values, "-", color="brown", lw=1)
    axes[2, 0].set_xlim(date_hm[0], date_hm[-1])
    axes[2, 0].set_ylabel(r"$\theta$ [-]")
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 0].tick_params(axis="x", labelrotation=60)

    axes[3, 0].plot(date_hm, ds_hm["q_ss"].isel(x=0, y=0).values, "-", color="grey", lw=1)
    axes[3, 0].set_xlim(date_hm[0], date_hm[-1])
    axes[3, 0].set_ylim(
        0,
    )
    axes[3, 0].set_ylabel(r"PERC [mm/day]")
    axes[3, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[3, 0].set_xlabel(r"Time [year-month]")
    axes[3, 0].tick_params(axis="x", labelrotation=60)

    axes[1, 1].plot(date_tm, ds_tm["M_transp"].isel(x=0, y=0).values * 0.01, "-", color="#31a354", lw=0.5)
    axes[1, 1].plot(date_tm, ds_tm["M_transp"].isel(x=1, y=0).values * 0.01, "-", color="#31a354", lw=1)
    axes[1, 1].plot(date_tm, ds_tm["M_transp"].isel(x=2, y=0).values * 0.01, "-", color="#31a354", lw=1.5)
    axes[1, 1].set_xlim(date_tm[0], date_tm[-1])
    axes[1, 1].set_ylim(0,)
    axes[1, 1].set_ylabel(r"[kg N/ha/day]")
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[1, 1].tick_params(axis="x", labelrotation=60)

    axes[2, 1].plot(date_tm, ds_tm["M_s"].isel(x=0, y=0).values * 0.01, "-", color="brown", lw=0.5)
    axes[2, 1].plot(date_tm, ds_tm["M_s"].isel(x=1, y=0).values * 0.01, "-", color="brown", lw=1)
    axes[2, 1].plot(date_tm, ds_tm["M_s"].isel(x=2, y=0).values * 0.01, "-", color="brown", lw=1.5)
    axes[2, 1].set_xlim(date_tm[0], date_tm[-1])
    axes[2, 1].set_ylim(0,)
    axes[2, 1].set_ylabel(r"[kg N/ha/day]")
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 1].tick_params(axis="x", labelrotation=60)

    axes[3, 1].plot(date_tm, ds_tm["M_q_ss"].isel(x=0, y=0).values * 0.01, "-", color="grey", lw=0.5)
    axes[3, 1].plot(date_tm, ds_tm["M_q_ss"].isel(x=1, y=0).values * 0.01, "-", color="grey", lw=1)
    axes[3, 1].plot(date_tm, ds_tm["M_q_ss"].isel(x=2, y=0).values * 0.01, "-", color="grey", lw=1.5)
    axes[3, 1].set_xlim(date_tm[0], date_tm[-1])
    axes[3, 1].set_ylim(0,)
    axes[3, 1].set_ylabel(r"[kg N/ha/day]")
    axes[3, 1].set_xlabel(r"Time [year-month]")
    axes[3, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[3, 1].tick_params(axis="x", labelrotation=60)

    axes[0, 2].set_axis_off()

    axes[1, 2].fill_between(
        date_tm,
        ds_tm["tt10_transp"].isel(x=0, y=0).values,
        ds_tm["tt90_transp"].isel(x=0, y=0).values,
        color="#31a354",
        edgecolor=None,
        alpha=0.2,
    )
    axes[1, 2].plot(date_tm, ds_tm["tt50_transp"].isel(x=0, y=0).values, "-", color="#31a354", lw=1)
    axes[1, 2].set_xlim(date_tm[0], date_tm[-1])
    axes[1, 2].set_ylabel(r"age [days]")
    axes[1, 2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[1, 2].tick_params(axis="x", labelrotation=60)
    axes[1, 2].set_ylim(
        0,
    )

    axes[2, 2].fill_between(
        date_tm,
        ds_tm["rt10_s"].isel(x=0, y=0).values,
        ds_tm["rt90_s"].isel(x=0, y=0).values,
        color="brown",
        edgecolor=None,
        alpha=0.2,
    )
    axes[2, 2].plot(date_tm, ds_tm["rt50_s"].isel(x=0, y=0).values, "-", color="brown", lw=1)
    axes[2, 2].set_xlim(date_tm[0], date_tm[-1])
    axes[2, 2].set_ylabel(r"age [days]")
    axes[2, 2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 2].tick_params(axis="x", labelrotation=60)
    axes[2, 2].set_ylim(
        0,
    )
    vals1 = onp.where(ds_tm["tt10_q_ss"].isel(x=0, y=0).values >= 1000, onp.nan, ds_tm["tt10_q_ss"].isel(x=0, y=0).values)
    vals2 = onp.where(ds_tm["tt90_q_ss"].isel(x=0, y=0).values >= 1000, onp.nan, ds_tm["tt90_q_ss"].isel(x=0, y=0).values)
    vals3 = onp.where(ds_tm["tt50_q_ss"].isel(x=0, y=0).values >= 1000, onp.nan, ds_tm["tt50_q_ss"].isel(x=0, y=0).values)
    axes[3, 2].fill_between(
        date_tm,
        vals1,
        vals2,
        color="grey",
        edgecolor=None,
        alpha=0.2,
    )
    axes[3, 2].plot(date_tm, vals3, "-", color="grey", lw=1)
    axes[3, 2].set_xlim(date_tm[0], date_tm[-1])
    axes[3, 2].set_ylabel(r"age [days]")
    axes[3, 2].set_xlabel(r"Time [year-month]")
    axes[3, 2].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[3, 2].tick_params(axis="x", labelrotation=60)
    axes[3, 2].set_ylim(0, 1000)

    fig.subplots_adjust(left=0.1, bottom=0.13, top=0.95, right=0.98, hspace=0.6, wspace=0.42)
    file = base_path_figs / "time_series_fluxes_nitrogen_mass_water_ages.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "time_series_fluxes_nitrogen_mass_water_ages.pdf"
    fig.savefig(file, dpi=300)


    fig, axes = plt.subplots(4, 2, figsize=(6, 6))
    axes[0, 0].plot(date_hm, onp.cumsum(ds_hm["prec"].isel(x=0, y=0).values), "-", color="blue", lw=1)
    axes[0, 0].set_ylabel(r"PRECIP [mm]")
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[0, 0].tick_params(axis="x", labelrotation=60)
    axes[0, 0].set_xlim(date_hm[0], date_hm[-1])
    axes[0, 0].set_ylim(
        0,
    )

    axes[0, 1].plot(date_tm, onp.cumsum(ds_tm["M_in"].isel(x=0, y=0).values * 0.01), "-", color="blue", lw=1)
    axes[0, 1].set_xlim(date_tm[0], date_tm[-1])
    axes[0, 1].set_ylabel(r"[kg N/ha/day]")
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[0, 1].tick_params(axis="x", labelrotation=60)
    axes[0, 1].set_ylim(
        0,
    )

    axes[1, 0].plot(date_hm, onp.cumsum(ds_hm["transp"].isel(x=0, y=0).values), "-", color="#31a354", lw=1)
    axes[1, 0].set_xlim(date_hm[0], date_hm[-1])
    axes[1, 0].set_ylabel(r"TRANSP [mm]")
    axes[1, 0].set_ylim(
        0,
    )
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[1, 0].tick_params(axis="x", labelrotation=60)

    axes[2, 0].plot(date_hm, ds_hm["theta"].isel(x=0, y=0).values, "-", color="brown", lw=1)
    axes[2, 0].set_xlim(date_hm[0], date_hm[-1])
    axes[2, 0].set_ylabel(r"$\theta$ [-]")
    axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 0].tick_params(axis="x", labelrotation=60)

    axes[3, 0].plot(date_hm, onp.cumsum(ds_hm["q_ss"].isel(x=0, y=0).values), "-", color="grey", lw=1)
    axes[3, 0].set_xlim(date_hm[0], date_hm[-1])
    axes[3, 0].set_ylim(
        0,
    )
    axes[3, 0].set_ylabel(r"PERC [mm]")
    axes[3, 0].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[3, 0].set_xlabel(r"Time [year-month]")
    axes[3, 0].tick_params(axis="x", labelrotation=60)

    axes[1, 1].plot(date_tm, onp.cumsum(ds_tm["M_transp"].isel(x=0, y=0).values * 0.01), "-", color="#31a354", lw=0.5)
    axes[1, 1].plot(date_tm, onp.cumsum(ds_tm["M_transp"].isel(x=1, y=0).values * 0.01), "-", color="#31a354", lw=1)
    axes[1, 1].plot(date_tm, onp.cumsum(ds_tm["M_transp"].isel(x=2, y=0).values * 0.01), "-", color="#31a354", lw=1.5)
    axes[1, 1].set_xlim(date_tm[0], date_tm[-1])
    axes[1, 1].set_ylim(0,)
    axes[1, 1].set_ylabel(r"[kg N/ha/day]")
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[1, 1].tick_params(axis="x", labelrotation=60)

    axes[2, 1].plot(date_tm, ds_tm["M_s"].isel(x=0, y=0).values * 0.01, "-", color="brown", lw=0.5)
    axes[2, 1].plot(date_tm, ds_tm["M_s"].isel(x=1, y=0).values * 0.01, "-", color="brown", lw=1)
    axes[2, 1].plot(date_tm, ds_tm["M_s"].isel(x=2, y=0).values * 0.01, "-", color="brown", lw=1.5)
    axes[2, 1].set_xlim(date_tm[0], date_tm[-1])
    axes[2, 1].set_ylim(0,)
    axes[2, 1].set_ylabel(r"[kg N/ha/day]")
    axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[2, 1].tick_params(axis="x", labelrotation=60)

    axes[3, 1].plot(date_tm, onp.cumsum(ds_tm["M_q_ss"].isel(x=0, y=0).values * 0.01), "-", color="grey", lw=0.5)
    axes[3, 1].plot(date_tm, onp.cumsum(ds_tm["M_q_ss"].isel(x=1, y=0).values * 0.01), "-", color="grey", lw=1)
    axes[3, 1].plot(date_tm, onp.cumsum(ds_tm["M_q_ss"].isel(x=2, y=0).values * 0.01), "-", color="grey", lw=1.5)
    axes[3, 1].set_xlim(date_tm[0], date_tm[-1])
    axes[3, 1].set_ylim(0,)
    axes[3, 1].set_ylabel(r"[kg N/ha/day]")
    axes[3, 1].set_xlabel(r"Time [year-month]")
    axes[3, 1].xaxis.set_major_formatter(mdates.DateFormatter("%y-%m"))
    axes[3, 1].tick_params(axis="x", labelrotation=60)

    fig.subplots_adjust(left=0.1, bottom=0.13, top=0.95, right=0.98, hspace=0.6, wspace=0.42)
    file = base_path_figs / "cumulated_time_series_fluxes_nitrogen_mass.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "cumulated_time_series_fluxes_nitrogen_mass.pdf"
    fig.savefig(file, dpi=300)


    fig, axes = plt.subplots(4, 2, figsize=(6, 6))
    df = _make_df_with_annual_sums(date_hm, ds_hm["prec"].isel(x=0, y=0).values)
    axes[0, 0].bar(df.index, df.loc[:, "vals"].values, color="blue", edgecolor="blue", width=180)
    axes[0, 0].set_ylabel("PRECIP\n[mm/year]")
    axes[0, 0].tick_params(axis="x", labelrotation=60)
    axes[0, 0].set_ylim(
        0,
    )

    df = _make_df_with_annual_sums(date_hm, ds_tm["M_in"].isel(x=0, y=0).values * 0.01)
    axes[0, 1].bar(df.index, df.loc[:, "vals"].values, color="blue", edgecolor="blue", width=180)
    axes[0, 1].set_ylabel(r"[kg N$O_3$-N/ha/year]")
    axes[0, 1].tick_params(axis="x", labelrotation=60)
    axes[0, 1].set_ylim(
        0,
    )

    df = _make_df_with_annual_sums(date_hm, ds_hm["transp"].isel(x=0, y=0).values)
    axes[1, 0].bar(df.index, df.loc[:, "vals"].values, color="#31a354", edgecolor="#31a354", width=180)
    axes[1, 0].set_ylabel("TRANSP\n[mm/year]")
    axes[1, 0].tick_params(axis="x", labelrotation=60)
    axes[1, 0].set_ylim(
        0,
    )

    df = _make_df_with_annual_avg(date_hm, ds_hm["theta"].isel(x=0, y=0).values)
    axes[2, 0].plot(df.index, df.loc[:, "vals"].values, color="brown", lw=1)
    axes[2, 0].set_ylabel(r"$\theta$ [-]")
    axes[2, 0].tick_params(axis="x", labelrotation=60)
    axes[2, 0].set_ylim(0.25, 0.4)

    df = _make_df_with_annual_sums(date_hm, ds_hm["q_ss"].isel(x=0, y=0).values)
    axes[3, 0].bar(df.index, df.loc[:, "vals"].values, color="grey", edgecolor="grey", width=180)
    axes[3, 0].set_ylabel("PERC\n[mm/year]")
    axes[3, 0].tick_params(axis="x", labelrotation=60)
    axes[3, 0].set_ylim(
        0,
    )

    df = _make_df_with_annual_sums(date_tm, ds_tm["M_transp"].isel(x=1, y=0).values * 0.01)
    axes[1, 1].bar(df.index, df.loc[:, "vals"].values, color="#31a354", edgecolor="#31a354", width=180)
    axes[1, 1].set_ylabel(r"[kg N$O_3$-N/ha/year]")
    axes[1, 1].tick_params(axis="x", labelrotation=60)
    axes[1, 1].set_ylim(
        0,
    )

    df = _make_df_with_annual_avg(date_tm, (ds_tm["M_s"].isel(x=1, y=0).values) * 0.01)
    axes[2, 1].bar(df.index, df.loc[:, "vals"].values, color="brown", edgecolor="brown", width=180)
    axes[2, 1].set_ylabel(r"[kg N$O_3$-N/ha/year]")
    axes[2, 1].tick_params(axis="x", labelrotation=60)
    axes[2, 1].set_ylim(
        0,
    )

    df = _make_df_with_annual_sums(date_tm, ds_tm["M_q_ss"].isel(x=1, y=0).values * 0.01)
    axes[3, 1].bar(df.index, df.loc[:, "vals"].values, color="grey", edgecolor="grey", width=180)
    axes[3, 1].set_ylabel(r"[kg N$O_3$-N/ha/year]")
    axes[3, 1].tick_params(axis="x", labelrotation=60)
    axes[3, 1].set_ylim(
        0,
    )

    fig.subplots_adjust(left=0.15, bottom=0.13, top=0.95, right=0.98, hspace=0.6, wspace=0.42)
    fig.tight_layout()
    file = base_path_figs / "annual_sums_fluxes_nitrogen_mass.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "annual_sums_fluxes_nitrogen_mass.pdf"
    fig.savefig(file, dpi=300)

    plt.close("all")
    return


if __name__ == "__main__":
    main()