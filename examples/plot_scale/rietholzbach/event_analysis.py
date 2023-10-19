from pathlib import Path
import os
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.dates as mdates
import click
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
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

_LABS_TM = {
    "complete-mixing": "CM",
    "piston": "PI",
    "advection-dispersion-power": "AD",
    "time-variant advection-dispersion-power": "AD-TV",
    "preferential-power": "PF",
    "older-preference-power": "OP",
    "advection-dispersion-kumaraswamy": "ADK",
    "time-variant advection-dispersion-kumaraswamy": "ADK-TV",
}

@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    tm_structures = [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
    ]

    # load observations (measured data)
    path_obs = Path(__file__).parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
    # assign date
    days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_obs = num2date(
        days_obs,
        units=f"days since {ds_obs['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_obs = ds_obs.assign_coords(Time=("Time", date_obs))
    df_obs = pd.DataFrame(index=date_obs)
    df_obs.loc[:, "d18O_prec"] = ds_obs["d18O_PREC"].isel(x=0, y=0).values
    df_obs.loc[:, "d18O_perc"] = ds_obs["d18O_PERC"].isel(x=0, y=0).values

    # load best monte carlo simulations
    states_hm1_file = base_path / "svat_monte_carlo" / "output" / "states_hm1.nc"
    ds_sim_hm1 = xr.open_dataset(states_hm1_file, engine="h5netcdf")
    # assign date
    days_sim_hm1 = ds_sim_hm1["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm1 = num2date(
        days_sim_hm1,
        units=f"days since {ds_sim_hm1['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm1 = ds_sim_hm1.assign_coords(Time=("Time", date_sim_hm1))
    # load best 10 monte carlo simulations
    states_hm10_file = base_path / "svat_monte_carlo" / "output" / "states_hm10.nc"
    ds_sim_hm10 = xr.open_dataset(states_hm10_file, engine="h5netcdf")
    # assign date
    days_sim_hm10 = ds_sim_hm10["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm10 = num2date(
        days_sim_hm10,
        units=f"days since {ds_sim_hm10['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm10 = ds_sim_hm10.assign_coords(Time=("Time", date_sim_hm10))

    # load best 100 monte carlo simulations
    states_hm100_file = base_path / "svat_monte_carlo" / "output" / "states_hm100.nc"
    ds_sim_hm100 = xr.open_dataset(states_hm100_file, engine="h5netcdf")
    # assign date
    days_sim_hm100 = ds_sim_hm100["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm100 = num2date(
        days_sim_hm100,
        units=f"days since {ds_sim_hm100['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm100 = ds_sim_hm100.assign_coords(Time=("Time", date_sim_hm100))

    states_hm_for_tm_file = (
        base_path
        / "svat_oxygen18_monte_carlo"
        / "output"
        / "states_hm_best_for_advection-dispersion-power.nc"
    )
    ds_sim_hm_for_tm = xr.open_dataset(states_hm_for_tm_file, engine="h5netcdf")
    # assign date
    days_sim_hm_for_tm = ds_sim_hm_for_tm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm_for_tm = num2date(
        days_sim_hm_for_tm,
        units=f"days since {ds_sim_hm_for_tm['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm_for_tm = ds_sim_hm_for_tm.assign_coords(Time=("Time", date_sim_hm_for_tm))

    # load HYDRUS-1D benchmarks
    # oxygen-18 simulations
    states_hydrus_file = base_path / "hydrus_benchmark" / "states_hydrus_18O.nc"
    ds_hydrus_18O = xr.open_dataset(states_hydrus_file, engine="h5netcdf")
    hours_hydrus_18O = ds_hydrus_18O["Time"].values / onp.timedelta64(60 * 60, "s")
    date_hydrus_18O = num2date(
        hours_hydrus_18O,
        units=f"hours since {ds_hydrus_18O['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_hydrus_18O = ds_hydrus_18O.assign_coords(Time=("Time", date_hydrus_18O))


    # load precipitation data
    prec_file = base_path / "observations" / "PREC.txt"
    df_PREC = pd.read_csv(
        prec_file,
        sep=r"\s+",
        skiprows=0,
        header=0,
        parse_dates=[[0, 1, 2, 3, 4]],
        index_col=0,
        na_values=-9999,
    )
    df_PREC.index = pd.to_datetime(df_PREC.index, format="%Y %m %d %H %M")
    df_PREC.index = df_PREC.index.rename("Index")

    df_events = pd.DataFrame(index=date_obs)
    for i in range(len(df_events.index)):
        df_events.loc[df_events.index[i], "PREC_max"] = df_PREC.loc[f"{df_events.index[i]} 00:00:00":f"{df_events.index[i]} 23:50:00", "PREC"].max()
        df_events.loc[df_events.index[i], "PREC"] = df_PREC.loc[f"{df_events.index[i]} 00:00:00":f"{df_events.index[i]} 23:50:00", "PREC"].sum()
        df_events.loc[df_events.index[i], "days_no_prec"] = 0
    for i in range(len(df_events.index)):
        if df_events.loc[df_events.index[i], "PREC"] > 0:
            t = 0
            for j in range(i+1, len(df_events.index)):
                if df_events.loc[df_events.index[j], "PREC"] <= 0:
                    t += 1
                    df_events.loc[df_events.index[i], "days_no_prec"] = t
                else:
                    break

    df_events.to_csv(base_path_figs / "events.csv", sep=";", index=True)    


    plt.close("all")
    return


if __name__ == "__main__":
    main()
