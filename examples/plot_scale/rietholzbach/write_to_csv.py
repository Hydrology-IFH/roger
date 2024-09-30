from pathlib import Path
import os
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.dates as mdates
import matplotlib.cm as cm
import click
import roger.tools.evaluation as eval_utils
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
    "talk",
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

    # load data from bromide experiment
    path_obs_br = Path(__file__).parent / "observations" / "bromide_breakthrough.csv"
    df_obs_br = pd.read_csv(path_obs_br, skiprows=1, sep=";", na_values="")

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

    df_hydrus = pd.DataFrame(index=date_hydrus_18O)
    df_hydrus.loc[:, "d18O"] = ds_hydrus_18O["d18O_perc"].values
    df_hydrus.columns = [
        ["[per mille]"],
        ["d18O"],
    ]
    file = base_path / "svat_oxygen18" / "output" / "hydrus_d18O.csv"
    df_hydrus.iloc[1:, :].to_csv(file, sep=";", decimal=".", index=True, header=True)



    # load RoGeR
    # oxygen-18 simulations
    states_roger_file = base_path / "svat_oxygen18" / "output" / "states_advection-dispersion-power.nc"
    ds_roger_18O = xr.open_dataset(states_roger_file, engine="h5netcdf")
    hours_roger_18O = ds_roger_18O["Time"].values / onp.timedelta64(60 * 60, "s")
    date_roger_18O = num2date(
        hours_roger_18O,
        units=f"hours since {ds_roger_18O['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_roger_18O = ds_roger_18O.assign_coords(Time=("Time", date_roger_18O))

    df_roger = pd.DataFrame(index=date_roger_18O)
    df_roger.loc[:, "d18O"] = ds_roger_18O["C_iso_q_ss"].isel(x=0, y=0).values
    df_roger.columns = [
        ["[per mille]"],
        ["d18O"],
    ]
    file = base_path / "svat_oxygen18" / "output" / "roger_d18O.csv"
    df_roger.iloc[2:, :].to_csv(file, sep=";", decimal=".", index=True, header=True)



    # df_sim_br_conc = pd.DataFrame(index=df_obs_br.index)
    # states_br_file = (
    #     base_path / "svat_bromide_benchmark" / "output" / f"states_{tms}_bromide_benchmark_stgallen.nc"
    # )
    # with xr.open_dataset(states_br_file, engine="h5netcdf", decode_times=False, group=f"1991") as ds:
    #     sim_vals = ds["C_q_ss_mmol_bs"].isel(x=0, y=0).values[315:716]
    #     sim_vals = onp.where(sim_vals < 0, onp.nan, sim_vals)
    #     df_sim_br_conc.loc[:, f"1991"] = sim_vals




if __name__ == "__main__":
    main()
