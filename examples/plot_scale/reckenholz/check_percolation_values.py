import os
from pathlib import Path
from cftime import num2date
import xarray as xr
import pandas as pd
import numpy as onp
import click
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


@click.command("main")
def main():
    # directory of figures
    base_path_figs = Path(__file__).parent / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    lys_experiments = ["lys2", "lys3", "lys4", "lys8", "lys9"]
    for lys_experiment in lys_experiments:

        # load observations (measured data)
        path_obs = Path(__file__).parent / "observations" / "reckenholz_lysimeter.nc"
        ds_obs = xr.open_dataset(path_obs, engine="h5netcdf", group=lys_experiment)

        # assign date
        days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_obs = num2date(
            days_obs,
            units=f"days since {ds_obs['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

        df_lys = pd.DataFrame(index=date_obs)
        df_lys.loc[:, "prec"] = ds_obs["PREC"].isel(x=0, y=0).values
        df_lys.loc[:, "pet"] = ds_obs["PET"].isel(x=0, y=0).values
        df_lys.loc[:, "perc"] = ds_obs["PERC"].isel(x=0, y=0).values
        df_lys.loc[:, "weight"] = ds_obs["WEIGHT"].isel(x=0, y=0).values
        df_lys.loc[df_lys.index[1]:, "dS"] = df_lys.loc[:, "weight"].values[1:] - df_lys.loc[:, "weight"].values[:-1]
        df_lys.loc[:, "perc_pet_ratio"] = df_lys.loc[:, "perc"] / (df_lys.loc[:, "perc"] + df_lys.loc[:, "pet"])
        df_lys = pd.DataFrame(index=date_obs).join(df_lys)
        # condition for plausible lysimeter seepage
        cond_perc = (df_lys.loc[:, "perc_pet_ratio"] >= 0.5) & ((df_lys.loc[:, "dS"] < (df_lys.loc[:, "pet"] + df_lys.loc[:, "perc"]) * (-1)) | (df_lys.loc[:, "dS"] <= df_lys.loc[:, "prec"]))
        cond_pet = (df_lys.loc[:, "perc_pet_ratio"] < 0.5) & ((df_lys.loc[:, "dS"] < (df_lys.loc[:, "pet"] + df_lys.loc[:, "perc"]) * (-1)) | (df_lys.loc[:, "dS"] <= df_lys.loc[:, "prec"]))
        cond = cond_perc | cond_pet
        df_lys.loc[~cond, :] = onp.nan

        df_lys_perc = df_lys.copy()
        df_lys_perc.loc[~cond_perc, :] = onp.nan

        df_lys_pet = df_lys.copy()
        df_lys_pet.loc[~cond_pet, :] = onp.nan

        # plot percolation to potential evapotranspiration ratio
        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        axes[0].plot(df_lys.index, df_lys.loc[:, "dS"], color="black", zorder=0)
        axes[0].scatter(df_lys_perc.index, df_lys_perc.loc[:, "dS"], color="blue", s=2, zorder=1, alpha=0.5)
        axes[0].scatter(df_lys_pet.index, df_lys_pet.loc[:, "dS"], color="green", s=2, zorder=1, alpha=0.5)
        axes[0].set_xlim((df_lys.index[0], df_lys.index[-1]))
        axes[0].set_ylabel(r"$\Delta$S [mm/day]")

        axes[1].scatter(df_lys_perc.index, df_lys_perc.loc[:, "perc"], color="blue", s=2, zorder=1)
        axes[1].scatter(df_lys_pet.index, df_lys_pet.loc[:, "pet"], color="green", s=2, zorder=0)
        axes[1].set_xlim((df_lys.index[0], df_lys.index[-1]))
        axes[1].set_ylabel("[mm/day]")

        axes[2].plot(df_lys.index, df_lys.loc[:, "perc_pet_ratio"], color="black", zorder=0)
        axes[2].scatter(df_lys_perc.index, df_lys_perc.loc[:, "perc_pet_ratio"], color="blue", s=2, zorder=1, alpha=0.5)
        axes[2].scatter(df_lys_pet.index, df_lys_pet.loc[:, "perc_pet_ratio"], color="green", s=2, zorder=1, alpha=0.5)
        axes[2].set_xlim((df_lys.index[0], df_lys.index[-1]))
        axes[2].set_ylabel("PERC/PET [-]")
        axes[2].set_xlabel("Time [year]")
        fig.tight_layout()
        file = base_path_figs / f"perc_pet_ratio_{lys_experiment}.png"
        fig.savefig(file, dpi=300)
    return


if __name__ == "__main__":
    main()
