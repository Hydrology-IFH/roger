from pathlib import Path
import os
import h5netcdf
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import roger.tools.evaluation as eval_utils
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


@click.option(
    "-tms",
    "--transport-model-structure",
    type=click.Choice(
        ["complete-mixing", "piston", "advection-dispersion-power", "time-variant_advection-dispersion-power"]
    ),
    default="piston",
)
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir, transport_model_structure):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    tms = transport_model_structure.replace("_", " ")
    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    years = onp.arange(1997, 2007).tolist()
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=onp.min(years), vmax=onp.max(years))
    # load hydrologic simulation
    states_hm_file = base_path / f"states_hm_best_for_{transport_model_structure}.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
    # assign date
    days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim_hm = num2date(
        days_sim_hm,
        units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))
    df_metrics_year = pd.DataFrame(index=years)
    fig, axes = plt.subplots(1, 1, figsize=(6, 2))
    # load observations
    br_obs_file = base_path.parent / "observations" / "bromide_breakthrough.csv"
    df_br_obs = pd.read_csv(br_obs_file, sep=";", skiprows=1, index_col=0)
    nrows = 100
    for nrow in range(nrows):
        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        for year in years:
            # load simulation
            states_tm_file = base_path_output / f"states_{transport_model_structure}_bromide_benchmark.nc"
            ds_sim_tm = xr.open_dataset(states_tm_file, group=f"{year}", engine="h5netcdf")
            # assign date
            days_sim_tm = ds_sim_tm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
            date_sim_tm = num2date(
                days_sim_tm,
                units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}",
                calendar="standard",
                only_use_cftime_datetimes=False,
            )
            ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
            alpha_transp = onp.round(ds_sim_tm["alpha_transp"].isel(x=nrow, y=0).values[0], 2)
            alpha_q = onp.round(ds_sim_tm["alpha_q"].isel(x=nrow, y=0).values[0], 2)
            click.echo(f"Calculate metrics for {tms}-{year} alpha_transp: {alpha_transp} alpha_q: {alpha_q} ...")
            # plot percolation rate (in l/h) and bromide concentration (mmol/l)
            idx = pd.date_range(start=f"1/1/{year}", end=f"31/12/{year+1}")
            df_perc_br_sim = pd.DataFrame(index=idx, columns=["perc", "Br_conc_mg", "Br_mg", "Br_conc_mmol"])
            # in mm per day
            df_perc_br_sim.loc[:, "perc"] = ds_sim_hm.sel(Time=slice(str(year), str(year + 1)))["q_ss"].isel(y=0).values
            # in mg per liter
            df_perc_br_sim.loc[:, "Br_conc_mg"] = ds_sim_tm["C_q_ss"].isel(x=nrow, y=0).values[1:]
            # in mg
            df_perc_br_sim.loc[:, "Br_mg"] = (
                ds_sim_tm["C_q_ss"].isel(x=nrow, y=0).values[1:]
                * ds_sim_hm.sel(Time=slice(str(year), str(year + 1)))["q_ss"].isel(y=0).values
            )
            # in mmol per liter
            df_perc_br_sim.loc[:, "Br_conc_mmol"] = df_perc_br_sim.loc[:, "Br_conc_mg"] / 79.904
            # daily samples from day 0 to day 220
            df_daily = df_perc_br_sim.loc[: df_perc_br_sim.index[315 + 220], :]
            # weekly samples after 220 days
            df_weekly = df_perc_br_sim.loc[df_perc_br_sim.index[316 + 220] :, "perc"].resample("7D").sum().to_frame()
            df_weekly.loc[:, "Br_mg"] = (
                df_perc_br_sim.loc[df_perc_br_sim.index[316 + 220] :, "Br_mg"].resample("7D").sum()
            )
            df_weekly.loc[:, "Br_conc_mg"] = (
                df_perc_br_sim.loc[df_perc_br_sim.index[316 + 220] :, "Br_mg"].resample("7D").sum()
                / df_perc_br_sim.loc[df_perc_br_sim.index[316 + 220] :, "perc"].resample("7D").sum()
            )
            df_weekly.loc[:, "Br_conc_mmol"] = df_weekly.loc[:, "Br_conc_mg"] / 79.904
            df_daily_weekly = pd.concat([df_daily, df_weekly])
            df_perc_br_sim = pd.DataFrame(index=idx).join(df_daily_weekly)
            df_perc_br_sim = df_perc_br_sim.iloc[315:716, :]
            df_perc_br_sim.index = range(len(df_perc_br_sim.index))
            axes.plot(
                df_perc_br_sim.dropna().index,
                df_perc_br_sim.dropna()["Br_conc_mmol"],
                ls="-",
                color=cmap(norm(year)),
                label=f"{year}",
            )

            # join observations on simulations
            obs_vals = df_br_obs.iloc[:, 0].values
            sim_vals = df_perc_br_sim.loc[:, "Br_conc_mmol"].values
            df_obs = pd.DataFrame(index=df_br_obs.index, columns=["obs"])
            df_obs.loc[:, "obs"] = obs_vals
            df_eval = eval_utils.join_obs_on_sim(df_perc_br_sim.index, sim_vals, df_obs)
            df_eval = df_eval.dropna()
            # calculate metrics
            obs_vals = df_eval.loc[:, "obs"].values
            sim_vals = df_eval.loc[:, "sim"].values
            # temporal correlation
            df_metrics_year.loc[year, "r"] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
            # tracer recovery (in %)
            df_metrics_year.loc[year, "Br_mass_recovery"] = onp.sum(
                ds_sim_tm["M_q_ss"].isel(x=nrow, y=0).values[315:716]
            ) / (79900 / 3.14)
            # average travel time of percolation (in days)
            df_metrics_year.loc[year, "ttavg"] = onp.nanmean(ds_sim_tm["ttavg_q_ss"].isel(x=nrow, y=0).values[315:716])
            # average median travel time of percolation (in days)
            df_metrics_year.loc[year, "tt50"] = onp.nanmedian(ds_sim_tm["ttavg_q_ss"].isel(x=nrow, y=0).values[315:716])

            # write simulated bulk sample to output file
            ds_sim_tm = ds_sim_tm.load()  # required to release file lock
            ds_sim_tm = ds_sim_tm.close()
            del ds_sim_tm
            states_tm_file = base_path_output / f"states_{transport_model_structure}_bromide_benchmark.nc"
            with h5netcdf.File(states_tm_file, "a", decode_vlen_strings=False) as f:
                try:
                    v = f.groups[f"{year}"].create_variable(
                        "C_q_ss_mmol_bs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                    )
                    v[nrow, 0, 315:716] = df_perc_br_sim.loc[:, "Br_conc_mmol"].values.astype(float)
                    v.attrs.update(long_name="bulk sample of bromide in percolation", units="mmol/l")
                except ValueError:
                    v = f.groups[f"{year}"].get("C_q_ss_mmol_bs")
                    v[nrow, 0, 315:716] = df_perc_br_sim.loc[:, "Br_conc_mmol"].values.astype(float)
                try:
                    v = f.groups[f"{year}"].create_variable(
                        "q_ss_bs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                    )
                    v[nrow, 0, 315:716] = df_perc_br_sim.loc[:, "perc"].values.astype(float)
                    v.attrs.update(long_name="bulk sample of percolation", units="mm/dt")
                except ValueError:
                    v = f.groups[f"{year}"].get("M_q_ss_bs")
                    v[nrow, 0, 315:716] = df_perc_br_sim.loc[:, "perc"].values.astype(float)
                try:
                    v = f.groups[f"{year}"].create_variable(
                        "M_q_ss_bs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                    )
                    v[nrow, 0, 315:716] = df_perc_br_sim.loc[:, "Br_mg"].values.astype(float)
                    v.attrs.update(long_name="bulk sample bromide mass in percolation", units="mg")
                except ValueError:
                    v = f.groups[f"{year}"].get("M_q_ss_bs")
                    v[nrow, 0, 315:716] = df_perc_br_sim.loc[:, "Br_mg"].values.astype(float)

        axes.set_ylabel("Br [mmol $l^{-1}$]")
        axes.set_xlabel("Time [days since injection]")
        axes.set_ylim(
            0,
        )
        axes.set_xlim((0, 400))
        axes.legend(fontsize=6, frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
        fig.tight_layout()
        file = f"bromide_breakthrough_{transport_model_structure}_alpha_transp_{alpha_transp}_alpha_q_{alpha_q}.png"
        path = base_path_figs / file
        fig.savefig(path, dpi=250)

        # write evaluation metrics to .csv
        path_csv = (
            base_path_output
            / f"bromide_metrics_{transport_model_structure}_alpha_transp_{alpha_transp}_alpha_q_{alpha_q}.csv"
        )
        df_metrics_year.to_csv(path_csv, header=True, index=True, sep=";")

    return


if __name__ == "__main__":
    main()
