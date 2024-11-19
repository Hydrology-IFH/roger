from pathlib import Path
import os
import glob
import datetime
import h5netcdf
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import click
import roger.tools.evaluation as eval_utils
import matplotlib as mpl
import seaborn as sns

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 7
mpl.rcParams["axes.titlesize"] = 9
mpl.rcParams["axes.labelsize"] = 8
mpl.rcParams["xtick.labelsize"] = 7
mpl.rcParams["ytick.labelsize"] = 7
mpl.rcParams["legend.fontsize"] = 7
mpl.rcParams["legend.title_fontsize"] = 8
sns.set_style("ticks")


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    transport_models = [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
    ]

    # load observations (measured data)
    path_obs = base_path.parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
    days_obs = ds_obs["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_obs = num2date(
        days_obs,
        units=f"days since {ds_obs['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_obs = ds_obs.assign_coords(Time=("Time", date_obs))

    # average observed soil water content of previous 5 days
    window = 5
    df_thetap = pd.DataFrame(index=date_obs, columns=["doy", "theta", "sc"])
    df_thetap.loc[:, "doy"] = df_thetap.index.day_of_year
    df_thetap.loc[:, "theta"] = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
    df_thetap.loc[df_thetap.index[window - 1] :, f"theta_avg{window}"] = (
        df_thetap.loc[:, "theta"].rolling(window=window).mean().iloc[window - 1 :].values
    )
    df_thetap.iloc[:window, 2] = onp.nan
    theta_lower = df_thetap.loc[:, f"theta_avg{window}"].quantile(0.1)
    theta_upper = df_thetap.loc[:, f"theta_avg{window}"].quantile(0.9)
    cond1 = df_thetap[f"theta_avg{window}"] < theta_lower
    cond2 = (df_thetap[f"theta_avg{window}"] >= theta_lower) & (df_thetap[f"theta_avg{window}"] < theta_upper)
    cond3 = df_thetap[f"theta_avg{window}"] >= theta_upper
    df_thetap.loc[cond1, "sc"] = 1  # dry
    df_thetap.loc[cond2, "sc"] = 2  # normal
    df_thetap.loc[cond3, "sc"] = 3  # wet

    for tm in transport_models:
        click.echo(f"Plot results of {tm}")
        tms = tm.replace(" ", "_")

        # load hydrologic simulation
        hm_file = base_path / "input" / f"SVAT_best_for_{tms}.nc"
        ds_sim_hm = xr.open_dataset(hm_file, engine="h5netcdf")
        days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        time_origin = ds_sim_hm["Time"].attrs["time_origin"]
        date_sim_hm = num2date(
            days_sim_hm,
            units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))
        ds_sim_hm.Time.attrs["time_origin"] = time_origin
        # load transport simulation
        tm_file = base_path_output / f"{tms}.nc"
        ds_sim_tm = xr.open_dataset(tm_file, engine="h5netcdf", decode_times=False)
        date_sim_tm = num2date(
            ds_sim_tm["Time"].values,
            units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))

        # compare observations and simulations
        nrows = ds_sim_tm.dims["x"]
        idx = ds_sim_tm.Time.values  # time index
        # figure to compare simulations with observations
        fig, ax = plt.subplots(figsize=(6, 2))
        # DataFrame with sampled model parameters and the corresponding metrics
        d18O_perc_bs = onp.zeros((nrows, 1, len(idx)))
        for nrow in range(nrows):
            df_idx_bs = pd.DataFrame(index=date_obs, columns=["sol"])
            df_idx_bs.loc[:, "sol"] = ds_obs["d18O_PERC"].isel(x=0, y=0).values
            idx_bs = df_idx_bs["sol"].dropna().index
            # calculate simulated oxygen-18 bulk sample
            df_perc_18O_obs = pd.DataFrame(index=date_obs, columns=["perc_obs", "d18O_perc_obs"])
            df_perc_18O_obs.loc[:, "perc_obs"] = ds_obs["PERC"].isel(x=0, y=0).values
            df_perc_18O_obs.loc[:, "d18O_perc_obs"] = ds_obs["d18O_PERC"].isel(x=0, y=0).values
            sample_no = pd.DataFrame(index=idx_bs, columns=["sample_no"])
            sample_no = sample_no.loc["1997":"2007"]
            sample_no["sample_no"] = range(len(sample_no.index))
            df_perc_18O_sim = pd.DataFrame(index=date_sim_tm, columns=["perc_sim", "d18O_perc_sim"])
            df_perc_18O_sim["perc_sim"] = ds_sim_hm["q_ss"].isel(y=0).values
            df_perc_18O_sim["d18O_perc_sim"] = ds_sim_tm["C_iso_q_ss"].isel(x=nrow, y=0).values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no)
            df_perc_18O_sim.loc[:, "sample_no"] = df_perc_18O_sim.loc[:, "sample_no"].fillna(method="bfill", limit=14)
            perc_sum = df_perc_18O_sim.groupby(["sample_no"]).sum().loc[:, "perc_sim"]
            sample_no["perc_sum"] = perc_sum.values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no["perc_sum"])
            df_perc_18O_sim.loc[:, "perc_sum"] = df_perc_18O_sim.loc[:, "perc_sum"].fillna(method="bfill", limit=14)
            df_perc_18O_sim["weight"] = df_perc_18O_sim["perc_sim"] / df_perc_18O_sim["perc_sum"]
            df_perc_18O_sim["d18O_weight"] = df_perc_18O_sim["d18O_perc_sim"] * df_perc_18O_sim["weight"]
            d18O_sample = df_perc_18O_sim.groupby(["sample_no"]).sum().loc[:, "d18O_weight"]
            sample_no["d18O_sample"] = d18O_sample.values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no["d18O_sample"])
            cond = df_perc_18O_sim["d18O_sample"] == 0
            df_perc_18O_sim.loc[cond, "d18O_sample"] = onp.NaN
            d18O_perc_bs[nrow, 0, :] = df_perc_18O_sim.loc[:, "d18O_sample"].values
            # calculate observed oxygen-18 bulk sample
            df_perc_18O_obs.loc[:, "d18O_perc_bs"] = df_perc_18O_obs["d18O_perc_obs"].fillna(method="bfill", limit=14)

            perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(["sample_no"]).sum().loc[:, "perc_obs"]
            sample_no["perc_obs_sum"] = perc_sample_sum_obs.values
            df_perc_18O_sim = df_perc_18O_sim.join(sample_no["perc_obs_sum"])
            df_perc_18O_sim.loc[:, "perc_obs_sum"] = df_perc_18O_sim.loc[:, "perc_obs_sum"].fillna(
                method="bfill", limit=14
            )

            # join observations on simulations
            for sc, sc1 in zip([0, 1, 2, 3], ["", "dry", "normal", "wet"]):
                obs_vals = ds_obs["d18O_PERC"].isel(x=0, y=0).values
                sim_vals = d18O_perc_bs[nrow, 0, :]
                df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
                df_obs.loc[:, "obs"] = obs_vals
                df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)

                if sc > 0:
                    df_rows = pd.DataFrame(index=df_eval.index).join(df_thetap)
                    rows = df_rows["sc"].values == sc
                    df_eval = df_eval.loc[rows, :]
                df_eval = df_eval.dropna()
            # plot observed and simulated d18O in percolation
            ax.plot(ds_sim_tm.Time.values, ds_sim_tm["C_iso_q_ss"].isel(x=nrow, y=0).values, color="red", zorder=2)
            ax.scatter(df_eval.index, df_eval.iloc[:, 0], color="red", s=4, zorder=1)
            ax.scatter(df_eval.index, df_eval.iloc[:, 1], color="blue", s=4, zorder=3)

        # write figure to .png
        ax.set_ylabel(r"$\delta^{18}$O [‰]")
        ax.set_xlabel("Time [year]")
        ax.set_xlim((ds_sim_tm.Time.values[0], ds_sim_tm.Time.values[-1]))
        fig.tight_layout()
        file = base_path_figs / f"d18O_perc_sim_obs_{tms}.png"
        fig.savefig(file, dpi=250)
        plt.close("all")

        # calculate upper interquartile travel time for each time step
        df_age = pd.DataFrame(index=idx[2:], columns=["MTT", "TT_50", "TT25", "TT75", "MRT", "RT_50", "RT25", "RT75"])
        df_age.loc[:, "MTT"] = ds_sim_tm["ttavg_q_ss"].isel(x=0, y=0).values[2:]
        df_age.loc[:, "TT_50"] = ds_sim_tm["tt50_q_ss"].isel(x=0, y=0).values[2:]
        df_age.loc[:, "TT10"] = ds_sim_tm["tt10_q_ss"].isel(x=0, y=0).values[2:]
        df_age.loc[:, "TT90"] = ds_sim_tm["tt90_q_ss"].isel(x=0, y=0).values[2:]
        df_age.loc[:, "MRT"] = ds_sim_tm["rtavg_s"].isel(x=0, y=0).values[2:]
        df_age.loc[:, "RT_50"] = ds_sim_tm["rt50_s"].isel(x=0, y=0).values[2:]
        df_age.loc[:, "RT10"] = ds_sim_tm["rt10_s"].isel(x=0, y=0).values[2:]
        df_age.loc[:, "RT90"] = ds_sim_tm["rt90_s"].isel(x=0, y=0).values[2:]
        df_age.loc[:, "q_ss"] = ds_sim_hm["q_ss"].isel(y=0).values[2:]

        # mean and median travel time over entire simulation period
        df_age_mean = pd.DataFrame(index=["avg"], columns=["MTT", "TT_50", "MRT", "RT_50"])
        df_age_mean.loc["avg", "MTT"] = onp.nanmean(df_age["MTT"].values)
        df_age_mean.loc["avg", "TT_50"] = onp.nanmean(df_age["TT_50"].values)
        df_age_mean.loc["avg", "MRT"] = onp.nanmean(df_age["MRT"].values)
        df_age_mean.loc["avg", "RT_50"] = onp.nanmean(df_age["RT_50"].values)
        file_str = "age_mean_perc_%s.csv" % (tms)
        path_csv = base_path_figs / file_str
        df_age_mean.to_csv(path_csv, header=True, index=True, sep="\t")

        # plot mean and median travel time
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 3))
        axes[0].plot(df_age.index, df_age["MTT"], ls="--", lw=2, color="magenta")
        axes[0].plot(df_age.index, df_age["TT_50"], ls=":", lw=2, color="purple")
        axes[0].fill_between(df_age.index, df_age["TT10"], df_age["TT90"], color="purple", edgecolor=None, alpha=0.2)
        tt_50 = str(int(df_age_mean.loc["avg", "TT_50"]))
        tt_mean = str(int(df_age_mean.loc["avg", "MTT"]))
        axes[0].text(
            0.88,
            0.93,
            r"$\overline{TT}_{50}$: %s days" % (tt_50),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[0].transAxes,
        )
        axes[0].text(
            0.88,
            0.83,
            r"$\overline{TT}$: %s days" % (tt_mean),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_ylabel("age\n[days]")
        axes[0].set_ylim(
            0,
        )
        axes[0].set_xlim((df_age.index[0], df_age.index[-1]))
        axes[1].bar(df_age.index, df_age["q_ss"], width=-1, align="edge", edgecolor="grey")
        axes[1].set_ylim(
            0,
        )
        axes[1].invert_yaxis()
        axes[1].set_xlim((df_age.index[0], df_age.index[-1]))
        axes[1].set_ylabel("Percolation\n[mm $day^{-1}$]")
        axes[1].set_xlabel(r"Time [year]")
        fig.tight_layout()
        file_str = "mean_median_tt_perc_%s.pdf" % (tms)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

        # plot mean and median travel time and residence time
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(6, 3))
        axes[0].plot(df_age.index, df_age["MRT"], ls="--", lw=2, color="magenta")
        axes[0].plot(df_age.index, df_age["RT_50"], ls=":", lw=2, color="purple")
        axes[0].fill_between(df_age.index, df_age["RT10"], df_age["RT90"], color="purple", edgecolor=None, alpha=0.2)
        rt_50 = str(int(df_age_mean.loc["avg", "RT_50"]))
        rt_mean = str(int(df_age_mean.loc["avg", "MRT"]))
        axes[0].text(
            0.88,
            0.93,
            r"$\overline{RT}_{50}$: %s days" % (rt_50),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[0].transAxes,
        )
        axes[0].text(
            0.88,
            0.83,
            r"$\overline{RT}$: %s days" % (rt_mean),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[0].transAxes,
        )
        axes[0].set_ylabel("age\n[days]")
        axes[0].set_ylim(
            0,
        )
        axes[0].set_xlim((df_age.index[0], df_age.index[-1]))
        axes[1].plot(df_age.index, df_age["MTT"], ls="--", lw=2, color="magenta")
        axes[1].plot(df_age.index, df_age["TT_50"], ls=":", lw=2, color="purple")
        axes[1].fill_between(df_age.index, df_age["TT10"], df_age["TT90"], color="purple", edgecolor=None, alpha=0.2)
        tt_50 = str(int(df_age_mean.loc["avg", "TT_50"]))
        tt_mean = str(int(df_age_mean.loc["avg", "MTT"]))
        axes[1].text(
            0.88,
            0.93,
            r"$\overline{TT}_{50}$: %s days" % (tt_50),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[1].transAxes,
        )
        axes[1].text(
            0.88,
            0.83,
            r"$\overline{TT}$: %s days" % (tt_mean),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_ylabel("age\n[days]")
        axes[1].set_ylim(
            0,
        )
        axes[1].set_xlim((df_age.index[0], df_age.index[-1]))
        axes[1].set_xlabel(r"Time [year]")
        fig.tight_layout()
        file_str = "mean_median_rt_tt_perc_%s.pdf" % (tms)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

        # plot observed and simulated d18O in percolation
        fig, ax = plt.subplots(figsize=(6, 1.5))
        # join observations on simulations
        obs_vals_bs = ds_obs["d18O_PERC"].isel(x=0, y=0).values
        sim_vals_bs = d18O_perc_bs[0, 0, :]
        sim_vals = ds_sim_tm["C_iso_q_ss"].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
        df_obs.loc[:, "obs"] = obs_vals_bs
        df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals_bs, df_obs)
        df_eval = df_eval.dropna()
        # plot observed and simulated d18O in percolation
        sim_vals = ds_sim_tm["C_iso_q_ss"].isel(x=0, y=0).values
        sim_vals = onp.where((sim_vals > 0), onp.nan, sim_vals)
        ax.plot(ds_sim_tm.Time.values, sim_vals, color="red", zorder=2)
        ax.scatter(df_eval.index, df_eval.iloc[:, 1], color="blue", s=1, zorder=3)
        # write figure to .png
        ax.set_ylabel(r"$\delta^{18}$O [‰]")
        ax.set_xlabel("Time [year]")
        ax.set_xlim((ds_sim_tm.Time.values[0], ds_sim_tm.Time.values[-1]))
        fig.tight_layout()
        file = base_path_figs / f"d18O_perc_sim_obs_{tms}.png"
        fig.savefig(file, dpi=250)
        plt.close("all")

        # plot numerical errors
        sd_dS_num_error = "{:.2e}".format(onp.std(ds_sim_tm["dS_num_error"].isel(y=0).values))
        max_dS_num_error = "{:.2e}".format(onp.max(ds_sim_tm["dS_num_error"].isel(y=0).values))
        sd_dC_num_error = "{:.2e}".format(onp.std(ds_sim_tm["dC_num_error"].isel(y=0).values))
        max_dC_num_error = "{:.2e}".format(onp.max(ds_sim_tm["dC_num_error"].isel(y=0).values))
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(6, 3))
        axes[0].plot(
            ds_sim_tm.Time.values, ds_sim_tm["dS_num_error"].isel(x=0, y=0).values, ls="-", lw=1, color="black"
        )
        axes[0].set_ylabel("Bias\n[mm]")
        axes[0].set_ylim(
            0,
        )
        axes[0].set_xlim((ds_sim_tm.Time.values[0], ds_sim_tm.Time.values[-1]))
        axes[0].text(
            0.75,
            0.93,
            r"Error SD: %s" % (sd_dS_num_error),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[0].transAxes,
        )
        axes[0].text(
            0.75,
            0.83,
            r"Error Max: %s" % (max_dS_num_error),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[0].transAxes,
        )
        axes[1].plot(
            ds_sim_tm.Time.values, ds_sim_tm["dC_num_error"].isel(x=0, y=0).values, ls="-", lw=1, color="black"
        )
        axes[1].set_ylabel("Bias\n[mg/l]")
        axes[1].set_ylim(
            0,
        )
        axes[1].set_xlim((ds_sim_tm.Time.values[0], ds_sim_tm.Time.values[-1]))
        axes[1].text(
            0.75,
            0.93,
            r"Error SD: %s" % (sd_dC_num_error),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[1].transAxes,
        )
        axes[1].text(
            0.75,
            0.83,
            r"Error Max: %s" % (max_dC_num_error),
            size=6,
            horizontalalignment="left",
            verticalalignment="center",
            transform=axes[1].transAxes,
        )
        axes[1].set_xlabel(r"Time [year]")
        fig.tight_layout()
        file_str = "num_errors_%s.pdf" % (tms)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

        # write simulated bulk sample to output file
        ds_sim_tm = ds_sim_tm.load()  # required to release file lock
        ds_sim_tm = ds_sim_tm.close()
        del ds_sim_tm
        tm_file = base_path_output / f"{tms}.nc"
        with h5netcdf.File(tm_file, "a", decode_vlen_strings=False) as f:
            try:
                v = f.create_variable("d18O_perc_bs", ("x", "y", "Time"), float, compression="gzip", compression_opts=1)
                v[:, :, :] = d18O_perc_bs
                v.attrs.update(long_name="bulk sample of d18O in percolation", units="permil")
            except ValueError:
                v = f.get("d18O_perc_bs")
                v[:, :, :] = d18O_perc_bs
                v.attrs.update(long_name="bulk sample of d18O in percolation", units="permil")
    return


if __name__ == "__main__":
    main()
