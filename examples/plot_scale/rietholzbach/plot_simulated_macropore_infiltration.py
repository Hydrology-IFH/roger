from pathlib import Path
import os
import scipy as sp
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.cm as cm
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

    data = pd.DataFrame(index=date_sim_hm100, columns=["PERC_obs [mm/day]", "PERC_sim [mm/day]"])
    data.iloc[1:, 0] = ds_obs["PERC"].isel(x=0, y=0).values
    data.iloc[:, 1] = ds_sim_hm100["q_ss"].isel(y=0).values
    file = base_path_figs / f"PERC.txt"
    data.iloc[1:, :].to_csv(file, header=True, index=True, sep="\t")

    file = base_path / "svat_monte_carlo" / "figures" / "params_metrics.txt"
    df_params_metrics = pd.read_csv(file, sep="\t")
    df_params_metrics100 = df_params_metrics.copy()
    df_params_metrics100.loc[:, "id"] = range(len(df_params_metrics100.index))
    df_params_metrics100 = df_params_metrics100.sort_values(by="KGE_multi", ascending=False)

    # plot macropore infiltration
    df_params_mp_100 = df_params_metrics100.loc[: df_params_metrics100.index[99], ["lmpv", "dmpv", "ks"]]
    for x in range(100):
        inf_tot = (
            ds_sim_hm100["inf_mat_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_sc_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        )
        inf_mp = ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        df_params_mp_100.loc[df_params_mp_100.index[x], "inf_mp_share"] = (onp.sum(inf_mp) / onp.sum(inf_tot)) * 100

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(6, 2))
    axes[0].scatter(df_params_mp_100["lmpv"], df_params_mp_100["inf_mp_share"], s=2, color="black")
    axes[0].set_ylim(
        0,
    )
    axes[0].set_ylabel(r"$INF_{mp}$ [%]")
    axes[0].set_xlabel(r"$l_{mpv}$ [mm]")
    axes[1].scatter(df_params_mp_100["dmpv"], df_params_mp_100["inf_mp_share"], s=2, color="black")
    axes[1].set_ylim(
        0,
    )
    axes[1].set_xlabel(r"$\rho_{mpv}$ [$m^{-2}$]")
    axes[2].scatter(df_params_mp_100["ks"], df_params_mp_100["inf_mp_share"], s=2, color="black")
    axes[2].set_ylim(
        0,
    )
    axes[2].set_xlabel(r"$k_s$ [mm $h^{-1}$]")
    fig.tight_layout()
    file = "inf_mp_share_vs_mp_params1.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    res = sp.stats.linregress(df_params_mp_100["ks"].values, df_params_mp_100["inf_mp_share"].values)
    xpred = onp.arange(5, 150)

    cmap = cm.get_cmap("Oranges")
    norm_lmpv = Normalize(vmin=10, vmax=1500)
    norm_dmpv = Normalize(vmin=10, vmax=300)
    fig, axes = plt.subplots(1, 1, sharey=True, figsize=(4, 3))
    for x in range(100):
        lmpv = df_params_mp_100.loc[df_params_mp_100.index[x], "lmpv"]
        dmpv = df_params_mp_100.loc[df_params_mp_100.index[x], "dmpv"]
        axes.plot(
            df_params_mp_100.loc[df_params_mp_100.index[x], "ks"],
            df_params_mp_100.loc[df_params_mp_100.index[x], "inf_mp_share"],
            "o",
            markersize=1 + norm_dmpv(dmpv) * 7,
            color=cmap(norm_lmpv(lmpv)),
        )
    axes.plot(xpred, res.intercept + res.slope * xpred, "black")
    rlab = r"$R^2$: %s" % (f"{res.rvalue**2:.2f}")
    axes.text(
        0.12, 0.93, rlab, fontsize=9, horizontalalignment="center", verticalalignment="center", transform=axes.transAxes
    )
    axes.set_ylim(
        0,
    )
    axes.set_ylabel(r"$INF_{mp}$ [%]")
    axes.set_xlabel(r"$k_s$ [mm $h^{-1}$]")
    axl = fig.add_axes([0.8, 0.48, 0.02, 0.5])
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm_lmpv, orientation="vertical", ticks=[10, 500, 1000, 1500])
    cb1.set_label(r"$l_{mpv}$ [mm]")

    (l1,) = axes.plot([], [], "or", markersize=1)
    (l2,) = axes.plot([], [], "or", markersize=2)
    (l3,) = axes.plot([], [], "or", markersize=6)
    (l4,) = axes.plot([], [], "or", markersize=8)
    labels = ["10", "100", "200", "300"]
    axes.legend(
        [l1, l2, l3, l4],
        labels,
        ncol=1,
        frameon=False,
        handlelength=2,
        borderpad=1.8,
        handletextpad=1,
        title=r"$\rho_{mpv}$ [$m^{-2}$]",
        scatterpoints=1,
        loc="lower right",
        bbox_to_anchor=(1.4, -0.15),
    )
    fig.subplots_adjust(bottom=0.15, right=0.75, left=0.12, top=0.98)
    file = "inf_mp_share_vs_mp_params.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    axes.plot(ds_sim_hm100["Time"].values, prec, lw=1, color="blue", ls="-", label="Precipitation")
    for x in range(100):
        sim_vals = (
            ds_sim_hm100["inf_mat_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_sc_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        )
        axes.plot(ds_sim_hm100["Time"].values, sim_vals, lw=1, color="red", ls="-", alpha=0.8)
        sim_vals = ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        axes.plot(ds_sim_hm100["Time"].values, sim_vals, lw=1, color="red", ls=":", alpha=1)
    axes.set_ylabel(r"[mm/day]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    fig.tight_layout()
    file = f"inf_mp_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"inf_mp_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    axes.plot(ds_sim_hm100["Time"].values, prec, lw=1, color="blue", ls="-", label="Precipitation")
    for x in range(100):
        sim_vals = (
            ds_sim_hm100["inf_mat_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_sc_rz"].isel(x=x, y=0).values
            + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        )
        sim_vals_cum = onp.cumsum(sim_vals)
        axes.plot(ds_sim_hm100["Time"].values, sim_vals_cum, lw=1, color="red", ls="-", alpha=0.8)
        sim_vals = ds_sim_hm100["inf_mp_rz"].isel(x=x, y=0).values + ds_sim_hm100["inf_ss"].isel(x=x, y=0).values
        sim_vals_cum = onp.cumsum(sim_vals)
        axes.plot(ds_sim_hm100["Time"].values, sim_vals_cum, lw=1, color="red", ls=":", alpha=1)
    axes.plot(
        ds_sim_hm100["Time"].values, sim_vals_cum, lw=1, color="red", ls="-", alpha=0.8, label="Total infiltration"
    )
    axes.plot(
        ds_sim_hm100["Time"].values, sim_vals_cum, lw=1, color="red", ls=":", alpha=1, label="Macropore infiltration"
    )
    axes.set_ylabel(r"[mm]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    axes.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    file = f"inf_mp_cumulated_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"inf_mp_cumulated_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    axes.plot(ds_sim_hm100["Time"].values, prec, lw=1, color="blue", ls="-", label="Precipitation")
    sim_vals = (
        ds_sim_hm100["inf_mat_rz"].isel(y=0).values
        + ds_sim_hm100["inf_mp_rz"].isel(y=0).values
        + ds_sim_hm100["inf_sc_rz"].isel(y=0).values
        + ds_sim_hm100["inf_ss"].isel(y=0).values
    )
    sim_vals_5 = onp.nanquantile(sim_vals, 0.05, axis=0)
    sim_vals_50 = onp.nanmedian(sim_vals, axis=0)
    sim_vals_95 = onp.nanquantile(sim_vals, 0.95, axis=0)
    axes.plot(ds_sim_hm100["Time"].values, sim_vals_50, lw=1, color="red", ls="-", label="Total infiltration")
    axes.fill_between(ds_sim_hm100["Time"].values, sim_vals_5, sim_vals_95, color="red", edgecolor=None, alpha=0.2)
    sim_vals = ds_sim_hm100["inf_mp_rz"].isel(y=0).values + ds_sim_hm100["inf_ss"].isel(y=0).values
    sim_vals = onp.cumsum(sim_vals, axis=1)
    sim_vals_5 = onp.nanquantile(sim_vals, 0.05, axis=0)
    sim_vals_50 = onp.nanmedian(sim_vals, axis=0)
    sim_vals_95 = onp.nanquantile(sim_vals, 0.95, axis=0)
    axes.plot(ds_sim_hm100["Time"].values, sim_vals_50, lw=1, color="orange", ls="-", label="Macropore infiltration")
    axes.fill_between(ds_sim_hm100["Time"].values, sim_vals_5, sim_vals_95, color="orange", edgecolor=None, alpha=0.2)
    axes.set_ylabel(r"[mm/day]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    axes.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    file = f"inf_mp_conf_int_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"inf_mp_conf_int_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    prec_cum = onp.cumsum(prec)
    axes.plot(ds_sim_hm100["Time"].values, prec_cum, lw=1, color="blue", ls="-", label="Precipitation")
    sim_vals = (
        ds_sim_hm100["inf_mat_rz"].isel(y=0).values
        + ds_sim_hm100["inf_mp_rz"].isel(y=0).values
        + ds_sim_hm100["inf_sc_rz"].isel(y=0).values
        + ds_sim_hm100["inf_ss"].isel(y=0).values
    )
    sim_vals_cum = onp.cumsum(sim_vals, axis=1)
    sim_vals_cum_5 = onp.nanquantile(sim_vals_cum, 0.05, axis=0)
    sim_vals_cum_50 = onp.nanmedian(sim_vals_cum, axis=0)
    sim_vals_cum_95 = onp.nanquantile(sim_vals_cum, 0.95, axis=0)
    axes.plot(ds_sim_hm100["Time"].values, sim_vals_cum_50, lw=1, color="red", ls="-", label="Total infiltration")
    axes.fill_between(
        ds_sim_hm100["Time"].values, sim_vals_cum_5, sim_vals_cum_95, color="red", edgecolor=None, alpha=0.2
    )
    sim_vals = ds_sim_hm100["inf_mp_rz"].isel(y=0).values + ds_sim_hm100["inf_ss"].isel(y=0).values
    sim_vals_cum = onp.cumsum(sim_vals, axis=1)
    sim_vals_cum_5 = onp.nanquantile(sim_vals_cum, 0.05, axis=0)
    sim_vals_cum_50 = onp.nanmedian(sim_vals_cum, axis=0)
    sim_vals_cum_95 = onp.nanquantile(sim_vals_cum, 0.95, axis=0)
    axes.plot(
        ds_sim_hm100["Time"].values, sim_vals_cum_50, lw=1, color="orange", ls="-", label="Macropore infiltration"
    )
    axes.fill_between(
        ds_sim_hm100["Time"].values, sim_vals_cum_5, sim_vals_cum_95, color="orange", edgecolor=None, alpha=0.2
    )
    axes.set_ylabel(r"[mm]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    axes.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    file = f"inf_mp_cumulated_conf_int_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = f"inf_mp_cumulated_conf_int_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(1, 1, sharex="col", figsize=(6, 1.5))
    prec = ds_sim_hm100["prec"].isel(x=0, y=0).values
    prec_cum = onp.cumsum(prec)
    axes.plot(ds_sim_hm100["Time"].values, prec_cum, lw=1, color="blue", ls="-", label="Precipitation")
    sim_vals = (
        ds_sim_hm100["inf_mat_rz"].isel(y=0).values
        + ds_sim_hm100["inf_mp_rz"].isel(x=0, y=0).values
        + ds_sim_hm100["inf_sc_rz"].isel(x=0, y=0).values
        + ds_sim_hm100["inf_ss"].isel(x=0, y=0).values
    )
    sim_vals_cum = onp.cumsum(sim_vals, axis=1)
    sim_vals_cum_5 = onp.nanquantile(sim_vals_cum, 0.05, axis=0)
    sim_vals_cum_50 = onp.nanmedian(sim_vals_cum, axis=0)
    sim_vals_cum_95 = onp.nanquantile(sim_vals_cum, 0.95, axis=0)
    axes.plot(ds_sim_hm100["Time"].values, sim_vals_cum_50, lw=1, color="red", ls="-", label="Total infiltration")
    axes.fill_between(
        ds_sim_hm100["Time"].values, sim_vals_cum_5, sim_vals_cum_95, color="red", edgecolor=None, alpha=0.2
    )
    sim_vals = ds_sim_hm100["inf_ss"].isel(y=0).values
    sim_vals_cum = onp.cumsum(sim_vals, axis=1)
    sim_vals_cum_5 = onp.nanquantile(sim_vals_cum, 0.05, axis=0)
    sim_vals_cum_50 = onp.nanmedian(sim_vals_cum, axis=0)
    sim_vals_cum_95 = onp.nanquantile(sim_vals_cum, 0.95, axis=0)
    axes.plot(
        ds_sim_hm100["Time"].values, sim_vals_cum_50, lw=1, color="orange", ls="-", label="Macropore infiltration"
    )
    axes.fill_between(
        ds_sim_hm100["Time"].values, sim_vals_cum_5, sim_vals_cum_95, color="orange", edgecolor=None, alpha=0.2
    )
    axes.set_ylabel(r"[mm]")
    axes.set_xlim((ds_sim_hm100["Time"].values[0], ds_sim_hm100["Time"].values[-1]))
    axes.set_ylim(
        0,
    )
    axes.set_xlabel(r"Time [year]")
    axes.legend(frameon=False, loc="upper left", fontsize=8)
    fig.tight_layout()
    file = "inf_mp_ss_cumulated_conf_int_best_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    file = "inf_mp_ss_cumulated_conf_int_best_100_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    fig.savefig(path, dpi=300)
    plt.close("all")
    return


if __name__ == "__main__":
    main()
