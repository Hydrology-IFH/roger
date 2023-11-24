import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import datetime
import glob
import h5netcdf
import roger
import roger.tools.labels as labs
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


@click.command("main")
def main():
    base_path = Path(__file__).parent
    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    meteo_stations = ["breitnau", "ihringen"]
    # merge model output into single file
    states_hm_file = base_path_output / "SVAT.nc"
    if not os.path.exists(states_hm_file):
        for meteo_station in meteo_stations:
            path = str(base_path_output / f"SVAT_{meteo_station}.*.nc")
            diag_files = glob.glob(path)
            with h5netcdf.File(states_hm_file, "a", decode_vlen_strings=False) as f:
                if meteo_station not in list(f.groups.keys()):
                    f.create_group(meteo_station)
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title="RoGeR model results for common parameter set and input from DWD stations Breitnau and Ihringen",
                    institution="University of Freiburg, Chair of Hydrology",
                    references="",
                    comment="",
                    model_structure="SVAT model with free drainage",
                    roger_version=f"{roger.__version__}",
                )
                # collect dimensions
                for dfs in diag_files:
                    with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                        # set dimensions with a dictionary
                        if not dfs.split("/")[-1].split(".")[1] == "constant":
                            dict_dim = {
                                "x": len(df.variables["x"]),
                                "y": len(df.variables["y"]),
                                "Time": len(df.variables["Time"]),
                            }
                            time = onp.array(df.variables.get("Time"))
                for dfs in diag_files:
                    with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                        if not f.groups[meteo_station].dimensions:
                            f.groups[meteo_station].dimensions = dict_dim
                            v = f.groups[meteo_station].create_variable(
                                "x", ("x",), float, compression="gzip", compression_opts=1
                            )
                            v.attrs["long_name"] = "Model run"
                            v.attrs["units"] = ""
                            v[:] = onp.arange(dict_dim["x"])
                            v = f.groups[meteo_station].create_variable(
                                "y", ("y",), float, compression="gzip", compression_opts=1
                            )
                            v.attrs["long_name"] = ""
                            v.attrs["units"] = ""
                            v[:] = onp.arange(dict_dim["y"])
                            v = f.groups[meteo_station].create_variable(
                                "Time", ("Time",), float, compression="gzip", compression_opts=1
                            )
                            var_obj = df.variables.get("Time")
                            v.attrs.update(time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"])
                            v[:] = time
                        for key in list(df.variables.keys()):
                            var_obj = df.variables.get(key)
                            if key not in list(dict_dim.keys()) and var_obj.ndim == 3:
                                v = f.groups[meteo_station].create_variable(
                                    key, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                                )
                                vals = onp.array(var_obj)
                                v[:, :, :] = vals.swapaxes(0, 2)
                                v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])

        with h5netcdf.File(states_hm_file, "a", decode_vlen_strings=False) as f:
            for meteo_station in meteo_stations:
                # water for infiltration
                try:
                    v = f.groups[meteo_station].create_variable(
                        "inf_in", ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                    )
                except ValueError:
                    v = f.groups[meteo_station].variables.get("inf_in")
                vals = (
                    onp.array(f.groups[meteo_station].variables.get("prec"))
                    - onp.array(f.groups[meteo_station].variables.get("int_rain_top"))
                    - onp.array(f.groups[meteo_station].variables.get("int_rain_ground"))
                    - onp.array(f.groups[meteo_station].variables.get("int_snow_top"))
                    - onp.array(f.groups[meteo_station].variables.get("int_snow_ground"))
                    - onp.array(f.groups[meteo_station].variables.get("snow_ground"))
                    + onp.array(f.groups[meteo_station].variables.get("q_snow"))
                )
                v[:, :, :] = vals
                v.attrs.update(long_name="infiltration input", units="mm/day")

                # initial soil water content
                try:
                    v = f.groups[meteo_station].create_variable(
                        "theta_init", ("x", "y"), float, compression="gzip", compression_opts=1
                    )
                except ValueError:
                    v = f.groups[meteo_station].variables.get("theta_init")
                vals = onp.array(f.groups[meteo_station].variables.get("theta"))
                v[:, :] = vals[:, :, 0]
                v.attrs.update(long_name="initial soil water content", units="-")
                try:
                    v = f.groups[meteo_station].create_variable(
                        "S_s_init", ("x", "y"), float, compression="gzip", compression_opts=1
                    )
                except ValueError:
                    v = f.groups[meteo_station].variables.get("S_s_init")
                vals = onp.array(f.groups[meteo_station].variables.get("S_s"))
                v[:, :] = vals[:, :, 0]
                v.attrs.update(long_name="initial soil water content", units="mm")
                # end soil water content
                try:
                    v = f.groups[meteo_station].create_variable(
                        "theta_end", ("x", "y"), float, compression="gzip", compression_opts=1
                    )
                except ValueError:
                    v = f.groups[meteo_station].variables.get("theta_end")
                vals = onp.array(f.groups[meteo_station].variables.get("theta"))
                v[:, :] = vals[:, :, -1]
                v.attrs.update(long_name="end soil water content", units="-")
                try:
                    v = f.groups[meteo_station].create_variable(
                        "S_s_end", ("x", "y"), float, compression="gzip", compression_opts=1
                    )
                except ValueError:
                    v = f.groups[meteo_station].variables.get("S_s_end")
                vals = onp.array(f.groups[meteo_station].variables.get("S_s"))
                v[:, :] = vals[:, :, -1]
                v.attrs.update(long_name="end soil water content", units="mm")

    # plot input data
    meteo_stations = ["breitnau", "ihringen"]
    for i, meteo_station in enumerate(meteo_stations):
        # load input
        input_file = base_path / "input" / meteo_station / "forcing.nc"
        ds_input = xr.open_dataset(input_file, engine="h5netcdf")
        hours = ds_input.Time.values / onp.timedelta64(60 * 60, "s")
        date = num2date(
            hours,
            units=f"hours since {ds_input['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_input = ds_input.assign_coords(date=("Time", date))
        # aggregate to daily
        ds_input_avg_daily = ds_input.resample(Time="1D").mean(dim="Time")
        ds_input_sum_daily = ds_input.resample(Time="1D").sum(dim="Time")

        days = ds_input_avg_daily.Time.values / onp.timedelta64(24 * 60 * 60, "s")
        fig, axs = plt.subplots(3, 1, figsize=(6, 4))
        axs[0].plot(days, ds_input_avg_daily.TA.values.flatten(), "-", color="orange", linewidth=1)
        axs[0].set_ylabel(r"TA [°C]")
        axs[0].set_xlim(days[0], days[-1])
        axs[1].bar(
            days, ds_input_sum_daily.PREC.values.flatten(), width=0.1, color="blue", align="edge", edgecolor="blue"
        )
        axs[1].set_ylabel("PRECIP\n[mm/day]")
        axs[1].set_xlim(days[0], days[-1])
        axs[2].bar(
            days, ds_input_sum_daily.PET.values.flatten(), width=0.1, color="green", align="edge", edgecolor="green"
        )
        axs[2].set_ylabel("PET\n[mm/day]")
        axs[2].set_xlim(days[0], days[-1])
        axs[2].set_xlabel("Time [days]")
        fig.tight_layout()
        file = base_path_figs / f"input_{meteo_station}.png"
        fig.savefig(file, dpi=250)
        plt.close(fig=fig)

    # plot parameters
    csv_file = base_path / "parameters.csv"
    df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
    df_lu_count = df_params.groupby("lu_id")["lu_id"].count()
    params = ["dmpv", "lmpv", "z_soil", "theta_ac", "theta_ufc", "theta_pwp", "ks"]
    _bins = [10, 12, 8, 15, 30, 25, 16]
    _range = [(60, 160), (200, 800), (200, 1000), (0.05, 0.2), (0.05, 0.35), (0.0, 0.25), (0, 80)]
    df_params = df_params.loc[:, params]
    fig, axs = plt.subplots(4, 2, figsize=(6, 6), sharey=True)
    labels = [labs._LABS[param] for param in params]
    axs.flatten()[0].bar(df_lu_count.index, df_lu_count.values, color="grey", width=0.6)
    axs.flatten()[0].set_xticks(df_lu_count.index, ("5", "6", "7", "8", "9", "10", "11", "12", "13"))
    axs.flatten()[0].set_xlabel("lu_id")
    for i, param in enumerate(params):
        if i == 2:
            axs.flatten()[3].bar([1, 2, 3], [2000, 2000, 2000], color="grey", width=0.2)
            axs.flatten()[3].set_xticks([1, 2, 3], ("300", "600", "900"))
            axs.flatten()[3].set_xlabel(r"$z_{soil}$ [mm]")
        else:
            axs.flatten()[i + 1].hist(df_params.loc[:, param], bins=_bins[i], range=_range[i], color="grey")
            axs.flatten()[i + 1].set_xlabel(labs._LABS[param])
    axs[0, 0].set_ylabel("# grid cells")
    axs[1, 0].set_ylabel("# grid cells")
    axs[2, 0].set_ylabel("# grid cells")
    axs[3, 0].set_ylabel("# grid cells")
    fig.tight_layout()
    file = base_path_figs / "parameters_hist1.png"
    fig.savefig(file, dpi=250)
    file = base_path_figs / "parameters_hist1.pdf"
    fig.savefig(file, dpi=250)

    csv_file = base_path / "parameters.csv"
    df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
    params = ["kf"]
    _bins = [10]
    _range = [(0, 5)]
    fig, axs = plt.subplots(1, 2, figsize=(6, 1.5), sharey=True)
    axs.flatten()[0].hist(df_params.loc[:, "kf"], bins=_bins[0], range=_range[0], color="grey")
    axs.flatten()[0].set_xlabel(labs._LABS["kf"])
    axs.flatten()[0].set_ylabel("# grid cells")
    axs.flatten()[1].axis("off")
    fig.tight_layout()
    file = base_path_figs / "parameters_hist2.png"
    fig.savefig(file, dpi=250)
    file = base_path_figs / "parameters_hist2.pdf"
    fig.savefig(file, dpi=250)

    theta_sat = (
        df_params.loc[:, "theta_pwp"].values.astype(float)
        + df_params.loc[:, "theta_ufc"].values.astype(float)
        + df_params.loc[:, "theta_ac"].values.astype(float)
    )
    theta_pwp = df_params.loc[:, "theta_pwp"].values.astype(float)

    # plot simulations
    meteo_stations = ["breitnau", "ihringen"]
    vars_sim = [
        "prec",
        "int_prec",
        "q_snow",
        "inf_in",
        "inf_mat",
        "inf_mp",
        "inf_sc",
        "q_hof",
        "q_sof",
        "q_ss",
        "pet",
        "aet",
        "evap_sur",
        "evap_soil",
        "transp",
    ]
    idx_percentiles = ["min", "q25", "median", "mean", "q75", "max"]
    ll_df_sim_sum = []
    ll_df_sim_sum_tot = []
    for i, meteo_station in enumerate(meteo_stations):
        # load simulation
        states_hm_file = base_path_output / "SVAT.nc"
        ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf", group=meteo_station)

        # assign date
        days_sim = ds_sim.Time.values / onp.timedelta64(24 * 60 * 60, "s")
        ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

        vals = ds_sim["theta"].isel(y=0).values
        max_vals = onp.max(vals, axis=1)
        test = onp.where(max_vals > theta_sat)
        onp.argmax(max_vals - theta_sat)
        min_vals = onp.min(vals, axis=1)
        test1 = onp.where(min_vals < theta_pwp)

        # sums per grid
        ds_sim_sum = ds_sim.sum(dim="Time")
        nx = ds_sim_sum.dims["x"]  # number of rows
        df = pd.DataFrame(index=range(nx))
        for var_sim in vars_sim:
            df.loc[:, var_sim] = ds_sim_sum[var_sim].values.flatten()

        df_percentiles = pd.DataFrame(index=idx_percentiles, columns=vars_sim)
        for var_sim in vars_sim:
            df_percentiles.loc["min", var_sim] = df.loc[:, var_sim].min()
            df_percentiles.loc["q25", var_sim] = df.loc[:, var_sim].quantile(0.25)
            df_percentiles.loc["median", var_sim] = df.loc[:, var_sim].median()
            df_percentiles.loc["mean", var_sim] = df.loc[:, var_sim].mean()
            df_percentiles.loc["q75", var_sim] = df.loc[:, var_sim].quantile(0.75)
            df_percentiles.loc["max", var_sim] = df.loc[:, var_sim].max()
        file = base_path_output / f"percentiles_{meteo_station}.csv"
        df_percentiles.to_csv(file, header=True, index=True, sep=";")

        file = base_path_output / f"summary_{meteo_station}.txt"
        df.to_csv(file, header=True, index=False, sep="\t")
        df.loc[:, "meteo_station"] = meteo_station
        df.loc[:, "idx"] = df.index

        ll_df_sim_sum.append(df)

        # total sums
        ds_sim_sum_tot = ds_sim.sum()
        df = pd.DataFrame(index=["sum"])
        for j, var_sim in enumerate(vars_sim):
            df.loc[:, var_sim] = ds_sim_sum_tot[var_sim].values
        df.loc[:, "meteo_station"] = meteo_station

        ll_df_sim_sum_tot.append(df)

        if meteo_station == "breitnau":
            color = "#f1b6bf"
        elif meteo_station == "ihringen":
            color = "#b83889"

        vars_sim_trace = [
            "S_s",
            "theta",
            "int_prec",
            "q_snow",
            "inf_in",
            "inf_mat",
            "inf_mp",
            "inf_sc",
            "q_hof",
            "q_sof",
            "q_ss",
            "aet",
            "evap_sur",
            "evap_soil",
            "transp",
        ]
        nx = ds_sim.dims["x"]
        days_sim = ds_sim.Time.values / onp.timedelta64(24 * 60 * 60, "s")
        for j, var_sim in enumerate(vars_sim_trace):
            fig, ax = plt.subplots(1, 1, figsize=(6, 2))
            vals = ds_sim[var_sim].isel(y=0).values
            median_vals = onp.median(vals, axis=0)
            min_vals = onp.min(vals, axis=0)
            max_vals = onp.max(vals, axis=0)
            p5_vals = onp.nanquantile(vals, 0.05, axis=0)
            p25_vals = onp.nanquantile(vals, 0.25, axis=0)
            p75_vals = onp.nanquantile(vals, 0.75, axis=0)
            p95_vals = onp.nanquantile(vals, 0.95, axis=0)
            ax.fill_between(
                days_sim[1:],
                min_vals[1:],
                max_vals[1:],
                edgecolor=color,
                facecolor=color,
                alpha=0.33,
                label="Min-Max interval",
            )
            ax.fill_between(
                days_sim[1:],
                p5_vals[1:],
                p95_vals[1:],
                edgecolor=color,
                facecolor=color,
                alpha=0.66,
                label="95% interval",
            )
            ax.fill_between(
                days_sim[1:],
                p25_vals[1:],
                p75_vals[1:],
                edgecolor=color,
                facecolor=color,
                alpha=1,
                label="75% interval",
            )
            ax.plot(days_sim[1:], median_vals[1:], color="black", label="Median", linewidth=1)
            ax.legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.93, 1.19))
            ax.set_xlabel("Time [days]")
            ax.set_ylabel(labs._Y_LABS_DAILY[var_sim])
            ax.set_xlim(days_sim[1], days_sim[-1])
            if var_sim == "theta":
                ax.hlines(onp.max(theta_sat), days_sim[1], days_sim[-1], color="black", linestyle="--", linewidth=1)
                ax.hlines(onp.min(theta_pwp), days_sim[1], days_sim[-1], color="black", linestyle="--", linewidth=1)
            fig.tight_layout()
            file = base_path_figs / f"trace_{var_sim}_{meteo_station}.png"
            fig.savefig(file, dpi=250)
            plt.close(fig)

    ds_sim = ds_sim.close()

    # concatenate dataframes
    df_sim_sum = pd.concat(ll_df_sim_sum, sort=False)
    df_sim_sum_tot = pd.concat(ll_df_sim_sum_tot, sort=False)

    # convert from wide to long
    df_sim_sum = pd.melt(df_sim_sum, id_vars=["meteo_station", "idx"])
    df_sim_sum_tot = pd.melt(df_sim_sum_tot, id_vars=["meteo_station"])
    for i, meteo_station in enumerate(meteo_stations):
        df_sim_sum_tot.loc[df_sim_sum_tot["meteo_station"] == meteo_station, "idx"] = range(len(vars_sim))

    # compare sums per grid
    ax = sns.catplot(
        x="variable",
        y="value",
        hue="meteo_station",
        data=df_sim_sum,
        kind="box",
        height=3,
        aspect=2,
        palette="RdPu",
        whis=[0, 100],
    )
    xticklabels = [labs._TICKLABS[var_sim] for var_sim in vars_sim]
    ax.set_xticklabels(xticklabels, rotation=30)
    ax.set(xlabel="", ylabel="[mm]")
    ax._legend.set_title("DWD station")
    sns.move_legend(ax, "upper right", bbox_to_anchor=(0.9, 0.95))
    file = base_path_figs / "sums_per_grid_box.png"
    ax.savefig(file, dpi=250)
    file = base_path_figs / "sums_per_grid_box.pdf"
    ax.savefig(file, dpi=250)


if __name__ == "__main__":
    main()
