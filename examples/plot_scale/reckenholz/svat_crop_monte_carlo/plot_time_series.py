import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import click
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import matplotlib as mpl

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

    # directory of results
    base_path_output = base_path.parent / "output" / "svat_crop_monte_carlo"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path.parent / "figures" / "svat_crop_monte_carlo"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    crops_lys2_lys3_lys8 = {2010: "winter barley",
                            2011: "sugar beet",
                            2012: "winter wheat",
                            2013: "winter rape",
                            2014: "winter triticale",
                            2015: "silage corn",
                            2016: "winter barley",
                            2017: "sugar beet"
    }

    crops_lys4_lys9 = {2010: "winter wheat",
                       2011: "winter rape & phacelia",
                       2012: "phacelia & silage corn",
                       2013: "beet root",
                       2014: "winter barley",
                       2015: "grass",
                       2016: "grass",
                       2017: "winter wheat"
    }

    fert_lys2_lys3_lys8 = {"lys2": "130% N-fertilized",
                           "lys3": "100% N-fertilized",
                           "lys8": "70% N-fertilized"
    }

    fert_lys4_lys9 = {"lys4": "Organic",
                      "lys9": "PEP-intensive"
    }

    _lys = {"lys2": "Lys 2",
            "lys3": "Lys 3",
            "lys4": "Lys 4",
            "lys8": "Lys 8",
            "lys9": "Lys 9",
    }

    lys_experiments = ["lys2", "lys3", "lys4", "lys8", "lys9"]
    for lys_experiment in lys_experiments:
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
        idx_best = df_params_metrics["KGE_q_ss"].idxmax()
        # load simulation
        sim_hm_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        ds_sim_hm = xr.open_dataset(sim_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim_hm = num2date(
            days_sim_hm,
            units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

        # load observations (measured data)
        path_obs = Path(__file__).parent.parent / "observations" / "reckenholz_lysimeter.nc"
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

        # compare best simulation with observations
        vars_obs = ["PERC", "THETA"]
        vars_sim = ["q_ss", "theta"]
        dict_obs_sim = {}
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            if var_sim == "theta":
                obs_vals = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
            else:
                obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
            df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
            df_obs.loc[:, "obs"] = obs_vals
            sim_vals = ds_sim_hm[var_sim].isel(x=idx_best, y=0).values
            # join observations on simulations
            df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
            # skip first seven days for warmup
            df_eval.loc[:f"{df_obs.index.year[1]}-01-07", :] = onp.nan
            dict_obs_sim[var_obs] = df_eval
            # plot observed and simulated time series
            fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
            file_str = "%s_%s.pdf" % (var_sim, lys_experiment)
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=300)
            if var_sim == "q_ss":
                # plot cumulated observed and simulated time series
                fig = eval_utils.plot_obs_sim_cum(df_eval.dropna(), labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
                file_str = "%s_cum_%s.pdf" % (var_sim, lys_experiment)
                path_fig = base_path_figs / file_str
                fig.savefig(path_fig, dpi=300)
                fig = eval_utils.plot_obs_sim_cum_year_facet(
                    df_eval.dropna(), labs._Y_LABS_CUM[var_sim], x_lab="Time\n[day-month-hydyear]"
                )
                file_str = "%s_cum_year_facet_%s.pdf" % (var_sim, lys_experiment)
                path_fig = base_path_figs / file_str
                fig.savefig(path_fig, dpi=300)
            plt.close("all")

        # compare best simulation with observations
        vars_obs = ["PERC", "THETA"]
        vars_sim = ["q_ss", "theta"]
        dict_obs_sim = {}
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]:
                if var_sim == "theta":
                    obs_vals = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
                else:
                    obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
                df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
                df_obs.loc[:, "obs"] = obs_vals
                sim_vals = ds_sim_hm[var_sim].isel(x=idx_best, y=0).values
                # join observations on simulations
                df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
                # skip first seven days for warmup
                df_eval.loc[f"{year}-01-01":f"{year}-01-07", :] = onp.nan
                df_eval = df_eval.loc[f"{year}-01-01":f"{year}-12-31", :]
                dict_obs_sim[var_obs] = df_eval
                # plot observed and simulated time series
                fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
                file_str = "%s_%s_%s.pdf" % (var_sim, lys_experiment, year)
                path_fig = base_path_figs / file_str
                fig.savefig(path_fig, dpi=300)
                if var_sim == "q_ss":
                    # plot cumulated observed and simulated time series
                    fig = eval_utils.plot_obs_sim_cum(df_eval.dropna(), labs._Y_LABS_CUM[var_sim], x_lab="Time [year]")
                    file_str = "%s_cum_%s_%s.pdf" % (var_sim, lys_experiment, year)
                    path_fig = base_path_figs / file_str
                    fig.savefig(path_fig, dpi=300)
                plt.close("all")

    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        # load parameters and metrics
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
        for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]:
            df_params_metrics.loc[:, f"RBS_r_{year}"] = 0.5 * df_params_metrics.loc[:, f"RBS_q_ss_{year}"].abs() + 0.5 * (1 - df_params_metrics.loc[:, f"r_q_ss_{year}"])
        df_params_metrics.loc[:, "RBS_r"] = 0.5 * df_params_metrics.loc[:, "RBS_q_ss_2011":"RBS_q_ss_2017"].abs().mean(axis=1) + 0.5 * (1 - df_params_metrics.loc[:, "r_q_ss_2011":"r_q_ss_2017"].mean(axis=1))
        idx_best = df_params_metrics["KGE_q_ss"].idxmax()
        # load simulation
        sim_hm_file = base_path_output / f"SVATCROP_{lys_experiment}.nc"
        ds_sim_hm = xr.open_dataset(sim_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = ds_sim_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date_sim_hm = num2date(
            days_sim_hm,
            units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

        # load observations (measured data)
        path_obs = Path(__file__).parent.parent / "observations" / "reckenholz_lysimeter.nc"
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

        # compare best simulation with observations
        years = [2011, 2012, 2013, 2014, 2015, 2016]
        vars_obs = ["PERC", "THETA"]
        vars_sim = ["q_ss", "theta"]
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            fig, axs = plt.subplots(6, 1, sharey=False, sharex=False, figsize=(6, 6))
            for i, year in enumerate(years):
                if var_sim == "theta":
                    obs_vals = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
                else:
                    obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
                df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
                df_obs.loc[:, "obs"] = obs_vals
                sim_vals = ds_sim_hm[var_sim].isel(x=idx_best, y=0).values
                # join observations on simulations
                df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
                # skip first seven days for warmup
                df_eval = df_eval.loc[f"{year}-01-01":f"{year}-12-31", :]
                # plot observed and simulated time series
                axs[i].plot(df_eval.index, df_eval["sim"], color="red")
                axs[i].plot(df_eval.index, df_eval["obs"], color="blue", ls="--")
                axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
                axs[i].set_ylabel('')
                axs[i].set_xlabel('')
                if var_sim == "q_ss":
                    axs[i].set_ylim(0)
                    metric_total = onp.round(df_params_metrics.loc[idx_best, "KGE_q_ss"], 2)
                    metric_year = onp.round(df_params_metrics.loc[idx_best, f"KGE_q_ss_{year}"], 2)
                    axs[i].text(0.9,
                                1.11,
                                f"KGE: {metric_year} ({metric_total})",
                                fontsize=8,
                                horizontalalignment="center",
                                verticalalignment="center",
                                transform=axs[i].transAxes)
                axs[i].text(0.5,
                            1.11,
                            f"{year}: {crops_lys2_lys3_lys8[year]} ({fert_lys2_lys3_lys8[lys_experiment]})",
                            fontsize=8,
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axs[i].transAxes)
                axs[i].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d-%m'))
            axs[1].set_ylabel(labs._Y_LABS_DAILY[var_sim])
            axs[-2].set_ylabel(labs._Y_LABS_DAILY[var_sim])
            axs[-1].set_xlabel('Time [day-month]')
            fig.tight_layout()
            file_str = "%s_%s_%s_%s.pdf" % (var_sim, lys_experiment, years[0], years[-1])
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=300)
            plt.close("all")

        # compare best simulation with observations
        years = [2011, 2012, 2013, 2014, 2015, 2016]
        vars_obs = ["PERC"]
        vars_sim = ["q_ss"]
        for var_obs, var_sim in zip(vars_obs, vars_sim):
            fig, axs = plt.subplots(6, 1, sharey=False, sharex=False, figsize=(6, 6))
            for i, year in enumerate(years):
                if var_sim == "theta":
                    obs_vals = onp.mean(ds_obs["THETA"].isel(x=0, y=0).values, axis=0)
                else:
                    obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
                df_obs = pd.DataFrame(index=date_obs, columns=["obs"])
                df_obs.loc[:, "obs"] = obs_vals
                sim_vals = ds_sim_hm[var_sim].isel(x=idx_best, y=0).values
                # join observations on simulations
                df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
                # skip first seven days for warmup
                df_eval = df_eval.loc[f"{year}-01-01":f"{year}-12-31", :]
                # plot observed and simulated time series
                axs[i].plot(df_eval.index, df_eval["sim"].cumsum(), color="red")
                axs[i].plot(df_eval.index, df_eval["obs"].cumsum(), color="blue")
                axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
                axs[i].set_ylabel('')
                axs[i].set_xlabel('')
                axs[i].text(0.5,
                            1.1,
                            f"{year}: {crops_lys2_lys3_lys8[year]} ({fert_lys2_lys3_lys8[lys_experiment]})",
                            fontsize=8,
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axs[i].transAxes)
                axs[i].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d-%m'))
                axs[i].set_ylim(0)
                metric_total = onp.round(df_params_metrics.loc[idx_best, "KGE_q_ss"], 2)
                metric_year = onp.round(df_params_metrics.loc[idx_best, f"KGE_q_ss_{year}"], 2)
                axs[i].text(0.9,
                            1.11,
                            f"KGE: {metric_year} ({metric_total})",
                            fontsize=8,
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axs[i].transAxes)
                rbs = onp.round(df_params_metrics.loc[idx_best, f"RBS_q_ss"], 2)
                rbs_year = onp.round(df_params_metrics.loc[idx_best, f"RBS_q_ss_{year}"], 2)
                axs[i].text(0.02,
                            0.85,
                            f"RBS: {rbs_year} ({rbs})",
                            fontsize=8,
                            horizontalalignment="left",
                            verticalalignment="center",
                            transform=axs[i].transAxes)
            axs[1].set_ylabel(labs._Y_LABS_CUM[var_sim])
            axs[-2].set_ylabel(labs._Y_LABS_CUM[var_sim])
            axs[-1].set_xlabel('Time [day-month]')
            fig.tight_layout()
            file_str = "%s_%s_%s_%s_cumulated.pdf" % (var_sim, lys_experiment, years[0], years[-1])
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=300)
            plt.close("all")

        # plot simulated variables of best simulation
        years = [2011, 2012, 2013, 2014, 2015, 2016]
        vars_sim = ["ground_cover", "z_root"]
        for var_sim in vars_sim:
            fig, axs = plt.subplots(6, 1, sharey=False, sharex=False, figsize=(6, 6))
            for i, year in enumerate(years):
                sim_vals = ds_sim_hm[var_sim].isel(x=idx_best, y=0).values
                # join observations on simulations
                df_eval = pd.DataFrame(index=date_sim_hm)
                df_eval.loc[:, "sim"] = sim_vals
                df_eval = df_eval.loc[f"{year}-01-01":f"{year}-12-31", :]
                # plot observed and simulated time series
                axs[i].plot(df_eval.index, df_eval["sim"], color="red")
                axs[i].set_xlim(df_eval.index[0], df_eval.index[-1])
                axs[i].set_ylabel('')
                axs[i].set_xlabel('')
                axs[i].text(0.5,
                            1.1,
                            f"{year}: {crops_lys2_lys3_lys8[year]} ({fert_lys2_lys3_lys8[lys_experiment]})",
                            fontsize=8,
                            horizontalalignment="center",
                            verticalalignment="center",
                            transform=axs[i].transAxes)
                axs[i].set_ylim(0,)
                axs[i].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d-%m'))
            axs[1].set_ylabel(labs._Y_LABS_DAILY[var_sim])
            axs[-2].set_ylabel(labs._Y_LABS_DAILY[var_sim])
            axs[-1].set_xlabel('Time [day-month]')
            fig.tight_layout()
            file_str = "%s_%s_%s_%s.pdf" % (var_sim, lys_experiment, years[0], years[-1])
            path_fig = base_path_figs / file_str
            fig.savefig(path_fig, dpi=300)
            plt.close("all")


    return


if __name__ == "__main__":
    main()
