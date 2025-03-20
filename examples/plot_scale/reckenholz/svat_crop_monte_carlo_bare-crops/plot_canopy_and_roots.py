import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import click
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
        base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")
        # base_path = Path(__file__).parent.parent

    # directory of results
    base_path_output = base_path / "output" / "svat_crop_monte_carlo"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = Path(__file__).parent.parent / "figures" / "svat_crop_monte_carlo"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    crops_lys2_lys3_lys8 = {2010: "winter barley",
                            2011: "sugar beet",
                            2012: "winter wheat",
                            2013: "winter rape",
                            2014: "winter triticale",
                            2015: "silage corn",
                            2016: "winter barley & green manure",
                            2017: "green manure & sugar beet"
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


    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        # load parameters and metrics
        df_metric = pd.read_csv(base_path_output / "KGE_bulk_samples.csv", sep=";")
        df_metric["E_multi"] = df_metric["avg"]
        df_metric.loc[:, "id"] = range(len(df_metric.index))
        df_metric = df_metric.sort_values(by=["E_multi"], ascending=False)
        idx_best100 = df_metric.loc[: df_metric.index[99], "id"].values.tolist()
        idx_best = idx_best100[0]        
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

        # plot simulated variables of best simulation
        years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
        vars_sim = ["ground_cover", "z_root"]
        for var_sim in vars_sim:
            fig, axs = plt.subplots(7, 1, sharey=False, sharex=False, figsize=(6, 6))
            for i, year in enumerate(years):
                sim_vals = ds_sim_hm[var_sim].isel(y=0).values[idx_best100, :].T
                # join observations on simulations
                df_eval = pd.DataFrame(index=date_sim_hm, columns=[f"sim{j}" for j in range(sim_vals.shape[1])])
                df_eval.iloc[:, :] = sim_vals
                df_eval = df_eval.loc[f"{year}-01-01":f"{year}-12-31", :]
                # plot observed and simulated time series
                sim_vals_min = onp.nanmin(df_eval.values.astype(onp.float64), axis=1)
                sim_vals_max = onp.nanmax(df_eval.values.astype(onp.float64), axis=1)
                sim_vals_median = onp.nanmedian(df_eval.loc[:, "sim0":].values.astype(onp.float64), axis=1)
                axs[i].fill_between(df_eval.index, sim_vals_min, sim_vals_max, color="red", alpha=0.5, zorder=0)
                axs[i].plot(df_eval.index, sim_vals_median, color="red", zorder=1)
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
                if var_sim == "z_root":
                    axs[i].invert_yaxis()
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
