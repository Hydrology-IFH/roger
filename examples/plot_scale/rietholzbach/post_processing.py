from pathlib import Path
import os
import scipy as sp
from SALib.analyze import sobol
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import yaml
import click
import seaborn as sns
import copy
sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
sns.set_context("paper", font_scale=1.5)
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs


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
    days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
    df_obs = pd.DataFrame(index=date_obs)
    df_obs.loc[:, 'd18O_prec'] = ds_obs['d18O_PREC'].isel(x=0, y=0).values
    df_obs.loc[:, 'd18O_perc'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values

    # measured oxygen-18 in precipitation and percolation
    d18O_prec_mean = onp.round(onp.nanmean(df_obs.loc[:, 'd18O_prec'].values), 2)
    d18O_perc_mean = onp.round(onp.nanmean(df_obs.loc[:, 'd18O_perc'].values), 2)
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    axs[0].plot(df_obs.index,
                df_obs.loc[:, 'd18O_prec'].fillna(method='bfill'),
                '-', color='blue')
    axs[0].plot(df_obs.index,
                df_obs.loc[:, 'd18O_prec'],
                '.', color='blue')
    axs[0].set_ylabel(r'$\delta^{18}$O [‰]')
    axs[0].set_ylim([-20,0])
    axs[0].set_xlim(df_obs.index[0], df_obs.index[-1])
    axs[1].plot(df_obs.index, df_obs.loc[:, 'd18O_perc'].fillna(method='bfill'),
             '-', color='grey')
    axs[1].plot(df_obs.index, df_obs.loc[:, 'd18O_perc'],
             '.', color='grey')
    axs[1].set_ylabel(r'$\delta^{18}$O [‰]')
    axs[1].set_xlabel('Time [year]')
    axs[1].set_ylim([-20,0])
    axs[1].set_xlim(df_obs.index[0], df_obs.index[-1])
    fig.tight_layout()
    fig.text(0.115, 0.92, "(a)", ha="center", va="center")
    fig.text(0.89, 0.615, r"$\overline{\delta^{18}O}_{prec}$: %s" % (d18O_prec_mean), ha="center", va="center")
    fig.text(0.89, 0.155, r"$\overline{\delta^{18}O}_{perc}$: %s" % (d18O_perc_mean), ha="center", va="center")
    fig.text(0.115, 0.46, "(b)", ha="center", va="center")
    file = base_path_figs / 'observed_d18O_prec_perc.pdf'
    fig.savefig(file, dpi=250)
    plt.close(fig=fig)

    # load best monte carlo run
    states_hm_file = base_path / "svat_monte_carlo" / "states_hm.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
    # assign date
    days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim_hm))

    # compare simulation and observation
    vars_obs = ['AET', 'PERC', 'dWEIGHT']
    vars_sim = ['aet', 'q_ss', 'dS']
    for var_obs, var_sim in zip(vars_obs, vars_sim):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = obs_vals
        sim_vals = ds_sim_hm[var_sim].isel(x=0, y=0).values
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
        df_eval = df_eval.iloc[3:, :]
        # plot observed and simulated time series
        fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
        file_str = '%s.pdf' % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)
        # plot cumulated observed and simulated time series
        fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time [year]')
        file_str = '%s_cum.pdf' % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)
        fig = eval_utils.plot_obs_sim_cum_year_facet(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
        file_str = '%s_cum_year_facet.pdf' % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

    vars_obs = ['PREC']
    vars_sim = ['prec']
    for var_obs, var_sim in zip(vars_obs, vars_sim):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = obs_vals
        # plot observed time series
        fig = eval_utils.plot_sim(df_obs, labs._Y_LABS_DAILY[var_sim])
        file_str = '%s.pdf' % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)
        # plot cumulated observed time series
        fig = eval_utils.plot_sim_cum(df_obs, labs._Y_LABS_CUM[var_sim], x_lab='Time [year]')
        file_str = '%s_cum.pdf' % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)
        fig = eval_utils.plot_sim_cum_year_facet(df_obs, labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
        file_str = '%s_cum_year_facet.pdf' % (var_sim)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

    vars_obs = ['TA']
    vars_sim = ['ta']
    for var_obs, var_sim in zip(vars_obs, vars_sim):
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = obs_vals
        # plot observed time series
        fig = eval_utils.plot_sim(df_obs, labs._Y_LABS_DAILY[var_sim])
        file_str = '%s.pdf' % (var_obs)
        path_fig = base_path_figs / file_str
        fig.savefig(path_fig, dpi=250)

    # load metrics of transport simulations
    dict_params_eff_tm_mc = {}
    tm_structures = ['complete-mixing', 'piston',
                     'preferential', 'advection-dispersion',
                     'time-variant preferential',
                     'time-variant advection-dispersion']
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        file = base_path / "svat_transport_monte_carlo" / "results" / f"params_eff_{tms}.txt"
        df_params_eff = pd.read_csv(file, sep="\t")
        dict_params_eff_tm_mc[tm_structure] = {}
        dict_params_eff_tm_mc[tm_structure]['params_eff'] = df_params_eff

    # compare best model runs
    fig, ax = plt.subplots(2, 3, sharey=True, figsize=(14, 7))
    for i, tm_structure in enumerate(tm_structures):
        # load transport simulation
        states_tm_file = base_path / "svat_transport_monte_carlo" / "states_tm.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, group=tm_structure, engine="h5netcdf")
        days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))
        # join observations on simulations
        obs_vals = ds_obs['d18O_PERC'].isel(x=0, y=0).values
        sim_vals = ds_sim_tm['d18O_perc_cs'].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = obs_vals
        df_sim = pd.DataFrame(index=date_sim_tm, columns=['sim'])
        df_sim.loc[:, 'sim'] = ds_sim_tm['C_q_ss'].isel(x=0, y=0).values
        df_sim = df_sim.iloc[1:, :]
        df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals, df_obs)
        df_eval = df_eval.dropna()
        ax.flatten()[i].plot(df_sim.index, df_sim.iloc[:, 0], color='red')
        ax.flatten()[i].scatter(df_eval.index, df_eval.iloc[:, 0], color='red', s=2)
        ax.flatten()[i].scatter(df_eval.index, df_eval.iloc[:, 1], color='blue', s=2)

    ax[0, 0].set_ylabel(r'$d_{18}O_{PERC}$ [permil]')
    ax[1, 0].set_ylabel(r'$d_{18}O_{PERC}$ [permil]')
    ax[1, 1].set_xlabel('Time')
    fig.tight_layout()
    file = base_path_figs / "d18O_perc_sim_obs_tm_structures.png"
    fig.savefig(file, dpi=250)

    # compare duration curve of 18O in percolation
    fig, ax = plt.subplots(2, 3, sharey=True, figsize=(14, 7))
    for i, tm_structure in enumerate(tm_structures):
        # load transport simulation
        states_tm_file = base_path / "svat_transport_monte_carlo" / "states_tm.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, group=tm_structure, engine="h5netcdf")
        days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))
        # join observations on simulations
        obs_vals = ds_obs['d18O_PERC'].isel(x=0, y=0).values
        sim_vals = ds_sim_tm['d18O_perc_cs'].isel(x=0, y=0).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = obs_vals
        df_sim = pd.DataFrame(index=date_sim_tm, columns=['sim'])
        df_sim.loc[:, 'sim'] = ds_sim_tm['C_q_ss'].isel(x=0, y=0).values
        df_sim = df_sim.iloc[1:, :]
        df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals, df_obs)
        df_eval = df_eval.dropna()
        obs = df_eval.sort_values(by=["obs"], ascending=True)
        sim = df_eval.sort_values(by=["sim"], ascending=True)

        # calculate exceedence probability
        ranks_obs = sp.stats.rankdata(obs["obs"], method="ordinal")
        ranks_obs = ranks_obs[::-1]
        prob_obs = [(ranks_obs[i] / (len(obs["obs"]) + 1)) for i in range(len(obs["obs"]))]

        ranks_sim = sp.stats.rankdata(sim["sim"], method="ordinal")
        ranks_sim = ranks_sim[::-1]
        prob_sim = [(ranks_sim[i] / (len(sim["sim"]) + 1)) for i in range(len(sim["sim"]))]

        ax.flatten()[i].plot(prob_obs, obs["obs"], color="blue", lw=1.5)
        ax.flatten()[i].plot(prob_sim, sim["sim"], color="red", lw=1, ls="-.", alpha=0.8)

    ax[0, 0].set_ylabel(r'$d_{18}O_{PERC}$ [permil]')
    ax[1, 0].set_ylabel(r'$d_{18}O_{PERC}$ [permil]')
    ax[1, 1].set_xlabel('Exceedence probabilty [-]')
    fig.tight_layout()
    file = base_path_figs / "fdc_d18O_perc_sim_obs_tm_structures.png"
    fig.savefig(file, dpi=250)

    # perform sensitivity analysis
    dict_params_eff_tm_sa = {}
    tm_structures = ['complete-mixing', 'piston',
                     'preferential', 'advection-dispersion',
                     'time-variant preferential',
                     'time-variant advection-dispersion']
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        file = base_path / "svat_transport_sensitivity" / "results" / f"params_eff_{tms}.txt"
        df_params_eff = pd.read_csv(file, sep="\t")
        dict_params_eff_tm_sa[tm_structure] = {}
        dict_params_eff_tm_sa[tm_structure]['params_eff'] = df_params_eff

    # sampled parameter space
    file_path = base_path / "svat_transport_sensitivity" / "param_bounds.yml"
    with open(file_path, 'r') as file:
        bounds = yaml.safe_load(file)

    tm_structures = ['preferential', 'advection-dispersion',
                     'time-variant preferential',
                     'time-variant advection-dispersion']
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        df_params_eff = dict_params_eff_tm_sa[tm_structure]['params_eff']
        df_params = df_params_eff.loc[:, bounds[tm_structure]['names']]
        n_params = len(bounds[tm_structure]['names'])
        df_eff = df_params_eff.iloc[:, n_params:]
        dict_si = {}
        for name in df_eff.columns:
            Y = df_eff[name].values
            Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]['names'])

        # plot sobol indices
        _LABS = {'KGE_C_q_ss': 'KGE',
                 'median_TT_q_ss': r'$tt_{50}$',
                 'median_SA_s': r'rt_{50}$',
                 }
        ncol = len(df_eff.columns)
        xaxis_labels = [labs._LABS[k].split(' ')[0] for k in bounds[tm_structure]['names']]
        cmap = cm.get_cmap('Greys')
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(14, 5))
        for i, name in enumerate(df_eff.columns):
            indices = dict_si[name][['S1', 'ST']]
            err = dict_si[name][['S1_conf', 'ST_conf']]
            indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
            ax[i].set_xticklabels(xaxis_labels)
            ax[i].set_title(_LABS[name])
            ax[i].legend(["First-order", "Total"], frameon=False)
        ax[-1].legend().set_visible(False)
        ax[-2].legend().set_visible(False)
        ax[-3].legend().set_visible(False)
        ax[0].set_ylabel('Sobol index [-]')
        fig.tight_layout()
        file = base_path_figs / f"sobol_indices_{tms}.png"
        fig.savefig(file, dpi=250)

        # make dotty plots
        nrow = len(df_eff.columns)
        ncol = bounds[tm_structure]['num_vars']
        fig, ax = plt.subplots(nrow, ncol, sharey='row', figsize=(14, 7))
        for i in range(nrow):
            for j in range(ncol):
                y = df_eff.iloc[:, i]
                x = df_params.iloc[:, j]
                sns.regplot(x=x, y=y, ax=ax[i, j], ci=None, color='k',
                            scatter_kws={'alpha': 0.2, 's': 4, 'color': 'grey'})
                ax[i, j].set_xlabel('')
                ax[i, j].set_ylabel('')

        for j in range(ncol):
            xlabel = labs._LABS[bounds[tm_structure]['names'][j].split(' ')[0]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel(r'$KGE$ [-]')
        ax[1, 0].set_ylabel(r'$\alpha$ [-]')
        ax[2, 0].set_ylabel(r'$\beta$ [-]')
        ax[3, 0].set_ylabel(r'r [-]')

        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        file = base_path_figs / f"dotty_plots_{tms}.png"
        fig.savefig(file, dpi=250)

    # HYDRUS-1D benchmark
    states_hydrus_file = base_path / "hydrus_benchmark" / "states_hydrus.nc"
    ds_hydrus = xr.open_dataset(states_hydrus_file, engine="h5netcdf")
    days_hydrus = (ds_hydrus['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_hydrus = num2date(days_hydrus, units=f"days since {ds_hydrus['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_hydrus = ds_hydrus.assign_coords(date=("Time", date_hydrus))

    # plot isotope ratios of precipitation and soil
    cmap = copy.copy(plt.cm.get_cmap('RdYlBu_r'))
    norm = mpl.colors.Normalize(vmin=-20, vmax=5)

    fig, axes = plt.subplots(2, 1, sharex=False, figsize=(14,7))
    axes[0].bar(date_hydrus, ds_hydrus['prec'].values, width=-1, color=cmap(norm(ds_hydrus['d18O_prec'].values)), align='edge', edgecolor=cmap(norm(ds_hydrus['d18O_prec'].values)))
    axes[0].set_ylabel('Precipitation\n[mm $day^{-1}$]')
    axes[0].set_xlim(date_hydrus[0], date_hydrus[-1])
    sns.heatmap(ds_hydrus['d18O_soil'].values, xticklabels=366, yticklabels=int(50/2), cmap='RdYlBu_r',
                vmax=5, vmin=-20, cbar=False, ax=axes[1])
    axes[1].set_ylabel('Soil depth\n[m]')
    axes[1].set_xlabel('Time [years]')

    axl = fig.add_axes([0.92, 0.3, 0.02, 0.3])
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
                                    orientation='vertical',
                                    ticks=[0, -5, -10, -15])
    cb1.set_label(r'$\delta^{18}$O [permil]')
    file = 'conc_prec_soil.png'
    path = base_path_figs / file
    fig.savefig(path, dpi=250)

    # plot isotope ratios of precipitation, soil and percolation
    cmap = copy.copy(plt.cm.get_cmap('RdYlBu_r'))
    norm = mpl.colors.Normalize(vmin=-20, vmax=5)

    fig, axes = plt.subplots(3, 1, sharex=False, figsize=(14,7))
    axes[0].bar(date_hydrus, ds_hydrus['prec'].values, width=-1, edgecolor=cmap(norm(ds_hydrus['d18O_prec'].values)), align='edge')
    axes[0].set_ylabel('Precipitation\n[mm $day^{-1}$]')
    axes[0].set_xlim(date_hydrus[0], date_hydrus[-1])
    sns.heatmap(ds_hydrus['d18O_soil'].values, xticklabels=366, yticklabels=int(50/2), cmap='RdYlBu_r',
                vmax=5, vmin=-20, cbar=False, ax=axes[1])
    axes[1].set_ylabel('Soil depth\n[m]')

    axes[2].bar(date_hydrus, ds_hydrus['perc'].values, width=-1, edgecolor=cmap(norm(ds_hydrus['d18O_perc'].values)), align='edge')
    axes[2].set_xlim(date_hydrus[0], date_hydrus[-1])
    axes[2].set_ylabel('Percolation\n[mm $day^{-1}$]')
    axes[2].set_ylim(0, )
    axes[2].invert_yaxis()
    axes[2].set_xlabel('Time [years]')

    axl = fig.add_axes([0.92, 0.33, 0.02, 0.3])
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
                                    orientation='vertical',
                                    ticks=[0, -5, -10, -15])
    cb1.set_label(r'$\delta^{18}$O [permil]')

    file = 'conc_prec_soil_perc.png'
    path = base_path_figs / file
    fig.savefig(path, dpi=250)


    # states_hydrus_tt_file = base_path / "hydrus_benchmark" / "states_tt_hydrus.nc"
    # ds_hydrus_tt = xr.open_dataset(states_hydrus_tt_file, engine="h5netcdf")

    return


if __name__ == "__main__":
    main()
