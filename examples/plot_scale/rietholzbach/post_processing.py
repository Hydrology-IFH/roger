from pathlib import Path
import os
import h5netcdf
import scipy as sp
from SALib.analyze import sobol
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.cm as cm
import yaml
import click
import copy
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
sns.set_style("ticks")

_LABS_HYDRUS = {
                'n': r'$n$ [-]',
                'alpha': r'$alfa$ [-]',
                'theta_sat_m': r'$\theta_{s}^{m}$ [-]',
                'theta_sat_im': r'$\theta_{s}^{im}$ [-]',
                'ks': r'$k_{s}$ [-]',
                'omega': r'$\omega$ [-]',
                'D_l': r'$D_l$ [-]',
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

    transport_models = ['complete-mixing', 'piston',
                        'advection-dispersion',
                        'time-variant advection-dispersion']

    # load observations (measured data)
    path_obs = Path(__file__).parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
    # assign date
    days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_obs = ds_obs.assign_coords(Time=("Time", date_obs))
    df_obs = pd.DataFrame(index=date_obs)
    df_obs.loc[:, 'd18O_prec'] = ds_obs['d18O_PREC'].isel(x=0, y=0).values
    df_obs.loc[:, 'd18O_perc'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values
    path_obs_br = Path(__file__).parent / "observations" / "bromide_breakthrough.csv"
    df_obs_br = pd.read_csv(path_obs_br, skiprows=1, sep=';', na_values='')

    # load best monte carlo simulations
    states_hm_file = base_path / "svat_monte_carlo" / "states_hm.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
    # load best 1% monte carlo simulations
    states_hm1_file = base_path / "svat_monte_carlo" / "states_hm1.nc"
    ds_sim_hm1 = xr.open_dataset(states_hm1_file, engine="h5netcdf")
    # assign date
    days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))

    # load HYDRUS-1D benchmarks
    # oxygen-18 simulations
    states_hydrus_file = base_path / "hydrus_benchmark" / "states_hydrus_18O.nc"
    ds_hydrus_18O = xr.open_dataset(states_hydrus_file, engine="h5netcdf")
    hours_hydrus_18O = (ds_hydrus_18O['Time'].values / onp.timedelta64(60 * 60, "s"))
    date_hydrus_18O = num2date(hours_hydrus_18O, units=f"hours since {ds_hydrus_18O['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_hydrus_18O = ds_hydrus_18O .assign_coords(Time=("Time", date_hydrus_18O))

    # travel time simulations
    states_hydrus_file = base_path / "hydrus_benchmark" / "states_hydrus_tt.nc"
    ds_hydrus_tt = xr.open_dataset(states_hydrus_file, engine="h5netcdf", decode_times=False)
    days_hydrus_tt = ds_hydrus_tt['Time'].values / 24
    date_hydrus_tt = num2date(days_hydrus_tt, units=f"hours since {ds_hydrus_tt['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_hydrus_tt = ds_hydrus_tt.assign_coords(Time=("Time", date_hydrus_tt))

    # # Monte Carlo simulations
    # states_hydrus_mc_file = base_path / "hydrus_benchmark" / "states_hydrus_mc.nc"
    # ds_hydrus_mc = xr.open_dataset(states_hydrus_mc_file, engine="h5netcdf")
    # days_hydrus_mc = (ds_hydrus_mc['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    # date_hydrus_mc = num2date(days_hydrus_mc, units=f"days since {ds_hydrus_mc['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    # ds_hydrus_mc = ds_hydrus_mc.assign_coords(date=("Time", date_hydrus_mc))

    # # average observed soil water content of previous 5 days
    # window = 5
    # df_thetap = pd.DataFrame(index=date_obs,
    #                          columns=['doy', 'theta', 'sc'])
    # df_thetap.loc[:, 'doy'] = df_thetap.index.day_of_year
    # df_thetap.loc[:, 'theta'] = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values, axis=0)
    # df_thetap.loc[df_thetap.index[window-1]:, 'theta'] = df_thetap.loc[:, 'theta'].rolling(window=window).mean().iloc[window-1:].values
    # df_thetap.iloc[:window, 1] = onp.nan
    # df_thetap_doy = df_thetap.groupby(by=["doy"], dropna=False).mean()
    # theta_p33 = df_thetap_doy.loc[:, 'theta'].quantile(0.33)
    # theta_p66 = df_thetap_doy.loc[:, 'theta'].quantile(0.66)
    # cond1 = (df_thetap['theta'] < theta_p33)
    # cond2 = (df_thetap['theta'] >= theta_p33) & (df_thetap['theta'] < theta_p66)
    # cond3 = (df_thetap['theta'] >= theta_p66)
    # df_thetap.loc[cond1, 'sc'] = 1  # dry
    # df_thetap.loc[cond2, 'sc'] = 2  # normal
    # df_thetap.loc[cond3, 'sc'] = 3  # wet

    # # measured oxygen-18 in precipitation and percolation
    # d18O_prec_mean = onp.round(onp.nanmean(df_obs.loc[:, 'd18O_prec'].values), 2)
    # d18O_perc_mean = onp.round(onp.nanmean(df_obs.loc[:, 'd18O_perc'].values), 2)
    # fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    # axs[0].plot(df_obs.index,
    #             df_obs.loc[:, 'd18O_prec'].fillna(method='bfill'),
    #             '-', color='blue')
    # axs[0].plot(df_obs.index,
    #             df_obs.loc[:, 'd18O_prec'],
    #             '.', color='blue')
    # axs[0].set_ylabel(r'$\delta^{18}$O [‰]')
    # axs[0].set_ylim([-20, 0])
    # axs[0].set_xlim(df_obs.index[0], df_obs.index[-1])
    # axs[1].plot(df_obs.index, df_obs.loc[:, 'd18O_perc'].fillna(method='bfill'),
    #             '-', color='grey')
    # axs[1].plot(df_obs.index, df_obs.loc[:, 'd18O_perc'],
    #             '.', color='grey')
    # axs[1].set_ylabel(r'$\delta^{18}$O [‰]')
    # axs[1].set_xlabel('Time [year]')
    # axs[1].set_ylim([-20, 0])
    # axs[1].set_xlim(df_obs.index[0], df_obs.index[-1])
    # fig.tight_layout()
    # fig.text(0.115, 0.92, "(a)", ha="center", va="center")
    # fig.text(0.89, 0.615, r"$\overline{\delta^{18}O}_{prec}$: %s" % (d18O_prec_mean), ha="center", va="center")
    # fig.text(0.89, 0.155, r"$\overline{\delta^{18}O}_{perc}$: %s" % (d18O_perc_mean), ha="center", va="center")
    # fig.text(0.115, 0.46, "(b)", ha="center", va="center")
    # file = base_path_figs / 'observed_d18O_prec_perc.pdf'
    # fig.savefig(file, dpi=250)
    # plt.close(fig=fig)

    # # dotty plots
    # file = base_path / "svat_monte_carlo" / "results" / "params_metrics.txt"
    # df_params_metrics = pd.read_csv(file, sep="\t")
    # df_params_metrics1 = df_params_metrics.copy()
    # df_params_metrics1.loc[:, 'id'] = range(len(df_params_metrics1.index))
    # df_params_metrics1 = df_params_metrics1.sort_values(by=['E_multi'], ascending=False)
    # idx_best1 = df_params_metrics1.loc[:df_params_metrics1.index[99], 'id'].values.tolist()
    # dict_metrics_best = {}
    # for sc in ['', 'dry', 'normal', 'wet']:
    #     dict_metrics_best[sc] = pd.DataFrame(index=range(len(idx_best1)))
    # for sc, sc1 in zip([0, 1, 2, 3], ['', 'dry', 'normal', 'wet']):
    #     df_metrics = df_params_metrics.loc[:, [f'KGE_aet{sc1}', f'r_dS{sc1}', f'KGE_q_ss{sc1}', f'E_multi{sc1}']]
    #     df_params = df_params_metrics.loc[:, ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']]
    #     nrow = len(df_metrics.columns)
    #     ncol = len(df_params.columns)
    #     fig, ax = plt.subplots(nrow, ncol, sharey='row', figsize=(14, 7))
    #     for i in range(nrow):
    #         for j in range(ncol):
    #             y = df_metrics.iloc[:, i]
    #             x = df_params.iloc[:, j]
    #             ax[i, j].scatter(x, y, s=4, c='grey', alpha=0.5)
    #             ax[i, j].set_xlabel('')
    #             ax[i, j].set_ylabel('')
    #             # best parameter set for individual evaluation metric at specific storage conditions
    #             df_params_metrics_sc1 = df_params_metrics.copy()
    #             df_params_metrics_sc1.loc[:, 'id'] = range(len(df_params_metrics1.index))
    #             df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
    #             idx_best_sc1 = df_params_metrics_sc1.loc[:df_params_metrics_sc1.index[99], 'id'].values.tolist()
    #             for idx_best_sc in idx_best_sc1:
    #                 y_best_sc = df_metrics.iloc[idx_best_sc, i]
    #                 x_best_sc = df_params.iloc[idx_best_sc, j]
    #                 ax[i, j].scatter(x_best_sc, y_best_sc, s=12, c='blue', alpha=0.8)
    #             # best parameter sets for multi-objective criteria
    #             for ii, idx_best in enumerate(idx_best1):
    #                 y_best = df_metrics.iloc[idx_best, i]
    #                 x_best = df_params.iloc[idx_best, j]
    #                 ax[i, j].scatter(x_best, y_best, s=12, c='red', alpha=1)
    #                 dict_metrics_best[sc1].loc[dict_metrics_best[sc1].index[ii], df_metrics.columns[i]] = df_params_metrics.loc[idx_best, df_metrics.columns[i]]

    #     for j in range(ncol):
    #         xlabel = labs._LABS[df_params.columns[j]]
    #         ax[-1, j].set_xlabel(xlabel)

    #     ax[0, 0].set_ylabel('$KGE_{ET}$\n [-]')
    #     ax[1, 0].set_ylabel(r'$r_{\Delta S}$ [-]')
    #     ax[2, 0].set_ylabel('$KGE_{PERC}$\n [-]')
    #     ax[3, 0].set_ylabel('$E_{multi}$\n [-]')

    #     fig.subplots_adjust(wspace=0.2, hspace=0.3)
    #     file = base_path_figs / f"dotty_plots_{sc1}.png"
    #     fig.savefig(file, dpi=250)

    # # write evaluation metrics for different storage condtions to .txt
    # df_avg_std = pd.DataFrame(columns=['KGE_aet', 'r_dS', 'KGE_q_ss', 'E_multi'])
    # for sc in ['', 'dry', 'normal', 'wet']:
    #     df_avg_std.loc[f'avg{sc}', :] = onp.mean(dict_metrics_best[sc].values, axis=0)
    #     df_avg_std.loc[f'std{sc}', :] = onp.std(dict_metrics_best[sc].values, axis=0)
    # file = base_path_figs / "metrics_best_1perc_avg_std.txt"
    # df_avg_std.to_csv(file, header=True, index=True, sep="\t")

    # # write average and standard deviation of best parameters to .txt
    # df_avg_std = pd.DataFrame(index=['dmpv', 'lmpv', 'theta_eff', 'frac_lp', 'frac_fp', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks'], columns=['avg', 'std'])
    # df_avg_std.loc[:, 'avg'] = onp.mean(df_params_metrics1.loc[:df_params_metrics1.index[99], ['dmpv', 'lmpv', 'theta_eff', 'frac_lp', 'frac_fp', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']].values, axis=0)
    # df_avg_std.loc[:, 'std'] = onp.std(df_params_metrics1.loc[:df_params_metrics1.index[99], ['dmpv', 'lmpv', 'theta_eff', 'frac_lp', 'frac_fp', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']].values, axis=0)
    # file = base_path_figs / "params_best_1perc_avg_std.txt"
    # df_avg_std.to_csv(file, header=True, index=True, sep="\t")

    # # # compare best simulation with observations
    # vars_obs = ['AET', 'PERC', 'dWEIGHT']
    # vars_sim = ['aet', 'q_ss', 'dS']
    # dict_obs_sim = {}
    # for var_obs, var_sim in zip(vars_obs, vars_sim):
    #     obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    #     df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    #     df_obs.loc[:, 'obs'] = obs_vals
    #     sim_vals = ds_sim_hm[var_sim].isel(x=0, y=0).values
    #     # join observations on simulations
    #     df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
    #     df_eval = df_eval.iloc[:, :]
    #     dict_obs_sim[var_sim] = df_eval
    #     # plot observed and simulated time series
    #     fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
    #     file_str = '%s.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    #     # plot cumulated observed and simulated time series
    #     fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time [year]')
    #     file_str = '%s_cum.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    #     fig = eval_utils.plot_obs_sim_cum_year_facet(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
    #     file_str = '%s_cum_year_facet.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    # plt.close('all')

    # # compare best 1% simulations with observations
    # vars_obs = ['AET', 'PERC', 'dWEIGHT']
    # vars_sim = ['aet', 'q_ss', 'dS']
    # dict_obs_sim1 = {}
    # for var_obs, var_sim in zip(vars_obs, vars_sim):
    #     obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    #     df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    #     df_obs.loc[:, 'obs'] = obs_vals
    #     sim_vals = ds_sim_hm1[var_sim].isel(y=0).values.T
    #     # join observations on simulations
    #     df_eval = eval_utils.join_obs_on_sim(date_sim_hm, sim_vals, df_obs)
    #     dict_obs_sim1[var_sim] = df_eval
    #     # plot observed and simulated time series
    #     fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
    #     file_str = '%s_best_1perc.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    #     # plot cumulated observed and simulated time series
    #     fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time [year]')
    #     file_str = '%s_cum_best_1perc.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    #     fig = eval_utils.plot_obs_sim_cum_year_facet(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
    #     file_str = '%s_cum_year_facet_best_1perc.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    # plt.close('all')

    # vars_obs = ['PREC']
    # vars_sim = ['prec']
    # dict_obs = {}
    # for var_obs, var_sim in zip(vars_obs, vars_sim):
    #     obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    #     df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    #     df_obs.loc[:, 'obs'] = obs_vals
    #     dict_obs[var_sim] = df_obs
    #     # plot observed time series
    #     fig = eval_utils.plot_sim(df_obs, labs._Y_LABS_DAILY[var_sim])
    #     file_str = '%s.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    #     # plot cumulated observed time series
    #     fig = eval_utils.plot_sim_cum(df_obs, labs._Y_LABS_CUM[var_sim], x_lab='Time [year]')
    #     file_str = '%s_cum.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    #     fig = eval_utils.plot_sim_cum_year_facet(df_obs, labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
    #     file_str = '%s_cum_year_facet.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    # plt.close('all')

    # vars_obs = ['TA']
    # vars_sim = ['ta']
    # for var_obs, var_sim in zip(vars_obs, vars_sim):
    #     obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    #     df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    #     df_obs.loc[:, 'obs'] = obs_vals
    #     # plot observed time series
    #     fig = eval_utils.plot_sim(df_obs, labs._Y_LABS_DAILY[var_sim])
    #     file_str = '%s.pdf' % (var_obs)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    # plt.close('all')

    # # compare HYDRUS-1D simulations with observations
    # vars_obs = ['AET', 'PERC', 'dWEIGHT']
    # vars_sim = ['aet', 'perc', 'dS']
    # dict_obs_sim_hydrus = {}
    # for var_obs, var_sim in zip(vars_obs, vars_sim):
    #     obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    #     df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    #     df_obs.loc[:, 'obs'] = obs_vals
    #     sim_vals = ds_hydrus_18O[var_sim].values
    #     # join observations on simulations
    #     df_eval = eval_utils.join_obs_on_sim(date_hydrus_18O, sim_vals, df_obs)
    #     df_eval = df_eval.iloc[:, :]
    #     dict_obs_sim_hydrus[var_sim] = df_eval
    #     # plot observed and simulated time series
    #     fig = eval_utils.plot_obs_sim(df_eval, labs._Y_LABS_DAILY[var_sim])
    #     file_str = 'hydrus_%s.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    #     # plot cumulated observed and simulated time series
    #     fig = eval_utils.plot_obs_sim_cum(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time [year]')
    #     file_str = 'hydrus_%s_cum.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    #     fig = eval_utils.plot_obs_sim_cum_year_facet(df_eval, labs._Y_LABS_CUM[var_sim], x_lab='Time\n[day-month-hydyear]')
    #     file_str = 'hydrus_%s_cum_year_facet.pdf' % (var_sim)
    #     path_fig = base_path_figs / file_str
    #     fig.savefig(path_fig, dpi=250)
    # plt.close('all')

    # # plot cumulated precipitation, evapotranspiration, soil storage change and percolation
    # fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 7))
    # axes[0].plot(dict_obs['prec'].index, dict_obs['prec'].cumsum(), lw=1.5, color='blue', ls='-', alpha=1)
    # axes[0].set_ylabel('PREC\n[mm]')
    # axes[0].set_xlim((dict_obs['prec'].index[0], dict_obs['prec'].index[-1]))
    # axes[0].set_ylim(0,)
    # axes[0].invert_yaxis()
    # ax2 = axes[0].twinx()
    # ax2.plot(dict_obs_sim['aet'].index, dict_obs_sim['aet']['obs'].cumsum(),
    #           lw=1.5, color='blue', ls='-', alpha=0.5)
    # ax2.plot(dict_obs_sim['aet'].index, dict_obs_sim['aet']['sim'].cumsum(),
    #           lw=1, color='red', ls='-.')
    # ax2.plot(dict_obs_sim_hydrus['aet'].index, dict_obs_sim_hydrus['aet']['sim'].cumsum(),
    #           lw=1, color='gray', ls='-.')
    # ax2.set_ylim(0,)
    # ax2.set_ylabel('ET\n[mm]')
    # axes[1].plot(dict_obs_sim['dS'].loc['2000':, :].index, dict_obs_sim['dS'].loc['2000':, 'obs'].cumsum(),
    #               lw=1.5, color='blue', ls='-', alpha=0.5)
    # axes[1].plot(dict_obs_sim['dS'].loc['2000':, :].index, dict_obs_sim['dS'].loc['2000':, 'sim'].cumsum(),
    #               lw=1, color='red', ls='-.')
    # axes[1].plot(dict_obs_sim_hydrus['dS'].loc['2000':, :].index, dict_obs_sim_hydrus['dS'].loc['2000':, 'sim'].cumsum(),
    #               lw=1, color='gray', ls='-.')
    # axes[1].set_ylabel('cum. $\Delta$S\n[mm]')
    # axes[1].set_xlim((dict_obs_sim['dS'].index[0], dict_obs_sim['dS'].index[-1]))
    # axes[2].plot(dict_obs_sim['q_ss'].index, dict_obs_sim['q_ss']['obs'].cumsum(),
    #               lw=1.5, color='blue', ls='-', alpha=0.5)
    # axes[2].plot(dict_obs_sim['q_ss'].index, dict_obs_sim['q_ss']['sim'].cumsum(),
    #               lw=1, color='red', ls='-.')
    # axes[2].plot(dict_obs_sim_hydrus['perc'].index, dict_obs_sim_hydrus['perc']['sim'].cumsum(),
    #               lw=1, color='gray', ls='-.')
    # axes[2].set_ylim(0,)
    # axes[2].invert_yaxis()
    # axes[2].set_xlim((dict_obs_sim['q_ss'].index[0], dict_obs_sim['q_ss'].index[-1]))
    # axes[2].set_ylabel('PERC\n[mm]')
    # axes[2].set_xlabel(r'Time [year]')
    # axes[0].text(0.015, 0.9, '(a)', size=15, horizontalalignment='center',
    #               verticalalignment='center', transform=axes[0].transAxes)
    # axes[1].text(0.015, 0.9, '(b)', size=15, horizontalalignment='center',
    #               verticalalignment='center', transform=axes[1].transAxes)
    # axes[2].text(0.015, 0.9, '(c)', size=15, horizontalalignment='center',
    #               verticalalignment='center', transform=axes[2].transAxes)
    # fig.tight_layout()
    # file = 'prec_et_dS_perc_obs_sim_cumulated.png'
    # path = base_path_figs / file
    # fig.savefig(path, dpi=250)

    # # compare best 1% simulations with observations
    # nx = ds_sim_hm1.dims['x']
    # fig, axes = plt.subplots(3, 1, sharex=True, figsize=(14, 7))
    # axes[0].plot(dict_obs['prec'].index, dict_obs['prec'].cumsum(), lw=1.5, color='blue', ls='-', alpha=1)
    # axes[0].set_ylabel('PREC\n[mm]')
    # axes[0].set_xlim((dict_obs['prec'].index[0], dict_obs['prec'].index[-1]))
    # axes[0].set_ylim(0,)
    # axes[0].invert_yaxis()
    # ax2 = axes[0].twinx()
    # for nrow in range(nx):
    #     ax2.plot(dict_obs_sim1['aet'].index, dict_obs_sim1['aet'].iloc[:, nrow].cumsum(),
    #               lw=1, color='red', ls='-', alpha=.8)
    # ax2.plot(dict_obs_sim1['aet'].index, dict_obs_sim1['aet']['obs'].cumsum(),
    #           lw=2, color='blue', ls='-', alpha=1)
    # ax2.plot(dict_obs_sim_hydrus['aet'].index, dict_obs_sim_hydrus['aet']['sim'].cumsum(),
    #           lw=2, color='gray', ls='-.')
    # ax2.set_ylim(0,)
    # ax2.set_ylabel('ET\n[mm]')
    # for nrow in range(nx):
    #     axes[1].plot(dict_obs_sim1['dS'].loc['2000':, :].index, dict_obs_sim1['dS'].loc['2000':, f'sim{nrow}'].cumsum(),
    #               lw=1, color='red', ls='-')
    # axes[1].plot(dict_obs_sim1['dS'].loc['2000':, :].index, dict_obs_sim1['dS'].loc['2000':, 'obs'].cumsum(),
    #               lw=2, color='blue', ls='-', alpha=1)
    # axes[1].plot(dict_obs_sim_hydrus['dS'].loc['2000':, :].index, dict_obs_sim_hydrus['dS'].loc['2000':, 'sim'].cumsum(),
    #               lw=2, color='gray', ls='-.', alpha=.8)
    # axes[1].set_ylabel('cum. $\Delta$S\n[mm]')
    # axes[1].set_xlim((dict_obs_sim['dS'].index[0], dict_obs_sim['dS'].index[-1]))
    # for nrow in range(nx):
    #     axes[2].plot(dict_obs_sim1['q_ss'].index, dict_obs_sim1['q_ss'].iloc[:, nrow].cumsum(),
    #               lw=1, color='red', ls='-', alpha=.8)
    # axes[2].plot(dict_obs_sim['q_ss'].index, dict_obs_sim['q_ss']['obs'].cumsum(),
    #               lw=2, color='blue', ls='-', alpha=1)
    # axes[2].plot(dict_obs_sim_hydrus['perc'].index, dict_obs_sim_hydrus['perc']['sim'].cumsum(),
    #               lw=2, color='gray', ls='-.')
    # axes[2].set_ylim(0,)
    # axes[2].invert_yaxis()
    # axes[2].set_xlim((dict_obs_sim1['q_ss'].index[0], dict_obs_sim1['q_ss'].index[-1]))
    # axes[2].set_ylabel('PERC\n[mm]')
    # axes[2].set_xlabel(r'Time [year]')
    # axes[0].text(0.015, 0.9, '(a)', size=15, horizontalalignment='center',
    #               verticalalignment='center', transform=axes[0].transAxes)
    # axes[1].text(0.015, 0.9, '(b)', size=15, horizontalalignment='center',
    #               verticalalignment='center', transform=axes[1].transAxes)
    # axes[2].text(0.015, 0.9, '(c)', size=15, horizontalalignment='center',
    #               verticalalignment='center', transform=axes[2].transAxes)
    # fig.tight_layout()
    # file = 'prec_et_dS_perc_obs_sim_cumulated_best_1perc.png'
    # path = base_path_figs / file
    # fig.savefig(path, dpi=250)

    # # load metrics of transport simulations
    # dict_params_metrics_tm_mc = {}
    # for tm_structure in transport_models:
    #     tms = tm_structure.replace(" ", "_")
    #     file = base_path / "svat_transport" / "results" / "deterministic" / "age_max_11" / f"params_metrics_{tms}.txt"
    #     df_params_metrics = pd.read_csv(file, sep="\t")
    #     dict_params_metrics_tm_mc[tm_structure] = {}
    #     dict_params_metrics_tm_mc[tm_structure]['params_metrics'] = df_params_metrics
    #
    # # compare best model runs
    # fig, ax = plt.subplots(2, 2, sharey=True, figsize=(14, 7))
    # for i, tm_structure in enumerate(transport_models):
    #     idx_best = dict_params_metrics_tm_mc[tm_structure]['params_metrics']['KGE_C_iso_q_ss'].idxmax()
    #     tms = tm_structure.replace(" ", "_")
    #     # load transport simulation
    #     states_tm_file = base_path / "svat_transport" / "deterministic" / "age_max_11" / f"states_{tms}.nc"
    #     ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    #     days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    #     date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    #     ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))
    #     # join observations on simulations
    #     obs_vals = ds_obs['d18O_PERC'].isel(x=0, y=0).values
    #     sim_vals = ds_sim_tm['d18O_perc_bs'].isel(x=idx_best, y=0).values
    #     sim_vals_hydrus = ds_hydrus_18O['d18O_perc_bs'].values
    #     df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    #     df_obs.loc[:, 'obs'] = obs_vals
    #     df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals, df_obs)
    #     df_eval = df_eval.dropna()
    #     df_eval_hydrus = eval_utils.join_obs_on_sim(date_hydrus_18O, sim_vals_hydrus, df_obs)
    #     df_eval_hydrus = df_eval_hydrus.dropna()
    #     ax.flatten()[i].plot(date_sim_tm, ds_sim_tm['C_iso_q_ss'].isel(x=idx_best, y=0).values, color='red')
    #     ax.flatten()[i].plot(date_hydrus_18O, ds_hydrus_18O['d18O_perc'].values, color='grey')
    #     ax.flatten()[i].scatter(df_eval.index, df_eval.iloc[:, 0], color='red', s=2)
    #     ax.flatten()[i].scatter(df_eval.index, df_eval.iloc[:, 1], color='blue', s=2)
    #     ax.flatten()[i].scatter(df_eval_hydrus.index, df_eval_hydrus.iloc[:, 0], color='grey', s=2)
    #
    # ax[0, 0].set_ylabel(r'$\delta^{18}$O [‰]')
    # ax[1, 0].set_ylabel(r'$\delta^{18}$O [‰]')
    # ax[1, 0].set_xlabel('Time [year]')
    # ax[1, 1].set_xlabel('Time [year]')
    # fig.tight_layout()
    # file = base_path_figs / "d18O_perc_sim_obs_transport_models.png"
    # fig.savefig(file, dpi=250)
    #
    # # compare duration curve of 18O in percolation
    # fig, ax = plt.subplots(2, 2, sharey=True, figsize=(14, 7))
    # for i, tm_structure in enumerate(transport_models):
    #     idx_best = dict_params_metrics_tm_mc[tm_structure]['params_metrics']['KGE_C_iso_q_ss'].idxmax()
    #     tms = tm_structure.replace(" ", "_")
    #     # load transport simulation
    #     states_tm_file = base_path / "svat_transport" / "deterministic" / "age_max_11" / f"states_{tms}.nc"
    #     ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    #     days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    #     date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    #     ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))
    #     # join observations on simulations
    #     obs_vals = ds_obs['d18O_PERC'].isel(x=0, y=0).values
    #     df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    #     df_obs.loc[:, 'obs'] = obs_vals
    #     df_sim = pd.DataFrame(index=date_sim_tm)
    #     df_sim.loc[:, 'sim0'] = ds_sim_tm['d18O_perc_bs'].isel(x=idx_best, y=0).values
    #     df_sim.loc[df_sim.index[1]:, 'sim1'] = ds_hydrus_18O['d18O_perc_bs'].values
    #     df_sim = df_sim.iloc[1:, :]
    #     df_eval = eval_utils.join_obs_on_sim(date_sim_tm[1:], df_sim.values, df_obs)
    #     df_eval = df_eval.dropna()
    #     obs = df_eval.sort_values(by=["obs"], ascending=True)
    #     sim0 = df_eval.sort_values(by=["sim0"], ascending=True)
    #     sim1 = df_eval.sort_values(by=["sim1"], ascending=True)
    #
    #     # calculate exceedence probability
    #     ranks_obs = sp.stats.rankdata(obs["obs"], method="ordinal")
    #     ranks_obs = ranks_obs[::-1]
    #     prob_obs = [(ranks_obs[i] / (len(obs["obs"]) + 1)) for i in range(len(obs["obs"]))]
    #
    #     ranks_sim0 = sp.stats.rankdata(sim0["sim0"], method="ordinal")
    #     ranks_sim0 = ranks_sim0[::-1]
    #     prob_sim0 = [(ranks_sim0[i] / (len(sim0["sim0"]) + 1)) for i in range(len(sim0["sim0"]))]
    #
    #     ranks_sim1 = sp.stats.rankdata(sim1["sim1"], method="ordinal")
    #     ranks_sim1 = ranks_sim1[::-1]
    #     prob_sim1 = [(ranks_sim1[i] / (len(sim1["sim1"]) + 1)) for i in range(len(sim1["sim1"]))]
    #
    #     ax.flatten()[i].plot(prob_obs, obs["obs"], color="blue", lw=1.5)
    #     ax.flatten()[i].plot(prob_sim0, sim0["sim0"], color="red", lw=1, ls="-.", alpha=0.8)
    #     ax.flatten()[i].plot(prob_sim1, sim1["sim1"], color="grey", lw=1, ls="-.", alpha=0.8)
    #     ax.flatten()[i].set_xlim(0, 1)
    #
    # ax[0, 0].set_ylabel(r'$\delta^{18}$O [‰]')
    # ax[1, 0].set_ylabel(r'$\delta^{18}$O [‰]')
    # ax[1, 0].set_xlabel('Exceedence probabilty [-]')
    # ax[1, 1].set_xlabel('Exceedence probabilty [-]')
    # fig.tight_layout()
    # file = base_path_figs / "fdc_d18O_perc_sim_obs_transport_models.png"
    # fig.savefig(file, dpi=250)

    # bromide benchmark
    years = onp.arange(1997, 2007).tolist()
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), dpi=250)
    for year in years:
        states_hydrus_br_file = base_path / "hydrus_benchmark" / "states_hydrus_bromide.nc"
        with xr.open_dataset(states_hydrus_br_file, engine="h5netcdf", decode_times=False, group=f'{year}') as ds:
            df_sim_br = pd.DataFrame(index=df_obs_br.index)
            df_sim_br.loc[:, "Br"] = ds["Br_perc_mmol"].values
        ax.plot(df_sim_br.dropna().index, df_sim_br.dropna()["Br"], color="grey", lw=1)
    ax.plot(df_obs_br.dropna().index, df_obs_br.dropna()["Br"], color="blue", lw=1.5)
    ax.set_ylim(0,)
    ax.set_xlim([0, 400])
    ax.set_ylabel(r'Bromide [mmol/l]')
    ax.set_xlabel(r'Time [days since injection]')
    fig.tight_layout()
    file = base_path_figs / "bromide_benchmark.png"
    fig.savefig(file, dpi=250)

    # travel time benchmark
    # plot cumulative backward travel time distributions
    TT = ds_hydrus_tt['TT_perc'].values
    fig, axs = plt.subplots()
    for i in range(len(date_hydrus_tt)):
        axs.plot(TT[i, :], lw=1, color='grey')
    axs.set_xlim((0, 4000))
    axs.set_ylim((0, 1))
    axs.set_ylabel('$P(T,t)$')
    axs.set_xlabel('T [days]')
    fig.tight_layout()
    file_str = 'TTD_hydrus.pdf'
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=250)
    
    # plot cumulative forward travel time distributions
    TT = ds_hydrus_tt['fTT_perc'].values
    fig, axs = plt.subplots()
    for i in range(len(date_hydrus_tt)):
        axs.plot(TT[i, :], lw=1, color='grey')
    axs.set_xlim((0, 4000))
    axs.set_ylim((0, 1))
    axs.set_ylabel('$P(T,t)$')
    axs.set_xlabel('T [days]')
    fig.tight_layout()
    file_str = 'fTTD_hydrus.pdf'
    path_fig = base_path_figs / file_str
    fig.savefig(path_fig, dpi=250)

    # # perform sensitivity analysis
    # dict_params_metrics_tm_sa = {}
    # for tm_structure in transport_models:
    #     tms = tm_structure.replace(" ", "_")
    #     file = base_path / "svat_transport_sensitivity" / "results" / f"params_metrics_{tms}.txt"
    #     df_params_metrics = pd.read_csv(file, sep="\t")
    #     dict_params_metrics_tm_sa[tm_structure] = {}
    #     dict_params_metrics_tm_sa[tm_structure]['params_metrics'] = df_params_metrics
    #
    # # sampled parameter space
    # file_path = base_path / "svat_transport_sensitivity" / "param_bounds.yml"
    # with open(file_path, 'r') as file:
    #     bounds = yaml.safe_load(file)
    #
    # for tm_structure in transport_models:
    #     tms = tm_structure.replace(" ", "_")
    #     df_params_metrics = dict_params_metrics_tm_sa[tm_structure]['params_metrics']
    #     df_params = df_params_metrics.loc[:, bounds[tm_structure]['names']]
    #     n_params = len(bounds[tm_structure]['names'])
    #     df_metrics = df_params_metrics.iloc[:, n_params:]
    #     dict_si = {}
    #     for name in df_metrics.columns:
    #         Y = df_metrics[name].values
    #         Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
    #         Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
    #         dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]['names'])
    #
    #     # plot sobol indices
    #     _LABS = {'KGE_C_q_ss': 'KGE',
    #              'median_TT_q_ss': r'$tt_{50}$',
    #              'median_SA_s': r'rt_{50}$',
    #              }
    #     ncol = len(df_metrics.columns)
    #     xaxis_labels = [labs._LABS[k].split(' ')[0] for k in bounds[tm_structure]['names']]
    #     cmap = cm.get_cmap('Greys')
    #     norm = Normalize(vmin=0, vmax=2)
    #     colors = cmap(norm([0.5, 1.5]))
    #     fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(14, 5))
    #     for i, name in enumerate(df_metrics.columns):
    #         indices = dict_si[name][['S1', 'ST']]
    #         err = dict_si[name][['S1_conf', 'ST_conf']]
    #         indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
    #         ax[i].set_xticklabels(xaxis_labels)
    #         ax[i].set_title(_LABS[name])
    #         ax[i].legend(["First-order", "Total"], frameon=False)
    #     ax[-1].legend().set_visible(False)
    #     ax[-2].legend().set_visible(False)
    #     ax[-3].legend().set_visible(False)
    #     ax[0].set_ylabel('Sobol index [-]')
    #     fig.tight_layout()
    #     file = base_path_figs / f"sobol_indices_{tms}.png"
    #     fig.savefig(file, dpi=250)
    #
    #     # make dotty plots
    #     nrow = len(df_metrics.columns)
    #     ncol = bounds[tm_structure]['num_vars']
    #     fig, ax = plt.subplots(nrow, ncol, sharey='row', figsize=(14, 7))
    #     for i in range(nrow):
    #         for j in range(ncol):
    #             y = df_metrics.iloc[:, i]
    #             x = df_params.iloc[:, j]
    #             sns.regplot(x=x, y=y, ax=ax[i, j], ci=None, color='k',
    #                         scatter_kws={'alpha': 0.2, 's': 4, 'color': 'grey'})
    #             ax[i, j].set_xlabel('')
    #             ax[i, j].set_ylabel('')
    #
    #     for j in range(ncol):
    #         xlabel = labs._LABS[bounds[tm_structure]['names'][j].split(' ')[0]]
    #         ax[-1, j].set_xlabel(xlabel)
    #
    #     ax[0, 0].set_ylabel(r'$KGE$ [-]')
    #     ax[1, 0].set_ylabel(r'$\alpha$ [-]')
    #     ax[2, 0].set_ylabel(r'$\beta$ [-]')
    #     ax[3, 0].set_ylabel(r'r [-]')
    #
    #     fig.subplots_adjust(wspace=0.2, hspace=0.3)
    #     file = base_path_figs / f"dotty_plots_{tms}.png"
    #     fig.savefig(file, dpi=250)

    # # dotty plots of HYDRUS-1D monte carlo simulations
    # file = base_path / "hydrus_benchmark" / "params_metrics.txt"
    # df_params_metrics_hydrus = pd.read_csv(file, sep="\t")
    # df_params_metrics_hydrus.loc[:, 'ks'] = df_params_metrics_hydrus.loc[:, 'ks'] * (10/24)
    # df_metrics_hydrus = df_params_metrics_hydrus.loc[:, ['kge_aet', 'kge_theta', 'kge_perc', 'kge_perc_18O', 'kge_multi']]
    # df_params_hydrus = df_params_metrics_hydrus.loc[:, ['theta_sat_m', 'alpha', 'n', 'ks', 'theta_sat_im', 'omega', 'D_l']]
    # nrow = len(df_metrics_hydrus.columns)
    # ncol = len(df_params_hydrus.columns)
    # idx_best = df_metrics_hydrus['kge_multi'].idxmax()
    # fig, ax = plt.subplots(nrow, ncol, sharey='row', figsize=(14, 7))
    # for i in range(nrow):
    #     for j in range(ncol):
    #         y = df_metrics_hydrus.iloc[:, i]
    #         x = df_params_hydrus.iloc[:, j]
    #         ax[i, j].scatter(x, y, s=4, c='grey', alpha=0.5)
    #         ax[i, j].set_xlabel('')
    #         ax[i, j].set_ylabel('')
    #         ax[i, j].set_ylim(-1, 1)
    #         # best parameter set for multi-objective criteria
    #         y_best = df_metrics_hydrus.iloc[idx_best, i]
    #         x_best = df_params_hydrus.iloc[idx_best, j]
    #         ax[i, j].scatter(x_best, y_best, s=12, c='red', alpha=0.8)

    # for j in range(ncol):
    #     xlabel = _LABS_HYDRUS[df_params_hydrus.columns[j]]
    #     ax[-1, j].set_xlabel(xlabel)

    # ax[0, 0].set_ylabel('$KGE_{ET}$\n [-]')
    # ylab_kge_theta = r'''$KGE_{\theta}$
    # [-]'''
    # ax[1, 0].set_ylabel(ylab_kge_theta)
    # ax[2, 0].set_ylabel('$KGE_{PERC}$\n [-]')
    # ax[3, 0].set_ylabel('$KGE_{\delta^{18}O_{perc}}$\n [-]')
    # ax[4, 0].set_ylabel('$KGE_{multi}$\n [-]')

    # fig.subplots_adjust(wspace=0.2, hspace=0.3)
    # file = base_path_figs / "dotty_plots_hydrus.png"
    # fig.savefig(file, dpi=250)

    # plot mean residence time along soil depth
    cmap = copy.copy(plt.cm.get_cmap('Blues_r'))
    norm = mpl.colors.Normalize(vmin=0, vmax=500)

    fig, axes = plt.subplots(1, 1, figsize=(10, 3))
    sns.heatmap(ds_hydrus_tt['mrt_s'].values, xticklabels=366, yticklabels=int(50/2), cmap='Blues_r',
                vmax=500, vmin=0, cbar=False, ax=axes)
    axes.set_yticks([0, 25, 50, 75, 100])
    axes.set_yticklabels([0, 0.5, 1, 1.5, 2])
    axes.set_xticklabels(list(range(1997, 2008)))
    axes.set_ylabel('Soil depth [m]')
    axes.set_xlabel('Time [year]')

    axl = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
                                    orientation='vertical',
                                    ticks=[0, 100, 200, 300, 400, 500])
    cb1.ax.invert_yaxis()
    cb1.set_label(r'age [days]')
    fig.subplots_adjust(bottom=0.15)
    file = 'mean_residence_time_soil.png'
    path = base_path_figs / file
    fig.savefig(path, dpi=250)

    # # plot isotope ratios of precipitation and soil
    # cmap = copy.copy(plt.cm.get_cmap('YlGnBu_r'))
    # norm = mpl.colors.Normalize(vmin=-20, vmax=5)
    #
    # fig, axes = plt.subplots(2, 1, sharex=False, figsize=(14, 7))
    # axes[0].bar(date_hydrus_18O, ds_hydrus_18O['prec'].values, width=-1, color=cmap(norm(ds_hydrus_18O['d18O_prec'].values)), align='edge', edgecolor=cmap(norm(ds_hydrus_18O['d18O_prec'].values)))
    # axes[0].set_ylabel('Precipitation\n[mm $day^{-1}$]')
    # axes[0].set_xlim(date_hydrus_18O[0], date_hydrus_18O[-1])
    # sns.heatmap(ds_hydrus_18O['d18O_soil'].values, xticklabels=366, yticklabels=int(50/2), cmap='YlGnBu_r',
    #             vmax=5, vmin=-20, cbar=False, ax=axes[1])
    # axes[1].set_yticks([0, 25, 50, 75, 100])
    # axes[1].set_yticklabels([0, 0.5, 1, 1.5, 2])
    # axes[1].set_xticklabels(list(range(1997, 2008)))
    # axes[1].set_ylabel('Soil depth\n[m]')
    # axes[1].set_xlabel('Time [years]')
    #
    # axl = fig.add_axes([0.92, 0.3, 0.02, 0.3])
    # cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
    #                              orientation='vertical',
    #                              ticks=[0, -5, -10, -15])
    # cb1.set_label(r'$\delta^{18}$O [‰]')
    # file = 'hydrus_d18O_prec_soil.png'
    # path = base_path_figs / file
    # fig.savefig(path, dpi=250)

    # # plot isotope ratios of precipitation, soil and percolation
    # cmap = copy.copy(plt.cm.get_cmap('YlGnBu_r'))
    # norm = mpl.colors.Normalize(vmin=-20, vmax=5)

    # fig, axes = plt.subplots(3, 1, sharex=False, figsize=(14, 7))
    # axes[0].bar(date_hydrus_18O, ds_hydrus_18O['prec'].values, width=-1, edgecolor=cmap(norm(ds_hydrus_18O['d18O_prec'].values)), align='edge')
    # axes[0].set_ylabel('Precipitation\n[mm $day^{-1}$]')
    # axes[0].set_xlim(date_hydrus_18O[0], date_hydrus_18O[-1])
    # sns.heatmap(ds_hydrus_18O['d18O_soil'].values, xticklabels=366, yticklabels=int(50/2), cmap='YlGnBu_r',
    #          vmax=5, vmin=-20, cbar=False, ax=axes[1])
    # axes[1].set_yticks([0, 25, 50, 75, 100])
    # axes[1].set_yticklabels([0, 0.5, 1, 1.5, 2])
    # axes[1].set_xticklabels(list(range(1997, 2008)))
    # axes[1].set_ylabel('Soil depth\n[m]')

    # axes[2].bar(date_hydrus_18O, ds_hydrus_18O['perc'].values, width=-1, edgecolor=cmap(norm(ds_hydrus_18O['d18O_perc'].values)), align='edge')
    # axes[2].set_xlim(date_hydrus_18O[0], date_hydrus_18O[-1])
    # axes[2].set_ylabel('Percolation\n[mm $day^{-1}$]')
    # axes[2].set_ylim(0, )
    # axes[2].invert_yaxis()
    # axes[2].set_xlabel('Time [year]')

    # axl = fig.add_axes([0.92, 0.33, 0.02, 0.3])
    # cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
    #                                 orientation='vertical',
    #                                 ticks=[0, -5, -10, -15])
    # cb1.set_label(r'$\delta^{18}$O [‰]')

    # file = 'hydrus_d18O_prec_soil_perc.png'
    # path = base_path_figs / file
    # fig.savefig(path, dpi=250)

    # # plot precipitation, soil water content and percolation
    # cmap = copy.copy(plt.cm.get_cmap('YlGnBu'))
    # norm = mpl.colors.Normalize(vmin=0.1, vmax=0.4)

    # fig, axes = plt.subplots(3, 1, sharex=False, figsize=(14, 7))
    # axes[0].bar(date_hydrus_18O, ds_hydrus_18O['prec'].values, width=-1, edgecolor='blue', align='edge')
    # axes[0].set_ylabel('Precipitation\n[mm $day^{-1}$]')
    # axes[0].set_xlim(date_hydrus_18O[0], date_hydrus_18O[-1])
    # sns.heatmap(ds_hydrus_18O['swc'].values, xticklabels=366, yticklabels=int(50/2), cmap='YlGnBu',
    #          vmax=0.4, vmin=0.1, cbar=False, ax=axes[1])
    # axes[1].set_yticks([0, 25, 50, 75, 100])
    # axes[1].set_yticklabels([0, 0.5, 1, 1.5, 2])
    # axes[1].set_xticklabels(list(range(1997, 2008)))
    # axes[1].set_ylabel('Soil depth\n[m]')

    # axes[2].bar(date_hydrus_18O, ds_hydrus_18O['perc'].values, width=-1, edgecolor='grey', align='edge')
    # axes[2].set_xlim(date_hydrus_18O[0], date_hydrus_18O[-1])
    # axes[2].set_ylabel('Percolation\n[mm $day^{-1}$]')
    # axes[2].set_ylim(0, )
    # axes[2].invert_yaxis()
    # axes[2].set_xlabel('Time [year]')

    # axl = fig.add_axes([0.92, 0.33, 0.02, 0.3])
    # cb1 = mpl.colorbar.ColorbarBase(axl, cmap=cmap, norm=norm,
    #                                 orientation='vertical',
    #                                 ticks=[0.1, 0.2, 0.3, 0.4])
    # cb1.set_label(r'$\theta$ [-]')

    # file = 'hydrus_prec_theta_perc.png'
    # path = base_path_figs / file
    # fig.savefig(path, dpi=250)

    plt.close('all')
    return


if __name__ == "__main__":
    main()
