from pathlib import Path
import os
import xarray as xr
from cftime import num2date
import pandas as pd
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import numpy as onp

# sampled parameter space
bounds = {
    'num_vars': 6,
    'names': ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks'],
    'bounds': [[1, 400],
               [1, 1500],
               [0.05, 0.33],
               [0.05, 0.33],
               [0.05, 0.33],
               [0.1, 120]]
}

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)

# load simulation
states_hm_file = base_path / "states_hm_sensitivity.nc"
ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")

# load observations (measured data)
path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/rietholzbach/rietholzbach_lysimeter.nc")
ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")


tm_structures = ['preferential', 'advection-dispersion',
                 'complete-mixing advection-dispersion',
                 'time-variant preferential',
                 'time-variant advection-dispersion']
for tm_structure in tm_structures:
    tms = tm_structure.replace(" ", "_")

    # load simulation
    states_tm_file = base_path / "states_tm_sensitivity_reverse.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, f"{tm_structure}", engine="h5netcdf")

    # assign date
    days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim_hm))
    ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    # DataFrame with sampled model parameters and the corresponding metrics
    nx = ds_sim_tm.dims['x']  # number of rows
    ny = ds_sim_tm.dims['y']  # number of columns
    df_params_eff = pd.DataFrame(index=range(nx * ny))
    # sampled model parameters
    for param in ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']:
        df_params_eff.loc[:, param] = ds_sim_hm[param].isel(y=0).values

    # compare observations and simulations
    ncol = 0
    idx = ds_sim_tm.Time  # time index
    for nrow in range(nx):
        # calculate simulated oxygen-18 composite sample
        df_perc_18O_obs = pd.DataFrame(index=idx, columns=['perc_obs', 'd18O_perc_obs'])
        df_perc_18O_obs.loc[:, 'perc_obs'] = ds_obs['PERC'].isel(x=nrow, y=ncol).values
        df_perc_18O_obs.loc[:, 'd18O_perc_obs'] = ds_obs['d18O_perc'].isel(x=nrow, y=ncol).values
        sample_no = pd.DataFrame(index=df_perc_18O_obs.dropna().index, columns=['sample_no'])
        sample_no = sample_no.loc['1997':'2007']
        sample_no['sample_no'] = range(len(sample_no.index))
        df_perc_18O_sim = pd.DataFrame(index=idx, columns=['perc_sim', 'd18O_perc_sim'])
        df_perc_18O_sim['perc_sim'] = ds_sim_hm['q_ss'].isel(x=nrow, y=ncol).values
        df_perc_18O_sim['d18O_perc_sim'] = ds_sim_tm['C_q_ss'].isel(x=nrow, y=ncol).values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no)
        df_perc_18O_sim.loc[:, 'sample_no'] = df_perc_18O_sim.loc[:, 'sample_no'].fillna(method='bfill', limit=14)
        perc_sum = df_perc_18O_sim.groupby(['sample_no']).sum().loc[:, 'perc_sim']
        sample_no['perc_sum'] = perc_sum.values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_sum'])
        df_perc_18O_sim.loc[:, 'perc_sum'] = df_perc_18O_sim.loc[:, 'perc_sum'].fillna(method='bfill', limit=14)
        df_perc_18O_sim['weight'] = df_perc_18O_sim['perc_sim'] / df_perc_18O_sim['perc_sum']
        df_perc_18O_sim['d18O_weight'] = df_perc_18O_sim['d18O_perc_sim'] * df_perc_18O_sim['weight']
        d18O_sample = df_perc_18O_sim.groupby(['sample_no']).sum().loc[:, 'd18O_weight']
        sample_no['d18O_sample'] = d18O_sample.values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no['d18O_sample'])
        df_perc_18O_sim.loc[:, 'd18O_sample'] = df_perc_18O_sim.loc[:, 'd18O_sample'].fillna(method='bfill', limit=14)
        cond = (df_perc_18O_sim['d18O_sample'] == 0)
        df_perc_18O_sim.loc[cond, 'd18O_sample'] = onp.NaN
        d18O_perc_cs = onp.zeros((1, 1, len(idx)))
        d18O_perc_cs[nrow, ncol, :] = df_perc_18O_sim.loc[:, 'd18O_sample'].values
        ds_sim_tm.assign(d18O_perc_cs=d18O_perc_cs)
        # calculate observed oxygen-18 composite sample
        df_perc_18O_obs.loc[:, 'd18O_perc_cs'] = df_perc_18O_obs['d18O_perc'].fillna(method='bfill', limit=14)

        perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
        sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
        df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)

        # join observations on simulations
        obs_vals = ds_obs['d18O_perc'].isel(x=nrow, y=ncol).values
        sim_vals = ds_sim_tm['d18O_perc_cs'].isel(x=nrow, y=ncol).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = ds_obs['d18O_perc'].isel(x=nrow, y=ncol).values
        df_eval = eval_utils.join_obs_on_sim(date_sim_tm, sim_vals, df_obs)
        df_eval = df_eval.dropna()

        # calculate metrics
        var_sim = 'C_q_ss'
        obs_vals = df_eval.loc[:, 'obs'].values
        sim_vals = df_eval.loc[:, 'sim'].values
        key_kge = 'KGE_' + var_sim
        df_params_eff.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
        key_kge_alpha = 'KGE_alpha_' + var_sim
        df_params_eff.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
        key_kge_beta = 'KGE_beta_' + var_sim
        df_params_eff.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
        key_r = 'r_' + var_sim
        df_params_eff.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)

    # write to .txt
    file = base_path_results / f"params_eff_{tm_structure}.txt"
    df_params_eff.to_csv(file, header=True, index=False, sep="\t")

    # perform sensitivity analysis
    df_params = df_params_eff.loc[:, bounds['names']]
    n_params = len(bounds['names'])
    df_eff = df_params_eff.iloc[:, n_params:]
    dict_si = {}
    for name in df_eff.columns:
        Y = df_eff[name].values
        Si = sobol.analyze(bounds, Y)
        Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
        dict_si[name] = pd.DataFrame(Si_filter, index=bounds['names'])

    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # plot sobol indices
    _LABS = {'KGE_C_q_ss': 'KGE',
             'KGE_beta_C_q_ss': '$\beta$',
             'KGE_alpha_C_q_ss': '$\alpha$',
             'r_C_q_ss': 'r',
             }
    ncol = len(df_eff.columns)
    xaxis_labels = [labs._LABS[k].split(' ')[0] for k in bounds['names']]
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
    ncol = bounds['num_vars']
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(14, 7))
    for i in range(nrow):
        for j in range(ncol):
            y = df_eff.iloc[:, i]
            x = df_params.iloc[:, j]
            sns.regplot(x=x, y=y, ax=ax[i, j], ci=None, color='k',
                        scatter_kws={'alpha': 0.2, 's': 4, 'color': 'grey'})
            ax[i, j].set_xlabel('')
            ax[i, j].set_ylabel('')

    for j in range(ncol):
        xlabel = labs._LABS[bounds['names'][j].split(' ')[0]]
        ax[-1, j].set_xlabel(xlabel)

    ax[0, 0].set_ylabel(r'$KGE$ [-]')
    ax[1, 0].set_ylabel(r'$\alpha$ [-]')
    ax[2, 0].set_ylabel(r'$\beta$ [-]')
    ax[3, 0].set_ylabel(r'r [-]')

    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    file = base_path_figs / f"dotty_plots_{tms}.png"
    fig.savefig(file, dpi=250)
