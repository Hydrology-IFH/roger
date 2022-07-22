import shutil
import glob
import os
from pathlib import Path
import datetime
from cftime import num2date
import h5netcdf
import xarray as xr
import pandas as pd
from de import de
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import yaml
import numpy as onp
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import roger
onp.random.seed(42)

base_path = Path(__file__).parent
base_path_data = Path('/Volumes/Gerics/') / base_path.parts[0].join(base_path.parts[-5:])

# sampled parameter space
file_path = base_path / "param_bounds.yml"
with open(file_path, 'r') as file:
    bounds = yaml.safe_load(file)

# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# merge model output into single file
path = str(base_path / "SVAT.*.nc")
diag_files = glob.glob(path)
states_hm_si_file = base_path / "states_hm_sensitivity.nc"
with h5netcdf.File(states_hm_si_file, 'w', decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title='RoGeR saltelli results at Rietholzbach Lysimeter site',
        institution='University of Freiburg, Chair of Hydrology',
        references='',
        comment='',
        model_structure='SVAT model with free drainage',
        roger_version=f'{roger.__version__}'
    )
    # collect dimensions
    for dfs in diag_files:
        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
            # set dimensions with a dictionary
            if not dfs.split('/')[-1].split('.')[1] == 'constant':
                dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time'])}
                time = onp.array(df.variables.get('Time'))
    for dfs in diag_files:
        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
            if not f.dimensions:
                f.dimensions = dict_dim
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'model run'
                v.attrs['units'] = ''
                v[:] = onp.arange(dict_dim["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = ''
                v.attrs['units'] = ''
                v[:] = onp.arange(dict_dim["y"])
                v = f.create_variable('Time', ('Time',), float)
                var_obj = df.variables.get('Time')
                v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                               units=var_obj.attrs["units"])
                v[:] = time
            for var_sim in list(df.variables.keys()):
                var_obj = df.variables.get(var_sim)
                if var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] > 1:
                    v = f.create_variable(var_sim, ('x', 'y', 'Time'), float)
                    vals = onp.array(var_obj)
                    v[:, :, :] = vals.swapaxes(0, 2)
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])
                elif var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] <= 2:
                    v = f.create_variable(var_sim, ('x', 'y'), float)
                    vals = onp.array(var_obj)
                    v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])

# move hydrologic states to directories of transport model
base_path_tm = base_path.parent / "svat_transport_sensitivity_reverse"
states_hm_si_file1 = base_path_tm / "states_hm_sensitivity.nc"
shutil.copy(states_hm_si_file, states_hm_si_file1)

# load simulation
states_hm_si_file2 = base_path_data / "states_hm_sensitivity.nc"
shutil.copy(states_hm_si_file, states_hm_si_file2)
ds_sim = xr.open_dataset(states_hm_si_file2, engine="h5netcdf")

# load observations (measured data)
path_obs = base_path.parent / "observations" / "rietholzbach_lysimeter.nc"
ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")

# assign date
days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_sim = num2date(days_sim, units=f"days since {ds_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_sim = ds_sim.assign_coords(date=("Time", date_sim))
ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

# DataFrame with sampled model parameters and the corresponding metrics
nx = ds_sim.dims['x']  # number of rows
ny = ds_sim.dims['y']  # number of columns
df_params_eff = pd.DataFrame(index=range(nx * ny))
# sampled model parameters
df_params_eff.loc[:, 'dmpv'] = ds_sim["dmpv"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'lmpv'] = ds_sim["lmpv"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'theta_ac'] = ds_sim["theta_ac"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'theta_ufc'] = ds_sim["theta_ufc"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'theta_pwp'] = ds_sim["theta_pwp"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'ks'] = ds_sim["ks"].isel(y=0).values.flatten()
# calculate metrics
vars_sim = ['aet', 'q_ss', 'theta', 'dS_s', 'dS']
vars_obs = ['AET', 'PERC', 'THETA', 'dWEIGHT', 'dWEIGHT']
for var_sim, var_obs in zip(vars_sim, vars_obs):
    if var_sim == 'theta':
        obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values, axis=0)
    elif var_sim == 'theta_rz':
        obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values[:5, :], axis=0)
    elif var_sim == 'theta_ss':
        obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values[5:, :], axis=0)
    else:
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    df_obs.loc[:, 'obs'] = obs_vals
    for nrow in range(nx * ny):
        sim_vals = ds_sim[var_sim].isel(x=nrow, y=0).values
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)
        df_eval = df_eval.dropna()

        if var_sim in ['theta_rz', 'theta_ss', 'theta']:
            Ni = len(df_eval.index)
            obs_vals = df_eval.loc[:, 'obs'].values
            sim_vals = df_eval.loc[:, 'sim'].values
            Nz = len(obs_vals)
            eff_swc = eval_utils.calc_kge(obs_vals, sim_vals)
            key_kge = 'KGE_' + var_sim
            df_params_eff.loc[nrow, key_kge] = (Nz / Ni) * eff_swc
        elif var_sim in ['dS', 'dS_s']:
            obs_vals = df_eval.loc[:, 'obs'].values
            sim_vals = df_eval.loc[:, 'sim'].values
            key_r = 'r_' + var_sim
            df_params_eff.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
        else:
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
            cond0 = (df_eval['obs'] == 0)
            if cond0.any():
                # number of data points
                N_obs = len(df_eval.index)
                # simulations and observations for which observed
                # values are exclusively zero
                df_obs0_sim = df_eval.loc[cond0, :]
                N_obs0 = (df_obs0_sim['obs'] == 0).sum()
                N_sim0 = (df_obs0_sim['sim'] == 0).sum()
                # share of observations with zero values
                key_p0 = 'p0_' + var_sim
                df_params_eff.loc[nrow, key_p0] = N_obs0 / N_obs
                # agreement of zero values
                N_obs0 = (df_obs0_sim['obs'] == 0).sum()
                N_sim0 = (df_obs0_sim['sim'] == 0).sum()
                ioa0 = 1 - (N_sim0 / N_obs0)
                key_ioa0 = 'ioa0_' + var_sim
                df_params_eff.loc[nrow, key_ioa0] = ioa0
                # mean absolute error from observations with zero values
                obs0_vals = df_obs0_sim.loc[:, 'obs'].values
                sim0_vals = df_obs0_sim.loc[:, 'sim'].values
                key_mae0 = 'MAE0_' + var_sim
                df_params_eff.loc[nrow, key_mae0] = eval_utils.calc_mae(obs0_vals,
                                                                   sim0_vals)
                # peak difference from observations with zero values
                key_pdiff0 = 'PDIFF0_' + var_sim
                df_params_eff.loc[nrow, key_pdiff0] = onp.max(sim0_vals)
                # simulations and observations with non-zero values
                cond_no0 = (df_eval['obs'] > 0)
                df_obs_sim_no0 = df_eval.loc[cond_no0, :]
                obs_vals_no0 = df_obs_sim_no0.loc[:, 'obs'].values
                sim_vals_no0 = df_obs_sim_no0.loc[:, 'sim'].values
                # number of data with non-zero observations
                N_no0 = len(df_obs_sim_no0.index)
                # mean absolute relative error
                key_mare = 'MARE_' + var_sim
                df_params_eff.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals_no0, sim_vals_no0)
                # mean relative bias
                key_brel_mean = 'brel_mean_' + var_sim
                brel_mean = de.calc_brel_mean(obs_vals_no0, sim_vals_no0)
                df_params_eff.loc[nrow, key_brel_mean] = brel_mean
                # residual relative bias
                brel_res = de.calc_brel_res(obs_vals_no0, sim_vals_no0)
                # area of relative residual bias
                key_b_area = 'b_area_' + var_sim
                b_area = de.calc_bias_area(brel_res)
                df_params_eff.loc[nrow, key_b_area] = b_area
                # temporal correlation
                key_temp_cor = 'temp_cor_' + var_sim
                temp_cor = de.calc_temp_cor(obs_vals_no0, sim_vals_no0)
                df_params_eff.loc[nrow, key_temp_cor] = temp_cor
                # diagnostic efficiency
                key_de = 'DE_' + var_sim
                df_params_eff.loc[nrow, key_de] = de.calc_de(obs_vals_no0, sim_vals_no0)
                # relative bias
                brel = de.calc_brel(obs_vals, sim_vals)
                # total bias
                key_b_tot = 'b_tot_' + var_sim
                b_tot = de.calc_bias_tot(brel)
                df_params_eff.loc[nrow, key_b_tot] = b_tot
                # bias of lower exceedance probability
                key_b_hf = 'b_hf_' + var_sim
                b_hf = de.calc_bias_hf(brel)
                df_params_eff.loc[nrow, key_b_hf] = b_hf
                # error contribution of higher exceedance probability
                key_err_hf = 'err_hf_' + var_sim
                err_hf = de.calc_err_hf(b_hf, b_tot)
                df_params_eff.loc[nrow, key_err_hf] = err_hf
                # bias of higher exceedance probability
                key_b_lf = 'b_lf_' + var_sim
                b_lf = de.calc_bias_lf(brel)
                df_params_eff.loc[nrow, key_b_lf] = b_lf
                # error contribution of lower exceedance probability
                key_err_lf = 'err_lf_' + var_sim
                err_lf = de.calc_err_hf(b_lf, b_tot)
                df_params_eff.loc[nrow, key_err_lf] = err_lf
                # direction of bias
                key_b_dir = 'b_dir_' + var_sim
                b_dir = de.calc_bias_dir(brel_res)
                df_params_eff.loc[nrow, key_b_dir] = b_dir
                # slope of bias
                key_b_slope = 'b_slope_' + var_sim
                b_slope = de.calc_bias_slope(b_area, b_dir)
                df_params_eff.loc[nrow, key_b_slope] = b_slope
                # (y, x) trigonometric inverse tangent
                key_phi = 'phi_' + var_sim
                df_params_eff.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)
                # combined diagnostic efficiency
                key_de0 = 'DE0_' + var_sim
                df_params_eff.loc[nrow, key_de0] = (N_no0 / N_obs) * df_params_eff.loc[nrow, key_de] + (N_obs0 / N_obs) * ioa0
            else:
                # share of observations with zero values
                key_p0 = 'p0_' + var_sim
                df_params_eff.loc[nrow, key_p0] = 0
                # mean absolute relative error
                key_mare = 'MARE_' + var_sim
                df_params_eff.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals, sim_vals)
                # mean relative bias
                key_brel_mean = 'brel_mean_' + var_sim
                brel_mean = de.calc_brel_mean(obs_vals, sim_vals)
                df_params_eff.loc[nrow, key_brel_mean] = brel_mean
                # residual relative bias
                brel_res = de.calc_brel_res(obs_vals, sim_vals)
                # area of relative residual bias
                key_b_area = 'b_area_' + var_sim
                b_area = de.calc_bias_area(brel_res)
                df_params_eff.loc[nrow, key_b_area] = b_area
                # temporal correlation
                key_temp_cor = 'temp_cor_' + var_sim
                temp_cor = de.calc_temp_cor(obs_vals, sim_vals)
                df_params_eff.loc[nrow, key_temp_cor] = temp_cor
                # diagnostic efficiency
                key_de = 'DE_' + var_sim
                df_params_eff.loc[nrow, key_de] = de.calc_de(obs_vals, sim_vals)
                # relative bias
                brel = de.calc_brel(obs_vals, sim_vals)
                # total bias
                key_b_tot = 'b_tot_' + var_sim
                b_tot = de.calc_bias_tot(brel)
                df_params_eff.loc[nrow, key_b_tot] = b_tot
                # bias of lower exceedance probability
                key_b_hf = 'b_hf_' + var_sim
                b_hf = de.calc_bias_hf(brel)
                df_params_eff.loc[nrow, key_b_hf] = b_hf
                # error contribution of higher exceedance probability
                key_err_hf = 'err_hf_' + var_sim
                err_hf = de.calc_err_hf(b_hf, b_tot)
                df_params_eff.loc[nrow, key_err_hf] = err_hf
                # bias of higher exceedance probability
                key_b_lf = 'b_lf_' + var_sim
                b_lf = de.calc_bias_lf(brel)
                df_params_eff.loc[nrow, key_b_lf] = b_lf
                # error contribution of lower exceedance probability
                key_err_lf = 'err_lf_' + var_sim
                err_lf = de.calc_err_hf(b_lf, b_tot)
                df_params_eff.loc[nrow, key_err_lf] = err_lf
                # direction of bias
                key_b_dir = 'b_dir_' + var_sim
                b_dir = de.calc_bias_dir(brel_res)
                df_params_eff.loc[nrow, key_b_dir] = b_dir
                # slope of bias
                key_b_slope = 'b_slope_' + var_sim
                b_slope = de.calc_bias_slope(b_area, b_dir)
                df_params_eff.loc[nrow, key_b_slope] = b_slope
                # (y, x) trigonometric inverse tangent
                key_phi = 'phi_' + var_sim
                df_params_eff.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)

# Calculate multi-objective metric
df_params_eff.loc[:, 'E_multi'] = 1/3 * df_params_eff.loc[:, 'r_dS'] + 1/3 * df_params_eff.loc[:, 'KGE_aet'] + 1/3 * df_params_eff.loc[:, 'KGE_q_ss']

# write .txt-file
file = base_path_results / "params_eff.txt"
df_params_eff.to_csv(file, header=True, index=False, sep="\t")

# perform sensitivity analysis
df_params = df_params_eff.loc[:, bounds['names']]
df_eff = df_params_eff.loc[:, ['KGE_aet', 'KGE_q_ss', 'r_dS', 'E_multi']]
dict_si = {}
for name in df_eff.columns:
    Y = df_eff[name].values
    Si = sobol.analyze(bounds, Y, calc_second_order=False)
    Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
    dict_si[name] = pd.DataFrame(Si_filter, index=bounds['names'])

# plot sobol indices
_LABS = {'KGE_aet': 'evapotranspiration',
         'KGE_q_ss': 'percolation',
         'r_dS': 'storage change',
         'E_multi': 'multi-objective metric',
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
file = base_path_figs / "sobol_indices.png"
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
    xlabel = labs._LABS[bounds['names'][j]]
    ax[-1, j].set_xlabel(xlabel)

ax[0, 0].set_ylabel('$KGE_{ET}$ [-]')
ax[1, 0].set_ylabel('$KGE_{PERC}$ [-]')
ax[2, 0].set_ylabel(r'$r_{\Delta S}$ [-]')
ax[3, 0].set_ylabel('$E_{multi}$\n [-]')

fig.subplots_adjust(wspace=0.2, hspace=0.3)
file = base_path_figs / "dotty_plots.png"
fig.savefig(file, dpi=250)

file1 = base_path / "param_bounds.yml"
file2 = base_path.parent / "svat_transport_sensitivity_reverse" / "param_bounds.yml"
shutil.copy(file1, file2)
