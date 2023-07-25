import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import datetime
import roger.tools.labels as labs
import matplotlib as mpl
import matplotlib.dates as mdates
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.api import abline_plot
import statsmodels.tools.eval_measures as smem
from scipy import stats
import pickle

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


def nanmeanweighted(y, w, axis=None):
    w1 = w / onp.nansum(w, axis=axis)
    w2 = onp.where(onp.isnan(w), 0, w1)
    w3 = onp.where(onp.isnan(y), 0, w2)
    y1 = onp.where(onp.isnan(y), 0, y)
    wavg = onp.sum(y1 * w3, axis=axis) / onp.sum(w3, axis=axis)

    return wavg


base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "output"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# load delta changes
delta_file = base_path_figs / "delta_changes.pkl"
with open(delta_file, 'rb') as handle:
    dict_deltas = pickle.load(handle)

glm_file = base_path_figs / "glm_results.pkl"
mlm_file = base_path_figs / "mlm_results.pkl"

# identifiers for simulations
locations = ["freiburg", "altheim", "kupferzell"]
Locations = ["Freiburg", "Altheim", "Kupferzell"]
land_cover_scenarios = ["grass", "corn", "corn_catch_crop", "crop_rotation"]
Land_cover_scenarios = ["Grass", "Corn", "Corn & Catch crop", "Crop rotation"]
climate_scenarios = ["CCCma-CanESM2_CCLM4-8-17", "MPI-M-MPI-ESM-LR_RCA4"]
periods = ["1985-2005", "2040-2060", "2080-2100"]
start_dates = [datetime.date(1985, 1, 1), datetime.date(2040, 1, 1), datetime.date(2080, 1, 1)]
end_dates = [datetime.date(2004, 12, 31), datetime.date(2059, 12, 31), datetime.date(2099, 12, 31)]

_lab = {
    "q_ss": "PERC",
    "transp": "TRANSP",
    "evap_soil": "$EVAP_{soil}$",
    "theta": r"$\theta$",
    "tt10_transp": "$TT_{10-TRANSP}$",
    "tt50_transp": "$TT_{50-TRANSP}$",
    "tt90_transp": "$TT_{90-TRANSP}$",
    "tt10_q_ss": "$TT_{10-PERC}$",
    "tt50_q_ss": "$TT_{50-PERC}$",
    "tt90_q_ss": "$TT_{90-PERC}$",
    "rt10_s": "$RT_{10}$",
    "rt50_s": "$RT_{50}$",
    "rt90_s": "$RT_{90}$",
    "M_transp": "$M_{TRANSP}$",
    "M_q_ss": "$M_{PERC}$",
    "dAvg": r"$\overline{\Delta}$",
    "dIPR": r"$\Delta IPR$",
    "dSum": r"$\Delta\sum$",
    "CCCma-CanESM2_CCLM4-8-17": "CCC",
    "MPI-M-MPI-ESM-LR_RCA4": "MPI",
    "nf": "NF",
    "ff": "FF",
    "RMSE": "Root mean squared error",
    "MAE": "Mean absolute error",
    "MEAE": "Median absolute error",
    "AIC": "AIC",
    "AICC": "AICc",
    "theta_pwp:theta_ufc": r"$\theta_{pwp}$ x $\theta_{ufc}$",
    "theta_pwp:theta_ac": r"$\theta_{pwp}$ x $\theta_{ac}$",
    "theta_ufc:theta_ac": r"$\theta_{ufc}$ x $\theta_{ac}$",
    "theta_pwp:theta_ufc:theta_ac": r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$",
}

_lab_unit1 = {
    "q_ss": "PERC [mm/day]",
    "transp": "TRANSP [mm/day]",
    "evap_soil": "$EVAP_{soil}$ [mm/day]",
    "theta": r"$\theta$ [-]",
    "tt10_transp": "$TT_{10-TRANSP}$ [days]",
    "tt50_transp": "$TT_{50-TRANSP}$ [days]",
    "tt90_transp": "$TT_{90-TRANSP}$ [days]",
    "tt10_q_ss": "$TT_{10-PERC}$ [days]",
    "tt50_q_ss": "$TT_{50-PERC}$ [days]",
    "tt90_q_ss": "$TT_{90-PERC}$ [days]",
    "rt10_s": "$RT_{10}$ [days]",
    "rt50_s": "$RT_{50}$ [days]",
    "rt90_s": "$RT_{90}$ [days]",
    "M_transp": "$M_{TRANSP}$ [mg]",
    "M_q_ss": "$M_{PERC}$ [mg]",
    "theta_ac": r"$\theta_{ac}$ [-]",
    "theta_ufc": r"$\theta_{ufc}$ [-]",
    "theta_pwp": r"$\theta_{pwp}$ [-]",
    "ks": "$k_s$ [mm/day]",
}

_lab_unit2 = {
    "q_ss": "PERC [mm]",
    "transp": "TRANSP [mm]",
    "evap_soil": "$EVAP_{soil}$ [mm]",
    "theta": r"$\theta$ [-]",
    "tt10_transp": "$TT_{10-TRANSP}$ [days]",
    "tt50_transp": "$TT_{50-TRANSP}$ [days]",
    "tt90_transp": "$TT_{90-TRANSP}$ [days]",
    "tt10_q_ss": "$TT_{10-PERC}$ [days]",
    "tt50_q_ss": "$TT_{50-PERC}$ [days]",
    "tt90_q_ss": "$TT_{90-PERC}$ [days]",
    "rt10_s": "$RT_{10}$ [days]",
    "rt50_s": "$RT_{50}$ [days]",
    "rt90_s": "$RT_{90}$ [days]",
    "M_transp": "$M_{TRANSP}$ [mg]",
    "M_q_ss": "$M_{PERC}$ [mg]",
}

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
cond_soil_depth_300 = df_params.loc[:, "z_soil"].values == 300
cond_soil_depth_600 = df_params.loc[:, "z_soil"].values == 600
cond_soil_depth_900 = df_params.loc[:, "z_soil"].values == 900
cond_soil_depth = onp.copy(cond_soil_depth_300)
cond_soil_depth[:] = True
soil_depths = ["all", "shallow", "medium", "deep"]
_soil_depths = {
    "all": cond_soil_depth,
    "shallow": cond_soil_depth_300,
    "medium": cond_soil_depth_600,
    "deep": cond_soil_depth_900,
}

# # identify impacted sites based on GLM
# for param in ['theta_pwp', 'theta_ufc', 'theta_ac', 'ks']:
#     vals = df_params.loc[:, param].values
#     fig, ax = plt.subplots(figsize=(3, 3))
#     ax.hist(vals, bins=25, color='black')
#     ax.set_xlabel(f'{_lab_unit1[param]}')
#     fig.tight_layout()
#     file = base_path_figs / "glm_parameters" / f"{param}_hist.png"
#     fig.savefig(file, dpi=300)

# glm_formulas = {"interaction_theta": "y ~ theta_pwp * theta_ufc * theta_ac + ks",
#                 "no_interaction": "y ~ theta_pwp + theta_ufc + theta_ac + ks",
#                 "theta": "y ~ theta_pwp + theta_ufc + theta_ac",
#                 "theta_pwp": "y ~ theta_pwp",
#                 "theta_ufc": "y ~ theta_ufc",
#                 "theta_ac": "y ~ theta_ac",
#                 "ks": "y ~ ks"}

# vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
# deltas = ["dAvg", "dIPR"]
# if not os.path.exists(glm_file):
#     dict_glm = {}
#     for glm_key in glm_formulas.keys():
#         dict_glm[glm_key] = {}
#         for location in locations:
#             dict_glm[glm_key][location] = {}
#             for land_cover_scenario in land_cover_scenarios:
#                 dict_glm[glm_key][location][land_cover_scenario] = {}
#                 for climate_scenario in climate_scenarios:
#                     dict_glm[glm_key][location][land_cover_scenario][climate_scenario] = {}
#                     for var_sim in vars_sim:
#                         dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim] = {}
#                         for delta in deltas:
#                             dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta] = {}
#                             for soil_depth in soil_depths:
#                                 dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth] = {}
#                                 cond = _soil_depths[soil_depth]
#                                 for future in ["nf", "ff"]:
#                                     dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
#                                         future
#                                     ] = {}
#                                     y = (
#                                         dict_deltas[location][land_cover_scenario][climate_scenario][var_sim]
#                                         .loc[cond, f"{delta}_{future}"]
#                                         .to_frame()
#                                     )
#                                     y = pd.DataFrame(data=y, dtype=onp.float64)
#                                     x = df_params.loc[cond, "theta_pwp":]
#                                     x = sm.add_constant(x)
#                                     # standardize the parameters
#                                     x_std = x.copy()
#                                     for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
#                                         x_std.loc[:, param] = (x_std.loc[:, param] - x_std.loc[:, param].mean()) / x_std.loc[
#                                             :, param
#                                         ].std()

#                                     dta = x_std.copy()
#                                     dta.loc[:, "y"] = y.values.flatten()

#                                     # fit the GLM model
#                                     glm = smf.glm(formula=glm_formulas[glm_key], data=dta, family=sm.families.Gaussian())
#                                     res = glm.fit()
#                                     nobs = res.nobs
#                                     yhat = res.mu

#                                     if glm_key == "interaction_theta":
#                                         ll = ['theta_pwp', 'theta_ufc', 'theta_ac', 'theta_pwp:theta_ufc', 'theta_pwp:theta_ac', 'theta_ufc:theta_ac', 'theta_pwp:theta_ufc:theta_ac', 'ks']
#                                         params = res.params.to_frame().T.loc[:, ll]
#                                     else:
#                                         params = res.params.to_frame().T

#                                     dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
#                                         future
#                                     ]["params"] = params / onp.sum(onp.abs(params.values.flatten()))
#                                     y = y.values.flatten()
#                                     dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
#                                         future
#                                     ]["MAE"] = smem.meanabs(y, yhat)
#                                     dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
#                                         future
#                                     ]["MEAE"] = smem.medianabs(y, yhat)
#                                     dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
#                                         future
#                                     ]["RMSE"] = smem.rmse(y, yhat)
#                                     ll_values = glm.loglike(res.params, scale=res.scale)
#                                     dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
#                                         future
#                                     ]["AIC"] = smem.aic(ll_values, nobs, res.params.shape[0])
#                                     dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
#                                         future
#                                     ]["AICC"] = smem.aicc(ll_values, nobs, res.params.shape[0])
#                                     dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
#                                         future
#                                     ]["VARE"] = smem.vare(y, yhat)

#                                     fig, ax = plt.subplots(figsize=(3, 3))
#                                     ax.scatter(yhat, y, color='black', s=4)
#                                     line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
#                                     abline_plot(model_results=line_fit, ax=ax, color='black')
#                                     ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} (RoGeR) [%]')
#                                     ax.set_xlabel(f'{_lab[delta]}{_lab_unit1[var_sim]} (GLM) [%]')
#                                     fig.tight_layout()
#                                     file = base_path_figs / "residuals" / f"{glm_key}_{location}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_line_fit.png"
#                                     fig.savefig(file, dpi=300)

#                                     fig, ax = plt.subplots(figsize=(3, 3))
#                                     ax.scatter(yhat, res.resid_pearson, color='black', s=4)
#                                     ax.set_ylabel(f'{_lab_unit1[var_sim]} (RoGeR) [%]')
#                                     ax.set_xlabel(f'{_lab[delta]}{_lab_unit1[var_sim]} (GLM)')
#                                     fig.tight_layout()
#                                     file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_residuals.png"
#                                     fig.savefig(file, dpi=300)

#                                     fig, ax = plt.subplots(figsize=(3, 3))
#                                     ax.scatter(x.loc[:, 'theta_ufc'].values, y, color='black', s=4)
#                                     ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
#                                     ax.set_xlabel(f'{_lab_unit1["theta_ufc"]}')
#                                     fig.tight_layout()
#                                     file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_theta_ufc.png"
#                                     fig.savefig(file, dpi=300)

#                                     fig, ax = plt.subplots(figsize=(3, 3))
#                                     ax.scatter(x.loc[:, 'theta_ac'].values, y, color='black', s=4)
#                                     ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
#                                     ax.set_xlabel(f'{_lab_unit1["theta_ac"]}')
#                                     fig.tight_layout()
#                                     file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_theta_ac.png"
#                                     fig.savefig(file, dpi=300)

#                                     fig, ax = plt.subplots(figsize=(3, 3))
#                                     ax.scatter(x.loc[:, 'theta_pwp'].values, y, color='black', s=4)
#                                     ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
#                                     ax.set_xlabel(f'{_lab_unit1["theta_pwp"]}')
#                                     fig.tight_layout()
#                                     file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_theta_pwp.png"
#                                     fig.savefig(file, dpi=300)

#                                     fig, ax = plt.subplots(figsize=(3, 3))
#                                     ax.scatter(x.loc[:, 'ks'].values, y, color='black', s=4)
#                                     ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
#                                     ax.set_xlabel(f'{_lab_unit1["ks"]}')
#                                     fig.tight_layout()
#                                     file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_ks.png"
#                                     fig.savefig(file, dpi=300)

#                                     fig, ax = plt.subplots(figsize=(3, 3))
#                                     resid = res.resid_deviance.copy()
#                                     resid_std = stats.zscore(resid)
#                                     ax.hist(resid_std, bins=25, color='black')
#                                     ax.set_xlabel(f'{_lab[delta]}{_lab_unit1[var_sim]} (GLM)')
#                                     fig.tight_layout()
#                                     file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_residuals_hist.png"
#                                     fig.savefig(file, dpi=300)
#                                     plt.close('all')

#     # Store data (serialize)
#     with open(glm_file, 'wb') as handle:
#         pickle.dump(dict_glm, handle, protocol=pickle.HIGHEST_PROTOCOL)
# else:
#     # Load data (deserialize)
#     with open(glm_file, 'rb') as handle:
#         dict_glm = pickle.load(handle)

# glm_key = "no_interaction"
# vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
# deltas = ["dAvg"]
# for var_sim in vars_sim:
#     for delta in deltas:
#         for soil_depth in soil_depths:
#             cond = _soil_depths[soil_depth]
#             fig, axes = plt.subplots(
#                 len(locations), len(land_cover_scenarios), sharex=True, sharey=True, figsize=(6, 4.5)
#             )
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     values_nf = dict_glm[glm_key][location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][delta][
#                         soil_depth
#                     ]["nf"]["params"].values.flatten()[1:]
#                     values_ff = dict_glm[glm_key][location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][delta][
#                         soil_depth
#                     ]["ff"]["params"].values.flatten()[1:]
#                     df_params_canesm = pd.DataFrame(index=range(8), columns=["value", "Parameter"])
#                     values = []
#                     for ii in range(4):
#                         values.append(values_nf[ii])
#                         values.append(values_ff[ii])
#                     df_params_canesm.loc[:, "value"] = values
#                     df_params_canesm.loc[:, "Parameter"] = [
#                         r"$\theta_{pwp}$ (NF)",
#                         r"$\theta_{pwp}$ (FF)",
#                         r"$\theta_{ufc}$ (NF)",
#                         r"$\theta_{ufc}$ (FF)",
#                         r"$\theta_{ac}$ (NF)",
#                         r"$\theta_{ac}$ (FF)",
#                         r"$k_s$ (NF)",
#                         r"$k_s$ (FF)",
#                     ]
#                     df_params_canesm.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"

#                     values_nf = dict_glm[glm_key][location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][delta][
#                         soil_depth
#                     ]["nf"]["params"].values.flatten()[1:]
#                     values_ff = dict_glm[glm_key][location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][delta][
#                         soil_depth
#                     ]["ff"]["params"].values.flatten()[1:]
#                     df_params_mpiesm = pd.DataFrame(index=range(8), columns=["value", "Parameter"])
#                     values = []
#                     for ii in range(4):
#                         values.append(values_nf[ii])
#                         values.append(values_ff[ii])
#                     df_params_mpiesm.loc[:, "value"] = values
#                     df_params_mpiesm.loc[:, "Parameter"] = [
#                         r"$\theta_{pwp}$ (NF)",
#                         r"$\theta_{pwp}$ (FF)",
#                         r"$\theta_{ufc}$ (NF)",
#                         r"$\theta_{ufc}$ (FF)",
#                         r"$\theta_{ac}$ (NF)",
#                         r"$\theta_{ac}$ (FF)",
#                         r"$k_s$ (NF)",
#                         r"$k_s$ (FF)",
#                     ]
#                     df_params_mpiesm.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"

#                     df_params = pd.concat([df_params_canesm, df_params_mpiesm], ignore_index=True)
#                     sns.barplot(
#                         x="Parameter",
#                         y="value",
#                         hue="Climate model",
#                         palette=["red", "blue"],
#                         data=df_params,
#                         ax=axes[i, j],
#                         errorbar=None
#                     )
#                     axes[i, j].legend([], [], frameon=False)
#                     axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
#                     axes[i, j].set_ylabel("")
#                     axes[i, j].set_xlabel("")
#                     axes[i, j].tick_params(axis="x", rotation=90)
#                 axes[i, 0].set_ylabel(f"{Locations[i]}\n{_lab[delta]} {_lab[var_sim]}")
#             fig.tight_layout()
#             file = base_path_figs / "glm_parameters" / f"{glm_key}_{var_sim}_{delta}_{soil_depth}_barplot.png"
#             fig.savefig(file, dpi=300)
#             plt.close("all")

# glm_key = "interaction_theta"
# vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
# deltas = ["dAvg"]
# for var_sim in vars_sim:
#     for delta in deltas:
#         for soil_depth in soil_depths:
#             cond = _soil_depths[soil_depth]
#             fig, axes = plt.subplots(
#                 len(locations), len(land_cover_scenarios), sharex=True, sharey=True, figsize=(6, 4.5)
#             )
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     values_nf = dict_glm[glm_key][location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][delta][
#                         soil_depth
#                     ]["nf"]["params"].values.flatten()[1:]
#                     values_ff = dict_glm[glm_key][location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][delta][
#                         soil_depth
#                     ]["ff"]["params"].values.flatten()[1:]
#                     df_params_canesm = pd.DataFrame(index=range(8), columns=["value", "Parameter"])
#                     values = []
#                     for ii in range(8):
#                         values.append(values_nf[ii])
#                         values.append(values_ff[ii])
#                     df_params_canesm.loc[:, "value"] = values
#                     df_params_canesm.loc[:, "Parameter"] = [
#                         r'$\theta_{pwp}$ (NF)',
#                         r'$\theta_{pwp}$ (FF)',
#                         r'$\theta_{ufc}$ (NF)',
#                         r'$\theta_{ufc}$ (FF)',
#                         r'$\theta_{ac}$ (NF)',
#                         r'$\theta_{ac}$ (FF)',
#                         r'$\theta_{pwp}$ x $\theta_{ufc}$ (NF)',
#                         r'$\theta_{pwp}$ x $\theta_{ufc}$ (FF)',
#                         r'$\theta_{pwp}$ x $\theta_{ac}$ (NF)',
#                         r'$\theta_{pwp}$ x $\theta_{ac}$ (FF)',
#                         r'$\theta_{ufc}$ x $\theta_{ac}$ (NF)',
#                         r'$\theta_{ufc}$ x $\theta_{ac}$ (FF)',
#                         r'$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (NF)',
#                         r'$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (FF)',
#                         r'$k_s$ (NF)',
#                         r'$k_s$ (FF)',
#                     ]
#                     df_params_canesm.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"

#                     values_nf = dict_glm[glm_key][location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][delta][
#                         soil_depth
#                     ]["nf"]["params"].values.flatten()[1:]
#                     values_ff = dict_glm[glm_key][location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][delta][
#                         soil_depth
#                     ]["ff"]["params"].values.flatten()[1:]
#                     df_params_mpiesm = pd.DataFrame(index=range(4), columns=["value", "Parameter"])
#                     values = []
#                     for ii in range(2):
#                         values.append(values_nf[ii])
#                         values.append(values_ff[ii])
#                     df_params_mpiesm.loc[:, "value"] = values
#                     df_params_mpiesm.loc[:, "Parameter"] = [
#                         r'$\theta_{pwp}$ (NF)',
#                         r'$\theta_{pwp}$ (FF)',
#                         r'$\theta_{ufc}$ (NF)',
#                         r'$\theta_{ufc}$ (FF)',
#                         r'$\theta_{ac}$ (NF)',
#                         r'$\theta_{ac}$ (FF)',
#                         r'$\theta_{pwp}$ x $\theta_{ufc}$ (NF)',
#                         r'$\theta_{pwp}$ x $\theta_{ufc}$ (FF)',
#                         r'$\theta_{pwp}$ x $\theta_{ac}$ (NF)',
#                         r'$\theta_{pwp}$ x $\theta_{ac}$ (FF)',
#                         r'$\theta_{ufc}$ x $\theta_{ac}$ (NF)',
#                         r'$\theta_{ufc}$ x $\theta_{ac}$ (FF)',
#                         r'$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (NF)',
#                         r'$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (FF)',
#                         r'$k_s$ (NF)',
#                         r'$k_s$ (FF)',
#                     ]

#                     df_params_mpiesm.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"

#                     df_params = pd.concat([df_params_canesm, df_params_mpiesm], ignore_index=True)
#                     sns.barplot(
#                         x="Parameter",
#                         y="value",
#                         hue="Climate model",
#                         palette=["red", "blue"],
#                         data=df_params,
#                         ax=axes[i, j],
#                         errorbar=None
#                     )
#                     axes[i, j].legend([], [], frameon=False)
#                     axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
#                     axes[i, j].set_ylabel("")
#                     axes[i, j].set_xlabel("")
#                     axes[i, j].tick_params(axis="x", rotation=90)
#                 axes[i, 0].set_ylabel(f"{Locations[i]}\n{_lab[delta]} {_lab[var_sim]}")
#             fig.tight_layout()
#             file = base_path_figs / "glm_parameters" / f"{glm_key}_{var_sim}_{delta}_{soil_depth}_barplot.png"
#             fig.savefig(file, dpi=300)
#             plt.close("all")

# glm_key = "no_interaction"
# vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
# deltas = ["dAvg"]
# norm = mpl.colors.Normalize(vmin=0, vmax=1)
# for delta in deltas:
#     for soil_depth in soil_depths:
#         cond = _soil_depths[soil_depth]
#         for future in ['nf', 'ff']:
#             fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex='col', sharey='row', figsize=(6, 4.5))
#             # axes for colorbar
#             axl = fig.add_axes([0.88, 0.3, 0.02, 0.5])
#             cb1 = mpl.colorbar.ColorbarBase(axl, cmap='Oranges', norm=norm,
#                                             orientation='vertical',
#                                             ticks=[0, 1, 2])
#             cb1.set_label(r'[-]')
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     params_cm = [r'$\theta_{pwp}$ (CCC)', r'$\theta_{pwp}$ (MPI)', r'$\theta_{ufc}$ (CCC)', r'$\theta_{ufc}$ (MPI)', r'$\theta_{ac}$ (CCC)', r'$\theta_{ac}$ (MPI)', r'$k_s$ (CCC)', r'$k_s$ (MPI)']
#                     params_canesm = [r'$\theta_{pwp}$ (CCC)', r'$\theta_{ufc}$ (CCC)', r'$\theta_{ac}$ (CCC)', r'$k_s$ (CCC)']
#                     params_mpim = [r'$\theta_{pwp}$ (MPI)', r'$\theta_{ufc}$ (MPI)', r'$\theta_{ac}$ (MPI)', r'$k_s$ (MPI)']
#                     df_params = pd.DataFrame(columns=params_cm)
#                     for ii, var_sim in enumerate(vars_sim):
#                         df_params.loc[f'{_lab[delta]}{_lab[var_sim]}', params_canesm] = dict_glm[glm_key][location][land_cover_scenario]['CCCma-CanESM2_CCLM4-8-17'][var_sim][delta][soil_depth][future]['params'].values.flatten()[1:]
#                         df_params.loc[f'{_lab[delta]}{_lab[var_sim]}', params_mpim] = dict_glm[glm_key][location][land_cover_scenario]['MPI-M-MPI-ESM-LR_RCA4'][var_sim][delta][soil_depth][future]['params'].values.flatten()[1:]
#                     df_params = df_params.astype('float32')
#                     df_params.iloc[:, :] = onp.abs(df_params.values)
#                     sns.heatmap(df_params, vmin=0, vmax=1, cmap='Oranges', cbar=False, ax=axes[i, j], square=True)
#                     axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                     axes[-1, j].set_xticklabels(params_cm, rotation=90)
#                     axes[i, j].set_ylabel('')
#                     axes[i, j].set_xlabel('')
#                 axes[i, 0].set_ylabel(f'{Locations[i]}')
#                 axes[i, 0].set_yticklabels(df_params.index.tolist(), rotation=0)
#             fig.subplots_adjust(bottom=0.2, left=0.2, right=0.85, top=0.9, hspace=-0.5, wspace=0.1)
#             file = base_path_figs / "glm_parameters" / f"{glm_key}_{delta}_{soil_depth}_{future}_heatmap.png"
#             fig.savefig(file, dpi=300)
#             plt.close('all')

# glm_key = "interaction_theta"
# vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
# deltas = ["dAvg"]
# norm = mpl.colors.Normalize(vmin=0, vmax=1)
# for delta in deltas:
#     for soil_depth in soil_depths:
#         cond = _soil_depths[soil_depth]
#         for future in ['nf', 'ff']:
#             fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex='col', sharey='row', figsize=(6, 4.5))
#             # axes for colorbar
#             axl = fig.add_axes([0.88, 0.3, 0.02, 0.5])
#             cb1 = mpl.colorbar.ColorbarBase(axl, cmap='Oranges', norm=norm,
#                                             orientation='vertical',
#                                             ticks=[0, 1, 2])
#             cb1.set_label(r'[-]')
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     params_cm = [r'$\theta_{pwp}$ (CCC)',
#                                  r'$\theta_{pwp}$ (MPI)',
#                                  r'$\theta_{ufc}$ (CCC)',
#                                  r'$\theta_{ufc}$ (MPI)',
#                                  r'$\theta_{ac}$ (CCC)',
#                                  r'$\theta_{ac}$ (MPI)',
#                                  r'$\theta_{pwp}$ x $\theta_{ufc}$ (CCC)',
#                                  r'$\theta_{pwp}$ x $\theta_{ufc}$ (MPI)',
#                                  r'$\theta_{pwp}$ x $\theta_{ac}$ (CCC)',
#                                  r'$\theta_{pwp}$ x $\theta_{ac}$ (MPI)',
#                                  r'$\theta_{ufc}$ x $\theta_{ac}$ (CCC)',
#                                  r'$\theta_{ufc}$ x $\theta_{ac}$ (MPI)',
#                                  r'$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (CCC)',
#                                  r'$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (MPI)',
#                                  r'$k_s$ (CCC)',
#                                  r'$k_s$ (MPI)']
#                     params_canesm = [r'$\theta_{pwp}$ (CCC)',
#                                      r'$\theta_{ufc}$ (CCC)',
#                                      r'$\theta_{ac}$ (CCC)',
#                                      r'$\theta_{pwp}$ x $\theta_{ufc}$ (CCC)',
#                                      r'$\theta_{pwp}$ x $\theta_{ac}$ (CCC)',
#                                      r'$\theta_{ufc}$ x $\theta_{ac}$ (CCC)',
#                                      r'$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (CCC)',
#                                      r'$k_s$ (CCC)']
#                     params_mpim = [r'$\theta_{pwp}$ (MPI)',
#                                    r'$\theta_{ufc}$ (MPI)',
#                                    r'$\theta_{ac}$ (MPI)',
#                                    r'$\theta_{pwp}$ x $\theta_{ufc}$ (MPI)',
#                                    r'$\theta_{pwp}$ x $\theta_{ac}$ (MPI)',
#                                    r'$\theta_{ufc}$ x $\theta_{ac}$ (MPI)',
#                                    r'$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (MPI)',
#                                    r'$k_s$ (MPI)']
#                     df_params = pd.DataFrame(columns=params_cm)
#                     for ii, var_sim in enumerate(vars_sim):
#                         print(dict_glm[glm_key][location][land_cover_scenario]['CCCma-CanESM2_CCLM4-8-17'][var_sim][delta][soil_depth][future]['params'].values[1:])
#                         df_params.loc[f'{_lab[delta]}{_lab[var_sim]}', params_canesm] = dict_glm[glm_key][location][land_cover_scenario]['CCCma-CanESM2_CCLM4-8-17'][var_sim][delta][soil_depth][future]['params'].values.flatten()[1:]
#                         df_params.loc[f'{_lab[delta]}{_lab[var_sim]}', params_mpim] = dict_glm[glm_key][location][land_cover_scenario]['MPI-M-MPI-ESM-LR_RCA4'][var_sim][delta][soil_depth][future]['params'].values.flatten()[1:]
#                     df_params = df_params.astype('float32')
#                     df_params.iloc[:, :] = onp.abs(df_params.values)
#                     sns.heatmap(df_params, vmin=0, vmax=1, cmap='Oranges', cbar=False, ax=axes[i, j], square=True)
#                     axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                     axes[-1, j].set_xticklabels(params_cm, rotation=90)
#                     axes[i, j].set_ylabel('')
#                     axes[i, j].set_xlabel('')
#                 axes[i, 0].set_ylabel(f'{Locations[i]}')
#                 axes[i, 0].set_yticklabels(df_params.index.tolist(), rotation=0)
#             fig.subplots_adjust(bottom=0.2, left=0.2, right=0.85, top=0.9, hspace=-0.5, wspace=0.1)
#             file = base_path_figs / "glm_parameters" / f"{glm_key}_{delta}_{soil_depth}_{future}_heatmap.png"
#             fig.savefig(file, dpi=300)
#             plt.close('all')

# metrics = ["MAE"]
# vars_sim = ["transp", "q_ss", "theta", "tt50_transp", "tt50_q_ss", "rt50_s"]
# deltas = ["dAvg"]
# norm = mpl.colors.Normalize(vmin=0, vmax=20)
# for glm_key in glm_formulas.keys():
#     for metric in metrics:
#         for delta in deltas:
#             for soil_depth in soil_depths:
#                 cond = _soil_depths[soil_depth]
#                 fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex='col', sharey='row', figsize=(6, 4.5))
#                 # axes for colorbar
#                 axl = fig.add_axes([0.88, 0.32, 0.02, 0.46])
#                 cb1 = mpl.colorbar.ColorbarBase(axl, cmap='Oranges', norm=norm,
#                                                 orientation='vertical')
#                 cb1.set_label(f'{_lab[metric]} [%]')
#                 for i, location in enumerate(locations):
#                     for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                         df_metric = pd.DataFrame()
#                         for var_sim in vars_sim:
#                             for climate_scenario in climate_scenarios:
#                                 for future in ['nf', 'ff']:
#                                     value = dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][future][metric]
#                                     df_metric.loc[f'{_lab[climate_scenario]} ({_lab[future]})', f'{_lab[var_sim]}'] = value
#                         sns.heatmap(df_metric, vmin=0, vmax=20, cmap='Oranges', cbar=False, ax=axes[i, j], square=True)
#                         axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                         axes[-1, j].set_xticklabels(df_metric.columns, rotation=90)
#                         axes[i, j].set_ylabel('')
#                         axes[i, j].set_xlabel('')
#                     axes[i, 0].set_ylabel(f'{Locations[i]}')
#                     axes[i, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
#                 fig.subplots_adjust(bottom=0.2, left=0.2, right=0.85, top=0.9, hspace=-0.5, wspace=0.1)
#                 file = base_path_figs / "glm_metrics" / f"{glm_key}_{metric}_{delta}_{soil_depth}_heatmap.png"
#                 fig.savefig(file, dpi=300)
#                 plt.close('all')


# metrics = ["RMSE"]
# vars_sim = ["transp", "q_ss", "theta", "tt50_transp", "tt50_q_ss", "rt50_s"]
# deltas = ["dAvg"]
# norm = mpl.colors.Normalize(vmin=0, vmax=30)
# for glm_key in glm_formulas.keys():
#     for metric in metrics:
#         for delta in deltas:
#             for soil_depth in soil_depths:
#                 cond = _soil_depths[soil_depth]
#                 fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex='col', sharey='row', figsize=(6, 4.5))
#                 # axes for colorbar
#                 axl = fig.add_axes([0.88, 0.32, 0.02, 0.46])
#                 cb1 = mpl.colorbar.ColorbarBase(axl, cmap='Oranges', norm=norm,
#                                                 orientation='vertical')
#                 cb1.set_label(f'{metric} [-]')
#                 for i, location in enumerate(locations):
#                     for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                         df_metric = pd.DataFrame()
#                         for var_sim in vars_sim:
#                             for climate_scenario in climate_scenarios:
#                                 for future in ['nf', 'ff']:
#                                     value = dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][future][metric]
#                                     df_metric.loc[f'{_lab[climate_scenario]} ({_lab[future]})', f'{_lab[var_sim]}'] = value
#                         sns.heatmap(df_metric, vmin=0, vmax=30, cmap='Oranges', cbar=False, ax=axes[i, j], square=True)
#                         axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                         axes[-1, j].set_xticklabels(df_metric.columns, rotation=90)
#                         axes[i, j].set_ylabel('')
#                         axes[i, j].set_xlabel('')
#                     axes[i, 0].set_ylabel(f'{Locations[i]}')
#                     axes[i, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
#                 fig.subplots_adjust(bottom=0.2, left=0.2, right=0.85, top=0.9, hspace=-0.5, wspace=0.1)
#                 file = base_path_figs / "glm_metrics" / f"{glm_key}_{metric}_{delta}_{soil_depth}_heatmap.png"
#                 fig.savefig(file, dpi=300)
#                 plt.close('all')

# metrics = ["AIC", "AICC"]
# vars_sim = ["transp", "q_ss", "theta", "tt50_transp", "tt50_q_ss", "rt50_s"]
# deltas = ["dAvg"]
# norm = mpl.colors.Normalize(vmin=0, vmax=1000)
# for glm_key in glm_formulas.keys():
#     for metric in metrics:
#         for delta in deltas:
#             for soil_depth in soil_depths:
#                 cond = _soil_depths[soil_depth]
#                 fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex='col', sharey='row', figsize=(6, 4.5))
#                 # axes for colorbar
#                 axl = fig.add_axes([0.88, 0.32, 0.02, 0.46])
#                 cb1 = mpl.colorbar.ColorbarBase(axl, cmap='Oranges', norm=norm,
#                                                 orientation='vertical')
#                 cb1.set_label(f'{_lab[metric]} [-]')
#                 for i, location in enumerate(locations):
#                     for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                         df_metric = pd.DataFrame()
#                         for var_sim in vars_sim:
#                             for climate_scenario in climate_scenarios:
#                                 for future in ['nf', 'ff']:
#                                     value = dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][future][metric]
#                                     df_metric.loc[f'{_lab[climate_scenario]} ({_lab[future]})', f'{_lab[var_sim]}'] = value
#                         sns.heatmap(df_metric, vmin=0, vmax=1000, cmap='Oranges', cbar=False, ax=axes[i, j], square=True)
#                         axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                         axes[-1, j].set_xticklabels(df_metric.columns, rotation=90)
#                         axes[i, j].set_ylabel('')
#                         axes[i, j].set_xlabel('')
#                     axes[i, 0].set_ylabel(f'{Locations[i]}')
#                     axes[i, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
#                 fig.subplots_adjust(bottom=0.2, left=0.2, right=0.85, top=0.9, hspace=-0.5, wspace=0.1)
#                 file = base_path_figs / "glm_metrics" / f"{glm_key}_{metric}_{delta}_{soil_depth}_heatmap.png"
#                 fig.savefig(file, dpi=300)
#                 plt.close('all')

mlm_formulas = {"interaction_theta": "~ theta_pwp * theta_ufc * theta_ac + ks",
                "no_interaction": "~ theta_pwp + theta_ufc + theta_ac + ks",
                "theta": "~ theta_pwp + theta_ufc + theta_ac",
                "theta_pwp": "~ theta_pwp",
                "theta_ufc": "~ theta_ufc",
                "theta_ac": "~ theta_ac",
                "ks": "~ ks"}

vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
deltas = ["dAvg", "dIPR"]

mlm_formulas = {"no_interaction": "~ theta_pwp + theta_ufc + theta_ac + ks"}

vars_sim = ["q_ss"]
deltas = ["dAvg"]

if not os.path.exists(mlm_file):
    dfs_mlms = []
    dict_mlm_soil = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_soil[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_soil[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_soil[mlm_key][var_sim][delta] = {}
                for land_cover_scenario in land_cover_scenarios:
                    dict_mlm_soil[mlm_key][var_sim][delta][land_cover_scenario] = {}
                    for soil_depth in soil_depths:
                        dict_mlm_soil[mlm_key][location][climate_scenario][var_sim][delta][land_cover_scenario][soil_depth] = {}
                        cond = _soil_depths[soil_depth]
                        for location in locations:
                            for climate_scenario in climate_scenarios:
                                for future in ['nf', 'ff']:
                                    y = (
                                        dict_deltas[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    x = sm.add_constant(x)
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (x_std.loc[:, param] - x_std.loc[:, param].mean()) / x_std.loc[
                                            :, param
                                        ].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = f"{location}_{climate_scenario}_{future}"
                                    dfs_mlms.append(dta)

                        df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                        mlm = smf.mixedlm(f'{delta} {mlm_formulas[mlm_key]}', df_mlm, groups=df_mlm["group"], re_formula=mlm_formulas[mlm_key])
                        res = mlm.fit()
                        print(res.summary())
                        dict_mlm_soil[mlm_key][var_sim][delta][land_cover_scenario] = res.params

    # Store data (serialize)
    with open(mlm_file, 'wb') as handle:
        pickle.dump(dict_mlm_soil, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load data (deserialize)
    with open(mlm_file, 'rb') as handle:
        dict_mlm_soil = pickle.load(handle)



dfs_mlms = []
dict_mlm_meteo = {}
for mlm_key in mlm_formulas.keys():
    dict_mlm_meteo[mlm_key] = {}
    for var_sim in vars_sim:
        dict_mlm_meteo[mlm_key][var_sim] = {}
        for delta in deltas:
            dict_mlm_meteo[mlm_key][var_sim][delta] = {}
            for land_cover_scenario in land_cover_scenarios:
                dict_mlm_meteo[mlm_key][var_sim][delta][land_cover_scenario] = {}
                for soil_depth in soil_depths:
                    dict_mlm_meteo[mlm_key][location][climate_scenario][var_sim][delta][land_cover_scenario][soil_depth] = {}
                    cond = _soil_depths[soil_depth]
                    for location in locations:
                        for climate_scenario in climate_scenarios:
                            for future in ['nf', 'ff']:
                                y = (
                                    dict_deltas[location][land_cover_scenario][climate_scenario][var_sim]
                                    .loc[cond, f"{delta}_{future}"]
                                    .to_frame()
                                )
                                y = pd.DataFrame(data=y, dtype=onp.float64)
                                x = pd.DataFrame(index=range(len(y.index)))
                                x.loc[:, "dPREC_avg"] = dict_deltas[location][land_cover_scenario][climate_scenario]["prec"].loc[:, f"dAvg_{future}"].values[0]
                                x.loc[:, "dTA_avg"] = dict_deltas[location][land_cover_scenario][climate_scenario]["ta"].loc[:, f"dAvg_{future}"].values[0]
                                x.loc[:, "dPREC_ipr"] = dict_deltas[location][land_cover_scenario][climate_scenario]["prec"].loc[:, f"dIPR_{future}"].values[0]
                                x.loc[:, "dTA_ipr"] = dict_deltas[location][land_cover_scenario][climate_scenario]["ta"].loc[:, f"dIPR_{future}"].values[0]
                                x = sm.add_constant(x)

                                dta = x.copy()
                                dta.loc[:, f"{delta}"] = y.values.flatten()
                                dta.loc[:, "group"] = f"{location}_{climate_scenario}_{future}"
                                dfs_mlms.append(dta)

                    df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                    mlm = smf.mixedlm(f'{delta} ~ dPREC + dTA', df_mlm, groups=df_mlm["group"])
                    res = mlm.fit()
                    print(res.summary())
                    dict_mlm_meteo[mlm_key][var_sim][delta][land_cover_scenario] = res.params
