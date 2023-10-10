import os
from pathlib import Path
import pandas as pd
import numpy as onp
import datetime
import matplotlib as mpl
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.graphics.api import abline_plot
import statsmodels.tools.eval_measures as smem
from scipy import stats
from sklearn.metrics import r2_score
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
base_path_output = base_path / "output"
if not os.path.exists(base_path_output):
    os.mkdir(base_path_output)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# load delta changes
delta_file = base_path_figs / "delta_changes.pkl"
with open(delta_file, "rb") as handle:
    dict_delta_changes = pickle.load(handle)

glm_location_file = base_path_figs / "glm_location_results.pkl"
mlm_random_intercepts_file = base_path_figs / "mlm_random_intercepts_results.pkl"
mlm_random_slopes_file = base_path_figs / "mlm_random_slopes_results.pkl"
mlm_random_intercepts_llcp_file = base_path_figs / "mlm_random_intercepts_llcp_results.pkl"
mlm_random_slopes_llcp_file = base_path_figs / "mlm_random_slopes_llcp_results.pkl"
mlm_random_intercepts_soil_llcp_file = base_path_figs / "mlm_random_intercepts_soil_llcp_results.pkl"
mlm_random_slopes_soil_llcp_file = base_path_figs / "mlm_random_slopes_soil_llcp_results.pkl"
mlm_random_intercepts_location_file = base_path_figs / "mlm_random_intercepts_location_results.pkl"
mlm_random_slopes_location_file = base_path_figs / "mlm_random_slopes_location_results.pkl"
mlm_random_intercepts_soil_location_file = base_path_figs / "mlm_random_intercepts_soil_location_results.pkl"
mlm_random_slopes_soil_location_file = base_path_figs / "mlm_random_slopes_soil_location_results.pkl"
mlm_random_intercepts_land_cover_scenario_file = base_path_figs / "mlm_random_intercepts_land_cover_scenario_results.pkl"
mlm_random_slopes_land_cover_scenario_file = base_path_figs / "mlm_random_slopes_land_cover_scenario_results.pkl"
mlm_random_intercepts_soil_land_cover_scenario_file = base_path_figs / "mlm_random_intercepts_soil_properties_land_cover_scenario_results.pkl"
mlm_random_slopes_soil_land_cover_scenario_file = base_path_figs / "mlm_random_slopes_soil_properties_land_cover_scenario_results.pkl"

# identifiers for simulations
locations = ["freiburg", "altheim", "kupferzell"]
Locations = ["Freiburg", "Altheim", "Kupferzell"]
land_cover_scenarios = ["grass", "corn", "corn_catch_crop", "crop_rotation"]
Land_cover_scenarios = ["Grass", "Corn", "Corn & Catch crop", "Crop rotation"]
climate_scenarios = ["CCCma-CanESM2_CCLM4-8-17", "MPI-M-MPI-ESM-LR_RCA4"]
periods = ["1985-2014", "2030-2059", "2070-2099"]
start_dates = [datetime.date(1985, 1, 1), datetime.date(2030, 1, 1), datetime.date(2070, 1, 1)]
end_dates = [datetime.date(2014, 12, 31), datetime.date(2059, 12, 31), datetime.date(2099, 12, 31)]

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
    "VARE": "Variance of error",
    "R2": "Coefficient of determination",
    "AIC": "AIC",
    "AICC": "AICc",
    "theta_pwp:theta_ufc": r"$\theta_{pwp}$ x $\theta_{ufc}$",
    "theta_pwp:theta_ac": r"$\theta_{pwp}$ x $\theta_{ac}$",
    "theta_ufc:theta_ac": r"$\theta_{ufc}$ x $\theta_{ac}$",
    "theta_pwp:theta_ufc:theta_ac": r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$",
    "grass": "Grass",
    "corn": "Corn",
    "corn_catch_crop": "Corn & Catch crop",
    "crop_rotation": "Crop rotation",
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

# load the RoGeR model parameters
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

# plot the distribution of the RoGeR model parameters
for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
    vals = df_params.loc[:, param].values
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.hist(vals, bins=25, color="black")
    ax.set_xlabel(f"{_lab_unit1[param]}")
    fig.tight_layout()
    file = base_path_figs / "glm_parameters" / f"{param}_hist.png"
    fig.savefig(file, dpi=300)

# GLM model structures
glm_location_formulas = {
    "interaction_theta": "y ~ theta_pwp * theta_ufc * theta_ac + ks",
    "no_interaction": "y ~ theta_pwp + theta_ufc + theta_ac + ks",
    # "theta": "y ~ theta_pwp + theta_ufc + theta_ac",
    # "theta_pwp": "y ~ theta_pwp",
    # "theta_ufc": "y ~ theta_ufc",
    # "theta_ac": "y ~ theta_ac",
    # "ks": "y ~ ks",
}

vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
deltas = ["dAvg", "dIPR"]
if not os.path.exists(glm_location_file):
    dict_glm = {}
    for glm_key in glm_location_formulas.keys():
        dict_glm[glm_key] = {}
        for location in locations:
            dict_glm[glm_key][location] = {}
            for land_cover_scenario in land_cover_scenarios:
                dict_glm[glm_key][location][land_cover_scenario] = {}
                for climate_scenario in climate_scenarios:
                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario] = {}
                    for var_sim in vars_sim:
                        dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim] = {}
                        for delta in deltas:
                            dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta] = {}
                            for soil_depth in soil_depths:
                                dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                    soil_depth
                                ] = {}
                                cond = _soil_depths[soil_depth]
                                for future in ["nf", "ff"]:
                                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                        soil_depth
                                    ][future] = {}
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    x = sm.add_constant(x)
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, "y"] = y.values.flatten()

                                    # fit the GLM model
                                    glm = smf.glm(
                                        formula=glm_location_formulas[glm_key], data=dta, family=sm.families.Gaussian()
                                    )
                                    res = glm.fit()
                                    print(res.summary())
                                    # number of data points
                                    nobs = res.nobs
                                    # average predicted values
                                    yhat = res.mu
                                    # response variable
                                    y = y.values.flatten()

                                    if glm_key == "interaction_theta":
                                        ll = [
                                            "theta_pwp",
                                            "theta_ufc",
                                            "theta_ac",
                                            "theta_pwp:theta_ufc",
                                            "theta_pwp:theta_ac",
                                            "theta_ufc:theta_ac",
                                            "theta_pwp:theta_ufc:theta_ac",
                                            "ks",
                                        ]
                                        params = res.params.to_frame().T.loc[:, ll]
                                    else:
                                        params = res.params.to_frame().T

                                    # relative importance of parameters
                                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                        soil_depth
                                    ][future]["params"] = params / onp.sum(onp.abs(params.values.flatten()))
                                    # evaluation metrics
                                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                        soil_depth
                                    ][future]["MAE"] = smem.meanabs(y, yhat)
                                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                        soil_depth
                                    ][future]["MEAE"] = smem.medianabs(y, yhat)
                                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                        soil_depth
                                    ][future]["RMSE"] = smem.rmse(y, yhat)
                                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                        soil_depth
                                    ][future]["VARE"] = smem.vare(y, yhat)
                                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                        soil_depth
                                    ][future]["R2"] = r2_score(y, yhat)
                                    ll_values = glm.loglike(res.params, scale=res.scale)
                                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                        soil_depth
                                    ][future]["AIC"] = smem.aic(ll_values, nobs, res.params.shape[0])
                                    dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][delta][
                                        soil_depth
                                    ][future]["AICC"] = smem.aicc(ll_values, nobs, res.params.shape[0])

                                    # plot the line fit
                                    fig, ax = plt.subplots(figsize=(3, 3))
                                    ax.scatter(yhat, y, color="black", s=4)
                                    line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
                                    abline_plot(model_results=line_fit, ax=ax, color="black")
                                    ax.set_xlabel(f"{_lab[delta]}{_lab_unit1[var_sim]} (RoGeR) [%]")
                                    ax.set_ylabel(f"{_lab[delta]}{_lab_unit1[var_sim]} (GLM) [%]")
                                    ax.set_xlim(-100, 100)
                                    ax.set_ylim(-100, 100)
                                    fig.tight_layout()
                                    file = (
                                        base_path_figs
                                        / "residuals"
                                        / "GLM"
                                        / f"{glm_key}_{location}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_line_fit.png"
                                    )
                                    fig.savefig(file, dpi=300)

                                    # plot the residuals
                                    fig, ax = plt.subplots(figsize=(3, 3))
                                    ax.scatter(y, res.resid_pearson, color="black", s=4)
                                    ax.set_ylabel("Residuals [%]")
                                    ax.set_xlabel(f"{_lab[delta]}{_lab_unit1[var_sim]}")
                                    fig.tight_layout()
                                    file = (
                                        base_path_figs
                                        / "residuals"
                                        / "GLM"
                                        / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_residuals.png"
                                    )
                                    fig.savefig(file, dpi=300)

                                    # fig, ax = plt.subplots(figsize=(3, 3))
                                    # ax.scatter(x.loc[:, 'theta_ufc'].values, y, color='black', s=4)
                                    # ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
                                    # ax.set_xlabel(f'{_lab_unit1["theta_ufc"]}')
                                    # fig.tight_layout()
                                    # file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_theta_ufc.png"
                                    # fig.savefig(file, dpi=300)

                                    # fig, ax = plt.subplots(figsize=(3, 3))
                                    # ax.scatter(x.loc[:, 'theta_ac'].values, y, color='black', s=4)
                                    # ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
                                    # ax.set_xlabel(f'{_lab_unit1["theta_ac"]}')
                                    # fig.tight_layout()
                                    # file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_theta_ac.png"
                                    # fig.savefig(file, dpi=300)

                                    # fig, ax = plt.subplots(figsize=(3, 3))
                                    # ax.scatter(x.loc[:, 'theta_pwp'].values, y, color='black', s=4)
                                    # ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
                                    # ax.set_xlabel(f'{_lab_unit1["theta_pwp"]}')
                                    # fig.tight_layout()
                                    # file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_theta_pwp.png"
                                    # fig.savefig(file, dpi=300)

                                    # fig, ax = plt.subplots(figsize=(3, 3))
                                    # ax.scatter(x.loc[:, 'ks'].values, y, color='black', s=4)
                                    # ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
                                    # ax.set_xlabel(f'{_lab_unit1["ks"]}')
                                    # fig.tight_layout()
                                    # file = base_path_figs / "residuals" / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_ks.png"
                                    # fig.savefig(file, dpi=300)

                                    # plot the distribution of residuals
                                    fig, ax = plt.subplots(figsize=(3, 3))
                                    resid = res.resid_deviance.copy()
                                    resid_std = stats.zscore(resid)
                                    ax.hist(resid_std, bins=25, color="black")
                                    ax.set_xlabel(f"{_lab[delta]}{_lab_unit1[var_sim]} (GLM)")
                                    fig.tight_layout()
                                    file = (
                                        base_path_figs
                                        / "residuals"
                                        / "GLM"
                                        / f"{glm_key}_{location}_{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_residuals_hist.png"
                                    )
                                    fig.savefig(file, dpi=300)
                                    plt.close("all")

    # Store data (serialize)
    with open(glm_location_file, "wb") as handle:
        pickle.dump(dict_glm, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(glm_location_file, "rb") as handle:
        dict_glm = pickle.load(handle)


# model structures of mixed linear effect models
mlm_formulas = {
    "interaction_theta": "~ theta_pwp * theta_ufc * theta_ac + ks",
    "no_interaction": "~ theta_pwp + theta_ufc + theta_ac + ks",
    # "theta": "~ theta_pwp + theta_ufc + theta_ac",
    # "theta_pwp": "~ theta_pwp",
    # "theta_ufc": "~ theta_ufc",
    # "theta_ac": "~ theta_ac",
    # "ks": "~ ks",
}

vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
deltas = ["dAvg", "dIPR"]

# group by soil properties
if not os.path.exists(mlm_random_intercepts_file):
    dict_mlm_random_intercepts = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_intercepts[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_intercepts[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_intercepts[mlm_key][var_sim][delta] = {}
                for location in locations:
                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location] = {}
                    for land_cover_scenario in land_cover_scenarios:
                        dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario] = {}
                        for climate_scenario in climate_scenarios:
                            dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario] = {}
                            for future in ["nf", "ff"]:
                                dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future] = {}
                                for soil_depth in soil_depths:
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth] = {}
                                    cond = _soil_depths[soil_depth]
                                    # merge the data for each land cover senario into a single dataframe
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    df_mlm = x_std.copy()
                                    df_mlm.loc[:, f"{delta}"] = y.values.flatten()
                                    df_mlm.loc[:, "group"] = range(len(y.values.flatten()))

                                    df_mlm = sm.add_constant(df_mlm)
                                    # fit a mixed linear model with random effects (random intercepts)
                                    mlm = smf.mixedlm(f"{delta} {mlm_formulas[mlm_key]}", df_mlm, groups=df_mlm["group"])
                                    res = mlm.fit()
                                    print(res.summary())

                                    # response variable
                                    y = df_mlm.loc[:, f"{delta}"].values
                                    # number of data points
                                    nobs = res.nobs
                                    # predicted values
                                    # yhat = res.predict()
                                    yhat = res.fittedvalues
                                    # fixed effect parameters and random effect parameters
                                    fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                                    re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                                    params = onp.concatenate((fe_params, re_params), axis=None)
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                                        "fixed_effects"
                                    ] = res.params.to_frame()
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                                        "random_effects"
                                    ] = pd.DataFrame(res.random_effects).var(axis=1)

                                    # relative importance of parameters
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                                        "params"
                                    ] = params / onp.sum(params)

                                    # contribution of the fixed effect parameters and random effect parameters
                                    dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                                    dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                                    dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["attribution"] = dfa
                                    # evaluation metrics
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["MAE"] = smem.meanabs(
                                        y, yhat
                                    )
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                                        "MEAE"
                                    ] = smem.medianabs(y, yhat)
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["RMSE"] = smem.rmse(
                                        y, yhat
                                    )
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["VARE"] = smem.vare(
                                        y, yhat
                                    )
                                    dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["R2"] = r2_score(
                                        y, yhat
                                    )


    # Store the data (serialize)
    with open(mlm_random_intercepts_file, "wb") as handle:
        pickle.dump(dict_mlm_random_intercepts, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_intercepts_file, "rb") as handle:
        dict_mlm_random_intercepts = pickle.load(handle)


if not os.path.exists(mlm_random_slopes_file):
    dict_mlm_random_slopes = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_slopes[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_slopes[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_slopes[mlm_key][var_sim][delta] = {}
                for location in locations:
                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location] = {}
                    for land_cover_scenario in land_cover_scenarios:
                        dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario] = {}
                        for climate_scenario in climate_scenarios:
                            dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario] = {}
                            for future in ["nf", "ff"]:
                                dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future] = {}
                                for soil_depth in soil_depths:
                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth] = {}
                                    cond = _soil_depths[soil_depth]
                                    # merge the data for each land cover senario into a single dataframe
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    df_mlm = x_std.copy()
                                    df_mlm.loc[:, f"{delta}"] = y.values.flatten()
                                    df_mlm.loc[:, "group"] = range(len(y.values.flatten()))

                                    df_mlm = sm.add_constant(df_mlm)
                                    # fit a mixed linear model with random effects (random intercepts and random slopes)
                                    mlm = smf.mixedlm(f'{delta} {mlm_formulas[mlm_key]}', df_mlm, groups=df_mlm["group"], re_formula=mlm_formulas[mlm_key])
                                    res = mlm.fit()
                                    print(res.summary())

                                    # response variable
                                    y = df_mlm.loc[:, f"{delta}"].values
                                    # number of data points
                                    nobs = res.nobs
                                    # predicted values
                                    # yhat = res.predict()
                                    yhat = res.fittedvalues
                                    # fixed effect parameters and random effect parameters
                                    fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                                    re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                                    params = onp.concatenate((fe_params, re_params), axis=None)
                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                                        "fixed_effects"
                                    ] = res.params.to_frame()
                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                                        "random_effects"
                                    ] = pd.DataFrame(res.random_effects).var(axis=1)

                                    # relative importance of parameters
                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                                        "params"
                                    ] = params / onp.sum(params)

                                    # contribution of the fixed effect parameters and random effect parameters
                                    dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                                    dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                                    dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)

                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["attribution"] = dfa
                                    # evaluation metrics
                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["MAE"] = smem.meanabs(
                                        y, yhat
                                    )
                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                                        "MEAE"
                                    ] = smem.medianabs(y, yhat)
                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["RMSE"] = smem.rmse(
                                        y, yhat
                                    )
                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["VARE"] = smem.vare(
                                        y, yhat
                                    )
                                    dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth]["R2"] = r2_score(
                                        y, yhat
                                    )

    # Store the data (serialize)
    with open(mlm_random_slopes_file, "wb") as handle:
        pickle.dump(dict_mlm_random_slopes, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_slopes_file, "rb") as handle:
        dict_mlm_random_slopes = pickle.load(handle)

# group by location, land cover scenario, climate model and future
if not os.path.exists(mlm_random_intercepts_llcp_file):
    dict_mlm_random_intercepts_llcp = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_intercepts_llcp[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_intercepts_llcp[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta] = {}
                for soil_depth in soil_depths:
                    dfs_mlms = []
                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][soil_depth] = {}
                    cond = _soil_depths[soil_depth]
                    # merge the data for each land cover senario into a single dataframe
                    for location in locations:
                        for land_cover_scenario in land_cover_scenarios:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = f"{location}_{land_cover_scenario}_{climate_scenario}_{future}"
                                    dfs_mlms.append(dta)

                    df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                    df_mlm = sm.add_constant(df_mlm)
                    # fit a mixed linear model with random effects (random intercepts)
                    mlm = smf.mixedlm(f"{delta} {mlm_formulas[mlm_key]}", df_mlm, groups=df_mlm["group"])
                    res = mlm.fit()
                    print(res.summary())

                    # response variable
                    y = df_mlm.loc[:, f"{delta}"].values
                    # number of data points
                    nobs = res.nobs
                    # predicted values
                    # yhat = res.predict()
                    yhat = res.fittedvalues
                    # fixed effect parameters and random effect parameters
                    fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                    re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                    params = onp.concatenate((fe_params, re_params), axis=None)
                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "fixed_effects"
                    ] = res.params.to_frame()
                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "random_effects"
                    ] = pd.DataFrame(res.random_effects).var(axis=1)

                    # relative importance of parameters
                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "params"
                    ] = params / onp.sum(params)

                    # contribution of the fixed effect parameters and random effect parameters
                    dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                    dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                    dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)

                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][soil_depth]["attribution"] = dfa
                    # evaluation metrics
                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][soil_depth]["MAE"] = smem.meanabs(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][soil_depth][
                        "MEAE"
                    ] = smem.medianabs(y, yhat)
                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][soil_depth]["RMSE"] = smem.rmse(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][soil_depth]["VARE"] = smem.vare(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_llcp[mlm_key][var_sim][delta][soil_depth]["R2"] = r2_score(
                        y, yhat
                    )

                    # plot the line fit
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.scatter(y, yhat, color="black", s=4)
                    line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
                    abline_plot(model_results=line_fit, ax=ax, color="black")
                    ax.set_xlabel(f"{_lab[delta]}{_lab_unit1[var_sim]} (RoGeR) [%]")
                    ax.set_ylabel(f"{_lab[delta]}{_lab_unit1[var_sim]} (MLM) [%]")
                    ax.set_xlim(-100, 100)
                    ax.set_ylim(-100, 100)
                    fig.tight_layout()
                    file = (
                        base_path_figs
                        / "residuals"
                        / "MLM"
                        / f"{mlm_key}_{var_sim}_{delta}_{soil_depth}_line_fit_for_random_intercepts.png"
                    )
                    fig.savefig(file, dpi=300)

                    # plot the residuals
                    resid = yhat - y
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.scatter(y, resid, color="black", s=4)
                    ax.set_ylabel("Residuals [%]")
                    ax.set_xlabel(f"{_lab[delta]}{_lab_unit1[var_sim]}")
                    fig.tight_layout()
                    file = (
                        base_path_figs
                        / "residuals"
                        / "MLM"
                        / f"{mlm_key}_{var_sim}_{delta}_{soil_depth}_residuals_for_random_intercepts.png"
                    )
                    fig.savefig(file, dpi=300)

                    # plot the distribution of residuals
                    fig, ax = plt.subplots(figsize=(3, 3))
                    resid_std = stats.zscore(resid)
                    ax.hist(resid_std, bins=25, color="black")
                    ax.set_xlabel(f"{_lab[delta]}{_lab_unit1[var_sim]} (MLM)")
                    fig.tight_layout()
                    file = (
                        base_path_figs
                        / "residuals"
                        / "MLM"
                        / f"{mlm_key}_{var_sim}_{delta}_{soil_depth}_residuals_hist_for_random_intercepts.png"
                    )
                    fig.savefig(file, dpi=300)
                    plt.close("all")

    # Store the data (serialize)
    with open(mlm_random_intercepts_llcp_file, "wb") as handle:
        pickle.dump(dict_mlm_random_intercepts_llcp, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_intercepts_llcp_file, "rb") as handle:
        dict_mlm_random_intercepts_llcp = pickle.load(handle)


if not os.path.exists(mlm_random_slopes_llcp_file):
    dict_mlm_random_slopes_llcp = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_slopes_llcp[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_slopes_llcp[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta] = {}
                for soil_depth in soil_depths:
                    dfs_mlms = []
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][soil_depth] = {}
                    cond = _soil_depths[soil_depth]
                    # merge the data for each land cover senario into a single dataframe
                    for land_cover_scenario in land_cover_scenarios:
                        for location in locations:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = f"{location}_{land_cover_scenario}_{climate_scenario}_{future}"
                                    dfs_mlms.append(dta)

                    df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                    df_mlm = sm.add_constant(df_mlm)
                    # fit a mixed linear model with random effects (random intercepts and random slopes)
                    mlm = smf.mixedlm(f'{delta} {mlm_formulas[mlm_key]}', df_mlm, groups=df_mlm["group"], re_formula=mlm_formulas[mlm_key])
                    res = mlm.fit()
                    print(res.summary())

                    # response variable
                    y = df_mlm.loc[:, f"{delta}"].values
                    # number of data points
                    nobs = res.nobs
                    # predicted values
                    # yhat = res.predict()
                    yhat = res.fittedvalues
                    # fixed effect parameters and random effect parameters
                    fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                    re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                    params = onp.concatenate((fe_params, re_params), axis=None)
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "fixed_effects"
                    ] = res.params.to_frame()
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "random_effects"
                    ] = pd.DataFrame(res.random_effects).var(axis=1)
                    # relative importance of parameters
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "params"
                    ] = params / onp.sum(params)
                    # contribution of the fixed effect parameters and random effect parameters
                    dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                    dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                    dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][soil_depth]["attribution"] = dfa
                    # evaluation metrics
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][soil_depth]["MAE"] = smem.meanabs(
                        y, yhat
                    )
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][soil_depth][
                        "MEAE"
                    ] = smem.medianabs(y, yhat)
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][soil_depth]["RMSE"] = smem.rmse(
                        y, yhat
                    )
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][soil_depth]["VARE"] = smem.vare(
                        y, yhat
                    )
                    dict_mlm_random_slopes_llcp[mlm_key][var_sim][delta][soil_depth]["R2"] = r2_score(
                        y, yhat
                    )

    # Store the data (serialize)
    with open(mlm_random_slopes_file, "wb") as handle:
        pickle.dump(dict_mlm_random_slopes_llcp, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_slopes_llcp_file, "rb") as handle:
        dict_mlm_random_slopes_llcp = pickle.load(handle)

# group by soil properties within location, land cover scenario, climate model and future
if not os.path.exists(mlm_random_intercepts_soil_llcp_file):
    dict_mlm_random_intercepts_soil_llcp = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_intercepts_soil_llcp[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta] = {}
                for soil_depth in soil_depths:
                    dfs_mlms = []
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][soil_depth] = {}
                    cond = _soil_depths[soil_depth]
                    # merge the data for each land cover senario into a single dataframe
                    for location in locations:
                        for land_cover_scenario in land_cover_scenarios:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = range(len(y.values.flatten()))
                                    dfs_mlms.append(dta)

                    df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                    df_mlm = sm.add_constant(df_mlm)
                    # fit a mixed linear model with random effects (random intercepts)
                    mlm = smf.mixedlm(f"{delta} {mlm_formulas[mlm_key]}", df_mlm, groups=df_mlm["group"])
                    res = mlm.fit()
                    print(res.summary())

                    # response variable
                    y = df_mlm.loc[:, f"{delta}"].values
                    # number of data points
                    nobs = res.nobs
                    # predicted values
                    # yhat = res.predict()
                    yhat = res.fittedvalues
                    # fixed effect parameters and random effect parameters
                    fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                    re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                    params = onp.concatenate((fe_params, re_params), axis=None)
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "fixed_effects"
                    ] = res.params.to_frame()
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "random_effects"
                    ] = pd.DataFrame(res.random_effects).var(axis=1)
                    # relative importance of parameters
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "params"
                    ] = params / onp.sum(params)
                    # contribution of the fixed effect parameters and random effect parameters
                    dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                    dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                    dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][soil_depth]["attribution"] = dfa
                    # evaluation metrics
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][soil_depth]["MAE"] = smem.meanabs(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][soil_depth][
                        "MEAE"
                    ] = smem.medianabs(y, yhat)
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][soil_depth]["RMSE"] = smem.rmse(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][soil_depth]["VARE"] = smem.vare(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][soil_depth]["R2"] = r2_score(
                        y, yhat
                    )

                    # plot the line fit
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.scatter(y, yhat, color="black", s=4)
                    line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
                    abline_plot(model_results=line_fit, ax=ax, color="black")
                    ax.set_xlabel(f"{_lab[delta]}{_lab_unit1[var_sim]} (RoGeR) [%]")
                    ax.set_ylabel(f"{_lab[delta]}{_lab_unit1[var_sim]} (MLM) [%]")
                    ax.set_xlim(-100, 100)
                    ax.set_ylim(-100, 100)
                    fig.tight_layout()
                    file = (
                        base_path_figs
                        / "residuals"
                        / "MLM"
                        / f"{mlm_key}_{var_sim}_{delta}_{soil_depth}_line_fit_for_random_intercepts.png"
                    )
                    fig.savefig(file, dpi=300)

                    # plot the residuals
                    resid = yhat - y
                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.scatter(y, resid, color="black", s=4)
                    ax.set_ylabel("Residuals [%]")
                    ax.set_xlabel(f"{_lab[delta]}{_lab_unit1[var_sim]}")
                    fig.tight_layout()
                    file = (
                        base_path_figs
                        / "residuals"
                        / "MLM"
                        / f"{mlm_key}_{var_sim}_{delta}_{soil_depth}_residuals_for_random_intercepts.png"
                    )
                    fig.savefig(file, dpi=300)

                    # plot the distribution of residuals
                    fig, ax = plt.subplots(figsize=(3, 3))
                    resid_std = stats.zscore(resid)
                    ax.hist(resid_std, bins=25, color="black")
                    ax.set_xlabel(f"{_lab[delta]}{_lab_unit1[var_sim]} (MLM)")
                    fig.tight_layout()
                    file = (
                        base_path_figs
                        / "residuals"
                        / "MLM"
                        / f"{mlm_key}_{var_sim}_{delta}_{soil_depth}_residuals_hist_for_random_intercepts.png"
                    )
                    fig.savefig(file, dpi=300)
                    plt.close("all")

    # Store the data (serialize)
    with open(mlm_random_intercepts_soil_llcp_file, "wb") as handle:
        pickle.dump(dict_mlm_random_intercepts_soil_llcp, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_intercepts_soil_llcp_file, "rb") as handle:
        dict_mlm_random_intercepts_soil_llcp = pickle.load(handle)


if not os.path.exists(mlm_random_slopes_soil_llcp_file):
    dict_mlm_random_slopes_soil_llcp = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_slopes_soil_llcp[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta] = {}
                for soil_depth in soil_depths:
                    dfs_mlms = []
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][soil_depth] = {}
                    cond = _soil_depths[soil_depth]
                    # merge the data for each land cover senario into a single dataframe
                    for land_cover_scenario in land_cover_scenarios:
                        for location in locations:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = range(len(y.values.flatten()))
                                    dfs_mlms.append(dta)

                    df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                    df_mlm = sm.add_constant(df_mlm)
                    # fit a mixed linear model with random effects (random intercepts and random slopes)
                    mlm = smf.mixedlm(f'{delta} {mlm_formulas[mlm_key]}', df_mlm, groups=df_mlm["group"], re_formula=mlm_formulas[mlm_key])
                    res = mlm.fit()
                    print(res.summary())

                    # response variable
                    y = df_mlm.loc[:, f"{delta}"].values
                    # number of data points
                    nobs = res.nobs
                    # predicted values
                    # yhat = res.predict()
                    yhat = res.fittedvalues
                    # fixed effect parameters and random effect parameters
                    fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                    re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                    params = onp.concatenate((fe_params, re_params), axis=None)
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "fixed_effects"
                    ] = res.params.to_frame()
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "random_effects"
                    ] = pd.DataFrame(res.random_effects).var(axis=1)
                    # relative importance of parameters
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "params"
                    ] = params / onp.sum(params)
                    # contribution of the fixed effect parameters and random effect parameters
                    dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                    dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                    dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][soil_depth]["attribution"] = dfa
                    # evaluation metrics
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][soil_depth]["MAE"] = smem.meanabs(
                        y, yhat
                    )
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][soil_depth][
                        "MEAE"
                    ] = smem.medianabs(y, yhat)
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][soil_depth]["RMSE"] = smem.rmse(
                        y, yhat
                    )
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][soil_depth]["VARE"] = smem.vare(
                        y, yhat
                    )
                    dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][soil_depth]["R2"] = r2_score(
                        y, yhat
                    )

    # Store the data (serialize)
    with open(mlm_random_slopes_file, "wb") as handle:
        pickle.dump(dict_mlm_random_slopes_soil_llcp, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_slopes_soil_llcp_file, "rb") as handle:
        dict_mlm_random_slopes_soil_llcp = pickle.load(handle)


# group by location
if not os.path.exists(mlm_random_intercepts_location_file):
    dict_mlm_random_intercepts_location = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_intercepts_location[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_intercepts_location[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_intercepts_location[mlm_key][var_sim][delta] = {}
                for soil_depth in soil_depths:
                    dfs_mlms = []
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][soil_depth] = {}
                    cond = _soil_depths[soil_depth]
                    # merge the data for each land cover senario into a single dataframe
                    for location in locations:
                        for land_cover_scenario in land_cover_scenarios:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = f"{location}"
                                    dfs_mlms.append(dta)

                    df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                    df_mlm = sm.add_constant(df_mlm)
                    # fit a mixed linear model with random effects (random intercepts)
                    mlm = smf.mixedlm(f"{delta} {mlm_formulas[mlm_key]}", df_mlm, groups=df_mlm["group"])
                    res = mlm.fit()
                    print(res.summary())

                    # response variable
                    y = df_mlm.loc[:, f"{delta}"].values
                    # number of data points
                    nobs = res.nobs
                    # predicted values
                    # yhat = res.predict()
                    yhat = res.fittedvalues
                    # fixed effect parameters and random effect parameters
                    fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                    re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                    params = onp.concatenate((fe_params, re_params), axis=None)
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "fixed_effects"
                    ] = res.params.to_frame()
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "random_effects"
                    ] = pd.DataFrame(res.random_effects).var(axis=1)
                    # relative importance of parameters
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "params"
                    ] = params / onp.sum(params)
                    # contribution of the fixed effect parameters and random effect parameters
                    dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                    dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                    dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][soil_depth]["attribution"] = dfa                    
                    # evaluation metrics
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][soil_depth]["MAE"] = smem.meanabs(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][soil_depth][
                        "MEAE"
                    ] = smem.medianabs(y, yhat)
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][soil_depth]["RMSE"] = smem.rmse(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][soil_depth]["VARE"] = smem.vare(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_location[mlm_key][var_sim][delta][soil_depth]["R2"] = r2_score(
                        y, yhat
                    )

    # Store the data (serialize)
    with open(mlm_random_intercepts_location_file, "wb") as handle:
        pickle.dump(dict_mlm_random_intercepts_location, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_intercepts_location_file, "rb") as handle:
        dict_mlm_random_intercepts_location = pickle.load(handle)


if not os.path.exists(mlm_random_slopes_location_file):
    dict_mlm_random_slopes_location = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_slopes_location[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_slopes_location[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_slopes_location[mlm_key][var_sim][delta] = {}
                for soil_depth in soil_depths:
                    dfs_mlms = []
                    dict_mlm_random_slopes_location[mlm_key][var_sim][delta][soil_depth] = {}
                    cond = _soil_depths[soil_depth]
                    # merge the data for each land cover senario into a single dataframe
                    for location in locations:
                        for land_cover_scenario in land_cover_scenarios:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = f"{location}"
                                    dfs_mlms.append(dta)

                        df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                        df_mlm = sm.add_constant(df_mlm)
                        # fit a mixed linear model with random effects (random intercepts and random slopes)
                        mlm = smf.mixedlm(f'{delta} {mlm_formulas[mlm_key]}', df_mlm, groups=df_mlm["group"], re_formula=mlm_formulas[mlm_key])
                        res = mlm.fit()
                        print(res.summary())

                        # response variable
                        y = df_mlm.loc[:, f"{delta}"].values
                        # number of data points
                        nobs = res.nobs
                        # predicted values
                        # yhat = res.predict()
                        yhat = res.fittedvalues
                        # fixed effect parameters and random effect parameters
                        fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                        re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                        params = onp.concatenate((fe_params, re_params), axis=None)
                        dict_mlm_random_slopes_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "fixed_effects"
                        ] = res.params.to_frame()
                        dict_mlm_random_slopes_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "random_effects"
                        ] = pd.DataFrame(res.random_effects).var(axis=1)
                        # relative importance of parameters
                        dict_mlm_random_slopes_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "params"
                        ] = params / onp.sum(params)
                        # contribution of the fixed effect parameters and random effect parameters
                        dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                        dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                        dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                        dict_mlm_random_slopes_location[mlm_key][var_sim][delta][soil_depth]["attribution"] = dfa
                        # evaluation metrics
                        dict_mlm_random_slopes_location[mlm_key][var_sim][delta][soil_depth]["MAE"] = smem.meanabs(
                            y, yhat
                        )
                        dict_mlm_random_slopes_location[mlm_key][var_sim][delta][soil_depth][
                            "MEAE"
                        ] = smem.medianabs(y, yhat)
                        dict_mlm_random_slopes_location[mlm_key][var_sim][delta][soil_depth]["RMSE"] = smem.rmse(
                            y, yhat
                        )
                        dict_mlm_random_slopes_location[mlm_key][var_sim][delta][soil_depth]["VARE"] = smem.vare(
                            y, yhat
                        )
                        dict_mlm_random_slopes_location[mlm_key][var_sim][delta][soil_depth]["R2"] = r2_score(
                            y, yhat
                        )

    # Store the data (serialize)
    with open(mlm_random_slopes_location_file, "wb") as handle:
        pickle.dump(dict_mlm_random_slopes_location, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_slopes_location_file, "rb") as handle:
        dict_mlm_random_slopes_location = pickle.load(handle)

# group by soil properties within location
if not os.path.exists(mlm_random_intercepts_soil_location_file):
    dict_mlm_random_intercepts_soil_location = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_intercepts_soil_location[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_intercepts_soil_location[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta] = {}
                for location in locations:
                    dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location] = {}
                    for soil_depth in soil_depths:
                        dfs_mlms = []
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][soil_depth] = {}
                        cond = _soil_depths[soil_depth]
                        # merge the data for each land cover senario into a single dataframe
                        for land_cover_scenario in land_cover_scenarios:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = range(len(y.values.flatten()))
                                    dfs_mlms.append(dta)

                        df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                        df_mlm = sm.add_constant(df_mlm)
                        # fit a mixed linear model with random effects (random intercepts)
                        mlm = smf.mixedlm(f"{delta} {mlm_formulas[mlm_key]}", df_mlm, groups=df_mlm["group"])
                        res = mlm.fit()
                        print(res.summary())

                        # response variable
                        y = df_mlm.loc[:, f"{delta}"].values
                        # number of data points
                        nobs = res.nobs
                        # predicted values
                        # yhat = res.predict()
                        yhat = res.fittedvalues
                        # fixed effect parameters and random effect parameters
                        fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                        re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                        params = onp.concatenate((fe_params, re_params), axis=None)
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "fixed_effects"
                        ] = res.params.to_frame()
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "random_effects"
                        ] = pd.DataFrame(res.random_effects).var(axis=1)
                        # relative importance of parameters
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "params"
                        ] = params / onp.sum(params)
                        # contribution of the fixed effect parameters and random effect parameters
                        dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                        dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                        dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][soil_depth]["attribution"] = dfa
                        # evaluation metrics
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][soil_depth]["MAE"] = smem.meanabs(
                            y, yhat
                        )
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][soil_depth][
                            "MEAE"
                        ] = smem.medianabs(y, yhat)
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][soil_depth]["RMSE"] = smem.rmse(
                            y, yhat
                        )
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][soil_depth]["VARE"] = smem.vare(
                            y, yhat
                        )
                        dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta][location][soil_depth]["R2"] = r2_score(
                            y, yhat
                        )

    # Store the data (serialize)
    with open(mlm_random_intercepts_soil_location_file, "wb") as handle:
        pickle.dump(dict_mlm_random_intercepts_soil_location, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_intercepts_soil_location_file, "rb") as handle:
        dict_mlm_random_intercepts_soil_location = pickle.load(handle)


if not os.path.exists(mlm_random_slopes_soil_location_file):
    dict_mlm_random_slopes_soil_location = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_slopes_soil_location[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_slopes_soil_location[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta] = {}
                for location in locations:
                    dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location] = {}
                    for soil_depth in soil_depths:
                        dfs_mlms = []
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][soil_depth] = {}
                        cond = _soil_depths[soil_depth]
                        # merge the data for each land cover senario into a single dataframe
                        for land_cover_scenario in land_cover_scenarios:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = range(len(y.values.flatten()))
                                    dfs_mlms.append(dta)

                        df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                        df_mlm = sm.add_constant(df_mlm)
                        # fit a mixed linear model with random effects (random intercepts and random slopes)
                        mlm = smf.mixedlm(f'{delta} {mlm_formulas[mlm_key]}', df_mlm, groups=df_mlm["group"], re_formula=mlm_formulas[mlm_key])
                        res = mlm.fit()
                        print(res.summary())

                        # response variable
                        y = df_mlm.loc[:, f"{delta}"].values
                        # number of data points
                        nobs = res.nobs
                        # predicted values
                        # yhat = res.predict()
                        yhat = res.fittedvalues
                        # fixed effect parameters and random effect parameters
                        fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                        re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                        params = onp.concatenate((fe_params, re_params), axis=None)
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "fixed_effects"
                        ] = res.params.to_frame()
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "random_effects"
                        ] = pd.DataFrame(res.random_effects).var(axis=1)
                        # relative importance of parameters
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "params"
                        ] = params / onp.sum(params)
                        # contribution of the fixed effect parameters and random effect parameters
                        dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                        dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                        dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][soil_depth]["attribution"] = dfa
                        # evaluation metrics
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][soil_depth]["MAE"] = smem.meanabs(
                            y, yhat
                        )
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][soil_depth][
                            "MEAE"
                        ] = smem.medianabs(y, yhat)
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][soil_depth]["RMSE"] = smem.rmse(
                            y, yhat
                        )
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][soil_depth]["VARE"] = smem.vare(
                            y, yhat
                        )
                        dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta][location][soil_depth]["R2"] = r2_score(
                            y, yhat
                        )

    # Store the data (serialize)
    with open(mlm_random_slopes_soil_location_file, "wb") as handle:
        pickle.dump(dict_mlm_random_slopes_soil_location, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_slopes_soil_location_file, "rb") as handle:
        dict_mlm_random_slopes_soil_location = pickle.load(handle)


# group by land cover scenario
if not os.path.exists(mlm_random_intercepts_land_cover_scenario_file):
    dict_mlm_random_intercepts_land_cover_scenario = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_intercepts_land_cover_scenario[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta] = {}
                for soil_depth in soil_depths:
                    dfs_mlms = []
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][soil_depth] = {}
                    cond = _soil_depths[soil_depth]
                    # merge the data for each land cover senario into a single dataframe
                    for land_cover_scenario in land_cover_scenarios:
                        for location in locations:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = f"{land_cover_scenario}"
                                    dfs_mlms.append(dta)

                    df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                    df_mlm = sm.add_constant(df_mlm)
                    # fit a mixed linear model with random effects (random intercepts)
                    mlm = smf.mixedlm(f"{delta} {mlm_formulas[mlm_key]}", df_mlm, groups=df_mlm["group"])
                    res = mlm.fit()
                    print(res.summary())

                    # response variable
                    y = df_mlm.loc[:, f"{delta}"].values
                    # number of data points
                    nobs = res.nobs
                    # predicted values
                    # yhat = res.predict()
                    yhat = res.fittedvalues
                    # fixed effect parameters and random effect parameters
                    fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                    re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                    params = onp.concatenate((fe_params, re_params), axis=None)
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "fixed_effects"
                    ] = res.params.to_frame()
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "random_effects"
                    ] = pd.DataFrame(res.random_effects).var(axis=1)
                    # relative importance of parameters
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "params"
                    ] = params / onp.sum(params)
                    # contribution of the fixed effect parameters and random effect parameters
                    dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                    dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                    dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["attribution"] = dfa
                    # evaluation metrics
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["MAE"] = smem.meanabs(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][soil_depth][
                        "MEAE"
                    ] = smem.medianabs(y, yhat)
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["RMSE"] = smem.rmse(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["VARE"] = smem.vare(
                        y, yhat
                    )
                    dict_mlm_random_intercepts_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["R2"] = r2_score(
                        y, yhat
                    )

    # Store the data (serialize)
    with open(mlm_random_intercepts_land_cover_scenario_file, "wb") as handle:
        pickle.dump(dict_mlm_random_intercepts_land_cover_scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_intercepts_land_cover_scenario_file, "rb") as handle:
        dict_mlm_random_intercepts_land_cover_scenario = pickle.load(handle)


if not os.path.exists(mlm_random_slopes_land_cover_scenario_file):
    dict_mlm_random_slopes_land_cover_scenario = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_slopes_land_cover_scenario[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta] = {}
                for soil_depth in soil_depths:
                    dfs_mlms = []
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][soil_depth] = {}
                    cond = _soil_depths[soil_depth]
                    # merge the data for each land cover senario into a single dataframe
                    for land_cover_scenario in land_cover_scenarios:
                        for location in locations:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = f"{land_cover_scenario}"
                                    dfs_mlms.append(dta)

                    df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                    df_mlm = sm.add_constant(df_mlm)
                    # fit a mixed linear model with random effects (random intercepts and random slopes)
                    mlm = smf.mixedlm(f'{delta} {mlm_formulas[mlm_key]}', df_mlm, groups=df_mlm["group"], re_formula=mlm_formulas[mlm_key])
                    res = mlm.fit()
                    print(res.summary())

                    # response variable
                    y = df_mlm.loc[:, f"{delta}"].values
                    # number of data points
                    nobs = res.nobs
                    # predicted values
                    # yhat = res.predict()
                    yhat = res.fittedvalues
                    # fixed effect parameters and random effect parameters
                    fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                    re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                    params = onp.concatenate((fe_params, re_params), axis=None)
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "fixed_effects"
                    ] = res.params.to_frame()
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "random_effects"
                    ] = pd.DataFrame(res.random_effects).var(axis=1)
                    # relative importance of parameters
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                        "params"
                    ] = params / onp.sum(params)
                    # contribution of the fixed effect parameters and random effect parameters
                    dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                    dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                    dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["attribution"] = dfa
                    # evaluation metrics
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["MAE"] = smem.meanabs(
                        y, yhat
                    )
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][soil_depth][
                        "MEAE"
                    ] = smem.medianabs(y, yhat)
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["RMSE"] = smem.rmse(
                        y, yhat
                    )
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["VARE"] = smem.vare(
                        y, yhat
                    )
                    dict_mlm_random_slopes_land_cover_scenario[mlm_key][var_sim][delta][soil_depth]["R2"] = r2_score(
                        y, yhat
                    )

    # Store the data (serialize)
    with open(mlm_random_slopes_land_cover_scenario_file, "wb") as handle:
        pickle.dump(dict_mlm_random_slopes_land_cover_scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_slopes_land_cover_scenario_file, "rb") as handle:
        dict_mlm_random_slopes_land_cover_scenario = pickle.load(handle)


# group by soil properties within land cover scenarios
if not os.path.exists(mlm_random_intercepts_soil_land_cover_scenario_file):
    dict_mlm_random_intercepts_soil_land_cover_scenario = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta] = {}
                for land_cover_scenario in land_cover_scenarios:
                    dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario] = {}
                    for soil_depth in soil_depths:
                        dfs_mlms = []
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth] = {}
                        cond = _soil_depths[soil_depth]
                        # merge the data for each land cover senario into a single dataframe
                        for location in locations:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = range(len(y.values.flatten()))
                                    dfs_mlms.append(dta)

                        df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                        df_mlm = sm.add_constant(df_mlm)
                        # fit a mixed linear model with random effects (random intercepts)
                        mlm = smf.mixedlm(f"{delta} {mlm_formulas[mlm_key]}", df_mlm, groups=df_mlm["group"])
                        res = mlm.fit()
                        print(res.summary())

                        # response variable
                        y = df_mlm.loc[:, f"{delta}"].values
                        # number of data points
                        nobs = res.nobs
                        # predicted values
                        # yhat = res.predict()
                        yhat = res.fittedvalues
                        # fixed effect parameters and random effect parameters
                        fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                        re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                        params = onp.concatenate((fe_params, re_params), axis=None)
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "fixed_effects"
                        ] = res.params.to_frame()
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "random_effects"
                        ] = pd.DataFrame(res.random_effects).var(axis=1)
                        # relative importance of parameters
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "params"
                        ] = params / onp.sum(params)
                        # contribution of the fixed effect parameters and random effect parameters
                        dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                        dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                        dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["attribution"] = dfa
                        # evaluation metrics
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["MAE"] = smem.meanabs(
                            y, yhat
                        )
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth][
                            "MEAE"
                        ] = smem.medianabs(y, yhat)
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["RMSE"] = smem.rmse(
                            y, yhat
                        )
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["VARE"] = smem.vare(
                            y, yhat
                        )
                        dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["R2"] = r2_score(
                            y, yhat
                        )

    # Store the data (serialize)
    with open(mlm_random_intercepts_soil_land_cover_scenario_file, "wb") as handle:
        pickle.dump(dict_mlm_random_intercepts_soil_land_cover_scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_intercepts_soil_land_cover_scenario_file, "rb") as handle:
        dict_mlm_random_intercepts_soil_land_cover_scenario = pickle.load(handle)


if not os.path.exists(mlm_random_slopes_soil_land_cover_scenario_file):
    dict_mlm_random_slopes_soil_land_cover_scenario = {}
    for mlm_key in mlm_formulas.keys():
        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key] = {}
        for var_sim in vars_sim:
            dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim] = {}
            for delta in deltas:
                dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta] = {}
                for land_cover_scenario in land_cover_scenarios:
                    dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][land_cover_scenario][delta] = {}
                    for soil_depth in soil_depths:
                        dfs_mlms = []
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth] = {}
                        cond = _soil_depths[soil_depth]
                        # merge the data for each land cover senario into a single dataframe
                        for location in locations:
                            for climate_scenario in climate_scenarios:
                                for future in ["nf", "ff"]:
                                    y = (
                                        dict_delta_changes[location][land_cover_scenario][climate_scenario][var_sim]
                                        .loc[cond, f"{delta}_{future}"]
                                        .to_frame()
                                    )
                                    y = pd.DataFrame(data=y, dtype=onp.float64)
                                    x = df_params.loc[cond, "theta_pwp":]
                                    # standardize the parameters
                                    x_std = x.copy()
                                    for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                        x_std.loc[:, param] = (
                                            x_std.loc[:, param] - x_std.loc[:, param].mean()
                                        ) / x_std.loc[:, param].std()

                                    dta = x_std.copy()
                                    dta.loc[:, f"{delta}"] = y.values.flatten()
                                    dta.loc[:, "group"] = range(len(y.values.flatten()))
                                    dfs_mlms.append(dta)

                        df_mlm = pd.concat(dfs_mlms, ignore_index=True)
                        df_mlm = sm.add_constant(df_mlm)
                        # fit a mixed linear model with random effects (random intercepts and random slopes)
                        mlm = smf.mixedlm(f'{delta} {mlm_formulas[mlm_key]}', df_mlm, groups=df_mlm["group"], re_formula=mlm_formulas[mlm_key])
                        res = mlm.fit()
                        print(res.summary())

                        # response variable
                        y = df_mlm.loc[:, f"{delta}"].values
                        # number of data points
                        nobs = res.nobs
                        # predicted values
                        # yhat = res.predict()
                        yhat = res.fittedvalues
                        # fixed effect parameters and random effect parameters
                        fe_params = onp.abs(res.params.to_frame().T.values.flatten())
                        re_params = onp.abs(pd.DataFrame(res.random_effects).var(axis=1).T.values.flatten())
                        params = onp.concatenate((fe_params, re_params), axis=None)
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "fixed_effects"
                        ] = res.params.to_frame()
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "random_effects"
                        ] = pd.DataFrame(res.random_effects).var(axis=1)
                        # relative importance of parameters
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][
                            "params"
                        ] = params / onp.sum(params)
                        # contribution of the fixed effect parameters and random effect parameters
                        dfa = pd.DataFrame(index=[0], columns=["fixed_effects", "random_effects"], dtype=onp.float64)
                        dfa.iloc[0, 0] = onp.sum(fe_params) / onp.sum(params)
                        dfa.iloc[0, 1] = onp.sum(re_params) / onp.sum(params)
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["attribution"] = dfa
                        # evaluation metrics
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["MAE"] = smem.meanabs(
                            y, yhat
                        )
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth][
                            "MEAE"
                        ] = smem.medianabs(y, yhat)
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["RMSE"] = smem.rmse(
                            y, yhat
                        )
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["VARE"] = smem.vare(
                            y, yhat
                        )
                        dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta][land_cover_scenario][soil_depth]["R2"] = r2_score(
                            y, yhat
                        )

    # Store the data (serialize)
    with open(mlm_random_slopes_soil_land_cover_scenario_file, "wb") as handle:
        pickle.dump(dict_mlm_random_slopes_soil_land_cover_scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    # Load the data (deserialize)
    with open(mlm_random_slopes_soil_land_cover_scenario_file, "rb") as handle:
        dict_mlm_random_slopes_soil_land_cover_scenario = pickle.load(handle)

# # bar plot of GLM parameters
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
#                     values_nf = dict_glm[glm_key][location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][
#                         delta
#                     ][soil_depth]["nf"]["params"].values.flatten()[1:]
#                     values_ff = dict_glm[glm_key][location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][
#                         delta
#                     ][soil_depth]["ff"]["params"].values.flatten()[1:]
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

#                     values_nf = dict_glm[glm_key][location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][
#                         delta
#                     ][soil_depth]["nf"]["params"].values.flatten()[1:]
#                     values_ff = dict_glm[glm_key][location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][
#                         delta
#                     ][soil_depth]["ff"]["params"].values.flatten()[1:]
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
#                         errorbar=None,
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
#                     values_nf = dict_glm[glm_key][location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][
#                         delta
#                     ][soil_depth]["nf"]["params"].values.flatten()[1:]
#                     values_ff = dict_glm[glm_key][location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][
#                         delta
#                     ][soil_depth]["ff"]["params"].values.flatten()[1:]
#                     df_params_canesm = pd.DataFrame(index=range(8), columns=["value", "Parameter"])
#                     values = []
#                     for ii in range(8):
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
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ (NF)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ (FF)",
#                         r"$\theta_{pwp}$ x $\theta_{ac}$ (NF)",
#                         r"$\theta_{pwp}$ x $\theta_{ac}$ (FF)",
#                         r"$\theta_{ufc}$ x $\theta_{ac}$ (NF)",
#                         r"$\theta_{ufc}$ x $\theta_{ac}$ (FF)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (NF)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (FF)",
#                         r"$k_s$ (NF)",
#                         r"$k_s$ (FF)",
#                     ]
#                     df_params_canesm.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"

#                     values_nf = dict_glm[glm_key][location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][
#                         delta
#                     ][soil_depth]["nf"]["params"].values.flatten()[1:]
#                     values_ff = dict_glm[glm_key][location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][
#                         delta
#                     ][soil_depth]["ff"]["params"].values.flatten()[1:]
#                     df_params_mpiesm = pd.DataFrame(index=range(4), columns=["value", "Parameter"])
#                     values = []
#                     for ii in range(2):
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
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ (NF)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ (FF)",
#                         r"$\theta_{pwp}$ x $\theta_{ac}$ (NF)",
#                         r"$\theta_{pwp}$ x $\theta_{ac}$ (FF)",
#                         r"$\theta_{ufc}$ x $\theta_{ac}$ (NF)",
#                         r"$\theta_{ufc}$ x $\theta_{ac}$ (FF)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (NF)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (FF)",
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
#                         errorbar=None,
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
#         for future in ["nf", "ff"]:
#             fig, axes = plt.subplots(
#                 len(locations), len(land_cover_scenarios), sharex="col", sharey="row", figsize=(6, 4.5)
#             )
#             # axes for colorbar
#             axl = fig.add_axes([0.88, 0.3, 0.02, 0.5])
#             cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical", ticks=[0, 1, 2])
#             cb1.set_label(r"[-]")
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     params_cm = [
#                         r"$\theta_{pwp}$ (CCC)",
#                         r"$\theta_{pwp}$ (MPI)",
#                         r"$\theta_{ufc}$ (CCC)",
#                         r"$\theta_{ufc}$ (MPI)",
#                         r"$\theta_{ac}$ (CCC)",
#                         r"$\theta_{ac}$ (MPI)",
#                         r"$k_s$ (CCC)",
#                         r"$k_s$ (MPI)",
#                     ]
#                     params_canesm = [
#                         r"$\theta_{pwp}$ (CCC)",
#                         r"$\theta_{ufc}$ (CCC)",
#                         r"$\theta_{ac}$ (CCC)",
#                         r"$k_s$ (CCC)",
#                     ]
#                     params_mpim = [
#                         r"$\theta_{pwp}$ (MPI)",
#                         r"$\theta_{ufc}$ (MPI)",
#                         r"$\theta_{ac}$ (MPI)",
#                         r"$k_s$ (MPI)",
#                     ]
#                     df_params = pd.DataFrame(columns=params_cm)
#                     for ii, var_sim in enumerate(vars_sim):
#                         df_params.loc[f"{_lab[delta]}{_lab[var_sim]}", params_canesm] = dict_glm[glm_key][location][
#                             land_cover_scenario
#                         ]["CCCma-CanESM2_CCLM4-8-17"][var_sim][delta][soil_depth][future]["params"].values.flatten()[1:]
#                         df_params.loc[f"{_lab[delta]}{_lab[var_sim]}", params_mpim] = dict_glm[glm_key][location][
#                             land_cover_scenario
#                         ]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][delta][soil_depth][future]["params"].values.flatten()[1:]
#                     df_params = df_params.astype("float32")
#                     df_params.iloc[:, :] = onp.abs(df_params.values)
#                     sns.heatmap(df_params, vmin=0, vmax=1, cmap="Oranges", cbar=False, ax=axes[i, j], square=True)
#                     axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
#                     axes[-1, j].set_xticklabels(params_cm, rotation=90)
#                     axes[i, j].set_ylabel("")
#                     axes[i, j].set_xlabel("")
#                 axes[i, 0].set_ylabel(f"{Locations[i]}")
#                 axes[i, 0].set_yticklabels(df_params.index.tolist(), rotation=0)
#             fig.subplots_adjust(bottom=0.2, left=0.2, right=0.85, top=0.9, hspace=-0.5, wspace=0.1)
#             file = base_path_figs / "glm_parameters" / f"{glm_key}_{delta}_{soil_depth}_{future}_heatmap.png"
#             fig.savefig(file, dpi=300)
#             plt.close("all")

# glm_key = "interaction_theta"
# vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
# deltas = ["dAvg"]
# norm = mpl.colors.Normalize(vmin=0, vmax=1)
# for delta in deltas:
#     for soil_depth in soil_depths:
#         cond = _soil_depths[soil_depth]
#         for future in ["nf", "ff"]:
#             fig, axes = plt.subplots(
#                 len(locations), len(land_cover_scenarios), sharex="col", sharey="row", figsize=(6, 4.5)
#             )
#             # axes for colorbar
#             axl = fig.add_axes([0.88, 0.3, 0.02, 0.5])
#             cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical", ticks=[0, 1, 2])
#             cb1.set_label(r"[-]")
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     params_cm = [
#                         r"$\theta_{pwp}$ (CCC)",
#                         r"$\theta_{pwp}$ (MPI)",
#                         r"$\theta_{ufc}$ (CCC)",
#                         r"$\theta_{ufc}$ (MPI)",
#                         r"$\theta_{ac}$ (CCC)",
#                         r"$\theta_{ac}$ (MPI)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ (CCC)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ (MPI)",
#                         r"$\theta_{pwp}$ x $\theta_{ac}$ (CCC)",
#                         r"$\theta_{pwp}$ x $\theta_{ac}$ (MPI)",
#                         r"$\theta_{ufc}$ x $\theta_{ac}$ (CCC)",
#                         r"$\theta_{ufc}$ x $\theta_{ac}$ (MPI)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (CCC)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (MPI)",
#                         r"$k_s$ (CCC)",
#                         r"$k_s$ (MPI)",
#                     ]
#                     params_canesm = [
#                         r"$\theta_{pwp}$ (CCC)",
#                         r"$\theta_{ufc}$ (CCC)",
#                         r"$\theta_{ac}$ (CCC)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ (CCC)",
#                         r"$\theta_{pwp}$ x $\theta_{ac}$ (CCC)",
#                         r"$\theta_{ufc}$ x $\theta_{ac}$ (CCC)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (CCC)",
#                         r"$k_s$ (CCC)",
#                     ]
#                     params_mpim = [
#                         r"$\theta_{pwp}$ (MPI)",
#                         r"$\theta_{ufc}$ (MPI)",
#                         r"$\theta_{ac}$ (MPI)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ (MPI)",
#                         r"$\theta_{pwp}$ x $\theta_{ac}$ (MPI)",
#                         r"$\theta_{ufc}$ x $\theta_{ac}$ (MPI)",
#                         r"$\theta_{pwp}$ x $\theta_{ufc}$ x $\theta_{ac}$ (MPI)",
#                         r"$k_s$ (MPI)",
#                     ]
#                     df_params = pd.DataFrame(columns=params_cm)
#                     for ii, var_sim in enumerate(vars_sim):
#                         print(
#                             dict_glm[glm_key][location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][
#                                 delta
#                             ][soil_depth][future]["params"].values[1:]
#                         )
#                         df_params.loc[f"{_lab[delta]}{_lab[var_sim]}", params_canesm] = dict_glm[glm_key][location][
#                             land_cover_scenario
#                         ]["CCCma-CanESM2_CCLM4-8-17"][var_sim][delta][soil_depth][future]["params"].values.flatten()[1:]
#                         df_params.loc[f"{_lab[delta]}{_lab[var_sim]}", params_mpim] = dict_glm[glm_key][location][
#                             land_cover_scenario
#                         ]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][delta][soil_depth][future]["params"].values.flatten()[1:]
#                     df_params = df_params.astype("float32")
#                     df_params.iloc[:, :] = onp.abs(df_params.values)
#                     sns.heatmap(df_params, vmin=0, vmax=1, cmap="Oranges", cbar=False, ax=axes[i, j], square=True)
#                     axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
#                     axes[-1, j].set_xticklabels(params_cm, rotation=90)
#                     axes[i, j].set_ylabel("")
#                     axes[i, j].set_xlabel("")
#                 axes[i, 0].set_ylabel(f"{Locations[i]}")
#                 axes[i, 0].set_yticklabels(df_params.index.tolist(), rotation=0)
#             fig.subplots_adjust(bottom=0.2, left=0.2, right=0.85, top=0.9, hspace=-0.5, wspace=0.1)
#             file = base_path_figs / "glm_parameters" / f"{glm_key}_{delta}_{soil_depth}_{future}_heatmap.png"
#             fig.savefig(file, dpi=300)
#             plt.close("all")

# # heatmap the root mean absolute error of GLMs
# metrics = ["MAE"]
# vars_sim = ["transp", "q_ss", "theta", "tt50_transp", "tt50_q_ss", "rt50_s"]
# deltas = ["dAvg", "dIPR"]
# norm = mpl.colors.Normalize(vmin=0, vmax=30)
# for glm_key in glm_location_formulas.keys():
#     for metric in metrics:
#         for delta in deltas:
#             for soil_depth in soil_depths:
#                 cond = _soil_depths[soil_depth]
#                 fig, axes = plt.subplots(
#                     len(locations), len(land_cover_scenarios), sharex="col", sharey="row", figsize=(6, 4.5)
#                 )
#                 # axes for colorbar
#                 axl = fig.add_axes([0.88, 0.32, 0.02, 0.46])
#                 cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
#                 cb1.set_label(f"{_lab[metric]} [%]")
#                 for i, location in enumerate(locations):
#                     for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                         df_metric = pd.DataFrame()
#                         for var_sim in vars_sim:
#                             for climate_scenario in climate_scenarios:
#                                 for future in ["nf", "ff"]:
#                                     value = dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][
#                                         delta
#                                     ][soil_depth][future][metric]
#                                     df_metric.loc[
#                                         f"{_lab[climate_scenario]} ({_lab[future]})", f"{_lab[var_sim]}"
#                                     ] = value
#                         sns.heatmap(df_metric, vmin=0, vmax=30, cmap="Oranges", cbar=False, ax=axes[i, j], square=True)
#                         axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
#                         axes[-1, j].set_xticklabels(df_metric.columns, rotation=90)
#                         axes[i, j].set_ylabel("")
#                         axes[i, j].set_xlabel("")
#                     axes[i, 0].set_ylabel(f"{Locations[i]}")
#                     axes[i, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
#                 fig.subplots_adjust(bottom=0.2, left=0.2, right=0.85, top=0.9, hspace=-0.5, wspace=0.1)
#                 file = base_path_figs / "glm_metrics" / f"{glm_key}_{metric}_{delta}_{soil_depth}_heatmap.png"
#                 fig.savefig(file, dpi=300)
#                 plt.close("all")

# # heatmap the variance of error of GLMs
# metrics = ["R2"]
# vars_sim = ["transp", "q_ss", "theta", "tt50_transp", "tt50_q_ss", "rt50_s"]
# deltas = ["dAvg", "dIPR"]
# norm = mpl.colors.Normalize(vmin=0, vmax=1)
# for glm_key in glm_location_formulas.keys():
#     for metric in metrics:
#         for delta in deltas:
#             for soil_depth in soil_depths:
#                 cond = _soil_depths[soil_depth]
#                 fig, axes = plt.subplots(
#                     len(locations), len(land_cover_scenarios), sharex="col", sharey="row", figsize=(6, 4.5)
#                 )
#                 # axes for colorbar
#                 axl = fig.add_axes([0.88, 0.32, 0.02, 0.46])
#                 cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
#                 cb1.set_label(f"{metric} [-]")
#                 for i, location in enumerate(locations):
#                     for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                         df_metric = pd.DataFrame()
#                         for var_sim in vars_sim:
#                             for climate_scenario in climate_scenarios:
#                                 for future in ["nf", "ff"]:
#                                     value = dict_glm[glm_key][location][land_cover_scenario][climate_scenario][var_sim][
#                                         delta
#                                     ][soil_depth][future][metric]
#                                     df_metric.loc[
#                                         f"{_lab[climate_scenario]} ({_lab[future]})", f"{_lab[var_sim]}"
#                                     ] = value
#                         sns.heatmap(df_metric, vmin=0, vmax=1, cmap="Oranges", cbar=False, ax=axes[i, j], square=True)
#                         axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
#                         axes[-1, j].set_xticklabels(df_metric.columns, rotation=90)
#                         axes[i, j].set_ylabel("")
#                         axes[i, j].set_xlabel("")
#                     axes[i, 0].set_ylabel(f"{Locations[i]}")
#                     axes[i, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
#                 fig.subplots_adjust(bottom=0.2, left=0.2, right=0.85, top=0.9, hspace=-0.5, wspace=0.1)
#                 file = base_path_figs / "glm_metrics" / f"{glm_key}_{metric}_{delta}_{soil_depth}_heatmap.png"
#                 fig.savefig(file, dpi=300)
#                 plt.close("all")

# heatmap the root mean absolute error of MLMs
metrics = ["MAE"]
vars_sim = ["transp", "q_ss", "theta", "tt50_transp", "tt50_q_ss", "rt50_s"]
deltas = ["dAvg", "dIPR"]
soil_depths = ["shallow", "medium", "deep"]
norm = mpl.colors.Normalize(vmin=0, vmax=50)
for mlm_key in mlm_formulas.keys():
    for metric in metrics:
        fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(6, 4.5))
        for ii, delta in enumerate(deltas):
            for jj, soil_depth in enumerate(soil_depths):
                # axes for colorbar
                axl = fig.add_axes([0.84, 0.32, 0.02, 0.46])
                cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
                cb1.set_label(f"{_lab[metric]} [-]")
                df_metric = pd.DataFrame()
                for var_sim in vars_sim:
                    for iloc, location in enumerate(locations):
                        for ilsc, land_cover_scenario in enumerate(land_cover_scenarios):
                            for ics, climate_scenario in enumerate(climate_scenarios):
                                value = dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][metric]
                                df_metric.loc[f"{iloc}{ilsc}{ics}", f"{_lab[var_sim]}"] = value
                sns.heatmap(df_metric, vmin=0, vmax=50, cmap="Oranges", cbar=False, ax=axes[jj, ii], square=True)
                axes[jj, ii].set_ylabel("")
                axes[jj, ii].set_xlabel("")
        axes[-1, 0].set_xticklabels(df_metric.columns, rotation=90)
        axes[-1, 1].set_xticklabels(df_metric.columns, rotation=90)
        axes[0, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[1, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[2, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[0, 0].set_title("(a)", loc="left", fontsize=9)
        axes[1, 0].set_title("(b)", loc="left", fontsize=9)
        axes[2, 0].set_title("(c)", loc="left", fontsize=9)
        axes[0, 1].set_title("(d)", loc="left", fontsize=9)
        axes[1, 1].set_title("(e)", loc="left", fontsize=9)
        axes[2, 1].set_title("(f)", loc="left", fontsize=9)
        fig.subplots_adjust(bottom=0.2, left=0.0, right=1.15, top=0.9, hspace=0.3, wspace=-0.7)
        file = base_path_figs / "mlm_metrics" / f"mlms_random_intercepts_{mlm_key}_{metric}_heatmap.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

for mlm_key in mlm_formulas.keys():
    for metric in metrics:
        fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(6, 4.5))
        for ii, delta in enumerate(deltas):
            for jj, soil_depth in enumerate(soil_depths):
                # axes for colorbar
                axl = fig.add_axes([0.84, 0.32, 0.02, 0.46])
                cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
                cb1.set_label(f"{_lab[metric]} [-]")
                df_metric = pd.DataFrame()
                for var_sim in vars_sim:
                    for iloc, location in enumerate(locations):
                        for ilsc, land_cover_scenario in enumerate(land_cover_scenarios):
                            for ics, climate_scenario in enumerate(climate_scenarios):
                                value = dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][metric]
                                df_metric.loc[f"{iloc}{ilsc}{ics}", f"{_lab[var_sim]}"] = value
                sns.heatmap(df_metric, vmin=0, vmax=50, cmap="Oranges", cbar=False, ax=axes[jj, ii], square=True)
                axes[jj, ii].set_ylabel("")
                axes[jj, ii].set_xlabel("")
        axes[-1, 0].set_xticklabels(df_metric.columns, rotation=90)
        axes[-1, 1].set_xticklabels(df_metric.columns, rotation=90)
        axes[0, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[1, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[2, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[0, 0].set_title("(a)", loc="left", fontsize=9)
        axes[1, 0].set_title("(b)", loc="left", fontsize=9)
        axes[2, 0].set_title("(c)", loc="left", fontsize=9)
        axes[0, 1].set_title("(d)", loc="left", fontsize=9)
        axes[1, 1].set_title("(e)", loc="left", fontsize=9)
        axes[2, 1].set_title("(f)", loc="left", fontsize=9)
        fig.subplots_adjust(bottom=0.2, left=0.0, right=1.15, top=0.9, hspace=0.3, wspace=-0.7)
        file = base_path_figs / "mlm_metrics" / f"mlms_random_slopes_{mlm_key}_{metric}_heatmap.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

for mlm_key in mlm_formulas.keys():
    for metric in metrics:
        fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(6, 4.5))
        for ii, delta in enumerate(deltas):
            for jj, soil_depth in enumerate(soil_depths):
                # axes for colorbar
                axl = fig.add_axes([0.84, 0.32, 0.02, 0.46])
                cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
                cb1.set_label(f"{_lab[metric]} [-]")
                df_metric = pd.DataFrame()
                for var_sim in vars_sim:
                    value = dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][soil_depth][metric]
                    df_metric.loc["Random intercepts", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta]["grass"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Grass)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta]["corn"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Corn)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta]["corn_catch_crop"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Corn & Catch crop)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta]["crop_rotation"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Crop rotation)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta]["freiburg"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Freiburg)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta]["altheim"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Altheim)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta]["kupferzell"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Kupferzell)", f"{_lab[var_sim]}"] = value
                sns.heatmap(df_metric, vmin=0, vmax=50, cmap="Oranges", cbar=False, ax=axes[jj, ii], square=True)
                axes[jj, ii].set_ylabel("")
                axes[jj, ii].set_xlabel("")
        axes[-1, 0].set_xticklabels(df_metric.columns, rotation=90)
        axes[-1, 1].set_xticklabels(df_metric.columns, rotation=90)
        axes[0, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[1, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[2, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[0, 0].set_title("(a)", loc="left", fontsize=9)
        axes[1, 0].set_title("(b)", loc="left", fontsize=9)
        axes[2, 0].set_title("(c)", loc="left", fontsize=9)
        axes[0, 1].set_title("(d)", loc="left", fontsize=9)
        axes[1, 1].set_title("(e)", loc="left", fontsize=9)
        axes[2, 1].set_title("(f)", loc="left", fontsize=9)
        fig.subplots_adjust(bottom=0.2, left=0.0, right=1.15, top=0.9, hspace=0.3, wspace=-0.7)
        file = base_path_figs / "mlm_metrics" / f"mlms_random_intercept_{mlm_key}_{metric}_soil_heatmap.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

for mlm_key in mlm_formulas.keys():
    for metric in metrics:
        fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(6, 4.5))
        for ii, delta in enumerate(deltas):
            for jj, soil_depth in enumerate(soil_depths):
                # axes for colorbar
                axl = fig.add_axes([0.84, 0.32, 0.02, 0.46])
                cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
                cb1.set_label(f"{_lab[metric]} [-]")
                df_metric = pd.DataFrame()
                for var_sim in vars_sim:
                    value = dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][soil_depth][metric]
                    df_metric.loc["Random slopes", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta]["grass"][soil_depth][metric]
                    df_metric.loc["Random slopes (Grass)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta]["corn"][soil_depth][metric]
                    df_metric.loc["Random slopes (Corn)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta]["corn_catch_crop"][soil_depth][metric]
                    df_metric.loc["Random slopes (Corn & Catch crop)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta]["crop_rotation"][soil_depth][metric]
                    df_metric.loc["Random slopes (Crop rotation)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta]["freiburg"][soil_depth][metric]
                    df_metric.loc["Random slopes (Freiburg)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta]["altheim"][soil_depth][metric]
                    df_metric.loc["Random slopes (Altheim)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta]["kupferzell"][soil_depth][metric]
                    df_metric.loc["Random slopes (Kupferzell)", f"{_lab[var_sim]}"] = value
                sns.heatmap(df_metric, vmin=0, vmax=50, cmap="Oranges", cbar=False, ax=axes[jj, ii], square=True)
                axes[jj, ii].set_ylabel("")
                axes[jj, ii].set_xlabel("")
        axes[-1, 0].set_xticklabels(df_metric.columns, rotation=90)
        axes[-1, 1].set_xticklabels(df_metric.columns, rotation=90)
        axes[0, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[1, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[2, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[0, 0].set_title("(a)", loc="left", fontsize=9)
        axes[1, 0].set_title("(b)", loc="left", fontsize=9)
        axes[2, 0].set_title("(c)", loc="left", fontsize=9)
        axes[0, 1].set_title("(d)", loc="left", fontsize=9)
        axes[1, 1].set_title("(e)", loc="left", fontsize=9)
        axes[2, 1].set_title("(f)", loc="left", fontsize=9)
        fig.subplots_adjust(bottom=0.2, left=0.0, right=1.15, top=0.9, hspace=0.3, wspace=-0.7)
        file = base_path_figs / "mlm_metrics" / f"mlms_random_slopes_{mlm_key}_{metric}_soil_heatmap.png"
        fig.savefig(file, dpi=300)
        plt.close("all")



# for metric in metrics:
#     for delta in deltas:
#         for soil_depth in soil_depths:
#             fig, axes = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(6, 4.5))
#             # axes for colorbar
#             axl = fig.add_axes([0.88, 0.32, 0.02, 0.46])
#             cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
#             cb1.set_label(f"{_lab[metric]} [%]")
#             for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                 df_metric = pd.DataFrame()
#                 for var_sim in vars_sim:
#                     for mlm_key in mlm_formulas.keys():
#                         value = dict_mlm_random_intercepts[mlm_key][var_sim][delta][land_cover_scenario][soil_depth][metric]
#                         df_metric.loc[f"{mlm_key}", f"{_lab[var_sim]}"] = value
#                 sns.heatmap(df_metric, vmin=0, vmax=50, cmap="Oranges", cbar=False, ax=axes.flatten()[j], square=True)
#                 axes.flatten()[j].set_title(f"{Land_cover_scenarios[j]}")
#                 axes.flatten()[j].set_ylabel("")
#                 axes.flatten()[j].set_xlabel("")
#             axes.flatten()[2].set_xticklabels(df_metric.columns, rotation=90)
#             axes.flatten()[3].set_xticklabels(df_metric.columns, rotation=90)
#             axes.flatten()[0].set_yticklabels(df_metric.index.tolist(), rotation=0)
#             axes.flatten()[2].set_yticklabels(df_metric.index.tolist(), rotation=0)
#             fig.subplots_adjust(bottom=0.2, left=0.05, right=1.1, top=0.9, hspace=0.25, wspace=-0.7)
#             file = base_path_figs / "mlm_metrics" / f"mlms_{metric}_{delta}_{soil_depth}_heatmap_for_soil.png"
#             fig.savefig(file, dpi=300)
#             plt.close("all")

# heatmap the coefficient of determination of MLMs
metrics = ["R2"]
vars_sim = ["transp", "q_ss", "theta", "tt50_transp", "tt50_q_ss", "rt50_s"]
deltas = ["dAvg", "dIPR"]
soil_depths = ["shallow", "medium", "deep"]
norm = mpl.colors.Normalize(vmin=0, vmax=1)
for mlm_key in mlm_formulas.keys():
    for metric in metrics:
        fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(6, 4.5))
        for ii, delta in enumerate(deltas):
            for jj, soil_depth in enumerate(soil_depths):
                # axes for colorbar
                axl = fig.add_axes([0.84, 0.32, 0.02, 0.46])
                cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
                cb1.set_label(f"{_lab[metric]} [-]")
                df_metric = pd.DataFrame()
                for var_sim in vars_sim:
                    for iloc, location in enumerate(locations):
                        for ilsc, land_cover_scenario in enumerate(land_cover_scenarios):
                            for ics, climate_scenario in enumerate(climate_scenarios):
                                value = dict_mlm_random_intercepts[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][metric]
                                df_metric.loc[f"{iloc}{ilsc}{ics}", f"{_lab[var_sim]}"] = value
                sns.heatmap(df_metric, vmin=0, vmax=1, cmap="Oranges", cbar=False, ax=axes[jj, ii], square=True)
                axes[jj, ii].set_ylabel("")
                axes[jj, ii].set_xlabel("")
        axes[-1, 0].set_xticklabels(df_metric.columns, rotation=90)
        axes[-1, 1].set_xticklabels(df_metric.columns, rotation=90)
        axes[0, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[1, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[2, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[0, 0].set_title("(a)", loc="left", fontsize=9)
        axes[1, 0].set_title("(b)", loc="left", fontsize=9)
        axes[2, 0].set_title("(c)", loc="left", fontsize=9)
        axes[0, 1].set_title("(d)", loc="left", fontsize=9)
        axes[1, 1].set_title("(e)", loc="left", fontsize=9)
        axes[2, 1].set_title("(f)", loc="left", fontsize=9)
        fig.subplots_adjust(bottom=0.2, left=0.0, right=1.15, top=0.9, hspace=0.3, wspace=-0.7)
        file = base_path_figs / "mlm_metrics" / f"mlms_random_intercepts_{mlm_key}_{metric}_heatmap.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

for mlm_key in mlm_formulas.keys():
    for metric in metrics:
        fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(6, 4.5))
        for ii, delta in enumerate(deltas):
            for jj, soil_depth in enumerate(soil_depths):
                # axes for colorbar
                axl = fig.add_axes([0.84, 0.32, 0.02, 0.46])
                cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
                cb1.set_label(f"{_lab[metric]} [-]")
                df_metric = pd.DataFrame()
                for var_sim in vars_sim:
                    for iloc, location in enumerate(locations):
                        for ilsc, land_cover_scenario in enumerate(land_cover_scenarios):
                            for ics, climate_scenario in enumerate(climate_scenarios):
                                value = dict_mlm_random_slopes[mlm_key][var_sim][delta][location][land_cover_scenario][climate_scenario][future][soil_depth][metric]
                                df_metric.loc[f"{iloc}{ilsc}{ics}", f"{_lab[var_sim]}"] = value
                sns.heatmap(df_metric, vmin=0, vmax=1, cmap="Oranges", cbar=False, ax=axes[jj, ii], square=True)
                axes[jj, ii].set_ylabel("")
                axes[jj, ii].set_xlabel("")
        axes[-1, 0].set_xticklabels(df_metric.columns, rotation=90)
        axes[-1, 1].set_xticklabels(df_metric.columns, rotation=90)
        axes[0, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[1, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[2, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[0, 0].set_title("(a)", loc="left", fontsize=9)
        axes[1, 0].set_title("(b)", loc="left", fontsize=9)
        axes[2, 0].set_title("(c)", loc="left", fontsize=9)
        axes[0, 1].set_title("(d)", loc="left", fontsize=9)
        axes[1, 1].set_title("(e)", loc="left", fontsize=9)
        axes[2, 1].set_title("(f)", loc="left", fontsize=9)
        fig.subplots_adjust(bottom=0.2, left=0.0, right=1.15, top=0.9, hspace=0.3, wspace=-0.7)
        file = base_path_figs / "mlm_metrics" / f"mlms_random_slopes_{mlm_key}_{metric}_heatmap.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

for mlm_key in mlm_formulas.keys():
    for metric in metrics:
        fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(6, 4.5))
        for ii, delta in enumerate(deltas):
            for jj, soil_depth in enumerate(soil_depths):
                # axes for colorbar
                axl = fig.add_axes([0.84, 0.32, 0.02, 0.46])
                cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
                cb1.set_label(f"{_lab[metric]} [-]")
                df_metric = pd.DataFrame()
                for var_sim in vars_sim:
                    value = dict_mlm_random_intercepts_soil_llcp[mlm_key][var_sim][delta][soil_depth][metric]
                    df_metric.loc["Random intercepts", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta]["grass"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Grass)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta]["corn"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Corn)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta]["corn_catch_crop"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Corn & Catch crop)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_land_cover_scenario[mlm_key][var_sim][delta]["crop_rotation"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Crop rotation)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta]["freiburg"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Freiburg)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta]["altheim"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Altheim)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_intercepts_soil_location[mlm_key][var_sim][delta]["kupferzell"][soil_depth][metric]
                    df_metric.loc["Random intercepts (Kupferzell)", f"{_lab[var_sim]}"] = value
                sns.heatmap(df_metric, vmin=0, vmax=1, cmap="Oranges", cbar=False, ax=axes[jj, ii], square=True)
                axes[jj, ii].set_ylabel("")
                axes[jj, ii].set_xlabel("")
        axes[-1, 0].set_xticklabels(df_metric.columns, rotation=90)
        axes[-1, 1].set_xticklabels(df_metric.columns, rotation=90)
        axes[0, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[1, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[2, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[0, 0].set_title("(a)", loc="left", fontsize=9)
        axes[1, 0].set_title("(b)", loc="left", fontsize=9)
        axes[2, 0].set_title("(c)", loc="left", fontsize=9)
        axes[0, 1].set_title("(d)", loc="left", fontsize=9)
        axes[1, 1].set_title("(e)", loc="left", fontsize=9)
        axes[2, 1].set_title("(f)", loc="left", fontsize=9)
        fig.subplots_adjust(bottom=0.2, left=0.0, right=1.15, top=0.9, hspace=0.3, wspace=-0.7)
        file = base_path_figs / "mlm_metrics" / f"mlms_random_intercepts_{mlm_key}_{metric}_soil_heatmap.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

for mlm_key in mlm_formulas.keys():
    for metric in metrics:
        fig, axes = plt.subplots(3, 2, sharex="col", sharey="row", figsize=(6, 4.5))
        for ii, delta in enumerate(deltas):
            for jj, soil_depth in enumerate(soil_depths):
                # axes for colorbar
                axl = fig.add_axes([0.84, 0.32, 0.02, 0.46])
                cb1 = mpl.colorbar.ColorbarBase(axl, cmap="Oranges", norm=norm, orientation="vertical")
                cb1.set_label(f"{_lab[metric]} [-]")
                df_metric = pd.DataFrame()
                for var_sim in vars_sim:
                    value = dict_mlm_random_slopes_soil_llcp[mlm_key][var_sim][delta][soil_depth][metric]
                    df_metric.loc["Random slopes", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta]["grass"][soil_depth][metric]
                    df_metric.loc["Random slopes (Grass)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta]["corn"][soil_depth][metric]
                    df_metric.loc["Random slopes (Corn)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta]["corn_catch_crop"][soil_depth][metric]
                    df_metric.loc["Random slopes (Corn & Catch crop)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_land_cover_scenario[mlm_key][var_sim][delta]["crop_rotation"][soil_depth][metric]
                    df_metric.loc["Random slopes (Crop rotation)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta]["freiburg"][soil_depth][metric]
                    df_metric.loc["Random slopes (Freiburg)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta]["altheim"][soil_depth][metric]
                    df_metric.loc["Random slopes (Altheim)", f"{_lab[var_sim]}"] = value
                    value = dict_mlm_random_slopes_soil_location[mlm_key][var_sim][delta]["kupferzell"][soil_depth][metric]
                    df_metric.loc["Random slopes (Kupferzell)", f"{_lab[var_sim]}"] = value
                sns.heatmap(df_metric, vmin=0, vmax=1, cmap="Oranges", cbar=False, ax=axes[jj, ii], square=True)
                axes[jj, ii].set_ylabel("")
                axes[jj, ii].set_xlabel("")
        axes[-1, 0].set_xticklabels(df_metric.columns, rotation=90)
        axes[-1, 1].set_xticklabels(df_metric.columns, rotation=90)
        axes[0, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[1, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[2, 0].set_yticklabels(df_metric.index.tolist(), rotation=0)
        axes[0, 0].set_title("(a)", loc="left", fontsize=9)
        axes[1, 0].set_title("(b)", loc="left", fontsize=9)
        axes[2, 0].set_title("(c)", loc="left", fontsize=9)
        axes[0, 1].set_title("(d)", loc="left", fontsize=9)
        axes[1, 1].set_title("(e)", loc="left", fontsize=9)
        axes[2, 1].set_title("(f)", loc="left", fontsize=9)
        fig.subplots_adjust(bottom=0.2, left=0.0, right=1.15, top=0.9, hspace=0.3, wspace=-0.7)
        file = base_path_figs / "mlm_metrics" / f"mlms_random_slopes_{mlm_key}_{metric}_soil_heatmap.png"
        fig.savefig(file, dpi=300)
        plt.close("all")


# # bar plot of fixed effects and random effects
# selected_mlms = [dict_mlm_random_intercepts_soil_llcp, dict_mlm_random_intercepts_soil_location, dict_mlm_random_intercepts_soil_land_cover_scenario]
# vars_sim = ["transp", "q_ss", "theta", "tt50_transp", "tt50_q_ss", "rt50_s"]
# deltas = ["dAvg", "dIPR"]
# mlm_key = "no_interaction"
# for soil_depth in soil_depths:
#     fig, axes = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(6, 6))
#     for j, delta in enumerate(deltas):
#         for i, dict_mlm in enumerate(selected_mlms):
#             ll_dfs = []
#             for var_sim in vars_sim:
#                 for soil_depth in soil_depths:
#                     df = pd.DataFrame(columns=["value", "variable", "effect"], index=range(4))
#                     df_as = dict_mlm[mlm_key][var_sim][delta][soil_depth]["attribution"]
#                     df.loc[0, "value"] = df_as.values.flatten()[0] * 100
#                     df.loc[1, "value"] = df_as.values.flatten()[1] * 100
#                     df.loc[:, "variable"] = f"{_lab[var_sim]}"
#                     df.loc[0, "effect"] = f"Fixed effect ({soil_depth})"
#                     df.loc[1, "effect"] = f"Random effect ({soil_depth})"
#                     ll_dfs.append(df)
#             df_attribution = pd.concat(ll_dfs, ignore_index=True)

#             g = sns.barplot(
#                 x="variable",
#                 y="value",
#                 hue="effect",
#                 palette=["#efedf5", "#fee6ce", "#bcbddc", "#fdae6b", "#756bb1", "#e6550d"],
#                 data=df_attribution,
#                 ax=axes[i, j],
#                 errorbar=None,
#                 width=1.0,
#             )
#             axes[i, j].legend([], [], frameon=False)
#             axes[i, j].set_ylabel("")
#             axes[i, j].set_xlabel("")
#             axes[i, j].set_ylim(0, 100)
#             axes[i, j].tick_params(axis="x", rotation=33)
#     axes[0, 0].set_ylabel("Change attribution [%]")
#     axes[1, 0].set_ylabel("Change attribution [%]")
#     axes[1, 0].set_ylabel("Change attribution [%]")

#     axes[0, 0].set_title("(a)", loc="left", fontsize=9)
#     axes[1, 0].set_title("(b)", loc="left", fontsize=9)
#     axes[2, 0].set_title("(c)", loc="left", fontsize=9)
#     axes[0, 1].set_title("(d)", loc="left", fontsize=9)
#     axes[1, 1].set_title("(e)", loc="left", fontsize=9)
#     axes[2, 1].set_title("(f)", loc="left", fontsize=9)
#     handels, labels = g.get_legend_handles_labels()
#     fig.legend(handels, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False)
#     fig.tight_layout()
#     fig.subplots_adjust(bottom=0.16)
#     file = base_path_figs / "change_attribution" / f"attribution_{mlm_key}_barplot.png"
#     fig.savefig(file, dpi=300)
#     plt.close("all")