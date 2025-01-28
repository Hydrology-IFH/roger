import os
from pathlib import Path
import pandas as pd
import yaml
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as onp
import click
import roger.tools.labels as labs
onp.random.seed(42)

_XLABS = {
        "lmpv": r"$l_{mpv}$",
        "dmpv": r"$\rho_{mpv}$",
        "theta_eff": r"$\theta_{eff}$",
        "frac_lp": r"$f_{lp}$",
        "theta_pwp": r"$\theta_{pwp}$",
        "ks": r"$k_{s}$",
        "c_root": r"$c_{root}$",
        "c_canopy": r"$c_{canopy}$",
        "k_transp": r"$k_{transp}$",
        "k_q": r"$k_{perc}$",
        "alpha_transp": r"$\alpha_{transp}$",
        "alpha_q": r"$\alpha_{perc}$",
        "km_denit": r"$k_{m-denit}$",
        "km_nit": r"$k_{m-nit}$",
        "kmin": r"$k_{min}$",
        "dmax_denit": r"$D_{max-denit}$",
        "dmax_nit": r"$D_{max-nit}$",
}

_UNITS = {
        "lmpv": r"$l_{mpv}$ [mm]",
        "dmpv": r"$\rho_{mpv}$ [1/$m^2$]",
        "theta_eff": r"$\theta_{eff}$ [-]",
        "frac_lp": r"$f_{lp}$ [-]",
        "theta_pwp": r"$\theta_{pwp}$ [-]",
        "ks": r"$k_{s}$ [mm/hour]",
        "c_root": r"$c_{root}$ [-]",
        "c_canopy": r"$c_{canopy}$ [-]",
        "k_transp": r"$k_{transp}$ [-]",
        "k_q": r"$k_{perc}$ [-]",
        "alpha_transp": r"$\alpha_{transp}$ [-]",
        "alpha_q": r"$\alpha_{perc}$ [-]",
        "km_denit": r"$k_{m-denit}$ [-]",
        "km_nit": r"$k_{m-nit}$ [-]",
        "kmin": r"$k_{min}$ [-]",
        "dmax_denit": r"$D_{max-denit}$ [-]",
        "dmax_nit": r"$D_{max-nit}$ [-]",
}


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # sampled parameter space
    file_path = base_path / "param_bounds.yml"
    with open(file_path, "r") as file:
        bounds = yaml.safe_load(file)
        bounds["num_vars"] = len(bounds["names"])

    # directory of results
    base_path_output = base_path.parent / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path.parent / "figures" / "svat_crop_nitrate_sobol"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)


    lys_experiments = ["lys3"]
    for lys_experiment in lys_experiments:
        # write .txt-file
        file = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz/output") / "svat_crop_nitrate_sobol" / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt"
        df_params_metrics = pd.read_csv(file, sep="\t")

        # perform sensitivity analysis
        df_metrics = df_params_metrics.loc[:, ["KGE_NO3_perc_bs", "KGE_NO3_perc_mass_bs", "MAE_N_uptake"]]
        dict_si = {}
        for name in df_metrics.columns:
            Y = df_metrics[name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"]).loc[:'c_canopy', :]

        # plot sobol indices
        _LABS = {"KGE_NO3_perc_bs": r"percolation (KGE$_{conc}$)",
                 "KGE_NO3_perc_mass_bs": r"percolation (KGE$_{mass}$)",
                 "MAE_N_uptake": "N-uptake (MAE)",
                }
        ncol = len(df_metrics.columns)
        xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][:8]]
        cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 3))
        for i, name in enumerate(df_metrics.columns):
            indices = dict_si[name][["S1", "ST"]]
            err = dict_si[name][["S1_conf", "ST_conf"]]
            indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
            ax[i].set_xticklabels(xaxis_labels)
            ax[i].set_title(_LABS[name])
            ax[i].legend(["First-order", "Total"], frameon=False)
        ax[-1].legend().set_visible(False)
        ax[-2].legend().set_visible(False)
        ax[-3].legend().set_visible(False)
        ax[0].set_ylabel("Sobol index [-]")
        fig.tight_layout()
        file = base_path_figs / "sobol_indices_soil_hyd_KGE_MAE.png"
        fig.savefig(file, dpi=300)

        df_metrics = df_params_metrics.loc[:, ["KGE_NO3_perc_bs", "KGE_NO3_perc_mass_bs", "MAE_N_uptake"]]
        dict_si = {}
        for name in df_metrics.columns:
            Y = df_metrics[name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"]).loc['k_transp':, :]

        # plot sobol indices
        _LABS = {"KGE_NO3_perc_bs": r"percolation (KGE$_{conc}$)",
                 "KGE_NO3_perc_mass_bs": r"percolation (KGE$_{mass}$)",
                 "MAE_N_uptake": "N-uptake (MAE)",
                }
        ncol = len(df_metrics.columns)
        xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][8:]]
        cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 3))
        for i, name in enumerate(df_metrics.columns):
            indices = dict_si[name][["S1", "ST"]]
            err = dict_si[name][["S1_conf", "ST_conf"]]
            indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
            ax[i].set_xticklabels(xaxis_labels)
            ax[i].set_title(_LABS[name])
            ax[i].legend(["First-order", "Total"], frameon=False)
        ax[-1].legend().set_visible(False)
        ax[-2].legend().set_visible(False)
        ax[-3].legend().set_visible(False)
        ax[0].set_ylabel("Sobol index [-]")
        fig.tight_layout()
        file = base_path_figs / "sobol_indices_transport_KGE_MAE.png"
        fig.savefig(file, dpi=300)

        df_metrics = df_params_metrics.loc[:, ["C_s_daily_avg", "C_q_ss_daily_avg"]]
        dict_si = {}
        for name in df_metrics.columns:
            Y = df_metrics[name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"]).loc[:'c_canopy', :]

        # plot sobol indices
        _LABS = {"C_s_daily_avg": r"Avg. soil conc.",
                 "C_q_ss_daily_avg": r"Avg. conc. of N leaching",
                }
        ncol = len(df_metrics.columns)
        xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][:8]]
        cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 3))
        for i, name in enumerate(df_metrics.columns):
            indices = dict_si[name][["S1", "ST"]]
            err = dict_si[name][["S1_conf", "ST_conf"]]
            indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
            ax[i].set_xticklabels(xaxis_labels)
            ax[i].set_title(_LABS[name])
            ax[i].legend(["First-order", "Total"], frameon=False)
        ax[-1].legend().set_visible(False)
        ax[-2].legend().set_visible(False)
        ax[0].set_ylabel("Sobol index [-]")
        fig.tight_layout()
        file = base_path_figs / "sobol_indices_soil_hyd_avg_conc.png"
        fig.savefig(file, dpi=300)

        df_metrics = df_params_metrics.loc[:, ["C_s_daily_avg", "C_q_ss_daily_avg"]]
        dict_si = {}
        for name in df_metrics.columns:
            Y = df_metrics[name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"]).loc['k_transp':, :]

        # plot sobol indices
        _LABS = {"C_s_daily_avg": r"Avg. soil conc.",
                 "C_q_ss_daily_avg": r"Avg. conc. of N leaching",
                }
        ncol = len(df_metrics.columns)
        xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][8:]]
        cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 3))
        for i, name in enumerate(df_metrics.columns):
            indices = dict_si[name][["S1", "ST"]]
            err = dict_si[name][["S1_conf", "ST_conf"]]
            indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
            ax[i].set_xticklabels(xaxis_labels)
            ax[i].set_title(_LABS[name])
            ax[i].legend(["First-order", "Total"], frameon=False)
        ax[-1].legend().set_visible(False)
        ax[-2].legend().set_visible(False)
        ax[0].set_ylabel("Sobol index [-]")
        fig.tight_layout()
        file = base_path_figs / "sobol_indices_transport_avg_conc.png"
        fig.savefig(file, dpi=300)


        df_metrics = df_params_metrics.loc[:, ["M_transp_annual_sum", "M_q_ss_annual_sum"]]
        dict_si = {}
        for name in df_metrics.columns:
            Y = df_metrics[name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"]).loc[:'c_canopy', :]

        # plot sobol indices
        _LABS = {"M_transp_annual_sum": r"Avg. N Uptake",
                 "M_q_ss_annual_sum": r"Avg. N leaching",
                }
        ncol = len(df_metrics.columns)
        xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][:8]]
        cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 3))
        for i, name in enumerate(df_metrics.columns):
            indices = dict_si[name][["S1", "ST"]]
            err = dict_si[name][["S1_conf", "ST_conf"]]
            indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
            ax[i].set_xticklabels(xaxis_labels)
            ax[i].set_title(_LABS[name])
            ax[i].legend(["First-order", "Total"], frameon=False)
        ax[-1].legend().set_visible(False)
        ax[-2].legend().set_visible(False)
        ax[0].set_ylabel("Sobol index [-]")
        fig.tight_layout()
        file = base_path_figs / "sobol_indices_soil_hyd_avg_loads.png"
        fig.savefig(file, dpi=300)


        df_metrics = df_params_metrics.loc[:, ["M_transp_annual_sum", "M_q_ss_annual_sum"]]
        dict_si = {}
        for name in df_metrics.columns:
            Y = df_metrics[name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"]).loc['k_transp':, :]

        # plot sobol indices
        _LABS = {"M_transp_annual_sum": r"Avg. N Uptake",
                 "M_q_ss_annual_sum": r"Avg. N leaching",
                }
        ncol = len(df_metrics.columns)
        xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][8:]]
        cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 3))
        for i, name in enumerate(df_metrics.columns):
            indices = dict_si[name][["S1", "ST"]]
            err = dict_si[name][["S1_conf", "ST_conf"]]
            indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
            ax[i].set_xticklabels(xaxis_labels)
            ax[i].set_title(_LABS[name])
            ax[i].legend(["First-order", "Total"], frameon=False)
        ax[-1].legend().set_visible(False)
        ax[-2].legend().set_visible(False)
        ax[0].set_ylabel("Sobol index [-]")
        fig.tight_layout()
        file = base_path_figs / "sobol_indices_transport_avg_loads.png"
        fig.savefig(file, dpi=300)


        df_metrics = df_params_metrics.loc[:, ["min_s_annual_sum", "nit_s_annual_sum", "denit_s_annual_sum"]]
        dict_si = {}
        for name in df_metrics.columns:
            Y = df_metrics[name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"]).loc[:'c_canopy', :]

        # plot sobol indices
        _LABS = {"min_s_annual_sum": r"Avg. soil N minerilization",
                 "nit_s_annual_sum": r"Avg. nitrification",
                 "denit_s_annual_sum": r"Avg. denitrification",
                }
        ncol = len(df_metrics.columns)
        xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][:8]]
        cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 5))
        for i, name in enumerate(df_metrics.columns):
            indices = dict_si[name][["S1", "ST"]]
            err = dict_si[name][["S1_conf", "ST_conf"]]
            indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
            ax[i].set_xticklabels(xaxis_labels)
            ax[i].set_title(_LABS[name])
            ax[i].legend(["First-order", "Total"], frameon=False)
        ax[-1].legend().set_visible(False)
        ax[-2].legend().set_visible(False)
        ax[-3].legend().set_visible(False)
        ax[0].set_ylabel("Sobol index [-]")
        fig.tight_layout()
        file = base_path_figs / "sobol_indices_soil_hyd_avg_loads_soilN_cycle.png"
        fig.savefig(file, dpi=300)


        df_metrics = df_params_metrics.loc[:, ["min_s_annual_sum", "nit_s_annual_sum", "denit_s_annual_sum"]]
        dict_si = {}
        for name in df_metrics.columns:
            Y = df_metrics[name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"]).loc[:'k_transp', :]

        # plot sobol indices
        _LABS = {"min_s_annual_sum": r"Avg. soil N minerilization",
                 "nit_s_annual_sum": r"Avg. nitrification",
                 "denit_s_annual_sum": r"Avg. denitrification",
                }
        ncol = len(df_metrics.columns)
        xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][8:]]
        cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 5))
        for i, name in enumerate(df_metrics.columns):
            indices = dict_si[name][["S1", "ST"]]
            err = dict_si[name][["S1_conf", "ST_conf"]]
            indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
            ax[i].set_xticklabels(xaxis_labels)
            ax[i].set_title(_LABS[name])
            ax[i].legend(["First-order", "Total"], frameon=False)
        ax[-1].legend().set_visible(False)
        ax[-2].legend().set_visible(False)
        ax[-3].legend().set_visible(False)
        ax[0].set_ylabel("Sobol index [-]")
        fig.tight_layout()
        file = base_path_figs / "sobol_indices_transport_avg_loads_soilN_cycle.png"
        fig.savefig(file, dpi=300)

        # df_metrics = df_params_metrics.loc[:, ["rt50_s_daily_avg", "tt50_transp_daily_avg", "tt50_q_ss_annual_avg"]]
        # dict_si = {}
        # for name in df_metrics.columns:
        #     Y = df_metrics[name].values
        #     Si = sobol.analyze(bounds, Y, calc_second_order=False)
        #     Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
        #     dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"][:8])

        # # plot sobol indices
        # _LABS = {"rt50_s_daily_avg": r"Median soil residence time",
        #          "tt50_transp_daily_avg": r"Median transpiration travel time",
        #          "tt50_q_ss_annual_avg": r"Median leaching travel time",
        #         }
        # ncol = len(df_metrics.columns)
        # xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][:8]]
        # cmap = plt.get_cmap("Greys")
        # norm = Normalize(vmin=0, vmax=2)
        # colors = cmap(norm([0.5, 1.5]))
        # fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 5))
        # for i, name in enumerate(df_metrics.columns):
        #     indices = dict_si[name][["S1", "ST"]]
        #     err = dict_si[name][["S1_conf", "ST_conf"]]
        #     indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
        #     ax[i].set_xticklabels(xaxis_labels)
        #     ax[i].set_title(_LABS[name])
        #     ax[i].legend(["First-order", "Total"], frameon=False)
        # ax[-1].legend().set_visible(False)
        # ax[-2].legend().set_visible(False)
        # ax[-3].legend().set_visible(False)
        # ax[0].set_ylabel("Sobol index [-]")
        # fig.tight_layout()
        # file = base_path_figs / "sobol_indices_soil_hyd_median_water_ages.png"
        # fig.savefig(file, dpi=300)


        # df_metrics = df_params_metrics.loc[:, ["rt50_s_daily_avg", "tt50_transp_daily_avg", "tt50_q_ss_annual_avg"]]
        # dict_si = {}
        # for name in df_metrics.columns:
        #     Y = df_metrics[name].values
        #     Si = sobol.analyze(bounds, Y, calc_second_order=False)
        #     Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
        #     dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"][8:])

        # # plot sobol indices
        # _LABS = {"rt50_s_daily_avg": r"Median soil residence time",
        #          "tt50_transp_daily_avg": r"Median transpiration travel time",
        #          "tt50_q_ss_annual_avg": r"Median leaching travel time",
        #         }
        # ncol = len(df_metrics.columns)
        # xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"][8:]]
        # cmap = plt.get_cmap("Greys")
        # norm = Normalize(vmin=0, vmax=2)
        # colors = cmap(norm([0.5, 1.5]))
        # fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 5))
        # for i, name in enumerate(df_metrics.columns):
        #     indices = dict_si[name][["S1", "ST"]]
        #     err = dict_si[name][["S1_conf", "ST_conf"]]
        #     indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
        #     ax[i].set_xticklabels(xaxis_labels)
        #     ax[i].set_title(_LABS[name])
        #     ax[i].legend(["First-order", "Total"], frameon=False)
        # ax[-1].legend().set_visible(False)
        # ax[-2].legend().set_visible(False)
        # ax[-3].legend().set_visible(False)
        # ax[0].set_ylabel("Sobol index [-]")
        # fig.tight_layout()
        # file = base_path_figs / "sobol_indices_transport_median_water_ages.png"
        # fig.savefig(file, dpi=300)
    return


if __name__ == "__main__":
    main()
