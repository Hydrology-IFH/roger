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
        'c_root': r"$c_{root}$",
        'c_canopy': r"$c_{canopy}$",
}

_UNITS = {
        "lmpv": r"$l_{mpv}$ [mm]",
        "dmpv": r"$\rho_{mpv}$ [1/$m^2$]",
        "theta_eff": r"$\theta_{eff}$ [-]",
        "frac_lp": r"$f_{lp}$ [-]",
        "theta_pwp": r"$\theta_{pwp}$ [-]",
        "ks": r"$k_{s}$ [mm/hour]",
        'c_root': r"$c_{root}$ [-]",
        'c_canopy': r"$c_{canopy}$ [-]",
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
    with open(file_path, 'r') as file:
        bounds = yaml.safe_load(file)

    # directory of results
    base_path_output = base_path.parent / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path.parent / "figures" / "svat_crop_sobol"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)


    lys_experiments = ["lys3"]
    for lys_experiment in lys_experiments:
        # write .txt-file
        file = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz/output") / "svat_crop_sobol" / f"params_eff_{lys_experiment}.txt"
        df_params_metrics = pd.read_csv(file, sep="\t")
        df_params_metrics['theta_eff'] = df_params_metrics['theta_ac'] + df_params_metrics['theta_ufc']
        df_params_metrics['frac_lp'] = df_params_metrics['theta_ac'] / df_params_metrics['theta_eff']

        df_params_metrics["E_dS"] = (1 - ((1 - df_params_metrics["r_S_perc_pet"])**2 + (1 - df_params_metrics["KGE_alpha_S_perc_pet"])**2)**(0.5))
        df_params_metrics["E_multi"] = 0.7 * df_params_metrics["KGE_q_ss_perc_pet"] + 0.3 * (1 - ((1 - df_params_metrics["r_S_perc_pet"])**2 + (1 - df_params_metrics["KGE_alpha_S_perc_pet"])**2)**(0.5))


        # perform sensitivity analysis
        df_params = df_params_metrics.loc[:, bounds['names']]
        df_eff = df_params_metrics.loc[:, ['KGE_q_ss_perc_pet', 'E_dS', 'E_multi']]
        dict_si = {}
        for name in df_eff.columns:
            Y = df_eff[name].values
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds['names'])

        # plot sobol indices
        _LABS = {'KGE_q_ss_perc_pet': 'percolation (KGE)',
                 'E_dS': r'storage ($\alpha$ & r)',
                 'E_multi': 'multi-objective metric',
                }
        ncol = len(df_eff.columns)
        xaxis_labels = [_XLABS[k].split(' ')[0] for k in bounds['names']]
        cmap = plt.get_cmap('Greys')
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
        file = base_path_figs / "sobol_indices_Emulti.png"
        fig.savefig(file, dpi=300)

        df_params = df_params_metrics.loc[:, bounds['names']]
        df_eff = df_params_metrics.loc[:, ['KGE_q_ss_perc_pet', 'KGE_alpha_S_all', 'r_S_all']]
        dict_si = {}
        for name in df_eff.columns:
            Y = df_eff[name].values
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds['names'])

        # plot sobol indices
        _LABS = {'KGE_q_ss_perc_pet': 'percolation (KGE)',
                 'KGE_alpha_S_all': r'storage ($KGE_{\alpha}$)',
                 'r_S_all': 'storage (r)',
                }
        ncol = len(df_eff.columns)
        xaxis_labels = [_XLABS[k].split(' ')[0] for k in bounds['names']]
        cmap = plt.get_cmap('Greys')
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
        file = base_path_figs / "sobol_indices_Emulti1.png"
        fig.savefig(file, dpi=300)


        df_eff = df_params_metrics.loc[:, ['KGE_alpha_q_ss_perc_pet', 'KGE_beta_q_ss_perc_pet', 'r_q_ss_perc_pet']]
        dict_si = {}
        for name in df_eff.columns:
            Y = df_eff[name].values
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds['names'])

        # plot sobol indices
        _LABS = {'KGE_alpha_q_ss_perc_pet': r'percolation ($KGE_{\alpha}$)',
                 'KGE_beta_q_ss_perc_pet': r'percolation ($KGE_{\beta}$)',
                 'r_q_ss_perc_pet': r'percolation ($r$)',
                }
        ncol = len(df_eff.columns)
        xaxis_labels = [_XLABS[k].split(' ')[0] for k in bounds['names']]
        cmap = plt.get_cmap('Greys')
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
        file = base_path_figs / "sobol_indices_KGE_perc.png"
        fig.savefig(file, dpi=300)


        df_eff = df_params_metrics.loc[:, ['KGE_alpha_q_ss_perc_dom', 'KGE_beta_q_ss_perc_dom', 'r_q_ss_perc_dom']]
        dict_si = {}
        for name in df_eff.columns:
            Y = df_eff[name].values
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds['names'])

        # plot sobol indices
        _LABS = {'KGE_alpha_q_ss_perc_dom': r'percolation ($KGE_{\alpha}$)',
                 'KGE_beta_q_ss_perc_dom': r'percolation ($KGE_{\beta}$)',
                 'r_q_ss_perc_dom': r'percolation ($r$)',
                }
        ncol = len(df_eff.columns)
        xaxis_labels = [_XLABS[k].split(' ')[0] for k in bounds['names']]
        cmap = plt.get_cmap('Greys')
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
        file = base_path_figs / "sobol_indices_KGE_perc_perc_dom.png"
        fig.savefig(file, dpi=300)

        df_eff = df_params_metrics.loc[:, ['KGE_alpha_q_ss_perc_pet_2012', 'KGE_beta_q_ss_perc_pet_2012', 'r_q_ss_perc_pet_2012']]
        dict_si = {}
        for name in df_eff.columns:
            Y = df_eff[name].values
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds['names'])

        # plot sobol indices
        _LABS = {'KGE_alpha_q_ss_perc_pet_2012': r'percolation ($KGE_{\alpha}$)',
                 'KGE_beta_q_ss_perc_pet_2012': r'percolation ($KGE_{\beta}$)',
                 'r_q_ss_perc_pet_2012': r'percolation ($r$)',
                }
        ncol = len(df_eff.columns)
        xaxis_labels = [_XLABS[k].split(' ')[0] for k in bounds['names']]
        cmap = plt.get_cmap('Greys')
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
        file = base_path_figs / "sobol_indices_KGE_perc_2012_winter_wheat.png"
        fig.savefig(file, dpi=300)


        df_eff = df_params_metrics.loc[:, ['KGE_alpha_q_ss_perc_pet_2015', 'KGE_beta_q_ss_perc_pet_2015', 'r_q_ss_perc_pet_2015']]
        dict_si = {}
        for name in df_eff.columns:
            Y = df_eff[name].values
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds['names'])

        # plot sobol indices
        _LABS = {'KGE_alpha_q_ss_perc_pet_2015': r'percolation ($KGE_{\alpha}$)',
                 'KGE_beta_q_ss_perc_pet_2015': r'percolation ($KGE_{\beta}$)',
                 'r_q_ss_perc_pet_2015': r'percolation ($r$)',
                }
        ncol = len(df_eff.columns)
        xaxis_labels = [_XLABS[k].split(' ')[0] for k in bounds['names']]
        cmap = plt.get_cmap('Greys')
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
        file = base_path_figs / "sobol_indices_KGE_perc_2015_silage_corn.png"
        fig.savefig(file, dpi=300)

        # make dotty plots
        nrow = len(df_eff.columns)
        ncol = 6
        fig, ax = plt.subplots(nrow, ncol, figsize=(14, 7), sharex='col', sharey='row')
        for i in range(nrow):
            for j in range(ncol):
                y = df_eff.iloc[:, i]
                x = df_params.iloc[:, j]
                sns.regplot(x=x, y=y, ax=ax[i, j], ci=None, color='k',
                            scatter_kws={'alpha': 0.2, 's': 4, 'color': 'grey'})
                ax[i, j].set_xlabel('')
                ax[i, j].set_ylabel('')

        for j in range(ncol):
            xlabel = _UNITS[bounds['names'][j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel('$KGE_{PERC}$ [-]')
        ax[1, 0].set_ylabel(r'$r_{\Delta S}$ [-]')
        ax[2, 0].set_ylabel('$E_{multi}$\n [-]')

        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        file = base_path_figs / "dotty_plots_Emulti.png"
        fig.savefig(file, dpi=300)
    return


if __name__ == "__main__":
    main()
