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
        file = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz/output") / "svat_crop_nitrate_sobol" / f"params_eff_{lys_experiment}.txt"
        df_params_metrics = pd.read_csv(file, sep="\t")

        # perform sensitivity analysis
        df_params = df_params_metrics.loc[:, bounds["names"]]
        df_eff = df_params_metrics.loc[:, ["KGE_NO3_perc_bs", "KGE_NO3_perc_mass_bs", "MAE_N_uptake"]]
        dict_si = {}
        for name in df_eff.columns:
            Y = df_eff[name].values
            Si = sobol.analyze(bounds, Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds["names"])

        # plot sobol indices
        _LABS = {"KGE_NO3_perc_bs": r"percolation (KGE$_{conc}$)",
                 "KGE_NO3_perc_mass_bs": r"percolation (KGE$_{mass}$)",
                 "MAE_N_uptake": "N-uptake (MAE)",
                }
        ncol = len(df_eff.columns)
        xaxis_labels = [_XLABS[k].split(" ")[0] for k in bounds["names"]]
        cmap = plt.get_cmap("Greys")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(14, 5))
        for i, name in enumerate(df_eff.columns):
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
        file = base_path_figs / "sobol_indices.png"
        fig.savefig(file, dpi=300)
    return


if __name__ == "__main__":
    main()
