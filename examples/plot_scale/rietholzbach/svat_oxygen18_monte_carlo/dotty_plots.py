from pathlib import Path
import os
import pandas as pd
import click
import roger.tools.labels as labs
import matplotlib as mpl
import seaborn as sns

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

sns.set_style("ticks")


@click.option("-ns", "--nsamples", type=int, default=10000)
@click.option(
    "-tms",
    "--transport-model-structure",
    type=click.Choice(
        [
            "complete-mixing",
            "piston",
            "preferential-power",
            "advection-dispersion-kumaraswamy",
            "time-variant_advection-dispersion-kumaraswamy",
            "advection-dispersion-power",
            "time-variant_advection-dispersion-power",
            "older-preference-power",
        ]
    ),
    default="advection-dispersion-power",
)
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(nsamples, transport_model_structure, tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    tms = transport_model_structure.replace("_", " ")
    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # dotty plots
    click.echo(f"Dotty plots for {tms} ...")
    file = base_path_figs / f"params_metrics_{transport_model_structure}.txt"
    df_params_metrics = pd.read_csv(file, header=0, index_col=False, sep="\t")
    df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
    if tms == "complete-mixing":
        df_params = df_params_metrics.loc[
            :, ["c1_mak", "c2_mak", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks"]
        ]
    elif tms == "piston":
        df_params = df_params_metrics.loc[
            :, ["c1_mak", "c2_mak", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks"]
        ]
    elif tms == "advection-dispersion-kumaraswamy":
        df_params = df_params_metrics.loc[:, ["a_transp", "b_transp", "a_q_rz", "b_q_rz", "a_q_ss", "b_q_ss"]]
    elif tms == "older-preference-power":
        df_params = df_params_metrics.loc[:, ["k_transp", "k_q_rz", "k_q_ss"]]
    elif tms == "time-variant advection-dispersion-kumaraswamy":
        df_params = df_params_metrics.loc[
            :, ["a_transp", "b1_transp", "b2_transp", "a1_q_rz", "a2_q_rz", "b_q_rz", "a1_q_ss", "a2_q_ss", "b_q_ss"]
        ]
    elif tms == "preferential-power":
        df_params = df_params_metrics.loc[:, ["k_transp", "k_q_rz", "k_q_ss"]]
    elif tms == "advection-dispersion-power":
        df_params = df_params_metrics.loc[:, ["k_transp", "k_q_rz", "k_q_ss"]]
    elif tms == "time-variant advection-dispersion-power":
        df_params = df_params_metrics.loc[:, ["k1_transp", "k2_transp", "k1_q_rz", "k2_q_rz", "k1_q_ss", "k2_q_ss"]]
    # select best model run
    idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
    nrow = len(df_metrics.columns)
    ncol = len(df_params.columns)
    fig, ax1 = plt.subplots(nrow, ncol, sharey=True, figsize=(ncol * 3.5, 3.5))
    if ncol > 1:
        ax = ax1.reshape(nrow, ncol)
        for i in range(nrow):
            for j in range(ncol):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, s=4, c="grey", alpha=0.5)
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")
                ax[i, j].set_ylim(-1, 1)
                # best model run
                y_best = df_metrics.iloc[idx_best, i]
                x_best = df_params.iloc[idx_best, j]
                ax[i, j].scatter(x_best, y_best, s=12, c="red", alpha=0.8)

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel(r"$KGE_{\delta^{18}O_{PERC}}$ [-]")

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_{transport_model_structure}.png"
        fig.savefig(file, dpi=250)
        plt.close("all")
    else:
        ax = ax1
        y = df_metrics.iloc[:, 0]
        x = df_params.iloc[:, 0]
        ax.scatter(x, y, s=4, c="grey", alpha=0.5)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_ylim(-1, 1)
        # best model run
        y_best = df_metrics.iloc[idx_best, 0]
        x_best = df_params.iloc[idx_best, 0]
        ax.scatter(x_best, y_best, s=12, c="red", alpha=0.8)
        xlabel = labs._LABS[df_params.columns[0]]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$KGE_{\delta^{18}O_{PERC}}$ [-]")
        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_{transport_model_structure}.png"
        fig.savefig(file, dpi=250)
        plt.close("all")
    return


if __name__ == "__main__":
    main()
