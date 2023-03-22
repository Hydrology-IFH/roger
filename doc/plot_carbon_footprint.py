import click
import numpy as np
import pandas as pd

import matplotlib as mpl
import seaborn as sns

mpl.use("agg")

import matplotlib.pyplot as plt  # noqa: E402

sns.set_style("ticks")

COMPONENT_COLORS = {
    "numpy": "orangered",
    "numpy-mpi": "coral",
    "jax": "#6baed6",
    "jax-mpi": "#2171b5",
    "jax-gpu": "#08306b",
}


def _set_size(x):
    return int(x.split('_')[-1])


def _set_backend(x):
    return str(x.split('_')[-2])


@click.option("--name", type=str, required=True)
@click.option("--nitt", type=float, default=1)
@click.option("--rescale", type=float, default=None)
@click.option("--file", type=click.Path(dir_okay=False, exists=True), required=True)
@click.option("--xaxis", type=click.Choice(["nproc", "size"]), required=True)
@click.option("--unit", type=click.Choice(["g", "kg", "t"]), default="g")
@click.command()
def plot_benchmarks(name, file, xaxis, unit, nitt, rescale):

    components = set()
    sizes = set()

    data = pd.read_csv(file, sep=";", index_col=0)

    if xaxis == "size":
        data["size"] = data["JobName_"].apply(_set_size)
    elif xaxis == "nproc":
        data["nproc"] = data["NCPUS_"] * data["NNodes_"]
    data["backend"] = data["JobName_"].apply(_set_backend)

    components = pd.unique(data["backend"]).tolist()

    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), dpi=250)
    last_coords = {}
    for component in components:
        data_component = data.loc[data["backend"] == component, :]
        if xaxis == "size":
            data_component = data_component.sort_values(by=['size'])
            xvals = data_component['size'].values
        elif xaxis == "nproc":
            data_component = data_component.sort_values(by=['nproc'])
            xvals = data_component['nproc'].values
        if unit == "g":
            yvals = data_component['carbonFootprint'].values / nitt
        elif unit == "kg":
            yvals = data_component['carbonFootprint'].values / nitt * 0.001
        elif unit == "t":
            yvals = data_component['carbonFootprint'].values / nitt * 0.000001

        if rescale:
            yvals *= rescale

        plt.plot(xvals, yvals, ".--", color=COMPONENT_COLORS[component], lw=1)

        finite_mask = np.isfinite(yvals)
        if finite_mask.any():
            last_coords[component] = (xvals[finite_mask][-1], yvals[finite_mask][-1])
        else:
            last_coords[component] = (xvals[0], 1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        if xaxis == "nproc":
            plt.xlabel("Number of MPI processes [# CPU cores]")
            mantissa, exponent = f"{list(sizes)[0]:.1e}".split("e")
            exponent = exponent.lstrip("+0")

        elif xaxis == "size":
            plt.xlabel("Grid size [# grid cells]")

        if rescale:
            plt.ylabel(f"Carbon footprint [{unit}$CO_2$e]")
        else:
            plt.ylabel(f"Carbon footprint [{unit}$CO_2$e/iteration]")

        fig.canvas.draw()

        # add annotations, make sure they don"t overlap
        last_text_pos = 0
        for component, (x, y) in sorted(last_coords.items(), key=lambda k: k[1][1]):
            trans = ax.transData
            _, tp = trans.transform((0, y))
            tp = max(tp, last_text_pos + 20)
            _, y = trans.inverted().transform((0, tp))

            y_last = np.round(y, 2)
            component_label = component + f' ({y_last} {unit}$CO_2$e)'

            plt.annotate(
                component_label,
                (x, y),
                xytext=(10, 0),
                textcoords="offset points",
                annotation_clip=False,
                color=COMPONENT_COLORS[component],
                va="center",
                weight="bold",
            )

            last_text_pos = tp

    if xaxis == "size":
        plt.xscale("log")

    fig.tight_layout()
    fig.savefig(f"carbon_footprint_{name}_{xaxis}.png")
    fig.savefig(f"carbon_footprint_{name}_{xaxis}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plot_benchmarks()
