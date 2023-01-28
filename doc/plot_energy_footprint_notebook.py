import json
import click
import numpy as np

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


@click.argument("INFILES", nargs=-1, type=click.Path(dir_okay=False, exists=True))
@click.option("--xaxis", type=click.Choice(["nproc", "size"]), required=True)
@click.option("--tdpc", type=float, default=45)
@click.option("--tdpm", type=float, default=0.3725)
@click.option("--mem", type=float, default=8)
@click.option("--ncores", type=int, default=4)
@click.option("--nitt", type=int, default=1)
@click.option("--unit", type=click.Choice(["Wh", "kWh"]), default="Wh")
@click.command()
def plot_energy_footprint(infiles, xaxis, tdpc, tdpm, mem, ncores, nitt, unit):
    benchmarks = set()
    components = set()
    sizes = set()
    nprocs = set()

    for infile in infiles:
        with open(infile) as f:
            data = json.load(f)

        meta = data["settings"]
        benchmarks |= set(meta["only"])
        components |= set(meta["backends"])
        sizes |= set(meta["sizes"])
        nprocs.add(meta["nproc"])

    if xaxis == "nproc":
        assert len(sizes) == 1
        xvals = np.array(sorted(nprocs))
    elif xaxis == "size":
        assert len(nprocs) == 1
        xvals = np.array(sorted(sizes))
    else:
        assert False

    component_data = {benchmark: {comp: np.full(len(xvals), np.nan) for comp in components} for benchmark in benchmarks}

    for infile in infiles:
        with open(infile) as f:
            data = json.load(f)

        for benchmark, bench_res in data["benchmarks"].items():
            for res in bench_res:
                if xaxis == "size":
                    # sizes are approximate, take the closest one
                    x_idx = np.argmin(np.abs(np.array(xvals) - res["size"]))
                else:
                    x_idx = xvals.tolist().index(data["settings"]["nproc"])

                time = float(res["per_iteration"]["mean"])
                component_data[benchmark][res["backend"]][x_idx] = time

    for benchmark in benchmarks:
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 4), dpi=150)

        last_coords = {}
        for component in components:
            if component in ['numpy-mpi', 'jax-mpi']:
                yvals = component_data[benchmark][component]/3600 * ((tdpc/ncores) * meta["nproc"] + component_data[benchmark][component]/3600 * mem * tdpm) * nitt
            else:
                yvals = component_data[benchmark][component]/3600 * (tdpc/ncores + component_data[benchmark][component]/3600 * mem * tdpm) * nitt

            plt.plot(xvals, yvals, ".--", color=COMPONENT_COLORS[component], lw=1)

            finite_mask = np.isfinite(yvals)
            if finite_mask.any():
                last_coords[component] = (xvals[finite_mask][-1], yvals[finite_mask][-1])
            else:
                last_coords[component] = (xvals[0], 1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        title_kwargs = dict(fontdict=dict(weight="bold", size=11), ha="left", x=0.05, y=1.05)
        if xaxis == "nproc":
            plt.xlabel("Number of MPI processes [# CPU cores]")
            mantissa, exponent = f"{list(sizes)[0]:.1e}".split("e")
            exponent = exponent.lstrip("+0")
            plt.title(f"Benchmark '{benchmark}' for size {mantissa} $\\times$ 10$^{{{exponent}}}$", **title_kwargs)

        elif xaxis == "size":
            nproc = list(nprocs)[0]
            plt.xlabel("Grid size [# grid cells]")
            plt.title(f"Benchmark '{benchmark}' on {nproc} processes", **title_kwargs)

        plt.ylabel(f"Energy footprint [{unit}/iteration]")
        plt.text(0.05, 1.05, "Energy (lower is better)", transform=ax.transAxes, va="top", color="0.4")

        plt.xscale("log")

        fig.canvas.draw()

        # add annotations, make sure they don"t overlap
        last_text_pos = 0
        for component, (x, y) in sorted(last_coords.items(), key=lambda k: k[1][1]):
            trans = ax.transData
            _, tp = trans.transform((0, y))
            tp = max(tp, last_text_pos + 20)
            _, y = trans.inverted().transform((0, tp))


            energy_last = np.round(y, 3)
            component_label = component + f' ({energy_last} {unit})'

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

        fig.tight_layout()

        fig.savefig(f"energy_footprint_{benchmark.split('_')[0]}_{xaxis}_notebook.png")
        fig.savefig(f"energy_footprint_{benchmark.split('_')[0]}_{xaxis}_notebook.pdf")
        plt.close(fig)


if __name__ == "__main__":
    plot_energy_footprint()
