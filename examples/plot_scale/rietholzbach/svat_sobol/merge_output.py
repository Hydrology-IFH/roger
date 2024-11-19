import glob
import os
from pathlib import Path
import datetime
from cftime import num2date
import h5netcdf
import xarray as xr
import pandas as pd
from SALib.analyze import sobol
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import yaml
import numpy as onp
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import click
import matplotlib as mpl
import seaborn as sns

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

onp.random.seed(42)


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
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # merge model output into single file
    tm_structures = [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
    ]
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        states_hm_si_file = base_path_output / f"SVAT_for_{tms}.nc"
        if not os.path.exists(states_hm_si_file):
            click.echo(f"Merge output files into {states_hm_si_file.as_posix()}")
            path = str(base_path_output / f"SVAT_for_{tms}.*.nc")
            diag_files = glob.glob(path)
            with h5netcdf.File(states_hm_si_file, "w", decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title="RoGeR saltelli results at Rietholzbach Lysimeter site",
                    institution="University of Freiburg, Chair of Hydrology",
                    references="",
                    comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
                    model_structure="SVAT model with free drainage",
                )
                # collect dimensions
                for dfs in diag_files:
                    with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                        f.attrs.update(roger_version=df.attrs["roger_version"])
                        # set dimensions with a dictionary
                        if not dfs.split("/")[-1].split(".")[1] == "constant":
                            dict_dim = {
                                "x": len(df.variables["x"]),
                                "y": len(df.variables["y"]),
                                "Time": len(df.variables["Time"]),
                            }
                            time = onp.array(df.variables.get("Time"))
                for dfs in diag_files:
                    with h5netcdf.File(dfs, "r", decode_vlen_strings=False) as df:
                        if not f.dimensions:
                            f.dimensions = dict_dim
                            v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
                            v.attrs["long_name"] = "model run"
                            v.attrs["units"] = ""
                            v[:] = onp.arange(dict_dim["x"])
                            v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
                            v.attrs["long_name"] = ""
                            v.attrs["units"] = ""
                            v[:] = onp.arange(dict_dim["y"])
                            v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
                            var_obj = df.variables.get("Time")
                            v.attrs.update(time_origin=var_obj.attrs["time_origin"], units=var_obj.attrs["units"])
                            v[:] = time
                        for var_sim in list(df.variables.keys()):
                            var_obj = df.variables.get(var_sim)
                            if var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] > 1:
                                v = f.create_variable(
                                    var_sim, ("x", "y", "Time"), float, compression="gzip", compression_opts=1
                                )
                                vals = onp.array(var_obj)
                                v[:, :, :] = vals.swapaxes(0, 2)
                                v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
                            elif (
                                var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] <= 2
                            ):
                                v = f.create_variable(
                                    var_sim, ("x", "y"), float, compression="gzip", compression_opts=1
                                )
                                vals = onp.array(var_obj)
                                v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                                v.attrs.update(long_name=var_obj.attrs["long_name"], units=var_obj.attrs["units"])
    return


if __name__ == "__main__":
    main()
