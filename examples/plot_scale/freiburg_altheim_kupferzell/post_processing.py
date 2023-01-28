import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import datetime
import glob
import h5netcdf
import roger
import roger.tools.labels as labs
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['legend.title_fontsize'] = 7
sns.set_style("ticks")
sns.plotting_context("paper", font_scale=1, rc={'font.size': 6.0,
                                                'axes.labelsize': 7.0,
                                                'axes.titlesize': 8.0,
                                                'xtick.labelsize': 6.0,
                                                'ytick.labelsize': 6.0,
                                                'legend.fontsize': 6.0,
                                                'legend.title_fontsize': 7.0})

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)