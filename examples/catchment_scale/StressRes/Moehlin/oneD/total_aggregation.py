import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import datetime
import numpy as onp
import yaml
import matplotlib as mpl
import seaborn as sns
import h5netcdf
import roger

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

base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# directory of output
base_path_output = Path("/Volumes/LaCie/roger/examples/catchment_scale/StressRes/Moehlin") / "output" / "oneD"
if not os.path.exists(base_path_output):
    os.mkdir(base_path_output)

# load configuration file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    config = yaml.safe_load(file)


# load RoGeR model parameters
params_hm_file = base_path / "parameters.nc"
ds_params = xr.open_dataset(params_hm_file, engine="h5netcdf")
dem = ds_params["dgm"].values
mask = onp.isfinite(dem)

# years to be aggregated
years = onp.arange(2013, 2023, dtype=int)
# variables to be aggregated
vars_sim = ["aet", "pet", "q_ss", "q_sub", "cpr_ss"]

# merge model output into single file
states_agg_file = base_path_output / "ONED_Moehlin_total.nc"
if not os.path.exists(states_agg_file):
    with h5netcdf.File(states_agg_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="Average of annual sums/means of RoGeR simulations for the Moehlin catchment, Germany (2013-2022)",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).",
            model_structure="1D model with capillary rise",
            roger_version=f"{roger.__version__}",
        )
        # collect dimensions
        # set dimensions with a dictionary
        dict_dim = {
            "x": config['nx'],
            "y": config['ny'],
            "Year": len(years),
        }
        if not f.dimensions:
            f.dimensions = dict_dim
            v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
            v.attrs["long_name"] = "x"
            v.attrs["units"] = "m"
            v[:] = onp.arange(dict_dim["x"]) * 25
            v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
            v.attrs["long_name"] = "y"
            v.attrs["units"] = "m"
            v[:] = onp.arange(dict_dim["y"]) * 25
            v = f.create_variable("Year", ("Year",), int, compression="gzip", compression_opts=1)
            v[:] = years

        # load hydrological simulations
        states_hm_file = base_path_output / f"{config['identifier']}.nc"
        ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf")

        # assign date
        days = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date = num2date(
            days,
            units=f"days since {ds_sim['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_sim = ds_sim.assign_coords(Time=("Time", date))
        for var_sim in vars_sim:
            v = f.create_variable(var_sim, ("x", "y"), float, compression="gzip", compression_opts=1)
            vals = ds_sim[var_sim].sel(Time=slice('2013', '2022')).resample(Time='YS').sum().mean(axis=-1)
            v[:, :] = onp.where(mask, vals, onp.nan)
            v.attrs.update(long_name=var_sim, units="mm/year")

        v = f.create_variable("inf", ("x", "y"), float, compression="gzip", compression_opts=1)
        vals1 = ds_sim["inf_mat_rz"].sel(Time=slice('2013', '2022')).resample(Time='YS').sum().mean(axis=-1)
        vals2 = ds_sim["inf_mp_rz"].sel(Time=slice('2013', '2022')).resample(Time='YS').sum().mean(axis=-1)
        vals3 = ds_sim["inf_sc_rz"].sel(Time=slice('2013', '2022')).resample(Time='YS').sum().mean(axis=-1)
        vals4 = ds_sim["inf_ss"].sel(Time=slice('2013', '2022')).resample(Time='YS').sum().mean(axis=-1)
        v[:, :] = onp.where(mask, vals1 + vals2 + vals3 + vals4, onp.nan)
        v.attrs.update(long_name="inf", units="mm/year")

        v = f.create_variable("inf_mat", ("x", "y"), float, compression="gzip", compression_opts=1)
        vals1 = ds_sim["inf_mat_rz"].sel(Time=slice('2013', '2022')).resample(Time='YS').sum().mean(axis=-1)
        v[:, :] = onp.where(mask, vals1, onp.nan)
        v.attrs.update(long_name="inf_mat", units="mm/year")

        v = f.create_variable("inf_mp", ("x", "y"), float, compression="gzip", compression_opts=1)
        vals1 = ds_sim["inf_mp_rz"].sel(Time=slice('2013', '2022')).resample(Time='YS').sum().mean(axis=-1)
        vals2 = ds_sim["inf_ss"].sel(Time=slice('2013', '2022')).resample(Time='YS').sum().mean(axis=-1)
        v[:, :] = onp.where(mask, vals1 + vals2, onp.nan)
        v.attrs.update(long_name="inf_mp", units="mm/year")

        v = f.create_variable("inf_sc", ("x", "y"), float, compression="gzip", compression_opts=1)
        vals1 = ds_sim["inf_sc_rz"].sel(Time=slice('2013', '2022')).resample(Time='YS').sum().mean(axis=-1)
        v[:, :] = onp.where(mask, vals1, onp.nan)
        v.attrs.update(long_name="inf_sc", units="mm/year")