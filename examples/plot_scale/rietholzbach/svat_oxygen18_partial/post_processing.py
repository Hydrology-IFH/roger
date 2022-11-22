from pathlib import Path
import os
import xarray as xr
from cftime import num2date
import numpy as onp
import click
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
sns.set_style("ticks")


@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(sas_solver, tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    age_max = "age_max_1500_days"
    metric_for_optimization = "optimized_with_KGE_multi_hm1"
    # directory of results
    base_path_results = base_path / "results"
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    base_path_results = base_path / "results" / sas_solver
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    base_path_results = base_path / "results" / sas_solver / age_max
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    base_path_results = base_path / "results" / sas_solver / age_max / metric_for_optimization
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)
    base_path_figs = base_path / "figures" / sas_solver
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)
    base_path_figs = base_path / "figures" / sas_solver / age_max
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)
    base_path_figs = base_path / "figures" / sas_solver / age_max / metric_for_optimization
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load observations (measured data)
    path_obs = base_path.parent / "observations" / "rietholzbach_lysimeter.nc"
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
    days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_obs = ds_obs.assign_coords(Time=("Time", date_obs))

    tm_structure = 'advection-dispersion'
    tms = tm_structure.replace(" ", "_")
    # load transport simulation
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{tms}_deterministic.average.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    params_tm_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_tm_file, engine="h5netcdf", group=tm_structure)

    b_q_ss = onp.unique(ds_params['b_q_ss'].isel(y=0).values)
    a_q_ss = onp.unique(ds_params['a_q_ss'].isel(y=0).values)
    params = onp.array(onp.meshgrid(b_q_ss, a_q_ss)).T.reshape(-1, 2)
    b_q_ss = params[:, 0].tolist()
    a_q_ss = params[:, 1].tolist()
    cmap = cm.get_cmap('Reds')
    norm = Normalize(vmin=0.1, vmax=10)

    for b in b_q_ss:
        rows = onp.where((ds_params['b_q_ss'].isel(y=0).values == b))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_b{onp.round(b, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    for a in a_q_ss:
        rows = onp.where((ds_params['a_q_ss'].isel(y=0).values == a))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_a{onp.round(a, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    tm_structure = 'preferential'
    tms = tm_structure.replace(" ", "_")
    # load transport simulation
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{tms}_deterministic.average.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    params_tm_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_tm_file, engine="h5netcdf", group=tm_structure)

    b_q_ss = onp.unique(ds_params['b_q_ss'].isel(y=0).values)
    a_q_ss = onp.unique(ds_params['a_q_ss'].isel(y=0).values)
    params = onp.array(onp.meshgrid(b_q_ss, a_q_ss)).T.reshape(-1, 2)
    b_q_ss = params[:, 0].tolist()
    a_q_ss = params[:, 1].tolist()
    cmap = cm.get_cmap('Reds')
    norm = Normalize(vmin=0.1, vmax=10)

    for b in b_q_ss:
        rows = onp.where((ds_params['b_q_ss'].isel(y=0).values == b))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_b{onp.round(b, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    for a in a_q_ss:
        rows = onp.where((ds_params['a_q_ss'].isel(y=0).values == a))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_a{onp.round(a, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    tm_structure = 'power'
    tms = tm_structure.replace(" ", "_")
    # load transport simulation
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{tms}_deterministic.average.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    params_tm_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_tm_file, engine="h5netcdf", group=tm_structure)

    k_transp = onp.unique(ds_params['k_transp'].isel(y=0).values)
    k_q_rz = onp.unique(ds_params['k_q_rz'].isel(y=0).values)
    params = onp.array(onp.meshgrid(k_transp, k_q_rz)).T.reshape(-1, 2)
    k_transp = params[:, 0].tolist()
    k_q_rz = params[:, 1].tolist()
    cmap = cm.get_cmap('Reds')
    norm = Normalize(vmin=0.5, vmax=5)

    for k1, k2 in zip(k_transp, k_q_rz):
        rows = onp.where((ds_params['k_transp'].isel(y=0).values == k1) & (ds_params['k_q_rz'].isel(y=0).values == k2))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['k_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['k_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['k_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['k_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_k{onp.round(k1, 1)}_k{onp.round(k2, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    tm_structure = 'older-preference'
    tms = tm_structure.replace(" ", "_")
    # load transport simulation
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{tms}_deterministic.average.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    params_tm_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_tm_file, engine="h5netcdf", group=tm_structure)

    for b in b_q_ss:
        rows = onp.where((ds_params['b_q_ss'].isel(y=0).values == b))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_b{onp.round(b, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    for a in a_q_ss:
        rows = onp.where((ds_params['a_q_ss'].isel(y=0).values == a))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_a{onp.round(a, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    tm_structure = 'preferential + advection-dispersion'
    tms = tm_structure.replace(" ", "_")
    # load transport simulation
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{tms}_deterministic.average.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    params_tm_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_tm_file, engine="h5netcdf", group=tm_structure)

    for b in b_q_ss:
        rows = onp.where((ds_params['b_q_ss'].isel(y=0).values == b))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_b{onp.round(b, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    for a in a_q_ss:
        rows = onp.where((ds_params['a_q_ss'].isel(y=0).values == a))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['b_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_a{onp.round(a, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    tm_structure = 'time-variant-transp'
    tms = tm_structure.replace(" ", "_")
    # load transport simulation
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{tms}_deterministic.average.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    params_tm_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_tm_file, engine="h5netcdf", group=tm_structure)

    c_transp = onp.unique(ds_params['c_transp'].isel(y=0).values)
    a_q_rz = onp.unique(ds_params['a_q_rz'].isel(y=0).values)
    params = onp.array(onp.meshgrid(c_transp, a_q_rz)).T.reshape(-1, 2)
    c_transp = params[:, 0].tolist()
    a_q_rz = params[:, 1].tolist()
    cmap = cm.get_cmap('Reds')
    norm = Normalize(vmin=2, vmax=10)

    for c, a in zip(c_transp, a_q_rz):
        rows = onp.where((ds_params['c_transp'].isel(y=0).values == c) & (ds_params['a_q_rz'].isel(y=0).values == a))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_c{onp.round(c, 1)}_a{onp.round(a, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')

    tm_structure = 'time-variant'
    tms = tm_structure.replace(" ", "_")
    # load transport simulation
    states_tm_file = base_path / sas_solver / age_max / metric_for_optimization / f"SVATTRANSPORT_{tms}_deterministic.average.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    params_tm_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_tm_file, engine="h5netcdf", group=tm_structure)

    c_transp = onp.unique(ds_params['c_transp'].isel(y=0).values)
    c_q_rz = onp.unique(ds_params['c_q_rz'].isel(y=0).values)
    params = onp.array(onp.meshgrid(c_transp, c_q_rz)).T.reshape(-1, 2)
    c_transp = params[:, 0].tolist()
    c_q_rz = params[:, 1].tolist()
    cmap = cm.get_cmap('Reds')
    norm = Normalize(vmin=2, vmax=10)

    for c1, c2 in zip(c_transp, c_q_rz):
        rows = onp.where((ds_params['c_transp'].isel(y=0).values == c1) & (ds_params['c_q_rz'].isel(y=0).values == c1))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='blue')
        ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel('$PERC_{rz}$\n$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
        ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time [year]')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_c{onp.round(c, 1)}_a{onp.round(a, 1)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)
    plt.close('all')
    #
    # for c1, c2 in zip(c_transp, c_q_rz):
    #     rows = onp.where((ds_params['c_transp'].isel(y=0).values == c1) & (ds_params['c_q_rz'].isel(y=0).values == c2))[0]
    #     fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
    #     for x in rows:
    #         ax.flatten()[0].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_evap_soil'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_q_ss'].isel(x=x, y=0).values)), lw=1)
    #     ax[0].set_ylabel('EVAP\n$\delta^{18}$O [‰]')
    #     ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_transp'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_q_ss'].isel(x=x, y=0).values)), lw=1)
    #     ax[1].set_ylabel('TRANSP\n$\delta^{18}$O [‰]')
    #     ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_q_ss'].isel(x=x, y=0).values)), lw=1)
    #     ax[2].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
    #     ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_cpr_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_q_ss'].isel(x=x, y=0).values)), lw=1)
    #     ax[3].set_ylabel('CPR\n$\delta^{18}$O [‰]')
    #     ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_q_ss'].isel(x=x, y=0).values)), lw=1)
    #     ax[4].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
    #     ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     ax[4].set_xlabel('Time [year]')
    #     fig.tight_layout()
    #     file = base_path_figs / f"d18O_uptake_{tms}_c{onp.round(c1, 1)}_c{onp.round(c2, 1)}_partial_q_ss.png"
    #     fig.savefig(file, dpi=250)
    # plt.close('all')
    #
    # c_q_rz = onp.unique(ds_params['c_q_rz'].isel(y=0).values)
    # c_q_ss = onp.unique(ds_params['c_q_ss'].isel(y=0).values)
    # params = onp.array(onp.meshgrid(c_q_rz, c_q_ss)).T.reshape(-1, 2)
    # c_q_rz = params[:, 0].tolist()
    # c_q_ss = params[:, 1].tolist()
    # cmap = cm.get_cmap('Greens')
    # norm = Normalize(vmin=1, vmax=10)
    #
    # for c1, c2 in zip(c_q_rz, c_q_ss):
    #     rows = onp.where((ds_params['c_q_rz'].isel(y=0).values == c1) & (ds_params['c_q_ss'].isel(y=0).values == c2))[0]
    #     fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
    #     ax.flatten()[0].plot(ds_sim_tm['Time'].values,
    #                          ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
    #                          '-', color='blue')
    #     ax.flatten()[0].set_ylabel('PREC\n$\delta^{18}$O [‰]')
    #     ax.flatten()[0].set_ylim([-20, 0])
    #     ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_transp'].isel(x=x, y=0).values)), lw=1)
    #     ax[1].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
    #     ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_transp'].isel(x=x, y=0).values)), lw=1)
    #     ax[2].set_ylabel(r'$\delta^{18}$O [‰]')
    #     ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_transp'].isel(x=x, y=0).values)), lw=1)
    #     ax[3].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
    #     ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_transp'].isel(x=x, y=0).values)), lw=1)
    #     ax.flatten()[4].scatter(ds_obs['Time'].values, ds_obs['d18O_PERC'].isel(x=0, y=0).values, color='blue', s=1)
    #     ax[4].set_ylabel('$PERC_{ss}$\n$\delta^{18}$O [‰]')
    #     ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     ax[4].set_xlabel('Time [year]')
    #     fig.tight_layout()
    #     file = base_path_figs / f"d18O_drain_{tms}_a{onp.round(a1, 1)}_a{onp.round(a2, 1)}_partial_transp.png"
    #     fig.savefig(file, dpi=250)
    # plt.close('all')
    #
    # for c1, c2 in zip(c_q_rz, c_q_ss):
    #     rows = onp.where((ds_params['c_q_rz'].isel(y=0).values == c1) & (ds_params['c_q_ss'].isel(y=0).values == c2))[0]
    #     fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
    #     for x in rows:
    #         ax.flatten()[0].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_evap_soil'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_transp'].isel(x=x, y=0).values)), lw=1)
    #     ax[0].set_ylabel('EVAP\n$\delta^{18}$O [‰]')
    #     ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_transp'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_transp'].isel(x=x, y=0).values)), lw=1)
    #     ax[1].set_ylabel('TRANSP\n$\delta^{18}$O [‰]')
    #     ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_transp'].isel(x=x, y=0).values)), lw=1)
    #     ax[2].set_ylabel('$S_{rz}$\n$\delta^{18}$O [‰]')
    #     ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_cpr_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_transp'].isel(x=x, y=0).values)), lw=1)
    #     ax[3].set_ylabel('CPR\n$\delta^{18}$O [‰]')
    #     ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     for x in rows:
    #         ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['c_transp'].isel(x=x, y=0).values)), lw=1)
    #     ax[4].set_ylabel('$S_{ss}$\n$\delta^{18}$O [‰]')
    #     ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    #     ax[4].set_xlabel('Time [year]')
    #     fig.tight_layout()
    #     file = base_path_figs / f"d18O_uptake_{tms}_c{onp.round(c1, 1)}_c{onp.round(c2, 1)}_partial_transp.png"
    #     fig.savefig(file, dpi=250)
    # plt.close('all')

    return


if __name__ == "__main__":
    main()
