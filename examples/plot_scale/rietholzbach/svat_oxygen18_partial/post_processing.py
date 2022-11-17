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


@click.option("-ns", "--nsamples", type=int, default=10000)
@click.option("-ss", "--split-size", type=int, default=1000)
@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(nsamples, split_size, sas_solver, tmp_dir):
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

    tm_structure = 'advection-dispersion'
    tms = tm_structure.replace(" ", "_")
    # load transport simulation
    states_tm_file = base_path / f"SVATTRANSPORT_{tms}_deterministic.average.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    params_tm_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_tm_file, engine="h5netcdf", group=tm_structure)

    b_transp = onp.unique(ds_params['b_transp'].isel(y=0).values).tolist()
    a_q_rz = onp.unique(ds_params['a_q_rz'].isel(y=0).values).tolist()
    cmap = cm.get_cmap('Reds_r')
    norm = Normalize(vmin=2, vmax=10)

    for b, a in zip(b_transp, a_q_rz):
        rows = onp.where((ds_params['b_transp'].isel(y=0).values == b) & (ds_params['a_q_rz'].isel(y=0).values == a))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='black')
        ax.flatten()[0].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[4].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_b{int(b)}_a{int(a)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)

    for b, a in zip(b_transp, a_q_rz):
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        for x in rows:
            ax.flatten()[0].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_evap_soil'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[0].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_transp'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_cpr_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[4].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time')
        fig.tight_layout()
        file = base_path_figs / f"d18O_uptake_{tms}_b{int(b)}_a{int(a)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)

    b_transp = onp.unique(ds_params['b_transp'].isel(y=0).values).tolist()
    a_q_rz = onp.unique(ds_params['a_q_rz'].isel(y=0).values).tolist()
    cmap = cm.get_cmap('Reds_r')
    norm = Normalize(vmin=2, vmax=10)

    for b, a in zip(b_transp, a_q_rz):
        rows = onp.where((ds_params['b_transp'].isel(y=0).values == b) & (ds_params['a_q_rz'].isel(y=0).values == a))[0]
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                             ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                             '-', color='black')
        ax.flatten()[0].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[0].set_ylim([-20, 0])
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[4].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time')
        fig.tight_layout()
        file = base_path_figs / f"d18O_drain_{tms}_b{int(b)}_a{int(a)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)

    for b, a in zip(b_transp, a_q_rz):
        fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
        for x in rows:
            ax.flatten()[0].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_evap_soil'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[0].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_transp'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[1].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[2].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_cpr_rz'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[3].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        for x in rows:
            ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(norm(ds_params['a_q_ss'].isel(x=x, y=0).values)), lw=1)
        ax[4].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
        ax[4].set_xlabel('Time')
        fig.tight_layout()
        file = base_path_figs / f"d18O_uptake_{tms}_b{int(b)}_a{int(a)}_partial_q_ss.png"
        fig.savefig(file, dpi=250)


    plt.close('all')
    return


if __name__ == "__main__":
    main()
