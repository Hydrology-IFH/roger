from pathlib import Path
import os
import glob
import h5netcdf
import datetime
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import roger.tools.evaluation as eval_utils
import click
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
sns.set_style("ticks")


@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir, sas_solver):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of results
    base_path_results = base_path / "results"
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    base_path_results = base_path / "results" / sas_solver
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)
    base_path_figs = base_path / "figures" / sas_solver
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # merge model output into single file
    tm_structures = ['complete-mixing', 'piston',
                     'advection-dispersion',
                     'time-variant advection-dispersion']
    years = onp.arange(1997, 2007).tolist()
    states_tm_file = base_path / sas_solver / "states_bromide_benchmark.nc"
    if not os.path.exists(states_tm_file):
        for tm_structure in tm_structures:
            tms = tm_structure.replace(" ", "_")
            for year in years:
                path = str(base_path / sas_solver / f'SVATTRANSPORT_{tms}_{year}_{sas_solver}.*.nc')
                diag_files = glob.glob(path)
                with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
                    click.echo(f'Merge output files of {tm_structure}-{year} into {states_tm_file.as_posix()}')
                    if f"{tm_structure}-{year}" not in list(f.groups.keys()):
                        f.create_group(f"{tm_structure}-{year}")
                    f.attrs.update(
                        date_created=datetime.datetime.today().isoformat(),
                        title='RoGeR transport simulations for virtual bromide experiments at Rietholzbach Lysimeter site',
                        institution='University of Freiburg, Chair of Hydrology',
                        references='',
                        comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                        model_structure='SVAT transport model with free drainage',
                        sas_solver=f'{sas_solver}',
                    )
                    # collect dimensions
                    for dfs in diag_files:
                        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                            f.attrs.update(
                                roger_version=df.attrs['roger_version']
                            )
                            # set dimensions with a dictionary
                            if not dfs.split('/')[-1].split('.')[1] == 'constant':
                                dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
                                time = onp.array(df.variables.get('Time'))
                    for dfs in diag_files:
                        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                            if not f.groups[f"{tm_structure}-{year}"].dimensions:
                                f.groups[f"{tm_structure}-{year}"].dimensions = dict_dim
                                v = f.groups[f"{tm_structure}-{year}"].create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                                v.attrs['long_name'] = 'Number of model run'
                                v.attrs['units'] = ''
                                v[:] = onp.arange(dict_dim["x"])
                                v = f.groups[f"{tm_structure}-{year}"].create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                                v.attrs['long_name'] = ''
                                v.attrs['units'] = ''
                                v[:] = onp.arange(dict_dim["y"])
                                v = f.groups[f"{tm_structure}-{year}"].create_variable('ages', ('ages',), float, compression="gzip", compression_opts=1)
                                v.attrs['long_name'] = 'Water ages'
                                v.attrs['units'] = 'days'
                                v[:] = onp.arange(1, dict_dim["ages"]+1)
                                v = f.groups[f"{tm_structure}-{year}"].create_variable('nages', ('nages',), float, compression="gzip", compression_opts=1)
                                v.attrs['long_name'] = 'Water ages (cumulated)'
                                v.attrs['units'] = 'days'
                                v[:] = onp.arange(0, dict_dim["nages"])
                                v = f.groups[f"{tm_structure}-{year}"].create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                                var_obj = df.variables.get('Time')
                                v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                               units=var_obj.attrs["units"])
                                v[:] = time
                            for var_sim in list(df.variables.keys()):
                                var_obj = df.variables.get(var_sim)
                                if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions:
                                    v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                                    vals = onp.array(var_obj)
                                    v[:, :, :] = vals.swapaxes(0, 2)
                                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                                   units=var_obj.attrs["units"])
                                elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions:
                                    v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
                                    vals = onp.array(var_obj)
                                    vals = vals.swapaxes(0, 3)
                                    vals = vals.swapaxes(1, 2)
                                    v[:, :, :] = vals[:, :, :, 0]
                                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                                   units=var_obj.attrs["units"])
                                elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                                    v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float, compression="gzip", compression_opts=1)
                                    vals = onp.array(var_obj)
                                    vals = vals.swapaxes(0, 3)
                                    vals = vals.swapaxes(1, 2)
                                    vals = vals.swapaxes(2, 3)
                                    v[:, :, :, :] = vals
                                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                                   units=var_obj.attrs["units"])
                                elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                                    v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float, compression="gzip", compression_opts=1)
                                    vals = onp.array(var_obj)
                                    vals = vals.swapaxes(0, 3)
                                    vals = vals.swapaxes(1, 2)
                                    vals = vals.swapaxes(2, 3)
                                    v[:, :, :, :] = vals
                                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                                   units=var_obj.attrs["units"])

    years = onp.arange(1997, 2007).tolist()
    cmap = cm.get_cmap('Reds')
    norm = Normalize(vmin=onp.min(years), vmax=onp.max(years))
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        # load hydrologic simulation
        states_hm_file = base_path / f"states_hm_best_for_{tms}.nc"
        ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
        # assign date
        days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))
        df_metrics_year = pd.DataFrame(index=years)
        fig, axes = plt.subplots(1, 1, figsize=(6, 2))
        for year in years:
            click.echo(f'Calculate metrics for {tm_structure}-{year} ...')
            # load observations
            br_obs_file = base_path.parent / "observations" / "bromide_breakthrough.csv"
            df_br_obs = pd.read_csv(br_obs_file, sep=';', skiprows=1, index_col=0)
            # load simulation
            states_tm_file = base_path / sas_solver / "states_bromide_benchmark.nc"
            ds_sim_tm = xr.open_dataset(states_tm_file, group=f"{tm_structure}-{year}", engine="h5netcdf")
            # assign date
            days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))

            # plot percolation rate (in l/h) and bromide concentration (mmol/l)
            idx = pd.date_range(start=f'1/1/{year}', end=f'31/12/{year+1}')
            df_perc_br_sim = pd.DataFrame(index=idx, columns=['perc', 'Br_conc_mg', 'Br_mg', 'Br_conc_mmol'])
            # in mm per day
            df_perc_br_sim.loc[:, 'perc'] = ds_sim_hm.sel(Time=slice(str(year), str(year + 1)))['q_ss'].isel(y=0).values
            # in mg per liter
            df_perc_br_sim.loc[:, 'Br_conc_mg'] = ds_sim_tm['C_q_ss'].isel(x=0, y=0).values[1:]
            # in mg
            df_perc_br_sim.loc[:, 'Br_mg'] = ds_sim_tm['C_q_ss'].isel(x=0, y=0).values[1:] * ds_sim_hm.sel(Time=slice(str(year), str(year + 1)))['q_ss'].isel(y=0).values
            # in mmol per liter
            df_perc_br_sim.loc[:, 'Br_conc_mmol'] = (df_perc_br_sim.loc[:, 'Br_conc_mg'] / 79.904)
            # daily samples from day 0 to day 220
            df_daily = df_perc_br_sim.loc[:df_perc_br_sim.index[315+220], :]
            # weekly samples after 220 days
            df_weekly = df_perc_br_sim.loc[df_perc_br_sim.index[316+220]:, 'perc'].resample('7D').sum().to_frame()
            df_weekly.loc[:, 'Br_mg'] = df_perc_br_sim.loc[df_perc_br_sim.index[316+220]:, 'Br_mg'].resample('7D').sum()
            df_weekly.loc[:, 'Br_conc_mg'] = df_perc_br_sim.loc[df_perc_br_sim.index[316+220]:, 'Br_mg'].resample('7D').sum() / df_perc_br_sim.loc[df_perc_br_sim.index[316+220]:, 'perc'].resample('7D').sum()
            df_weekly.loc[:, 'Br_conc_mmol'] = (df_weekly.loc[:, 'Br_conc_mg'] / 79.904)
            df_daily_weekly = pd.concat([df_daily, df_weekly])
            df_perc_br_sim = pd.DataFrame(index=idx).join(df_daily_weekly)
            df_perc_br_sim = df_perc_br_sim.iloc[315:716, :]
            df_perc_br_sim.index = range(len(df_perc_br_sim.index))
            axes.plot(df_perc_br_sim.dropna().index, df_perc_br_sim.dropna()['Br_conc_mmol'], ls='-', color=cmap(norm(year)), label=f'{year}')

            # join observations on simulations
            obs_vals = df_br_obs.iloc[:, 0].values
            sim_vals = df_perc_br_sim.loc[:, 'Br_conc_mmol'].values
            df_obs = pd.DataFrame(index=df_br_obs.index, columns=['obs'])
            df_obs.loc[:, 'obs'] = obs_vals
            df_eval = eval_utils.join_obs_on_sim(df_perc_br_sim.index, sim_vals, df_obs)
            df_eval = df_eval.dropna()
            # calculate metrics
            obs_vals = df_eval.loc[:, 'obs'].values
            sim_vals = df_eval.loc[:, 'sim'].values
            # temporal correlation
            df_metrics_year.loc[year, 'r'] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
            # tracer recovery (in %)
            df_metrics_year.loc[year, 'Br_mass_recovery'] = onp.sum(ds_sim_tm['M_q_ss'].isel(x=0, y=0).values[315:716]) / (79900 / 3.14)
            # average travel time of percolation (in days)
            df_metrics_year.loc[year, 'ttavg'] = onp.nanmean(ds_sim_tm['ttavg_q_ss'].isel(x=0, y=0).values[315:716])
            # average median travel time of percolation (in days)
            df_metrics_year.loc[year, 'tt50'] = onp.nanmedian(ds_sim_tm['ttavg_q_ss'].isel(x=0, y=0).values[315:716])

            # write simulated bulk sample to output file
            ds_sim_tm = ds_sim_tm.load()  # required to release file lock
            ds_sim_tm = ds_sim_tm.close()
            del ds_sim_tm
            states_tm_file = base_path / sas_solver / "states_bromide_benchmark.nc"
            with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
                try:
                    v = f.groups[f"{tm_structure}-{year}"].create_variable('C_q_ss_mmol_bs', ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                    v[0, 0, 315:716] = df_perc_br_sim.loc[:, 'Br_conc_mmol'].values
                    v.attrs.update(long_name="bulk sample of bromide in percolation",
                                   units="mmol/l")
                except ValueError:
                    v = f.groups[f"{tm_structure}-{year}"].get('C_q_ss_mmol_bs')
                    v[0, 0, 315:716] = df_perc_br_sim.loc[:, 'Br_conc_mmol'].values
                try:
                    v = f.groups[f"{tm_structure}-{year}"].create_variable('q_ss_bs', ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                    v[0, 0, 315:716] = df_perc_br_sim.loc[:, 'perc'].values
                    v.attrs.update(long_name="bulk sample of percolation",
                                   units="mm/dt")
                except ValueError:
                    v = f.groups[f"{tm_structure}-{year}"].get('M_q_ss_bs')
                    v[0, 0, 315:716] = df_perc_br_sim.loc[:, 'perc'].values
                try:
                    v = f.groups[f"{tm_structure}-{year}"].create_variable('M_q_ss_bs', ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                    v[0, 0, 315:716] = df_perc_br_sim.loc[:, 'Br_mg'].values
                    v.attrs.update(long_name="bulk sample bromide mass in percolation",
                                   units="mg")
                except ValueError:
                    v = f.groups[f"{tm_structure}-{year}"].get('M_q_ss_bs')
                    v[0, 0, 315:716] = df_perc_br_sim.loc[:, 'Br_mg'].values

        axes.set_ylabel('Br [mmol $l^{-1}$]')
        axes.set_xlabel('Time [days since injection]')
        axes.set_ylim(0,)
        axes.set_xlim((0, 400))
        axes.legend(fontsize=6, frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
        fig.tight_layout()
        file = f'bromide_breakthrough_{tms}.png'
        path = base_path_figs / file
        fig.savefig(path, dpi=250)

        # write evaluation metrics to .csv
        path_csv = base_path_results / f"bromide_metrics_{tms}.csv"
        df_metrics_year.to_csv(path_csv, header=True, index=True, sep=";")

    return


if __name__ == "__main__":
    main()
