import os
from pathlib import Path
import glob
import h5netcdf
import datetime
import numpy as onp

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# merge nitrate model output into single file
lys_experiments = ["lys2", "lys3", "lys4", "lys8", "lys9"]
tm_structures = ['complete-mixing', 'piston',
                 'preferential', 'complete-mixing + advection-dispersion',
                 'time-variant preferential',
                 'time-variant complete-mixing + advection-dispersion']
for lys_experiment in lys_experiments:
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        path = str(base_path / f"SVATCROPTRANSPORT_{tms}_{lys_experiment}_nitrate.*.nc")
        diag_files = glob.glob(path)
        states_tm_file = base_path / "states_tm_monte_carlo_rev_nitrate.nc"
        states_hm_file = base_path / "states_hm.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            if lys_experiment not in list(f.groups.keys()):
                f.create_group(lys_experiment)
            if tm_structure not in list(f.groups[lys_experiment].groups.keys()):
                f.groups[lys_experiment].create_group(tm_structure)
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR nitrate transport model Monte Carlo simulations reverse at Reckenholz Lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='SVAT transport model with free drainage and crop phenology/crop rotation'
            )
            # collect dimensions
            for dfs in diag_files:
                with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                    # set dimensions with a dictionary
                    if not dfs.split('/')[-1].split('.')[1] == 'constant':
                        dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
                        time = onp.array(df.variables.get('Time'))
            for dfs in diag_files:
                with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                    # set dimensions with a dictionary
                    if not f.groups[lys_experiment].groups[tm_structure].dimensions:
                        f.groups[lys_experiment].groups[tm_structure].dimensions = dict_dim
                        v = f.groups[lys_experiment].groups[tm_structure].create_variable('x', ('x',), float)
                        v.attrs['long_name'] = 'model run'
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["x"])
                        v = f.groups[lys_experiment].groups[tm_structure].create_variable('y', ('y',), float)
                        v.attrs['long_name'] = ''
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["y"])
                        v = f.groups[lys_experiment].groups[tm_structure].create_variable('Time', ('Time',), float)
                        var_obj = df.variables.get('Time')
                        v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                       units=var_obj.attrs["units"])
                        v[:] = time
                        v = f.groups[lys_experiment].groups[tm_structure].create_variable('ages', ('ages',), float)
                        v.attrs['long_name'] = 'Water ages'
                        v.attrs['units'] = 'days'
                        v[:] = onp.arange(1, dict_dim["ages"]+1)
                        v = f.groups[lys_experiment].groups[tm_structure].create_variable('nages', ('nages',), float)
                        v.attrs['long_name'] = 'Water ages (cumulated)'
                        v.attrs['units'] = 'days'
                        v[:] = onp.arange(0, dict_dim["nages"])
                        v = f.groups[lys_experiment].groups[tm_structure].create_variable('n_sas_params', ('n_sas_params',), float)
                        v.attrs['long_name'] = 'Number of SAS parameters'
                        v.attrs['units'] = ''
                    for var_sim in list(df.variables.keys()):
                        var_obj = df.variables.get(var_sim)
                        if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] > 2:
                            v = f.groups[lys_experiment].groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time'), float)
                            vals = onp.array(var_obj)
                            v[:, :, :] = vals.swapaxes(0, 2)
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] <= 2:
                            v = f.groups[lys_experiment].groups[tm_structure].create_variable(var_sim, ('x', 'y', 'n_sas_params'), float)
                            vals = onp.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            v[:, :, :] = vals[:, :, :, 0]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[lys_experiment].groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float)
                            vals = onp.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[lys_experiment].groups[tm_structure].create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float)
                            vals = onp.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
