from pathlib import Path
import pandas as pd
import h5netcdf
import datetime

base_path = Path(__file__).parent
csv_file = base_path / "initvals.csv"
df_initvals = pd.read_csv(csv_file, sep=";", header=0, index_col=0)

nc_file = base_path / "initvals.nc"
with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title='Initial values of Reckenholz lysimeters',
        institution='University of Freiburg, Chair of Hydrology',
        references='',
        comment=''
    )
    for lys_experiment in list(df_initvals.columns):
        f.create_group(lys_experiment)
        # set dimensions with a dictionary
        dict_dim = {'x': 1, 'y': 1}
        if not f.dimensions:
            f.dimensions = dict_dim
            v = f.groups[lys_experiment].create_variable('x', ('x',), float)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = dict_dim["x"]
            v = f.groups[lys_experiment].create_variable('y', ('y',), float)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'
            v[:] = dict_dim["y"]

        for key in list(df_initvals.index):
                v = f.groups[lys_experiment].create_variable(key, ('x', 'y'), float)
                v[:, :] = df_initvals.loc[key, lys_experiment]
                v.attrs.update(long_name=key,
                                units="-")
