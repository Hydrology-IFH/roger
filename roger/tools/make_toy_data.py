import os
import numpy as onp
import datetime
from datetime import timedelta
import pandas as pd
import scipy.special as sps
import h5netcdf
from cftime import date2num
onp.random.seed(42)


def sin_func(t, amp, phase, off):
    return amp * onp.sin(2*onp.pi*t - phase) + off


def make_toy_forcing(base_path, ndays=10, nrows=1, ncols=1,
                     event_type='rain', enable_groundwater_boundary=False,
                     enable_crop_phenology=False,
                     float_type="float64"):
    """
    Make toy forcing with synthetically generated data.
    """
    rng = onp.random.default_rng(42)
    if event_type == 'rain':
        # generate random rainfall
        prec_rnd = rng.uniform(0, 1, 18)
        n_prec_rnd = len(prec_rnd)
        n_prec = ndays*24*6
        prec = onp.zeros((n_prec))
        prec[12:12+n_prec_rnd] = prec_rnd
        prec[int(n_prec/2):int(n_prec/2)+n_prec_rnd] = prec_rnd
        idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
        # generate random air temperature
        idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
        ta = rng.uniform(15, 20, ndays)
        # generate random potential evapotranspiration
        pet = rng.uniform(2, 3, ndays)

    elif event_type == 'snow':
        # generate random rainfall
        prec_rnd = rng.uniform(0, 1, 18)
        n_prec_rnd = len(prec_rnd)
        n_prec = ndays*24*6
        prec = onp.zeros((n_prec))
        prec[12:12+n_prec_rnd] = prec_rnd
        prec[int(n_prec/2):int(n_prec/2)+n_prec_rnd] = prec_rnd
        idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
        # generate random air temperature
        idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
        ta = rng.uniform(-3, -1, ndays)
        # generate random potential evapotranspiration
        pet = rng.uniform(1, 2, ndays)

    elif event_type == 'snow+rain':
        # generate random rainfall
        prec_rnd = rng.uniform(0, 1, 18)
        n_prec_rnd = len(prec_rnd)
        n_prec = ndays*24*6
        prec = onp.zeros((n_prec))
        prec[12:12+n_prec_rnd] = prec_rnd
        prec[int(n_prec/2):int(n_prec/2)+n_prec_rnd] = prec_rnd
        idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
        # generate random air temperature
        idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
        ta = rng.uniform(0, 3, ndays)
        ta[:2] = -1
        # generate random potential evapotranspiration
        pet = rng.uniform(1, 2, ndays)

    elif event_type == 'heavyrain':
        prec_rnd = rng.uniform(0.1, 6, 12*6)
        n_prec_rnd = len(prec_rnd)
        n_prec = ndays*24*6
        prec = onp.zeros((n_prec))
        prec[12:12+n_prec_rnd] = prec_rnd
        prec[int(n_prec/2):int(n_prec/2)+n_prec_rnd] = prec_rnd
        idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
        # generate random air temperature
        idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
        ta = rng.uniform(15, 20, ndays)
        # generate random potential evapotranspiration
        pet = rng.uniform(2, 3, ndays)

    elif event_type == 'mixed':
        x1 = onp.unique(rng.randint(low=0, high=ndays * 24 * 6, size=(int(100 * (ndays/365)),)))
        x2 = onp.zeros((int(100 * (ndays/365)),), dtype=int)
        for i in range(int(100 * (ndays/365)) - 1):
            if x1[i+1] - x1[i] <= 1:
                high = 2
            else:
                high = x1[i+1] - x1[i]
            x2[i] = x1[i] + rng.randint(low=1, high=high)
        x2[x2 > ndays * 24 * 6] = ndays * 24 * 6
        x2[-1] = rng.randint(low=x1[-1] + 1, high=ndays * 24 * 6)

        prec = onp.zeros((ndays * 24 * 6,))
        for i, ii in zip(x1, x2):
            lam = rng.weibull(1, 1)
            if lam > 5.5:
                prec[i:ii] = rng.poisson(lam, ii - i) * 0.5
            else:
                prec[i:ii] = rng.poisson(lam, ii - i) * rng.uniform(0.01, 0.1, ii - i)
        idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
        # generate random air temperature
        idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
        ta_init = -5
        ta_off = 35
        pet_init = 1.5
        pet_off = 4
        ta = onp.zeros((ndays,))
        pet = onp.zeros((ndays,))
        scale = onp.sin(onp.linspace(0, onp.pi, 365))
        ii = 0
        for i in range(ndays):
            if i % 365 == 0:
                ii = 0
            ta[i] = rng.uniform(ta_init - 1 + scale[ii] * ta_off, ta_init + 1 + scale[ii] * ta_off, 1)
            # generate random potential evapotranspiration
            pet[i] = rng.uniform(pet_init - 1 + scale[ii] * pet_off, pet_init + 1 + scale[ii] * pet_off, 1)
            ii += 1

    elif event_type == 'norain':
        # generate random rainfall
        prec = onp.zeros((ndays*24*6))
        idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
        # generate random air temperature
        idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
        ta = rng.uniform(15, 20, ndays)
        # generate random potential evapotranspiration
        pet = rng.uniform(2, 3, ndays)

    df_prec = pd.DataFrame(index=idx_prec, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'PREC'])
    df_prec.loc[:, 'YYYY'] = df_prec.index.year
    df_prec.loc[:, 'MM'] = df_prec.index.month
    df_prec.loc[:, 'DD'] = df_prec.index.day
    df_prec.loc[:, 'hh'] = df_prec.index.hour
    df_prec.loc[:, 'mm'] = df_prec.index.minute
    df_prec.loc[:, 'PREC'] = prec

    df_ta = pd.DataFrame(index=idx_ta_pet, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'TA'])
    df_ta.loc[:, 'YYYY'] = df_ta.index.year
    df_ta.loc[:, 'MM'] = df_ta.index.month
    df_ta.loc[:, 'DD'] = df_ta.index.day
    df_ta.loc[:, 'hh'] = df_ta.index.hour
    df_ta.loc[:, 'mm'] = df_ta.index.minute
    df_ta.loc[:, 'TA'] = ta
    if enable_crop_phenology:
        df_ta.loc[:, 'TA_min'] = ta - 3
        df_ta.loc[:, 'TA_max'] = ta + 3

    df_pet = pd.DataFrame(index=idx_ta_pet, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'PET'])
    df_pet.loc[:, 'YYYY'] = df_pet.index.year
    df_pet.loc[:, 'MM'] = df_pet.index.month
    df_pet.loc[:, 'DD'] = df_pet.index.day
    df_pet.loc[:, 'hh'] = df_pet.index.hour
    df_pet.loc[:, 'mm'] = df_pet.index.minute
    df_pet.loc[:, 'PET'] = pet

    if enable_crop_phenology:
        df_meteo = df_prec['PREC'].to_frame().join([df_ta.loc[:, 'TA':'TA_max'], df_pet['PET'].to_frame()])
    else:
        df_meteo = df_prec['PREC'].to_frame().join([df_ta['TA'].to_frame(), df_pet['PET'].to_frame()])
    df_meteo = df_meteo.ffill()
    # downscale daily PET to 10 minutes
    df_meteo.loc[:, 'PET'] = (df_meteo.loc[:, 'PET'] / 24) / 6

    input_dir = base_path / "input"
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    nc_file = input_dir / "forcing.nc"
    with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='Meteorological toy forcing',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment=''
        )
        # set dimensions with a dictionary
        dict_dim = {'x': nrows, 'y': ncols, 'Time': len(df_meteo.index), 'scalar': 1}
        f.dimensions = dict_dim
        v = f.create_variable('PREC', ('x', 'y', 'Time'), float_type)
        arr = df_meteo['PREC'].astype(float_type).values
        v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
        v.attrs['long_name'] = 'Precipitation'
        v.attrs['units'] = 'mm/10 minutes'
        v = f.create_variable('TA', ('x', 'y', 'Time'), float_type)
        arr = df_meteo['TA'].astype(float_type).values
        v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
        v.attrs['long_name'] = 'Air temperature'
        v.attrs['units'] = 'degC'
        v = f.create_variable('PET', ('x', 'y', 'Time'), float_type)
        arr = df_meteo['PET'].astype(float_type).values
        v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
        v.attrs['long_name'] = 'Potential Evapotranspiration'
        v.attrs['units'] = 'mm/10 minutes'
        v = f.create_variable('dt', ('Time',), float_type)
        v[:] = 10 * 60
        v.attrs['long_name'] = 'time step'
        v.attrs['units'] = 'seconds'
        v = f.create_variable('YEAR', ('Time',), int)
        v[:] = df_meteo.index.year.astype(int).values
        v.attrs['units'] = 'year'
        v = f.create_variable('MONTH', ('Time',), int)
        v[:] = df_meteo.index.month.astype(int).values
        v.attrs['units'] = 'month'
        v = f.create_variable('DOY', ('Time',), int)
        v[:] = df_meteo.index.dayofyear.astype(int).values
        v.attrs['units'] = 'day of year'
        v = f.create_variable('Time', ('Time',), float_type)
        time_origin = df_meteo.index[0] - timedelta(hours=1)
        v.attrs['time_origin'] = f"{time_origin}"
        v.attrs['units'] = 'hours'
        v[:] = date2num(df_meteo.index.tolist(), units=f"hours since {time_origin}", calendar='standard')
        v = f.create_variable('x', ('x',), int)
        v.attrs['long_name'] = 'x'
        v.attrs['units'] = ''
        v[:] = onp.arange(nrows)
        v = f.create_variable('y', ('y',), int)
        v.attrs['long_name'] = 'y'
        v.attrs['units'] = ''
        v[:] = onp.arange(ncols)
        if enable_crop_phenology:
            v = f.create_variable('TA_min', ('x', 'y', 'Time'), float_type)
            arr = df_meteo['TA_min'].values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'minimum air temperature'
            v.attrs['units'] = 'degC'
            v = f.create_variable('TA_max', ('x', 'y', 'Time'), float_type)
            arr = df_meteo['TA_max'].values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'maximum air temperature'
            v.attrs['units'] = 'degC'
        if enable_groundwater_boundary:
            v = f.create_variable('z_gw', ('x', 'y', 'Time'), float_type)
            v[:, :, :] = 3
            v.attrs['long_name'] = 'depth of groundwater table'
            v.attrs['units'] = 'm'


def make_toy_forcing_event(base_path, ta=10, nhours=5, dt=10, nrows=1, ncols=1, event_type='rain', rain_sum=10, heavyrain_sum=60, float_type="float64"):
    """
    Make toy forcing for a single event with synthetically generated data.
    """
    rng = onp.random.default_rng(42)
    n_prec = int(nhours * (60/dt))  # number of rainfall intervals

    if event_type == 'rain':
        # generate random rainfall
        pp = rng.uniform(0.1, 1, n_prec)
        scale = rain_sum / onp.sum(pp)
        prec = pp * scale

    elif event_type == 'block-rain':
        prec = (rain_sum / nhours) / (60/dt)

    elif event_type == 'rain-with-break':
        # generate random rainfall
        pp = rng.uniform(0.1, 1, n_prec)
        start = int(n_prec/2) - 2
        end = int(n_prec/2) + 2
        pp[start:end] = 0
        scale = rain_sum / onp.sum(pp)
        prec = pp * scale

    elif event_type == 'heavyrain':
        # generate random rainfall
        pp = rng.uniform(0.1, 1, n_prec)
        scale = heavyrain_sum / onp.sum(pp)
        prec = pp * scale

    elif event_type == 'heavyrain-normal':
        # generate rainfall with normal distribution
        mu = 2
        sigma = 0.5
        s = rng.normal(mu, sigma, 1000)
        _, bins = onp.histogram(s, bins=n_prec-1)
        pp = 1/(sigma * onp.sqrt(2 * onp.pi)) * onp.exp(-(bins - mu)**2 / (2 * sigma**2))
        scale = heavyrain_sum / onp.sum(pp)
        prec = pp * scale

    elif event_type == 'heavyrain-gamma':
        # generate rainfall with light tail
        shape, scale = 2., 2.
        s = rng.gamma(shape, scale, 1000)
        _, bins = onp.histogram(s, bins=n_prec-1)
        pp = bins**(shape-1)*(onp.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
        scale = heavyrain_sum / onp.sum(pp)
        prec = pp * scale

    elif event_type == 'heavyrain-gamma-reverse':
        # generate rainfall with heavy tail
        shape, scale = 2., 2.
        s = rng.gamma(shape, scale, 1000)
        _, bins = onp.histogram(s, bins=n_prec-1)
        pp = bins**(shape-1)*(onp.exp(-bins/scale) / (sps.gamma(shape)*scale**shape))
        scale = heavyrain_sum / onp.sum(pp)
        prec = pp[::-1] * scale

    elif event_type == 'block-heavyrain':
        prec = (heavyrain_sum / nhours) / (60/dt)

    idx = onp.arange(n_prec) * dt
    df_prec = pd.DataFrame(index=idx, columns=['DD', 'hh', 'mm', 'PREC'])
    df_prec.loc[:, 'DD'] = idx / (60 * 60)
    df_prec.loc[:, 'hh'] = idx / 60
    df_prec.loc[:, 'mm'] = idx
    df_prec.loc[:, 'PREC'] = prec
    df_prec.loc[0, 'PREC'] = 0

    df_ta = pd.DataFrame(index=idx, columns=['DD', 'hh', 'mm', 'TA'])
    df_ta.loc[:, 'DD'] = idx / (60 * 60)
    df_ta.loc[:, 'hh'] = idx / 60
    df_ta.loc[:, 'mm'] = idx
    df_ta.loc[:, 'TA'] = ta

    df_meteo = df_prec.join(df_ta["TA"].to_frame())
    df_meteo = df_meteo.ffill()

    input_dir = base_path / "input"
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)

    nc_file = input_dir / "forcing.nc"
    with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='Meteorological toy forcing of a single event',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment=''
        )
        # set dimensions with a dictionary
        dict_dim = {'x': nrows, 'y': ncols, 'Time': len(df_meteo.index), 'scalar': 1}
        f.dimensions = dict_dim
        v = f.create_variable('PREC', ('x', 'y', 'Time'), float_type)
        arr = df_meteo['PREC'].astype(float_type).values
        v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
        v.attrs['long_name'] = 'Precipitation'
        v.attrs['units'] = 'mm/dt'
        v = f.create_variable('TA', ('x', 'y', 'Time'), float_type)
        arr = df_meteo['TA'].astype(float_type).values
        v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
        v.attrs['long_name'] = 'Air temperature'
        v.attrs['units'] = 'degC'
        v = f.create_variable('dt', ('Time',), int)
        time_steps = onp.around(onp.diff(df_meteo["hh"].values) * 60 * 60, 1)
        v[:-1] = time_steps.astype(int)
        v[-1] = time_steps[-1].astype(int)
        v.attrs['long_name'] = 'time step'
        v.attrs['units'] = 'seconds'
        v = f.create_variable('Time', ('Time',), float_type)
        v.attrs['units'] = 'hours'
        v[:] = df_meteo["hh"].values
        v = f.create_variable('x', ('x',), int)
        v.attrs['long_name'] = 'Zonal coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(nrows)
        v = f.create_variable('y', ('y',), int)
        v.attrs['long_name'] = 'Meridonial coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(ncols)


def make_toy_forcing_tracer(base_path, tracer="Br", start_date='1/10/2010', ndays=10, nrows=1, ncols=1,
                            float_type="float64"):
    """
    Make toy forcing with synthetically generated data.
    """
    rng = onp.random.default_rng(42)
    input_dir = base_path / "input"
    if not os.path.exists(input_dir):
        os.mkdir(input_dir)
    nc_file = input_dir / "forcing_tracer.nc"
    if tracer == "Br":
        idx = pd.date_range(start=start_date, periods=ndays, freq='D')
        df_tracer = pd.DataFrame(index=idx, columns=['Br'])
        df_tracer.iloc[:, 0] = 0
        df_tracer.iloc[2, 0] = 1000
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='Bromide toy forcing',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            dict_dim = {'x': nrows, 'y': ncols, 'Time': len(df_tracer.index), 'scalar': 1}
            f.dimensions = dict_dim
            v = f.create_variable(tracer, ('x', 'y', 'Time'), float_type)
            arr = df_tracer[tracer].astype(float_type).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'Bromide'
            v.attrs['units'] = 'mg'
            v = f.create_variable('Time', ('Time',), float_type)
            time_origin = df_tracer.index[0] - timedelta(hours=24)
            v.attrs['time_origin'] = f"{time_origin}"
            v.attrs['units'] = 'hours'
            v[:] = date2num(df_tracer.index.tolist(), units=f"hours since {time_origin}", calendar='standard')
            v = f.create_variable('x', ('x',), int)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(nrows)
            v = f.create_variable('y', ('y',), int)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'

    elif tracer == "Cl":
        idx = pd.date_range(start=start_date, periods=ndays, freq='D')
        df_tracer = pd.DataFrame(index=idx, columns=['Cl'])
        df_tracer.iloc[:, 0] = rng.uniform(0.1, 1, ndays)
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='Chloride toy forcing',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            dict_dim = {'x': nrows, 'y': ncols, 'Time': len(df_tracer.index), 'scalar': 1}
            f.dimensions = dict_dim
            v = f.create_variable(tracer, ('x', 'y', 'Time'), float_type)
            arr = df_tracer[tracer].astype(float_type).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'Chloride'
            v.attrs['units'] = 'mg'
            v = f.create_variable('Time', ('Time',), float_type)
            time_origin = df_tracer.index[0] - timedelta(hours=24)
            v.attrs['time_origin'] = f"{time_origin}"
            v.attrs['units'] = 'hours'
            v[:] = date2num(df_tracer.index.tolist(), units=f"hours since {time_origin}", calendar='standard')
            v = f.create_variable('x', ('x',), int)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(nrows)
            v = f.create_variable('y', ('y',), int)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'

    elif tracer == "d2H":
        idx = pd.date_range(start=start_date, periods=ndays, freq='D')
        df_tracer = pd.DataFrame(index=idx, columns=['d2H'])
        offset = -75 + rng.uniform(-2, 2, ndays)
        amp = 35 + rng.uniform(-2, 2, ndays)
        t = (onp.arange(0, ndays) % 365) / 365
        df_tracer.iloc[:, 0] = sin_func(t, amp, 1.5, offset)
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='Deuterium toy forcing',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            dict_dim = {'x': nrows, 'y': ncols, 'Time': len(df_tracer.index), 'scalar': 1}
            f.dimensions = dict_dim
            v = f.create_variable(tracer, ('x', 'y', 'Time'), float_type)
            arr = df_tracer[tracer].astype(float_type).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'Deuterium'
            v.attrs['units'] = 'permil'
            v = f.create_variable('Time', ('Time',), float_type)
            time_origin = df_tracer.index[0] - timedelta(hours=24)
            v.attrs['time_origin'] = f"{time_origin}"
            v.attrs['units'] = 'hours'
            v[:] = date2num(df_tracer.index.tolist(), units=f"hours since {time_origin}", calendar='standard')
            v = f.create_variable('x', ('x',), int)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(nrows)
            v = f.create_variable('y', ('y',), int)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'

    elif tracer == "d18O":
        idx = pd.date_range(start=start_date, periods=ndays, freq='D')
        df_tracer = pd.DataFrame(index=idx, columns=['d18O'])
        offset = -10 + rng.uniform(-0.5, 0.5, ndays)
        amp = 4.3 + rng.uniform(-0.5, 0.5, ndays)
        t = (onp.arange(0, ndays) % 365) / 365
        df_tracer.iloc[:, 0] = sin_func(t, amp, 60, offset)
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='Oxygen-18 toy forcing',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            dict_dim = {'x': nrows, 'y': ncols, 'Time': len(df_tracer.index), 'scalar': 1}
            f.dimensions = dict_dim
            v = f.create_variable(tracer, ('x', 'y', 'Time'), float_type)
            arr = df_tracer[tracer].astype(float_type).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'Oxygen-18'
            v.attrs['units'] = 'permil'
            v = f.create_variable('Time', ('Time',), float_type)
            time_origin = df_tracer.index[0] - timedelta(hours=24)
            v.attrs['time_origin'] = f"{time_origin}"
            v.attrs['units'] = 'hours'
            v[:] = date2num(df_tracer.index.tolist(), units=f"hours since {time_origin}", calendar='standard')
            v = f.create_variable('x', ('x',), int)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(nrows)
            v = f.create_variable('y', ('y',), int)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'

    elif tracer == "NO3":
        idx = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
        df_tracer = pd.DataFrame(index=idx, columns=['Nmin'])
        df_tracer.iloc[:, 0] = 0
        df_tracer.iloc[2, 0] = 100
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='Nitrate toy forcing',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            dict_dim = {'x': nrows, 'y': ncols, 'Time': len(df_tracer.index), 'scalar': 1}
            f.dimensions = dict_dim
            v = f.create_variable(tracer, ('x', 'y', 'Time'), float_type)
            arr = df_tracer[tracer].astype(float_type).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'Mineral nitrogen fertilizer'
            v.attrs['units'] = 'kg/ha'
            v = f.create_variable('Norg', ('x', 'y', 'Time'), float_type)
            v[:, :, :] = 0
            v.attrs['long_name'] = 'Organic nitrogen fertilizer'
            v.attrs['units'] = 'kg/ha'
            v = f.create_variable('Time', ('Time',), float_type)
            time_origin = df_tracer.index[0] - timedelta(hours=24)
            v.attrs['time_origin'] = f"{time_origin}"
            v.attrs['units'] = 'hours'
            v[:] = date2num(df_tracer.index.tolist(), units=f"hours since {time_origin}", calendar='standard')
            v = f.create_variable('x', ('x',), int)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(nrows)
            v = f.create_variable('y', ('y',), int)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'
