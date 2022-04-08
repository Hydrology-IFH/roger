from roger.core.operators import numpy as npx
import roger.tools.event_classification as ecl
import numpy as onp
import pandas as pd
import h5netcdf
from cftime import date2num
import datetime

import scipy.interpolate
import scipy.spatial


def interpolate(coords, var, interp_coords, missing_value=None, fill=True, kind="linear"):
    """Interpolate globally defined data to a different (regular) grid.

    Arguments:
       coords: Tuple of coordinate arrays for each dimension.
       var (:obj:`ndarray` of dim (nx1, ..., nxd)): Variable data to interpolate.
       interp_coords: Tuple of coordinate arrays to interpolate to.
       missing_value (optional): Value denoting cells of missing data in ``var``.
          Is replaced by `NaN` before interpolating. Defaults to `None`, which means
          no replacement is taking place.
       fill (bool, optional): Whether `NaN` values should be replaced by the nearest
          finite value after interpolating. Defaults to ``True``.
       kind (str, optional): Order of interpolation. Supported are `nearest` and
          `linear` (default).

    Returns:
       :obj:`ndarray` containing the interpolated values on the grid spanned by
       ``interp_coords``.

    """
    if len(coords) != len(interp_coords) or len(coords) != var.ndim:
        raise ValueError("Dimensions of coordinates and values do not match")

    if missing_value is not None:
        invalid_mask = npx.isclose(var, missing_value)
        var = npx.where(invalid_mask, npx.nan, var)

    if var.ndim > 1 and coords[0].ndim == 1:
        interp_grid = npx.rollaxis(npx.array(npx.meshgrid(*interp_coords, indexing="ij")), 0, len(interp_coords) + 1)
    else:
        interp_grid = interp_coords

    coords = [onp.array(c) for c in coords]
    var = scipy.interpolate.interpn(
        coords, onp.array(var), interp_grid, bounds_error=False, fill_value=npx.nan, method=kind
    )
    var = npx.asarray(var)

    if fill:
        var = fill_holes(var)

    return var


def fill_holes(data):
    """A simple inpainting function that replaces NaN values in `data` with the
    nearest finite value.
    """
    data = onp.array(data)
    dim = data.ndim
    flag = ~onp.isnan(data)

    slcs = [slice(None)] * dim

    while onp.any(~flag):
        for i in range(dim):
            slcs1 = slcs[:]
            slcs2 = slcs[:]
            slcs1[i] = slice(0, -1)
            slcs2[i] = slice(1, None)

            slcs1 = tuple(slcs1)
            slcs2 = tuple(slcs2)

            # replace from the right
            repmask = onp.logical_and(~flag[slcs1], flag[slcs2])
            data[slcs1][repmask] = data[slcs2][repmask]
            flag[slcs1][repmask] = True

            # replace from the left
            repmask = onp.logical_and(~flag[slcs2], flag[slcs1])
            data[slcs2][repmask] = data[slcs1][repmask]
            flag[slcs2][repmask] = True

    return npx.asarray(data)


def get_uniform_grid_steps(total_length, stepsize):
    """Get uniform grid step sizes in an interval.

    Arguments:
        total_length (float): total length of the resulting grid
        stepsize (float): grid step size

    Returns:
        :obj:`ndarray` of grid steps

    Example:
        >>> uniform_steps = uniform_grid_setup(6., 0.25)
        >>> uniform_steps
        [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
          0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
          0.25,  0.25,  0.25,  0.25,  0.25,  0.25 ]

    """
    if total_length % stepsize:
        raise ValueError("total length must be an integer multiple of stepsize")
    return stepsize * npx.ones(int(total_length / stepsize))


def write_forcing(input_dir, nrows=1, ncols=1, hpi=5, end_prec_event=36, sf=3,
                  ta_fm=0, uniform=True, enable_film_flow=False,
                  enable_crop_phenology=False,
                  prec_correction=None, z_soil=None,
                  a=None, rain_sum_ff=100, max_dur=72,
                  z_soil_max=5000, float_type="float64"):
    """Runs event classification and writes forcing data

    Args
    ----------
    input_dir : Path
        path to directory with input data

    nrows : int, optional
        number of rows

    ncols : int, optional
        number of columns

    hpi : int, optional
        threshold for classification of heavy precipitation event
        (in mm/10min; default: 5)

    end_prec_event : int, optional
        temporal threshold when event ends (in 10min; default: 36)
        i.e. time after which no precipitation occured.

    sf : float, optional
        snow melt factor (-)

    ta_fm : float, optional
        freeze-melt threshold (default = 0; in °C)

    uniform : bool, optional
        True if time series are used as input data

    enable_film_flow : bool, optional
        if True film flow events are classified

    enable_crop_phenology : bool, optional
        if True daily minimum and maximum is required

    prec_correction : str, optional
        if True precipitation is corrected according to Richter (1995)

    z_soil : float, optional
        soil depth

    a : float, optional
        film flow parameter

    rain_sum_ff : float, optional
        if rainfall sum of the event is greater than the provided threshold,
        film flow approach will be applied

    max_dur : float, optional
        time after rainfall pulse (in hours)

    z_soil_max : float, optional
        maximum soil depth to scale rainfall threshold (in mm)
    """
    if uniform:
        df_PREC, df_PET, df_TA = ecl.read_meteo(input_dir)
        validate(df_PREC)
        validate(df_PET)
        validate(df_TA)
        df_meteo = df_PREC.join(df_TA)
        df_meteo = df_meteo.ffill()
        if prec_correction:
            prec_corr = ecl.precipitation_correction(df_meteo['PREC'].values,
                                                     df_meteo['TA'].values,
                                                     df_meteo.index.month,
                                                     horizontal_shielding=prec_correction)
            df_meteo['PREC'] = prec_corr

        # event classification
        df_events = ecl.event_classification(df_meteo)

        if enable_film_flow:
            df_events = ecl.film_flow_event_classification(df_events, z_soil, a,
                                                           rain_sum_ff=rain_sum_ff,
                                                           max_dur=max_dur,
                                                           z_soil_max=z_soil_max)

        # seamless variable time index for precipitation time series
        df_meteo_events = ecl.make_variable_time_index(df_events, enable_film_flow=enable_film_flow)
        df_meteo_events = ecl.time_delta(df_meteo_events)

        # join temperature and evapotranspiration on precipitation
        df_meteo_events = ecl.join_meteo(df_meteo_events, df_PET, df_TA)

        dict_events = {}
        # number of time steps
        no_time_steps = len(df_meteo_events.index)
        # time steps (in hours)
        time_steps = onp.ones((no_time_steps), dtype=float_type)
        time_steps[:] = df_meteo_events['dt'].astype(float_type).values
        dict_events['time_steps'] = time_steps
        # years
        years = onp.zeros((no_time_steps), dtype=int)
        years[:] = df_meteo_events.index.year.astype(int).values
        dict_events['years'] = years
        # months
        months = onp.zeros((no_time_steps), dtype=int)
        months[:] = df_meteo_events.index.month.astype(int).values
        dict_events['months'] = months
        # days of year
        days_of_year = onp.zeros((no_time_steps), dtype=int)
        days_of_year[:] = df_meteo_events.index.dayofyear.astype(int).values
        dict_events['days_of_year'] = days_of_year
        # hours
        hours = onp.zeros((no_time_steps))
        hours[:] = df_meteo_events.index.hour.astype(int).values
        dict_events['hours'] = hours
        # event ID
        event_id = onp.zeros((no_time_steps), dtype=int)
        event_id[:] = df_meteo_events.event_no.astype(int).values
        dict_events['EVENT_ID'] = event_id

        if (df_meteo_events['event_no'] > 0).any():
            _, frequency_events = onp.unique(df_meteo_events['event_no'][df_meteo_events['event_no'] > 0], return_counts=True)
            # number of iterations of longest event
            nitt_event = onp.max(frequency_events)
            dict_events['nitt_event'] = nitt_event
        else:
            dict_events['nitt_event'] = 1

        if enable_film_flow:
            # event ID
            event_id_ff = onp.zeros((no_time_steps), dtype=int)
            event_id_ff[:] = df_meteo_events.event_no_ff.astype(int).values
            dict_events['EVENT_ID_FF'] = event_id_ff
            if (df_meteo_events['event_no_ff'] > 0).any():
                event_nos_ff = onp.unique(df_meteo_events['event_no_ff'][df_meteo_events['event_no_ff'] > 0])
                nevent_ff = len(event_nos_ff)
                dict_events['nevent_ff'] = nevent_ff
            else:
                dict_events['nevent_ff'] = 1

        file = input_dir / "EVENTS.txt"
        df_meteo_events.to_csv(file, header=True, index=False, sep=" ")

        nc_file = input_dir.parent / "forcing.nc"
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='model forcing',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            f.dimensions = {'x': nrows, 'y': ncols, 'time': len(df_meteo_events.index), 'scalar': 1}
            v = f.create_variable('PREC', ('x', 'y', 'time'), float_type)
            arr = df_meteo_events['PREC'].astype(float_type).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'Precipitation'
            v.attrs['units'] = 'mm/dt'
            v = f.create_variable('TA', ('x', 'y', 'time'), float_type)
            arr = df_meteo_events['TA'].astype(float_type).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'air temperature'
            v.attrs['units'] = 'degC'
            v = f.create_variable('PET', ('x', 'y', 'time'), float_type)
            arr = df_meteo_events['PET'].astype(float_type).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'Potential Evapotranspiration'
            v.attrs['units'] = 'mm/dt'
            v = f.create_variable('dt', ('time',), float_type)
            v[:] = dict_events['time_steps']
            v.attrs['long_name'] = 'time step (!not constant)'
            v.attrs['units'] = 'hour'
            v = f.create_variable('year', ('time',), int)
            v[:] = dict_events['years']
            v.attrs['units'] = 'year'
            v = f.create_variable('month', ('time',), int)
            v[:] = dict_events['months']
            v.attrs['units'] = 'month'
            v = f.create_variable('doy', ('time',), int)
            v[:] = dict_events['days_of_year']
            v.attrs['units'] = 'day of year'
            v = f.create_variable('EVENT_ID', ('time',), int)
            v[:] = dict_events['EVENT_ID']
            v.attrs['units'] = ''
            v = f.create_variable('time', ('time',), float_type)
            v.attrs['time_origin'] = f"{df_meteo_events.index[0]}"
            v.attrs['units'] = 'hours'
            v[:] = date2num(df_meteo_events.index.tolist(), units=f"hours since {df_meteo_events.index[0]}", calendar='standard')
            v = f.create_variable('x', ('x',), int)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(nrows)
            v = f.create_variable('y', ('y',), int)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(ncols)
            v = f.create_variable('nitt_event', ('scalar',), int)
            v[:] = dict_events['nitt_event']
            if enable_film_flow:
                v = f.create_variable('EVENT_ID_FF', ('time',), int)
                v[:] = dict_events['EVENT_ID_FF']
                v.attrs['units'] = ''
                v = f.create_variable('nevent_ff', ('scalar',), int)
                v[:] = dict_events['nevent_ff']
            if enable_crop_phenology:
                v = f.create_variable('TA_min', ('x', 'y', 'time'), float_type)
                arr = df_meteo_events['TA_min'].values - 3
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'minimum air temperature'
                v.attrs['units'] = 'degC'
                v = f.create_variable('TA_max', ('x', 'y', 'time'), float_type)
                arr = df_meteo_events['TA_max'].values + 3
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'maximum air temperature'
                v.attrs['units'] = 'degC'


def validate(data: pd.DataFrame):
    """Check if Dataframe has correct type and is numerical.

    This function checks if the input is a pd.DataFrame throws an error in
    case of incorrect data.

    Args
    ----------
    data : pd.DataFrame
        model input data

    Raises
    ----------
    ValueError : Error
        In case non-numerical data is passed
    """
    if isinstance(data, (pd.DataFrame)):
        non_numeric = data.isin([onp.nan, onp.inf, -onp.inf]).any().values

        if any(non_numeric):
            raise ValueError('File contains non-numeric data.')
