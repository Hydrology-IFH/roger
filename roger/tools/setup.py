from roger.core.operators import numpy as npx
from roger.io_tools.csv import read_meteo
import roger.tools.labels as labs
from roger import roger_sync, logger
import time
import os
from pathlib import Path
import numpy as onp
import pandas as pd
import h5netcdf
from cftime import date2num
import datetime
from datetime import timedelta

import scipy.interpolate
import scipy.spatial


def read_tracer_input(path_to_dir: Path, tracer: str):
    """Importing the solute input data

    Data is imported from .txt files and stored in dataframes. Format of NA/NaN
    values is -9999.

    Args
    ----------
    path_to_dir : Path
        path to directions which contains input data

    tracer : str
        name of tracer (e.g. d18O)

    Returns
    ----------
    df_tracer : pd.DataFrame
        solute input (in G/L or permil)
    """
    if not os.path.isdir(path_to_dir):
        print(path_to_dir, 'does not exist')

    sol_file = "%s.txt" % (tracer)
    sol_path = path_to_dir / sol_file

    df_tracer = pd.read_csv(sol_path, sep=r"\s+", skiprows=0, header=0, parse_dates=[[0, 1, 2, 3, 4]],
                            index_col=0, names=['YYYY', 'MM', 'DD', 'hh', 'mm', tracer],
                            na_values=-9999)
    df_tracer.index = pd.to_datetime(df_tracer.index, format='%Y %m %d %H %M')

    return df_tracer


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


@roger_sync
def write_forcing_tracer(input_dir, tracer, nrows=1, ncols=1, uniform=True, float_type="float64"):
    """Writes tracer forcing data

    Args
    ----------
    input_dir : Path
        path to directory with input data

    tracer : str
        name of tracer (e.g. d18O)

    nrows : int, optional
        number of rows

    ncols : int, optional
        number of columns

    uniform : bool, optional
        True if time series are used as input data
    """
    input_path = input_dir / "forcing_tracer.nc"
    if os.path.exists(input_path):
        logger.warning("Use available tracer forcing.\n")
        return

    if uniform:
        if tracer in ['Nmin', 'Norg', 'NO3']:
            df_tracer = read_tracer_input(input_dir, 'Nmin')
            df_tracer1 = read_tracer_input(input_dir, 'Norg')
        else:
            df_tracer = read_tracer_input(input_dir, tracer)

        nc_file = input_dir / "forcing_tracer.nc"
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title=f'{tracer} forcing',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            dict_dim = {'x': nrows, 'y': ncols, 'Time': len(df_tracer.index), 'scalar': 1}
            f.dimensions = dict_dim
            if tracer in ['Nmin', 'Norg', 'NO3']:
                v = f.create_variable('Nmin', ('x', 'y', 'Time'), float_type)
                arr = df_tracer['Nmin'].astype(float_type).values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = labs._LONG_NAME['Nmin']
                v.attrs['units'] = labs._UNITS['Nmin']
                v = f.create_variable('Norg', ('x', 'y', 'Time'), float_type)
                arr = df_tracer1['Norg'].astype(float_type).values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = labs._LONG_NAME['Norg']
                v.attrs['units'] = labs._UNITS['Norg']
            else:
                v = f.create_variable(tracer, ('x', 'y', 'Time'), float_type)
                arr = df_tracer[tracer].astype(float_type).values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = labs._LONG_NAME[tracer]
                v.attrs['units'] = labs._UNITS[tracer]
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


@roger_sync
def write_crop_rotation(input_dir, nrows=1, ncols=1, float_type="float64"):
    """Writes crop rotation data

    Args
    ----------
    input_dir : Path
        path to directory with input data

    nrows : int, optional
        number of rows

    ncols : int, optional
        number of columns
    """
    input_path = input_dir / "crop_rotation.nc"
    if os.path.exists(input_path):
        logger.warning("Use available crop rotation.\n")
        return

    csv_file = input_dir / 'crop_rotation.csv'
    crops = pd.read_csv(csv_file, sep=';', skiprows=1)
    nc_file = input_dir / 'crop_rotation.nc'
    with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='Crop rotation',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment=''
        )
        # set dimensions with a dictionary
        dict_dim = {'x': nrows, 'y': ncols, 'year_season': len(crops.columns[1:])}
        f.dimensions = dict_dim
        arr = onp.full((nrows, ncols, len(crops.columns[1:])), 598, dtype=int)
        idx = onp.arange(nrows * ncols).reshape((nrows, ncols))
        for row in range(nrows):
            for col in range(ncols):
                arr[row, col, :] = crops.iloc[idx[row, col], 1:]

        v = f.create_variable('crop', ('x', 'y', 'year_season'), int)
        v[:, :,  :] = arr
        v.attrs['long_name'] = 'crop'
        v.attrs['units'] = ''
        v = f.create_variable('year_season', ('year_season',), 'S11')
        v.attrs['units'] = 'year_season'
        v[:] = onp.array(crops.columns[1:].astype('S11').values, dtype='S11')
        v = f.create_variable('x', ('x',), float_type)
        v.attrs['long_name'] = 'Zonal coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(nrows)
        v = f.create_variable('y', ('y',), float_type)
        v.attrs['long_name'] = 'Meridonial coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(ncols)


@roger_sync
def write_forcing_event(input_dir, nrows=1, ncols=1, uniform=True, prec_correction=False, float_type="float64"):
    """Writes forcing data for a single event (i.e. no event classification is
    required)

    Args
    ----------
    input_dir : Path
        path to directory with input data

    nrows : int, optional
        number of rows

    ncols : int, optional
        number of columns

    uniform : bool, optional
        True if time series are used as input data

    prec_correction : str, optional
        if True precipitation is corrected according to Richter (1995)
    """
    input_path = input_dir / "forcing.nc"
    if os.path.exists(input_path):
        logger.warning("Use available forcing.\n")
        return

    if uniform:
        if not os.path.isdir(input_dir):
            raise ValueError(input_dir, 'does not exist')

        PREC_path = input_dir / "PREC.txt"
        df_PREC = pd.read_csv(PREC_path, sep=r"\s+", skiprows=0, header=0, na_values=-9999)

        TA_path = input_dir / "TA.txt"
        df_TA = pd.read_csv(TA_path, sep=r"\s+", skiprows=0, header=0, na_values=-9999)

        validate(df_PREC)
        validate(df_TA)
        df_meteo = df_PREC.join(df_TA["TA"].to_frame())
        df_meteo = df_meteo.ffill()
        if prec_correction:
            prec_corr = precipitation_correction(df_meteo['PREC'].values,
                                                 df_meteo['TA'].values,
                                                 df_meteo.index.month,
                                                 horizontal_shielding=prec_correction)
            df_meteo['PREC'] = prec_corr

        nc_file = input_dir / "forcing.nc"
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='Meteorological forcing',
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


def precipitation_correction(prec, ta, month, horizontal_shielding="b1"):
    """Correction of precipitation according to Richter (1995).

    Args
    ----------
    prec : onp.ndarray
        precipitation at time step t (in mm)

    ta : onp.ndarray
        air temperature at time step t (in celsius)

    month : int
        month at time step t

    horizontal_shielding : str
        b1 = open location
        b2 = slightly protected
        b3 = moderately protected
        b4 = strongly protected

    Returns
    ----------
    prec_corr : onp.ndarray
        corrected precipitation at time step t (in mm)

    Reference
    ----------
    Richter, D.: Ergebnisse methodischer Untersuchungen zur Korrektur des
    systematischen Meßfehlers des Hellmann-Niederschlagsmessers, Berichte des
    Deutschen Wetterdienstes, Selbstverlag des Deutschen Wetterdienstes,
    Offenbach am Main, 1995.
    """
    # look-up table for correction
    LUT_PREC_CORR = pd.DataFrame(index=['summer', 'winter', 'mixed', 'snow'],
                                 columns=['eps', 'b1', 'b2', 'b3', 'b4'])
    LUT_PREC_CORR.loc[:, 'eps'] = [0.38, 0.46, 0.55, 0.82]
    LUT_PREC_CORR.loc[:, 'b1'] = [0.345, 0.34, 0.535, 0.72]
    LUT_PREC_CORR.loc[:, 'b2'] = [0.31, 0.28, 0.39, 0.51]
    LUT_PREC_CORR.loc[:, 'b3'] = [0.28, 0.24, 0.305, 0.33]
    LUT_PREC_CORR.loc[:, 'b4'] = [0.245, 0.19, 0.185, 0.21]

    # correction factor
    dprec = onp.zeros((prec.shape))
    eps = LUT_PREC_CORR.loc['snow', 'eps']
    b = LUT_PREC_CORR.loc['snow', horizontal_shielding]
    dprec = onp.where((ta <= -0.7), b * prec**eps, dprec)

    eps = LUT_PREC_CORR.loc['mixed', 'eps']
    b = LUT_PREC_CORR.loc['mixed', horizontal_shielding]
    dprec = onp.where((ta > -0.7) & (ta < 3.0), b * prec**eps, dprec)

    eps = LUT_PREC_CORR.loc['winter', 'eps']
    b = LUT_PREC_CORR.loc['winter', horizontal_shielding]
    dprec = onp.where((ta >= 3.0) & onp.isin(ta, onp.array([9, 10, 11, 12, 1, 2])), b * prec**eps, dprec)

    eps = LUT_PREC_CORR.loc['summer', 'eps']
    b = LUT_PREC_CORR.loc['summer', horizontal_shielding]
    dprec = onp.where((ta >= 3.0) & onp.isin(ta, onp.array([3, 4, 5, 6, 7, 8])), b * prec**eps, dprec)

    # corrected precipitation
    prec_corr = prec + dprec

    return prec_corr


def validate(data):
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
            raise ValueError('File contains non-numeric values.')


@roger_sync
def write_forcing(input_dir, nrows=1, ncols=1, uniform=True,
                  enable_crop_phenology=False,
                  prec_correction=None, float_type="float64"):
    """Runs event classification and writes forcing data

    Args
    ----------
    input_dir : Path
        path to directory with input data

    nrows : int, optional
        number of rows

    ncols : int, optional
        number of columns

    uniform : bool, optional
        True if time series are used as input data

    enable_crop_phenology : bool, optional
        if True daily minimum and maximum is required

    prec_correction : str, optional
        if True precipitation is corrected according to Richter (1995)
    """
    input_path = input_dir / "forcing.nc"
    if os.path.exists(input_path):
        logger.warning("Use available forcing.\n")
        return

    if uniform:
        _lock = True
        while _lock:
            try:
                df_PREC, df_PET, df_TA, df_RS = read_meteo(input_dir)
                _lock = False
                break
            except BlockingIOError:
                _lock = True
                logger.debug("Wait for input files. Files might be used by some other process")
                time.wait(10)

        validate(df_PREC)
        validate(df_TA)
        if df_PET:
            validate(df_PET)
            df_meteo = df_PREC.join([df_TA, df_PET.loc[:, 'PET'].to_frame()])
            df_meteo = df_meteo.ffill()
            # downscale daily PET to 10 minutes
            df_meteo.loc[:, 'PET'] = (df_meteo.loc[:, 'PET'] / 24) / 6
        else:
            validate(df_RS)
            df_meteo = df_PREC.join([df_TA, df_RS.loc[:, 'RS'].to_frame()])
            df_meteo = df_meteo.ffill()

        if prec_correction:
            prec_corr = precipitation_correction(df_meteo['PREC'].values,
                                                 df_meteo['TA'].values,
                                                 df_meteo.index.month,
                                                 horizontal_shielding=prec_correction)
            df_meteo['PREC'] = prec_corr

        nc_file = input_dir / "forcing.nc"
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='Meteorological forcing',
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
            if isinstance(df_PET, pd.DataFrame):
                v = f.create_variable('PET', ('x', 'y', 'Time'), float_type)
                arr = df_meteo['PET'].astype(float_type).values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'Potential Evapotranspiration'
                v.attrs['units'] = 'mm/10 minutes'
            if isinstance(df_RS, pd.DataFrame):
                v = f.create_variable('RS', ('x', 'y', 'Time'), float_type)
                arr = df_meteo['RS'].astype(float_type).values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'Solar radiation'
                v.attrs['units'] = 'MJ/m2'
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
