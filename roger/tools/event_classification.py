import roger.lookuptables as lut
import numpy as onp
import pandas as pd
from datetime import datetime, timedelta

from roger.io_tools.csv import read_meteo


def join_meteo(prec_var_time: pd.DataFrame, df_pet: pd.DataFrame,
               df_ta: pd.DataFrame, tres_input='d'):
    """Join meteorological input on variable time index.

    Args
    ----------
    prec_var_time : pd.DataFrame
        precipitation with variable time index

    df_pet : pd.DataFrame
        daily potential evapotranspiration

    df_ta : pd.DataFrame
        daily air temperature

    tres_input : str, optional
        temporal resolution of input data.

    Returns
    ----------
    var_time : pd.DataFrame
        precipitation, evapotranspiration and air temperature with
        variable time index
    """
    if tres_input == 'd':
        var_time = prec_var_time.join([df_pet['PET'], df_ta])
        if (len(df_ta.columns) == 1):
            var_time['TA'] = var_time['TA'].ffill()
        elif (len(df_ta.columns) == 3):
            var_time['TA'] = var_time['TA'].ffill()
            var_time['TA_min'] = var_time['TA_min'].ffill()
            var_time['TA_max'] = var_time['TA_max'].ffill()
        var_time['PET'] = var_time['PET'].ffill()
        var_time.loc[var_time['h'] == True, 'PET'] = var_time.loc[var_time['h'] == True, 'PET']/24.
        var_time.loc[var_time['10mins'] == True, 'PET'] = var_time.loc[var_time['10mins'] == True, 'PET']/144.

    elif tres_input == 'h':
        var_time = prec_var_time.join([df_pet['PET'], df_ta])
        idx = var_time.loc[(var_time['h'] == True)].index
        if (len(df_ta.columns) == 1):
            var_time['TA'] = var_time['TA'].ffill()
            ta_d = df_ta['TA'].resample('D').mean().to_frame()
            var_time.loc[idx, 'TA'] = ta_d.loc[idx, 'TA']
        elif (len(df_ta.columns) == 3):
            var_time['TA'] = var_time['TA'].ffill()
            var_time['TA_min'] = var_time['TA_min'].ffill()
            var_time['TA_max'] = var_time['TA_max'].ffill()
            ta_d = df_ta['TA'].resample('D').mean().to_frame()
        var_time['PET'] = var_time['PET'].ffill()
        et_d = df_pet['PET'].resample('D').sum().to_frame()
        idx = var_time.loc[(var_time['d'] == True)].index
        if (len(df_ta.columns) == 1):
            var_time.loc[idx, 'TA'] = ta_d.loc[idx, 'TA']
        elif (len(df_ta.columns) == 3):
            var_time.loc[idx, 'TA'] = ta_d.loc[idx, 'TA']
            var_time.loc[idx, 'TA_min'] = ta_d.loc[idx, 'TA_min']
            var_time.loc[idx, 'TA_max'] = ta_d.loc[idx, 'TA_max']
        var_time.loc[idx, 'PET'] = et_d.loc[idx, 'PET']
        var_time.loc[var_time['10mins'] == True, 'PET'] = var_time.loc[var_time['10mins'] == True, 'PET']/6.

    var_time = var_time.reset_index().drop_duplicates(subset='index',
                                                      keep='first').set_index('index')
    return var_time


def event_classification(pta, hpi=5, end_prec_event=36, sf=3, ta_fm=0):
    """Event classification for spatially uniform distributed precipitation

    Args
    ----------
    pta : pandas.DataFrame
        precipitation and air temperature recorded at 10 mins interval
        (contains only three columns: datetime.index,
        precipitation (in mm), air temperature (in celsius))

    hpi : int, optional
        threshold for classification of heavy precipitation event
        (in mm/10min; default: 5)

    end_prec_event : int, optional
        temporal threshold when event ends (in 10min; default: 36). Time after
        which no precipitation occured.

    sf : float, optional
        snow melt factor (-)

    ta_fm : float, optional
        freeze-melt threshold (default = 0; in °C)

    Returns
    ----------
    prec_event : pandas.DataFrame
        event numbers and event type
    """
    # array with precipitation values
    prec_arr = pta.iloc[:, 0].values
    # array with air temperature values
    ta_arr = pta.iloc[:, 1].values

    df_event = pd.DataFrame(index=pta.index, columns=['rng_idx', 'TA', 'PREC', 'snowfall', 'snow_melt', 'rain_', 'snow_', 'snow_melt_'])
    df_event.loc[:, 'rng_idx'] = range(len(df_event.index))
    df_event.loc[:, 'PREC'] = prec_arr
    df_event.loc[:, 'TA'] = ta_arr
    df_event.loc[:, 'snowfall'] = 0
    df_event.loc[:, 'snow_melt'] = 0
    df_event.loc[:, 'rain_'] = onp.NaN
    df_event.loc[:, 'snow_'] = onp.NaN
    df_event.loc[:, 'snow_melt_'] = onp.NaN
    df_event.loc[:, 'event_'] = onp.NaN
    mask = (df_event['PREC'] > 0) & (df_event['TA'] > 0)
    df_event.loc[mask, 'rain_'] = True
    df_event.loc[:, 'rain_'] = df_event.loc[:, 'rain_'].fillna(method="ffill", limit=end_prec_event)
    mask = df_event['rain_'] & (df_event['TA'] <= 0)
    df_event.loc[mask, 'rain_'] = False
    df_event.loc[:, 'rain_'] = df_event.loc[:, 'rain_'].fillna(False)
    mask = (df_event['PREC'] > 0) & (df_event['TA'] <= 0)
    df_event.loc[mask, 'snowfall'] = df_event.loc[mask, 'PREC']
    df_event.loc[mask, 'snow_'] = True
    df_event.loc[:, 'snow_'] = df_event.loc[:, 'snow_'].fillna(method="ffill", limit=end_prec_event)
    df_event.loc[:, 'snow_'] = df_event.loc[:, 'snow_'].fillna(False)
    mask = (df_event['TA'] > 0)
    df_event.loc[mask, 'snow_melt'] = df_event.loc[mask, 'TA'] * sf * 1/6
    mask = (df_event['TA'] > 0) & (df_event['snow_'] == True) & (df_event['snow_'].shift(-1) == False)
    df_event.loc[mask, 'snow_melt_'] = True
    df_event.loc[:, 'rain_'] = df_event.loc[:, 'rain_'].fillna(False)
    df_event.loc[:, 'snow_melt_'] = df_event.loc[:, 'snow_melt_'].fillna(False)
    mask = (df_event['rain_'] == True) & (df_event['rain_'].shift(1) == False)
    df_event.loc[mask, 'start_rain_'] = True
    df_event.loc[:, 'start_rain_'] = df_event.loc[:, 'start_rain_'].fillna(False)
    mask = (df_event['rain_'] == True) & (df_event['rain_'].shift(-1) == False)
    df_event.loc[mask, 'end_rain_'] = True
    df_event.loc[:, 'end_rain_'] = df_event.loc[:, 'end_rain_'].fillna(False)
    mask = (df_event['snow_'] == True) & (df_event['snow_'].shift(1) == False)
    df_event.loc[mask, 'start_snow_'] = True
    df_event.loc[:, 'start_snow_'] = df_event.loc[:, 'start_snow_'].fillna(False)
    mask = (df_event['snow_'] == True) & (df_event['snow_'].shift(-1) == False)
    df_event.loc[mask, 'end_snow_'] = True
    df_event.loc[:, 'end_snow_'] = df_event.loc[:, 'end_snow_'].fillna(False)
    start_snow_melt = [0] + [i for i in df_event.rng_idx if df_event.snow_melt_[i]]
    end_snow_melt = []
    if len(start_snow_melt) > 1:
        for i in range(1, len(start_snow_melt)):
            sum_snow = df_event.loc[df_event.index[start_snow_melt[i-1]]:df_event.index[start_snow_melt[i]], 'snowfall'].sum()
            csum_melt = onp.cumsum(df_event.loc[df_event.index[start_snow_melt[i]]:, 'TA'] * 3 * 1/6)
            end = onp.where(csum_melt > sum_snow)[0][0]
            if end:
                end_snow_melt.append(start_snow_melt[i] + end)
            else:
                end_snow_melt.append(start_snow_melt[i] + 1)
        del start_snow_melt[0]
        for i in end_snow_melt:
            df_event.loc[df_event.index[i], 'end_snow_melt_'] = True
        for s, e in zip(start_snow_melt, end_snow_melt):
            df_event.loc[df_event.index[s]:df_event.index[e], 'snow_melt_'] = True
        df_event.loc[:, 'end_snow_melt_'] = df_event.loc[:, 'end_snow_melt_'].fillna(False)
    mask = (df_event['rain_'] == True) | (df_event['snow_'] == True) | (df_event['snow_melt_'] == True)
    df_event.loc[mask, 'event_'] = True
    df_event.loc[:, 'event_'] = df_event.loc[:, 'event_'].fillna(False)
    mask = (df_event['event_'] == True) & (df_event['event_'].shift(1) == False)
    df_event.loc[mask, 'start_event_'] = True
    df_event.loc[:, 'start_event_'] = df_event.loc[:, 'start_event_'].fillna(False)
    mask = (df_event['event_'] == True) & (df_event['event_'].shift(-1) == False)
    df_event.loc[mask, 'end_event_'] = True
    df_event.loc[:, 'end_event_'] = df_event.loc[:, 'end_event_'].fillna(False)
    df_event.loc[:, 'event_no'] = onp.NaN
    mask = (df_event['start_event_'] == True)
    df_event.loc[mask, 'event_no'] = range(1, onp.sum(mask) + 1)
    mask = (df_event['event_'] == True)
    df_event.loc[mask, 'event_no'] = df_event.loc[mask, 'event_no'].fillna(method="ffill")
    df_event.loc[:, 'event_no'] = df_event.loc[:, 'event_no'].fillna(0)
    df_event.loc[:, 'event_type'] = 0

    df_event = df_event.astype({'event_no': int, 'PREC': float, 'TA': float, 'event_type': int})

    event_nos = onp.unique(df_event['event_no'].values).tolist()[1:]

    for en in event_nos:
        mask = (df_event['event_no'] == en)
        mask_no_prec = (df_event['event_no'] == en) & (df_event['PREC'] <= 0)
        # heavy rain
        if (df_event.loc[mask, 'PREC'] >= hpi).any() and df_event.loc[mask, 'rain_'].all():
            df_event.loc[mask, 'event_type'] = 10
            df_event.loc[mask_no_prec, 'event_type'] = 20
        # rain
        if (df_event.loc[mask, 'PREC'] < hpi).all() and df_event.loc[mask, 'rain_'].all():
            df_event.loc[mask, 'event_type'] = 30
            df_event.loc[mask_no_prec, 'event_type'] = 40
        # snowfall
        if df_event.loc[mask, 'snow_'].all():
            df_event.loc[mask, 'event_type'] = 50
        # rain-on-snow
        if df_event.loc[mask, 'rain_'].any() and df_event.loc[mask, 'snow_melt_'].any():
            df_event.loc[mask, 'event_type'] = 30
            df_event.loc[mask_no_prec, 'event_type'] = 60
        # snow melt
        if df_event.loc[mask, 'snow_melt_'].all():
            df_event.loc[mask, 'event_type'] = 60

    df_event = df_event.loc[:, ['PREC', 'event_type', 'event_no']]

    return df_event


def film_flow_event_classification(prec_event, z_soil, a, rain_sum_ff=100, max_dur=72, z_soil_max=5000):
    """Event classification for film flow. Extend events for film flow.

    Args
    ----------
    prec_event : pandas.DataFrame
        precipitation with event numbers and corresponding event types

    z_soil : float
        soil depth

    a : float
        film flow parameter

    rain_sum_ff : float, optional
        if rainfall sum of the event is greater than the provided threshold,
        film flow approach will be applied

    max_dur : float, optional
        time after rainfall pulse (in hours)

    z_soil_max : float, optional
        maximum soil depth to scale rainfall threshold (in mm)

    Returns
    ----------
    prec_event : pandas.DataFrame
        event numbers and event type
    """
    event_nos = pd.unique(prec_event['event_no'])
    for event_no in event_nos:
        cond = (prec_event['event_no'] == event_no)
        rain_sum = prec_event.loc[cond, 'PREC'].sum()
        t_rain = onp.sum(cond)
        rain_int = rain_sum / t_rain
        qs = rain_int / 600 / 1000
        v_wf = a * qs**(2/3)
        v_pf = 3 * v_wf
        t_soil_wf = z_soil / v_wf
        t_soil_pf = t_rain + z_soil / v_pf
        if (rain_sum > (z_soil / z_soil_max) * rain_sum_ff) | (t_soil_wf < t_soil_pf):
            start_event_ff = prec_event.loc[cond].index[0]
            end_event = prec_event.loc[cond].index[-1]
            # datetime at end of event
            end_event_ff1 = end_event + timedelta(hours=max_dur)
            # round to next hour
            end_event_ff = end_event_ff1.ceil(freq='H')
            if end_event_ff > prec_event.index[-1]:
                end_event_ff = prec_event.index[-1]
            prec_event[f'#{event_no}'] = 0

            prec_event.loc[start_event_ff:end_event_ff, f'#{event_no}'] = event_no

            # set event types for film flow
            cond_prec = (prec_event[f'#{event_no}'] == event_no) & (prec_event['PREC'] > 0) & (prec_event.index >= start_event_ff) & (prec_event.index <= end_event_ff)
            cond_no_prec = (prec_event[f'#{event_no}'] == event_no) & (prec_event['PREC'] == 0) & (prec_event.index >= start_event_ff) & (prec_event.index <= end_event_ff)
            cond_no_event = (prec_event[f'#{event_no}'] == 0) & (prec_event['PREC'] == 0) & (prec_event.index >= start_event_ff) & (prec_event.index <= end_event_ff)
            prec_event.loc[cond_prec, 'event_type'] = 70
            prec_event.loc[cond_prec, 'event_no'] = event_no
            prec_event.loc[cond_no_prec, 'event_type'] = 80
            prec_event.loc[cond_no_prec, 'event_no'] = event_no
            prec_event.loc[cond_no_event, 'event_type'] = 0

    return prec_event


def make_variable_time_index(prec_event, enable_film_flow=False):
    """Generating a seamless time index for precipitation time series with
    variable time steps.

    The time series is aggregated at different time intervals. A daily time
    step is used if there is no event while there is a hourly time step
    during an non-heavy precipitation event. For heavy precipitation events
    the time step is increased to 10 mins interval.

    Args
    ----------
    prec_event : pandas.DataFrame
        precipitation with event numbers and corresponding event types

    Returns
    ----------
    prec_var_time_idx : pandas.DataFrame
        precipitation with seamlessly variable time steps
    """
    prec_10mins = prec_event.copy()

    prec_10mins['event_no'] = 0
    # add new columns to condicate whether there an event takes place
    prec_10mins['event'] = False
    # columns condicate time step of aggregation
    prec_10mins['10mins'] = False  # 10 mins time step
    prec_10mins['h'] = False  # hourly time step
    prec_10mins['d'] = True  # daily time step

    # array with event numbers
    event_nos_zeros = onp.unique(prec_event['event_no'].values)
    event_nos_trim = onp.trim_zeros(event_nos_zeros)
    event_nos = onp.unique(event_nos_trim)

    # 10 mins index for complete hours
    for i, eno in enumerate(event_nos):
        prec_eno = prec_10mins.loc[(prec_event['event_no'] == eno), :]
        cond = prec_eno.index
        new_cond_10mins = pd.date_range(start=datetime(cond[0].year, cond[0].month, cond[0].day, cond[0].hour, 0),
                                       end=datetime(cond[-1].year, cond[-1].month, cond[-1].day, cond[-1].hour, 50),
                                       freq='10T')
        new_prec_10mins = pd.DataFrame(index=new_cond_10mins)
        new_prec_10mins.loc[:, 'event_no'] = eno
        event_type = onp.min(prec_eno['event_type'].values)
        new_prec_10mins.loc[:, 'event_type'] = event_type
        prec_10mins.loc[new_cond_10mins, 'event_no'] = eno
        prec_10mins.loc[new_cond_10mins, 'event_type'] = event_type

        if event_type in [10, 20, 70, 80]:
            new_prec_10mins.loc[:, 'event'] = True
            prec_10mins.loc[new_cond_10mins, 'event'] = new_prec_10mins['event']
            new_prec_10mins.loc[:, '10mins'] = True
            prec_10mins.loc[new_cond_10mins, '10mins'] = new_prec_10mins['10mins']
            new_prec_10mins.loc[:, 'd'] = False
            prec_10mins.loc[new_cond_10mins, 'd'] = new_prec_10mins['d']
        elif event_type in [30, 40, 60]:
            new_prec_10mins.loc[:, 'event'] = True
            prec_10mins.loc[new_cond_10mins, 'event'] = new_prec_10mins['event']
            new_prec_10mins.loc[:, 'h'] = True
            prec_10mins.loc[new_cond_10mins, 'h'] = new_prec_10mins['h']
            new_prec_10mins.loc[:, 'd'] = False
            prec_10mins.loc[new_cond_10mins, 'd'] = new_prec_10mins['d']
        elif event_type in [50]:
            new_prec_10mins.loc[:, 'event'] = True
            prec_10mins.loc[new_cond_10mins, 'event'] = new_prec_10mins['event']
            new_prec_10mins.loc[:, 'h'] = False
            prec_10mins.loc[new_cond_10mins, 'h'] = new_prec_10mins['h']
            new_prec_10mins.loc[:, 'd'] = False
            prec_10mins.loc[new_cond_10mins, 'd'] = new_prec_10mins['d']

    # aggregate to hourly values
    prec_hourly_sum = prec_10mins.resample('h').sum()
    prec_hourly = prec_10mins.resample('h').max()
    prec_hourly['PREC'] = prec_hourly_sum['PREC']

    # daily index for events
    prec_daily = prec_hourly.resample('d').max()  # daily aggregation to create index
    cond_prec_daily_event = (prec_daily['event'] == True)
    prec_daily_event = prec_daily[cond_prec_daily_event]

    # hourly index for complete days
    for i, d_cond in enumerate(prec_daily_event.index):
        new_cond_h = pd.date_range(start=datetime(d_cond.year, d_cond.month, d_cond.day, 0),
                                  end=datetime(d_cond.year, d_cond.month, d_cond.day, 23),
                                  freq='H')
        new_prec_h = pd.DataFrame(index=new_cond_h)
        new_prec_h['event_no'] = prec_hourly.loc[new_cond_h, 'event_no'].values
        new_prec_h['event_type'] = prec_hourly.loc[new_cond_h, 'event_type'].values
        new_prec_h['event'] = True
        new_prec_h['h'] = True
        new_prec_h['d'] = False

        new_prec_h['event_no'].replace(0, onp.nan, inplace=True)
        new_prec_h['event_type'].replace(0, onp.nan, inplace=True)

        new_prec_h['event_no'] = new_prec_h['event_no'].replace(0, onp.nan)
        new_prec_h['event_type'] = new_prec_h['event_type'].replace(0, onp.nan)

        new_prec_h['event_no'].fillna(method='ffill', inplace=True)
        new_prec_h['event_type'].fillna(method='ffill', inplace=True)

        new_prec_h['event_no'].replace(onp.nan, 0, inplace=True)
        new_prec_h['event_type'].replace(onp.nan, 0, inplace=True)

        prec_hourly.loc[new_cond_h, 'event_no'] = new_prec_h['event_no']
        prec_hourly.loc[new_cond_h, 'event_type'] = new_prec_h['event_type']
        prec_hourly.loc[new_cond_h, 'event'] = new_prec_h['event']
        prec_hourly.loc[new_cond_h, 'h'] = new_prec_h['h']
        prec_hourly.loc[new_cond_h, 'd'] = new_prec_h['d']

    # aggregate to daily values
    prec_daily_sum = prec_hourly_sum.resample('d').sum()
    prec_daily = prec_hourly.resample('d').max()
    prec_daily['PREC'] = prec_daily_sum['PREC']
    prec_daily['d'] = (prec_daily['event'] == False)

    # dataframe with heavy precipitation events
    cond_prec_10mins = (prec_10mins['10mins'] == True)
    prec_10mins_new = prec_10mins[cond_prec_10mins]

    # dataframe with non-heavy precipitation events
    cond_prec_hourly = ((prec_hourly['h'] == True) & (prec_hourly['10mins'] == False))
    prec_hourly_new = prec_hourly[cond_prec_hourly]

    # dataframe with precipitation if no event occurs
    cond_prec_daily = (prec_daily['d'] == True)
    prec_daily_new = prec_daily[cond_prec_daily]

    # concatenate dataframes
    frames = [prec_daily_new, prec_hourly_new, prec_10mins_new]
    prec_var_time_idx = pd.concat(frames)
    prec_var_time_idx.sort_index(inplace=True)
    prec_var_time_idx['h'] = ((prec_var_time_idx['10mins'] == False) & (prec_var_time_idx['d'] == False))
    prec_var_time_idx = prec_var_time_idx.reset_index().drop_duplicates(subset='index',
                                                                  keep='first').set_index('index')

    # check whether aggregation is correct
    # cumulated sums of 10 mins series and time series with variable
    # datetime index are compared
    if int(prec_var_time_idx['PREC'].sum()) != int(prec_10mins['PREC'].sum()):
        raise ValueError('Aggregation is wrong.')

    if enable_film_flow:
        cond = (prec_var_time_idx['event_type'] == 70) | (prec_var_time_idx['event_type'] == 80)
        prec_var_time_idx.loc[:, 'event_no_ff'] = 0
        prec_var_time_idx.loc[cond, 'event_no_ff'] = prec_var_time_idx.loc[cond, 'event_no']
        prec_var_time_idx.loc[cond, 'event_no'] = 0

    return prec_var_time_idx


def time_delta(events):
    """Assign column with timedelta (dt). Unit of timedelta is 1 hour.

    Args
    ----------
    events : pandas.DataFrame
        events with seamlessly variable time steps
    """
    events['dt'] = 60 * 60

    cond_prec_daily = (events['d'] == True)
    prec_daily = events[cond_prec_daily]
    events.loc[prec_daily.index, 'dt'] = 24* 60 * 60

    cond_prec_10mins = (events['10mins'] == True)
    prec_10mins = events[cond_prec_10mins]
    events.loc[prec_10mins.index, 'dt'] = 60 * 10

    return events


def calc_event_classification(input_dir, hpi=5, end_prec_event=36, sf=3, ta_fm=0,
                              enable_film_flow=False, z_soil=None, a=None,
                              rain_sum_ff=100, max_dur=72, z_soil_max=5000
                              ):
    """Calculates event classification.

    Args
    ----------
    input_dir : Path
        directory to input files

    hpi : int, optional
        threshold for classification of heavy precipitation event
        (in mm/10min; default: 5)

    end_prec_event : int, optional
        temporal threshold when event ends (in 10min; default: 36). Time after
        which no precipitation occured.

    sf : float, optional
        snow melt factor (-)

    ta_fm : float, optional
        freeze-melt threshold (default = 0; in °C)

    enable_film_flow : bool
        if True film flow events will be classified.

    z_soil : float
        soil depth

    a : float
        film flow parameter

    rain_sum_ff : float, optional
        if rainfall sum of the event is greater than the provided threshold,
        film flow approach will be applied

    max_dur : float, optional
        time after rainfall pulse (in hours)

    z_soil_max : float, optional
        maximum soil depth to scale rainfall threshold (in mm)
    """
    # import meteo data
    df_PREC, df_PET, df_Ta = read_meteo(input_dir)
    df_PREC['PREC'] = df_PREC['PREC'].values
    df_meteo = df_PREC.join(df_Ta)
    df_meteo = df_meteo.ffill()

    # event classification
    prec_event = event_classification(df_meteo)

    if enable_film_flow:
        prec_event = film_flow_event_classification(prec_event, z_soil, a,
                                                    rain_sum_ff=rain_sum_ff,
                                                    max_dur=max_dur,
                                                    z_soil_max=z_soil_max)

    # seamless variable time index for precipitation time series
    events = make_variable_time_index(prec_event, enable_film_flow=enable_film_flow)
    events = time_delta(events)

    # join temperature and evapotranspiration on precipitation
    events = join_meteo(events, df_PET, df_Ta)

    results = {}
    results['meteo'] = events
    # number of time steps
    no_time_steps = len(events.index)
    # time steps (in hours)
    time_steps = onp.ones((no_time_steps))
    time_steps[:] = events['dt'].values
    results['time_steps'] = time_steps
    # years
    years = onp.zeros((no_time_steps), dtype=onp.int32)
    years[:] = events.index.year.values
    results['years'] = years
    # months
    months = onp.zeros((no_time_steps), dtype=onp.int32)
    months[:] = events.index.month.values
    results['months'] = months
    # days of year
    days_of_year = onp.zeros((no_time_steps), dtype=onp.int32)
    days_of_year[:] = events.index.dayofyear.values
    results['days_of_year'] = days_of_year
    # hours
    hours = onp.zeros((no_time_steps))
    hours[:] = events.index.hour.values
    results['hours'] = hours

    if (events['event_no'] > 0).any():
        _, frequency_events = onp.unique(events['event_no'][events['event_no'] > 0], return_counts = True)
        # number of iterations of longest event
        nitt_event = onp.max(frequency_events)
        results['nitt_event'] = nitt_event
    else:
        results['nitt_event'] = 1

    if enable_film_flow:
        if (events['event_no_ff'] > 0).any():
            event_nos_ff = onp.unique(events['event_no_ff'][events['event_no_ff'] > 0])
            nevent_ff = len(event_nos_ff)
            results['nevent_ff'] = nevent_ff
        else:
            results['nevent_ff'] = 1

    return results


def precipitation_correction(prec, ta, month, horizontal_shielding="b1"):
    """Correction of precipitation according to Richter (1995).

    Args
    ----------
    prec : onp.ndarray
        precipitation at time step t (in mm)

    ta : onp.ndarray
        air temperature at time step t (in celsius)

    month : int
        month at time step t (in celsius)

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
    if (ta <= -0.7):
        eps = lut.LUT_PREC_CORR.loc['snow', 'eps']
        b = lut.LUT_PREC_CORR.loc['snow', horizontal_shielding]
    elif ((ta > -0.7) & (ta < 3.0)):
        eps = lut.LUT_PREC_CORR.loc['mixed', 'eps']
        b = lut.LUT_PREC_CORR.loc['mixed', horizontal_shielding]
    elif ((ta >= 3.0) & (month in [9, 10, 11, 12, 1, 2])):
        eps = lut.LUT_PREC_CORR.loc['winter', 'eps']
        b = lut.LUT_PREC_CORR.loc['winter', horizontal_shielding]
    elif ((ta >= 3.0) & (month in [3, 4, 5, 6, 7, 8])):
        eps = lut.LUT_PREC_CORR.loc['summer', 'eps']
        b = lut.LUT_PREC_CORR.loc['summer', horizontal_shielding]

    dprec = b * prec**eps
    prec_corr = prec + dprec

    return prec_corr
