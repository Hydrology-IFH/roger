import numpy as onp
import datetime
import pandas as pd
import h5netcdf
from cftime import date2num

from roger.core.event_classification import calc_event_classification
from roger.io_tools import yml
onp.random.seed(42)

def make_setup(base_path, identifier="dummy", ndays=10, nrows=1, ncols=1,
               event_type='rain', enable_groundwater_boundary=False,
               enable_film_flow=False, enable_crop_phenology=False,
               enable_crop_rotation=False, enable_lateral_flow=False,
               enable_groundwater=False, enable_routing=False,
               enable_offline_transport=False,
               enable_bromide=False,
               enable_chloride=False,
               enable_deuterium=False,
               enable_oxygen18=False,
               enable_nitrate=False,
               tm_structure=None,
               z_soil=None,
               a=None,
               rain_sum_ff=100,
               max_dur=72,
               z_soil_max=5000,
               nrows=1,
               ncols=1):
    """
    Make dummy setup with maximum two events.
    """
    if not enable_offline_transport:
        # generate config file
        yml.write_config(base_path, identifier, None, nrows=nrows, ncols=ncols,
                         enable_groundwater_boundary=enable_groundwater_boundary,
                         enable_film_flow=enable_film_flow,
                         enable_crop_phenology=enable_crop_phenology,
                         enable_crop_rotation=enable_crop_rotation,
                         enable_lateral_flow=enable_lateral_flow,
                         enable_groundwater=enable_groundwater,
                         enable_routing=enable_routing,
                         enable_offline_transport=enable_offline_transport,
                         enable_bromide=enable_bromide,
                         enable_chloride=enable_chloride,
                         enable_deuterium=enable_deuterium,
                         enable_oxygen18=enable_oxygen18,
                         enable_nitrate=enable_nitrate)

        # model parameters
        head_units_params = ['Unit']
        head_columns_params = ['No']
        df_params = pd.DataFrame(index=range(nrows * ncols),
                                 columns=head_columns_params)
        df_params.loc[:, 'No'] = range(1, nrows * ncols + 1)

        # write initial values file
        head_units_initvals = ['Unit']
        head_columns_initvals = ['No']
        df_initvals = pd.DataFrame(index=range(nrows * ncols),
                                   columns=head_columns_initvals)
        df_initvals.loc[:, 'No'] = range(1, nrows * ncols + 1)

        if event_type == 'rain':
            # generate random rainfall
            prec_rnd = onp.random.uniform(0, 1, 18)
            n_prec_rnd = len(prec_rnd)
            n_prec = ndays*24*6
            prec = onp.zeros((n_prec))
            prec[12:12+n_prec_rnd] = prec_rnd
            prec[int(n_prec/2):int(n_prec/2)+n_prec_rnd] = prec_rnd
            idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
            # generate random air temperature
            idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            ta = onp.random.uniform(15, 20, ndays)
            # generate random potential evapotranspiration
            pet = onp.random.uniform(2, 3, ndays)

        elif event_type == 'snow':
            # generate random rainfall
            prec_rnd = onp.random.uniform(0, 1, 18)
            n_prec_rnd = len(prec_rnd)
            n_prec = ndays*24*6
            prec = onp.zeros((n_prec))
            prec[12:12+n_prec_rnd] = prec_rnd
            prec[int(n_prec/2):int(n_prec/2)+n_prec_rnd] = prec_rnd
            idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
            # generate random air temperature
            idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            ta = onp.random.uniform(-3, -1, ndays)
            # generate random potential evapotranspiration
            pet = onp.random.uniform(1, 2, ndays)

        elif event_type == 'snow+rain':
            # generate random rainfall
            prec_rnd = onp.random.uniform(0, 1, 18)
            n_prec_rnd = len(prec_rnd)
            n_prec = ndays*24*6
            prec = onp.zeros((n_prec))
            prec[12:12+n_prec_rnd] = prec_rnd
            prec[int(n_prec/2):int(n_prec/2)+n_prec_rnd] = prec_rnd
            idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
            # generate random air temperature
            idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            ta = onp.random.uniform(0, 3, ndays)
            ta[:2] = -1
            # generate random potential evapotranspiration
            pet = onp.random.uniform(1, 2, ndays)

        elif event_type == 'heavyrain':
            prec_rnd = onp.random.uniform(0.1, 6, 12*6)
            n_prec_rnd = len(prec_rnd)
            n_prec = ndays*24*6
            prec = onp.zeros((n_prec))
            prec[12:12+n_prec_rnd] = prec_rnd
            prec[int(n_prec/2):int(n_prec/2)+n_prec_rnd] = prec_rnd
            idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
            # generate random air temperature
            idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            ta = onp.random.uniform(15, 20, ndays)
            # generate random potential evapotranspiration
            pet = onp.random.uniform(2, 3, ndays)

        elif event_type == 'mixed':
            x1 = onp.unique(onp.random.randint(low=0, high=ndays * 24 * 6, size=(int(100 * (ndays/365)),)))
            x2 = onp.zeros((int(100 * (ndays/365)),), dtype=int)
            for i in range(int(100 * (ndays/365)) - 1):
                if x1[i+1] - x1[i] <= 1:
                    high = 2
                else:
                    high = x1[i+1] - x1[i]
                x2[i] = x1[i] + onp.random.randint(low=1, high=high)
            x2[x2 > ndays * 24 * 6] = ndays * 24 * 6
            x2[-1] = onp.random.randint(low=x1[-1] + 1, high=ndays * 24 * 6)

            prec = onp.zeros((ndays * 24 * 6,))
            for i, ii in zip(x1, x2):
                lam = onp.random.weibull(1, 1)
                if lam > 5.5:
                    prec[i:ii] = onp.random.poisson(lam, ii - i) * 0.5
                else:
                    prec[i:ii] = onp.random.poisson(lam, ii - i) * onp.random.uniform(0.01, 0.1, ii - i)
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
                ta[i] = onp.random.uniform(ta_init - 1 + scale[ii] * ta_off, ta_init + 1 + scale[ii] * ta_off, 1)
                # generate random potential evapotranspiration
                pet[i] = onp.random.uniform(pet_init - 1 + scale[ii] * pet_off, pet_init + 1 + scale[ii] * pet_off, 1)
                ii += 1

        elif event_type == 'norain':
            # generate random rainfall
            prec = onp.zeros((ndays*24*6))
            idx_prec = pd.date_range(start='1/1/2018', periods=ndays*24*6, freq='10T')
            # generate random air temperature
            idx_ta_pet = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            ta = onp.random.uniform(15, 20, ndays)
            # generate random potential evapotranspiration
            pet = onp.random.uniform(2, 3, ndays)

        df_prec = pd.DataFrame(index=idx_prec, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'PREC'])
        df_prec.loc[:, 'YYYY'] = df_prec.index.year
        df_prec.loc[:, 'MM'] = df_prec.index.month
        df_prec.loc[:, 'DD'] = df_prec.index.day
        df_prec.loc[:, 'hh'] = df_prec.index.hour
        df_prec.loc[:, 'mm'] = df_prec.index.minute
        df_prec.loc[:, 'PREC'] = prec
        file = base_path / "input" / "PREC.txt"
        df_prec.to_csv(file, header=True, index=False, sep=" ")

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
        file = base_path / "input" / "TA.txt"
        df_ta.to_csv(file, header=True, index=False, sep=" ")

        df_pet = pd.DataFrame(index=idx_ta_pet, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'PET'])
        df_pet.loc[:, 'YYYY'] = df_pet.index.year
        df_pet.loc[:, 'MM'] = df_pet.index.month
        df_pet.loc[:, 'DD'] = df_pet.index.day
        df_pet.loc[:, 'hh'] = df_pet.index.hour
        df_pet.loc[:, 'mm'] = df_pet.index.minute
        df_pet.loc[:, 'PET'] = pet
        file = base_path / "input" / "PET.txt"
        df_pet.to_csv(file, header=True, index=False, sep=" ")

        input_dir = base_path / "input"
        if not enable_film_flow:
            res = calc_event_classification(input_dir)
            NITT_EVENT = res['nitt_event']
            df_events = pd.DataFrame(index=res['meteo'].index, columns=['YYYY', 'MM', 'DOY', 'hh', 'EVENT_ID', 'time_step', 'PREC', 'PET', 'TA'])
            df_events.loc[:, 'time_step'] = res['time_steps']
            df_events.loc[:, 'YYYY'] = res['years']
            df_events.loc[:, 'MM'] = res['months']
            df_events.loc[:, 'DOY'] = res['days_of_year']
            df_events.loc[:, 'hh'] = res['hours']
            df_events.loc[:, 'EVENT_ID'] = res['meteo']['event_no'].values
            df_events.loc[:, 'PREC'] = res['meteo']['PREC'].values
            df_events.loc[:, 'TA'] = res['meteo']['TA'].values
            if enable_crop_phenology:
                df_events.loc[:, 'TA_min'] = res['meteo']['TA_min'].values
                df_events.loc[:, 'TA_max'] = res['meteo']['TA_max'].values
            df_events.loc[:, 'PET'] = res['meteo']['PET'].values
            df_events.loc[:, 'itt'] = range(len(df_events.index))

        elif enable_film_flow:
            res = calc_event_classification(input_dir, enable_film_flow=enable_film_flow,
                                            z_soil=z_soil, a=a, rain_sum_ff=rain_sum_ff,
                                            max_dur=max_dur, z_soil_max=z_soil_max)
            NITT_EVENT = res['nitt_event']
            NEVENT_FF = res['nevent_ff']
            df_events = pd.DataFrame(index=res['meteo'].index, columns=['YYYY', 'MM', 'DOY', 'hh', 'EVENT_ID', 'EVENT_ID_FF', 'time_step', 'PREC', 'PET', 'TA'])
            df_events.loc[:, 'time_step'] = res['time_steps']
            df_events.loc[:, 'YYYY'] = res['years']
            df_events.loc[:, 'MM'] = res['months']
            df_events.loc[:, 'DOY'] = res['days_of_year']
            df_events.loc[:, 'hh'] = res['hours']
            df_events.loc[:, 'EVENT_ID'] = res['meteo']['event_no'].values
            df_events.loc[:, 'EVENT_ID_FF'] = res['meteo']['event_no_ff'].values
            df_events.loc[:, 'PREC'] = res['meteo']['PREC'].values
            df_events.loc[:, 'TA'] = res['meteo']['TA'].values
            if enable_crop_phenology:
                df_events.loc[:, 'TA_min'] = res['meteo']['TA_min'].values
                df_events.loc[:, 'TA_max'] = res['meteo']['TA_max'].values
            df_events.loc[:, 'PET'] = res['meteo']['PET'].values
            df_events.loc[:, 'itt'] = range(len(df_events.index))

        if enable_groundwater_boundary:
            offset = onp.sin(onp.linspace(-onp.pi, onp.pi, ndays)) / 2
            df_events.loc[:, 'Z_GW'] = 4 + offset

        file = base_path / "input" / "EVENTS.txt"
        df_events.to_csv(file, header=True, index=False, sep=" ")

        nc_file = base_path / "forcing.nc"
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='test forcing',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            f.dimensions = {'x': nrows, 'y': ncols, 'time': len(df_events.index), 'scalar': 1}
            v = f.create_variable('PREC', ('x', 'y', 'time'), float)
            arr = df_events['PREC'].astype(float).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'Precipitation'
            v.attrs['units'] = 'mm/dt'
            v = f.create_variable('TA', ('x', 'y', 'time'), float)
            arr = df_events['TA'].astype(float).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'air temperature'
            v.attrs['units'] = 'degC'
            v = f.create_variable('PET', ('x', 'y', 'time'), float)
            arr = df_events['PET'].astype(float).values
            v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
            v.attrs['long_name'] = 'Potential Evapotranspiration'
            v.attrs['units'] = 'mm/dt'
            v = f.create_variable('dt', ('time',), float)
            v[:] = df_events['time_step'].astype(float).values
            v.attrs['long_name'] = 'time step (!not constant)'
            v.attrs['units'] = 'hour'
            v = f.create_variable('year', ('time',), int)
            v[:] = df_events['YYYY'].astype(int).values
            v.attrs['units'] = 'year'
            v = f.create_variable('month', ('time',), int)
            v[:] = df_events['MM'].astype(int).values
            v.attrs['units'] = 'month'
            v = f.create_variable('doy', ('time',), int)
            v[:] = df_events['DOY'].astype(int).values
            v.attrs['units'] = 'day of year'
            v = f.create_variable('EVENT_ID', ('time',), int)
            v[:] = df_events['EVENT_ID'].astype(int).values
            v.attrs['units'] = ''
            v = f.create_variable('time', ('time',), float)
            v.attrs['time_origin'] = f"{df_events.index[0]}"
            v.attrs['units'] = 'hours'
            v[:] = date2num(df_events.index.tolist(), units=f"hours since {df_events.index[0]}", calendar='standard')
            v = f.create_variable('x', ('x',), float)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(f.dimensions["x"])
            v = f.create_variable('y', ('y',), float)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(f.dimensions["y"])
            v = f.create_variable('nitt_event', ('scalar',), int)
            v[:] = NITT_EVENT
            if enable_film_flow:
                v = f.create_variable('EVENT_ID_FF', ('time',), int)
                v[:] = df_events['EVENT_ID_FF'].astype(int).values
                v.attrs['units'] = ''
                v = f.create_variable('nevent_ff', ('scalar',), int)
                v[:] = NEVENT_FF
            if enable_groundwater_boundary:
                arr = df_events['Z_GW'].values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'Groundwater head'
                v.attrs['units'] = 'm'
            if enable_crop_phenology:
                v = f.create_variable('TA_min', ('x', 'y', 'time'), float)
                arr = df_events['TA_min'].values - 3
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'minimum air temperature'
                v.attrs['units'] = 'degC'
                v = f.create_variable('TA_max', ('x', 'y', 'time'), float)
                arr = df_events['TA_max'].values + 3
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'maximum air temperature'
                v.attrs['units'] = 'degC'

        # write parameter file
        head_units_params += ['',
                              '[%]', '[-]', '[mm]',
                              '[mm]', '[1/m2]', '[mm]',
                              '[-]', '[-]', '[-]',
                              '[mm/h]', '[mm/h]']
        head_columns_params += ['lu_id',
                                'sealing', 'slope', 'S_dep_tot',
                                'z_soil', 'dmpv', 'lmpv',
                                'theta_ac', 'theta_ufc', 'theta_pwp',
                                'ks', 'kf']

        df_params.loc[:, 'lu_id'] = 8
        df_params.loc[:, 'sealing'] = 0
        df_params.loc[:, 'slope'] = 0
        df_params.loc[:, 'S_dep_tot'] = 0
        df_params.loc[:, 'z_soil'] = 2200
        df_params.loc[:, 'dmpv'] = 75
        df_params.loc[:, 'lmpv'] = 300
        df_params.loc[:, 'theta_ac'] = 0.13
        df_params.loc[:, 'theta_ufc'] = 0.24
        df_params.loc[:, 'theta_pwp'] = 0.23
        df_params.loc[:, 'ks'] = 86
        df_params.loc[:, 'kf'] = 2500

        # write initial values file
        head_units_initvals += ['[mm]', '[mm]', '[mm]', '[mm]', '[mm]', '[mm]',
                                '[mm]', '[-]', '[-]']
        head_columns_initvals += ['S_int_top', 'swe_top', 'S_int_ground',
                                  'swe_ground', 'S_dep', 'S_snow',
                                  'swe', 'theta_rz', 'theta_ss']

        df_initvals.loc[:, 'S_int_top'] = 0
        df_initvals.loc[:, 'swe_top'] = 0
        df_initvals.loc[:, 'S_int_ground'] = 0
        df_initvals.loc[:, 'swe_ground'] = 0
        df_initvals.loc[:, 'S_dep'] = 0
        df_initvals.loc[:, 'S_snow'] = 0
        df_initvals.loc[:, 'swe'] = 0
        df_initvals.loc[:, 'theta_rz'] = 0.46
        df_initvals.loc[:, 'theta_ss'] = 0.47

        if enable_lateral_flow and not enable_offline_transport:
            head_units_initvals += ['[mm]']
            head_columns_initvals += ['z_sat']
            df_initvals.loc[:, 'z_sat'] = 0

        if enable_groundwater and not enable_offline_transport:
            head_units_initvals += ['[m]', '[m/m]']
            head_columns_initvals += ['z_gw', 'beta']
            df_initvals.loc[:, 'z_gw'] = 10
            df_initvals.loc[:, 'beta'] = 0.01

        if enable_film_flow:
            head_units_params += ['[-]', '[-]']
            head_columns_params += ['a_ff', 'c_ff']
            df_params.loc[:, 'a_ff'] = 0.19
            df_params.loc[:, 'c_ff'] = 0.01

        if enable_lateral_flow:
            head_units_params += ['[1/m2]']
            head_columns_params += ['dmph']
            df_params.loc[:, 'dmph'] = 50

        if enable_groundwater:
            df_params.loc[:, 'kf'] = 900
            head_units_params += ['[mm/h]', '[mm/h]', '[-]', '[-]', '[-]', '[m]']
            head_columns_params += ['k_gw', 'k_leak', 'bdec', 'n0', 'npor', 'z_tot_gw']
            df_params.loc[:, 'k_gw'] = 900
            df_params.loc[:, 'k_leak'] = 0
            df_params.loc[:, 'bdec'] = 4
            df_params.loc[:, 'n0'] = 0.25
            df_params.loc[:, 'npor'] = 0.3
            df_params.loc[:, 'z_tot_gw'] = 20

        if enable_crop_phenology:
            df_params.loc[:, 'lu_id'] = 557

        if enable_crop_rotation:
            head_units_crops = ['']
            head_columns_crops = ['No']
            head_units_crops += ['[year_season]', '[year_season]', '[year_season]', '[year_season]', '[year_season]', '[year_season]']
            year_season = ['2017_summer', '2017_winter', '2018_summer', '2018_winter', '2019_summer', '2019_winter']
            head_columns_crops += year_season
            df_crops = pd.DataFrame(index=range(nrows * ncols),
                                    columns=head_columns_crops)
            df_crops.loc[:, 'No'] = range(1, nrows * ncols + 1)
            # df_crops.loc[:, '2017_summer'] = 599
            # df_crops.loc[:, '2017_winter'] = 557
            # df_crops.loc[:, '2018_summer'] = 599
            # df_crops.loc[:, '2018_winter'] = 557
            # df_crops.loc[:, '2019_summer'] = 599
            # df_crops.loc[:, '2019_winter'] = 556
            df_crops.loc[:, '2017_summer'] = 599
            df_crops.loc[:, '2017_winter'] = 599
            df_crops.loc[:, '2018_summer'] = 563
            df_crops.loc[:, '2018_winter'] = 564
            df_crops.loc[:, '2019_summer'] = 539
            df_crops.loc[:, '2019_winter'] = 564
            df_crops.columns = pd.MultiIndex.from_tuples(
                zip(head_units_crops, head_columns_crops))
            file = base_path / 'crop_rotation.csv'
            df_crops.to_csv(file, header=True, index=False, sep=";")
            csv_file = base_path / 'crop_rotation.csv'
            crops = pd.read_csv(csv_file, sep=';', skiprows=1)
            nc_file = base_path / 'crop_rotation.nc'
            with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title='dummy model parameters',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment=''
                )
                # set dimensions with a dictionary
                f.dimensions = {'x': nrows, 'y': ncols, 'year_season': len(crops.columns[1:])}
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
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["y"])

    elif enable_offline_transport:
        # generate config file
        yml.write_config(base_path, identifier, None, nrows=nrows, ncols=ncols,
                         enable_groundwater_boundary=enable_groundwater_boundary,
                         enable_film_flow=enable_film_flow,
                         enable_crop_phenology=enable_crop_phenology,
                         enable_crop_rotation=enable_crop_rotation,
                         enable_lateral_flow=enable_lateral_flow,
                         enable_groundwater=enable_groundwater,
                         enable_routing=enable_routing,
                         enable_offline_transport=enable_offline_transport,
                         enable_bromide=enable_bromide,
                         enable_chloride=enable_chloride,
                         enable_deuterium=enable_deuterium,
                         enable_oxygen18=enable_oxygen18,
                         enable_nitrate=enable_nitrate,
                         tm_structure=tm_structure)

        # model parameters
        head_units_params = ['Unit']
        head_columns_params = ['No']
        df_params = pd.DataFrame(index=range(nrows * ncols),
                                 columns=head_columns_params)
        df_params.loc[:, 'No'] = range(1, nrows * ncols + 1)

        # write initial values file
        head_units_initvals = ['Unit']
        head_columns_initvals = ['No']
        df_initvals = pd.DataFrame(index=range(nrows * ncols),
                                   columns=head_columns_initvals)
        df_initvals.loc[:, 'No'] = range(1, nrows * ncols + 1)

        # write SAS parameter files
        head_units_sas_params_transp = ['Unit']
        head_columns_sas_params_transp = ['No']
        df_sas_params_transp = pd.DataFrame(index=range(nrows * ncols),
                                            columns=head_columns_sas_params_transp)
        df_sas_params_transp.loc[:, 'No'] = range(1, nrows * ncols + 1)

        head_units_sas_params_q_rz = ['Unit']
        head_columns_sas_params_q_rz = ['No']
        df_sas_params_q_rz = pd.DataFrame(index=range(nrows * ncols),
                                          columns=head_columns_sas_params_q_rz)
        df_sas_params_q_rz.loc[:, 'No'] = range(1, nrows * ncols + 1)

        head_units_sas_params_q_ss = ['Unit']
        head_columns_sas_params_q_ss = ['No']
        df_sas_params_q_ss = pd.DataFrame(index=range(nrows * ncols),
                                          columns=head_columns_sas_params_q_ss)
        df_sas_params_q_ss.loc[:, 'No'] = range(1, nrows * ncols + 1)

        if enable_film_flow:
            pass

        if enable_groundwater_boundary:
            pass

        if enable_lateral_flow:
            pass

        if enable_groundwater:
            pass

        if tm_structure == 'complete-mixing':
            head_units_sas_params_transp += ['']
            head_columns_sas_params_transp += ['sas_function']
            df_sas_params_transp.loc[:, 'sas_function'] = 1

            head_units_sas_params_q_rz += ['']
            head_columns_sas_params_q_rz += ['sas_function']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 1

            head_units_sas_params_q_ss += ['']
            head_columns_sas_params_q_ss += ['sas_function']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 1

        elif tm_structure == 'piston':
            head_units_sas_params_transp += ['']
            head_columns_sas_params_transp += ['sas_function']
            df_sas_params_transp.loc[:, 'sas_function'] = 21

            head_units_sas_params_q_rz += ['']
            head_columns_sas_params_q_rz += ['sas_function']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 22

            head_units_sas_params_q_ss += ['']
            head_columns_sas_params_q_ss += ['sas_function']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 22

        elif tm_structure == 'complete-mixing + advection-dispersion':
            head_units_sas_params_transp += ['', '[-]', '[-]']
            head_columns_sas_params_transp += ['sas_function', 'a', 'b']
            df_sas_params_transp.loc[:, 'sas_function'] = 1

            head_units_sas_params_q_rz += ['', '[-]', '[-]']
            head_columns_sas_params_q_rz += ['sas_function', 'a', 'b']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 3
            df_sas_params_q_rz.loc[:, 'a'] = 2
            df_sas_params_q_rz.loc[:, 'b'] = 1

            head_units_sas_params_q_ss += ['', '[-]', '[-]']
            head_columns_sas_params_q_ss += ['sas_function', 'a', 'b']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 3
            df_sas_params_q_ss.loc[:, 'a'] = 3
            df_sas_params_q_ss.loc[:, 'b'] = 1

        elif tm_structure == 'complete-mixing + time-variant advection-dispersion':
            head_units_sas_params_transp += ['', '[-]', '[-]']
            head_columns_sas_params_transp += ['sas_function', 'a', 'b']
            df_sas_params_transp.loc[:, 'sas_function'] = 1

            head_units_sas_params_q_rz += ['', '[-]', '[-]']
            head_columns_sas_params_q_rz += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 32
            df_sas_params_q_rz.loc[:, 'lower_limit'] = 1
            df_sas_params_q_rz.loc[:, 'upper_limit'] = 3

            head_units_sas_params_q_ss += ['', '[-]', '[-]']
            head_columns_sas_params_q_ss += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 32
            df_sas_params_q_ss.loc[:, 'lower_limit'] = 1
            df_sas_params_q_ss.loc[:, 'upper_limit'] = 3

        elif tm_structure == 'advection-dispersion':
            head_units_sas_params_transp += ['', '[-]', '[-]']
            head_columns_sas_params_transp += ['sas_function', 'a', 'b']
            df_sas_params_transp.loc[:, 'sas_function'] = 3
            df_sas_params_transp.loc[:, 'a'] = 1
            df_sas_params_transp.loc[:, 'b'] = 10

            head_units_sas_params_q_rz += ['', '[-]', '[-]']
            head_columns_sas_params_q_rz += ['sas_function', 'a', 'b']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 3
            df_sas_params_q_rz.loc[:, 'a'] = 2
            df_sas_params_q_rz.loc[:, 'b'] = 1

            head_units_sas_params_q_ss += ['', '[-]', '[-]']
            head_columns_sas_params_q_ss += ['sas_function', 'a', 'b']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 3
            df_sas_params_q_ss.loc[:, 'a'] = 3
            df_sas_params_q_ss.loc[:, 'b'] = 1

        elif tm_structure == 'time-variant advection-dispersion':
            head_units_sas_params_transp += ['', '[-]', '[-]']
            head_columns_sas_params_transp += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_transp.loc[:, 'sas_function'] = 31
            df_sas_params_transp.loc[:, 'lower_limit'] = 1
            df_sas_params_transp.loc[:, 'upper_limit'] = 30

            head_units_sas_params_q_rz += ['', '[-]', '[-]']
            head_columns_sas_params_q_rz += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 32
            df_sas_params_q_rz.loc[:, 'lower_limit'] = 1
            df_sas_params_q_rz.loc[:, 'upper_limit'] = 2

            head_units_sas_params_q_ss += ['', '[-]', '[-]']
            head_columns_sas_params_q_ss += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 32
            df_sas_params_q_ss.loc[:, 'lower_limit'] = 1
            df_sas_params_q_ss.loc[:, 'upper_limit'] = 3

        elif tm_structure == 'preferential':
            head_units_sas_params_transp += ['', '[-]', '[-]']
            head_columns_sas_params_transp += ['sas_function', 'a', 'b']
            df_sas_params_transp.loc[:, 'sas_function'] = 3
            df_sas_params_transp.loc[:, 'a'] = 1
            df_sas_params_transp.loc[:, 'b'] = 30

            head_units_sas_params_q_rz += ['', '[-]', '[-]']
            head_columns_sas_params_q_rz += ['sas_function', 'a', 'b']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 3
            df_sas_params_q_rz.loc[:, 'a'] = 1
            df_sas_params_q_rz.loc[:, 'b'] = 5

            head_units_sas_params_q_ss += ['', '[-]', '[-]']
            head_columns_sas_params_q_ss += ['sas_function', 'a', 'b']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 3
            df_sas_params_q_ss.loc[:, 'a'] = 1
            df_sas_params_q_ss.loc[:, 'b'] = 3

        elif tm_structure == 'time-variant preferential':
            head_units_sas_params_transp += ['', '[-]', '[-]']
            head_columns_sas_params_transp += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_transp.loc[:, 'sas_function'] = 31
            df_sas_params_transp.loc[:, 'lower_limit'] = 1
            df_sas_params_transp.loc[:, 'upper_limit'] = 30

            head_units_sas_params_q_rz += ['', '[-]', '[-]']
            head_columns_sas_params_q_rz += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 31
            df_sas_params_q_rz.loc[:, 'lower_limit'] = 1
            df_sas_params_q_rz.loc[:, 'upper_limit'] = 3

            head_units_sas_params_q_ss += ['', '[-]', '[-]']
            head_columns_sas_params_q_ss += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 31
            df_sas_params_q_ss.loc[:, 'lower_limit'] = 1
            df_sas_params_q_ss.loc[:, 'upper_limit'] = 3

        elif tm_structure == 'preferential + advection-dispersion':
            head_units_sas_params_transp += ['', '[-]', '[-]']
            head_columns_sas_params_transp += ['sas_function', 'a', 'b']
            df_sas_params_transp.loc[:, 'sas_function'] = 3
            df_sas_params_transp.loc[:, 'a'] = 1
            df_sas_params_transp.loc[:, 'b'] = 30

            head_units_sas_params_q_rz += ['', '[-]', '[-]']
            head_columns_sas_params_q_rz += ['sas_function', 'a', 'b']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 3
            df_sas_params_q_rz.loc[:, 'a'] = 1
            df_sas_params_q_rz.loc[:, 'b'] = 5

            head_units_sas_params_q_ss += ['', '[-]', '[-]']
            head_columns_sas_params_q_ss += ['sas_function', 'a', 'b']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 3
            df_sas_params_q_ss.loc[:, 'a'] = 3
            df_sas_params_q_ss.loc[:, 'b'] = 1

        elif tm_structure == 'time-variant':
            head_units_sas_params_transp += ['', '[-]', '[-]']
            head_columns_sas_params_transp += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_transp.loc[:, 'sas_function'] = 35
            df_sas_params_transp.loc[:, 'lower_limit'] = 1
            df_sas_params_transp.loc[:, 'upper_limit'] = 30

            head_units_sas_params_q_rz += ['', '[-]', '[-]']
            head_columns_sas_params_q_rz += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_q_rz.loc[:, 'sas_function'] = 35
            df_sas_params_q_rz.loc[:, 'lower_limit'] = 1
            df_sas_params_q_rz.loc[:, 'upper_limit'] = 3

            head_units_sas_params_q_ss += ['', '[-]', '[-]']
            head_columns_sas_params_q_ss += ['sas_function', 'lower_limit', 'upper_limit']
            df_sas_params_q_ss.loc[:, 'sas_function'] = 35
            df_sas_params_q_ss.loc[:, 'lower_limit'] = 1
            df_sas_params_q_ss.loc[:, 'upper_limit'] = 3

        # write to csv
        df_sas_params_transp.columns = pd.MultiIndex.from_tuples(
            zip(head_units_sas_params_transp, head_columns_sas_params_transp))
        file = base_path / "sas_parameters_transp.csv"
        df_sas_params_transp.to_csv(file, header=True, index=False, sep=";")

        df_sas_params_q_rz.columns = pd.MultiIndex.from_tuples(
            zip(head_units_sas_params_q_rz, head_columns_sas_params_q_rz))
        file = base_path / "sas_parameters_q_rz.csv"
        df_sas_params_q_rz.to_csv(file, header=True, index=False, sep=";")

        df_sas_params_q_ss.columns = pd.MultiIndex.from_tuples(
            zip(head_units_sas_params_q_ss, head_columns_sas_params_q_ss))
        file = base_path / "sas_parameters_q_ss.csv"
        df_sas_params_q_ss.to_csv(file, header=True, index=False, sep=";")

        # write to nc
        csv_file = base_path / 'sas_parameters_transp.csv'
        params = pd.read_csv(csv_file, sep=';', skiprows=1)
        nc_file = base_path / 'sas_parameters_transp.nc'
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='dummy SAS parameters of transpiration',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            f.dimensions = {'x': nrows, 'y': ncols}
            v = f.create_variable('x', ('x',), float)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(f.dimensions["x"])
            v = f.create_variable('y', ('y',), float)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(f.dimensions["y"])
            for var_name in params.columns:
                v = f.create_variable(var_name, ('x', 'y'), float)
                v[:, :] = params[var_name].values.reshape((nrows, ncols)).astype(float)

        csv_file = base_path / 'sas_parameters_q_rz.csv'
        params = pd.read_csv(csv_file, sep=';', skiprows=1)
        nc_file = base_path / 'sas_parameters_q_rz.nc'
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='dummy SAS parameters of percolation from root zone',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            f.dimensions = {'x': nrows, 'y': ncols}
            v = f.create_variable('x', ('x',), float)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(f.dimensions["x"])
            v = f.create_variable('y', ('y',), float)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(f.dimensions["y"])
            for var_name in params.columns:
                v = f.create_variable(var_name, ('x', 'y'), float)
                v[:, :] = params[var_name].values.reshape((nrows, ncols)).astype(float)

        csv_file = base_path / 'sas_parameters_q_ss.csv'
        params = pd.read_csv(csv_file, sep=';', skiprows=1)
        nc_file = base_path / 'sas_parameters_q_ss.nc'
        with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='dummy SAS parameters of percolation from subsoil',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment=''
            )
            # set dimensions with a dictionary
            f.dimensions = {'x': nrows, 'y': ncols}
            v = f.create_variable('x', ('x',), float)
            v.attrs['long_name'] = 'Zonal coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(f.dimensions["x"])
            v = f.create_variable('y', ('y',), float)
            v.attrs['long_name'] = 'Meridonial coordinate'
            v.attrs['units'] = 'meters'
            v[:] = onp.arange(f.dimensions["y"])
            for var_name in params.columns:
                v = f.create_variable(var_name, ('x', 'y'), float)
                v[:, :] = params[var_name].values.reshape((nrows, ncols)).astype(float)

        if enable_bromide:
            # generate bromide injection pulse
            idx_br = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            br = onp.zeros((ndays))
            br[2] = 30
            df_br = pd.DataFrame(index=idx_br, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'Br'])
            df_br.loc[:, 'YYYY'] = df_br.index.year
            df_br.loc[:, 'MM'] = df_br.index.month
            df_br.loc[:, 'DD'] = df_br.index.day
            df_br.loc[:, 'hh'] = df_br.index.hour
            df_br.loc[:, 'mm'] = df_br.index.minute
            df_br.loc[:, 'Br'] = br
            file = base_path / "input" / "Br.txt"
            df_br.to_csv(file, header=True, index=False, sep=" ")
            nc_file = base_path / "tracer_input.nc"
            with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title='test bromide tracer input',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment=''
                )
                # set dimensions with a dictionary
                f.dimensions = {'x': nrows, 'y': ncols, 'time': len(df_br.index), 'scalar': 1}
                v = f.create_variable('time', ('time',), float)
                v.attrs['time_origin'] = f"{df_br.index[0]}"
                v.attrs['units'] = 'days'
                v[:] = date2num(df_br.index.tolist(), units=f"days since {df_br.index[0]}", calendar='standard')
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["y"])
                v = f.create_variable('Br', ('x', 'y', 'time'), float)
                arr = df_br['Br'].values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'mass of bromide tracer injection'
                v.attrs['units'] = 'mg'

            # write parameter file
            head_units_params += ['[-]', '[-]']
            head_columns_params += ['alpha_transp', 'alpha_q']
            df_params.loc[:, 'alpha_transp'] = 1
            df_params.loc[:, 'alpha_q'] = 1

            # write initial values file
            head_units_initvals += ['[mg/l]', '[mg/l]']
            head_columns_initvals += ['C_rz', 'C_ss']
            df_initvals.loc[:, 'C_rz'] = 0
            df_initvals.loc[:, 'C_ss'] = 0

        if enable_chloride:
            # generate chloride concentration
            idx_cl = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            cl = onp.zeros((ndays))
            cl[:] = 3
            df_cl = pd.DataFrame(index=idx_cl, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'Cl'])
            df_cl.loc[:, 'YYYY'] = df_cl.index.year
            df_cl.loc[:, 'MM'] = df_cl.index.month
            df_cl.loc[:, 'DD'] = df_cl.index.day
            df_cl.loc[:, 'hh'] = df_cl.index.hour
            df_cl.loc[:, 'mm'] = df_cl.index.minute
            df_cl.loc[:, 'Cl'] = cl
            file = base_path / "input" / "Cl.txt"
            df_cl.to_csv(file, header=True, index=False, sep=" ")
            nc_file = base_path / "tracer_input.nc"
            with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title='test chloride tracer input',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment=''
                )
                # set dimensions with a dictionary
                f.dimensions = {'x': nrows, 'y': ncols, 'time': len(df_cl.index), 'scalar': 1}
                v = f.create_variable('time', ('time',), float)
                v.attrs['time_origin'] = f"{df_cl.index[0]}"
                v.attrs['units'] = 'days'
                v[:] = date2num(df_cl.index.tolist(), units=f"days since {df_cl.index[0]}", calendar='standard')
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["y"])
                v = f.create_variable('Cl', ('x', 'y', 'time'), float)
                arr = df_cl['Cl'].values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'chloride concentration of precipitation'
                v.attrs['units'] = 'mg/l'

            # write parameter file
            head_units_params += ['[-]', '[-]']
            head_columns_params += ['alpha_transp', 'alpha_q']
            df_params.loc[:, 'alpha_transp'] = 1
            df_params.loc[:, 'alpha_q'] = 1

            # write initial values file
            head_units_initvals += ['[mg/l]', '[mg/l]']
            head_columns_initvals += ['C_rz', 'C_ss']
            df_initvals.loc[:, 'C_rz'] = 3
            df_initvals.loc[:, 'C_ss'] = 3

        if enable_oxygen18:
            # generate oxygen-18 input signal
            idx_d18O = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            d18O = onp.random.uniform(-16, -14, ndays)
            df_d18O = pd.DataFrame(index=idx_d18O, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'd18O'])
            df_d18O.loc[:, 'YYYY'] = df_d18O.index.year
            df_d18O.loc[:, 'MM'] = df_d18O.index.month
            df_d18O.loc[:, 'DD'] = df_d18O.index.day
            df_d18O.loc[:, 'hh'] = df_d18O.index.hour
            df_d18O.loc[:, 'mm'] = df_d18O.index.minute
            df_d18O.loc[:, 'd18O'] = d18O
            file = base_path / "input" / "d18O.txt"
            df_d18O.to_csv(file, header=True, index=False, sep=" ")
            nc_file = base_path / "tracer_input.nc"
            with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title='test oxygen-18 tracer input',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment=''
                )
                # set dimensions with a dictionary
                f.dimensions = {'x': nrows, 'y': ncols, 'time': len(df_d18O.index), 'scalar': 1}
                v = f.create_variable('time', ('time',), float)
                v.attrs['time_origin'] = f"{df_d18O.index[0]}"
                v.attrs['units'] = 'days'
                v[:] = date2num(df_d18O.index.tolist(), units=f"days since {df_d18O.index[0]}", calendar='standard')
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["y"])
                v = f.create_variable('d18O', ('x', 'y', 'time'), float)
                arr = df_d18O['d18O'].values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'oxygen-18 signal of precipitation'
                v.attrs['units'] = 'permil'

            # write initial values file
            head_units_initvals += ['[permil]', '[permil]']
            head_columns_initvals += ['C_rz', 'C_ss']
            df_initvals.loc[:, 'C_rz'] = -7
            df_initvals.loc[:, 'C_ss'] = -13

        if enable_deuterium:
            # generate deuterium input signal
            idx_d2H = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            d2H = onp.random.uniform(-81, -79, ndays)
            df_d2H = pd.DataFrame(index=idx_d2H, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'd2H'])
            df_d2H.loc[:, 'YYYY'] = df_d2H.index.year
            df_d2H.loc[:, 'MM'] = df_d2H.index.month
            df_d2H.loc[:, 'DD'] = df_d2H.index.day
            df_d2H.loc[:, 'hh'] = df_d2H.index.hour
            df_d2H.loc[:, 'mm'] = df_d2H.index.minute
            df_d2H.loc[:, 'd2H'] = d2H
            file = base_path / "input" / "d2H.txt"
            df_d2H.to_csv(file, header=True, index=False, sep=" ")
            nc_file = base_path / "tracer_input.nc"
            with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title='test deuterium tracer input',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment=''
                )
                # set dimensions with a dictionary
                f.dimensions = {'x': nrows, 'y': ncols, 'time': len(df_d2H.index), 'scalar': 1}
                v = f.create_variable('time', ('time',), float)
                v.attrs['time_origin'] = f"{df_d2H.index[0]}"
                v.attrs['units'] = 'days'
                v[:] = date2num(df_d2H.index.tolist(), units=f"days since {df_d2H.index[0]}", calendar='standard')
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["y"])
                v = f.create_variable('d2H', ('x', 'y', 'time'), float)
                arr = df_d2H['d2H'].values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'deuterium signal of precipitation'
                v.attrs['units'] = 'permil'

            # write initial values file
            head_units_initvals += ['[permil]', '[permil]']
            head_columns_initvals += ['C_rz', 'C_ss']
            df_initvals.loc[:, 'C_rz'] = -40
            df_initvals.loc[:, 'C_ss'] = -60

        if enable_nitrate:
            # generate nitrogen fertilizer input signal
            idx_Nin = pd.date_range(start='1/1/2018', periods=ndays, freq='D')
            Nmin = onp.zeros((ndays))
            Nmin[2] = 60
            Norg = onp.zeros((ndays))
            df_Nin = pd.DataFrame(index=idx_Nin, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'Nmin', 'Norg'])
            df_Nin.loc[:, 'YYYY'] = df_Nin.index.year
            df_Nin.loc[:, 'MM'] = df_Nin.index.month
            df_Nin.loc[:, 'DD'] = df_Nin.index.day
            df_Nin.loc[:, 'hh'] = df_Nin.index.hour
            df_Nin.loc[:, 'mm'] = df_Nin.index.minute
            df_Nin.loc[:, 'Nmin'] = Nmin
            df_Nin.loc[:, 'Norg'] = Norg
            file = base_path / "input" / "Nin.txt"
            df_Nin.to_csv(file, header=True, index=False, sep=" ")
            nc_file = base_path / "tracer_input.nc"
            with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
                f.attrs.update(
                    date_created=datetime.datetime.today().isoformat(),
                    title='test nitrogen tracer input',
                    institution='University of Freiburg, Chair of Hydrology',
                    references='',
                    comment=''
                )
                # set dimensions with a dictionary
                f.dimensions = {'x': nrows, 'y': ncols, 'time': len(df_Nin.index), 'scalar': 1}
                v = f.create_variable('time', ('time',), float)
                v.attrs['time_origin'] = f"{df_Nin.index[0]}"
                v.attrs['units'] = 'days'
                v[:] = date2num(df_Nin.index.tolist(), units=f"days since {df_Nin.index[0]}", calendar='standard')
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = onp.arange(f.dimensions["y"])
                v = f.create_variable('Nmin', ('x', 'y', 'time'), float)
                arr = df_Nin['Nmin'].values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'Mineral nitrogen fertilizer'
                v.attrs['units'] = 'kg N/ha'
                v = f.create_variable('Norg', ('x', 'y', 'time'), float)
                arr = df_Nin['Norg'].values
                v[:, :, :] = arr[onp.newaxis, onp.newaxis, :]
                v.attrs['long_name'] = 'Organic nitrogen fertilizer'
                v.attrs['units'] = 'kg N/ha'

                # write parameter file
                head_units_params += ['[kg N ha-1 year-1]', '[kg N ha-1 year-1]',
                                      '[kg N ha-1 year-1]', '[kg N ha-1 year-1]',
                                      '[kg N ha-1 year-1]', '[kg N ha-1 year-1]',
                                      '[kg N ha-1 year-1]', '[kg N ha-1 year-1]',
                                      '[kg N ha-1 year-1]', '[kg N ha-1 year-1]',
                                      '[-]', '[-]', '[mg/l]']
                head_columns_params += ['km_denit_rz', 'dmax_denit_rz', 'km_nit_rz',
                                        'dmax_nit_rz', 'kmin_rz', 'km_denit_ss',
                                        'dmax_denit_ss', 'km_nit_ss', 'dmax_nit_ss',
                                        'kmin_ss', 'alpha_transp', 'alpha_q', 'N_dep']
                df_params.loc[:, 'km_denit_rz'] = 1.75
                df_params.loc[:, 'dmax_denit_rz'] = 10
                df_params.loc[:, 'km_nit_rz'] = 1.75
                df_params.loc[:, 'dmax_nit_rz'] = 10
                df_params.loc[:, 'kmin_rz'] = 50
                df_params.loc[:, 'km_denit_ss'] = 1.75
                df_params.loc[:, 'dmax_denit_ss'] = 10
                df_params.loc[:, 'km_nit_ss'] = 1.75
                df_params.loc[:, 'dmax_nit_ss'] = 10
                df_params.loc[:, 'kmin_ss'] = 50
                df_params.loc[:, 'alpha_transp'] = 1
                df_params.loc[:, 'alpha_q'] = 1
                df_params.loc[:, 'Ndep'] = 5

                # write initial values file
                head_units_initvals += ['[mg/l]', '[mg/l]', '[kg N/ha]', '[kg N/ha]']
                head_columns_initvals += ['C_rz', 'C_ss', 'Nmin_rz', 'Nmin_ss']
                df_initvals.loc[:, 'C_rz'] = 30
                df_initvals.loc[:, 'C_ss'] = 30
                df_initvals.loc[:, 'Nmin_rz'] = 90
                df_initvals.loc[:, 'Nmin_ss'] = 50

    # write to csv
    df_params.columns = pd.MultiIndex.from_tuples(
        zip(head_units_params, head_columns_params))
    file = base_path / 'parameters.csv'
    df_params.to_csv(file, header=True, index=False, sep=";")

    df_initvals.columns = pd.MultiIndex.from_tuples(
        zip(head_units_initvals, head_columns_initvals))
    file = base_path / 'initvals.csv'
    df_initvals.to_csv(file, header=True, index=False, sep=";")

    csv_file = base_path / 'parameters.csv'
    params = pd.read_csv(csv_file, sep=';', skiprows=1)
    nc_file = base_path / 'parameters.nc'
    with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='dummy model parameters',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment=''
        )
        # set dimensions with a dictionary
        f.dimensions = {'x': nrows, 'y': ncols}
        v = f.create_variable('x', ('x',), float)
        v.attrs['long_name'] = 'Zonal coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(f.dimensions["x"])
        v = f.create_variable('y', ('y',), float)
        v.attrs['long_name'] = 'Meridonial coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(f.dimensions["y"])
        for var_name in params.columns:
            if var_name in ['lu_id', 'sealing', 'slope', 'dmpv', 'dmph']:
                v = f.create_variable(var_name, ('x', 'y'), int)
                v[:, :] = params[var_name].values.reshape((nrows, ncols)).astype(int)
            else:
                v = f.create_variable(var_name, ('x', 'y'), float)
                v[:, :] = params[var_name].values.reshape((nrows, ncols)).astype(float)

    csv_file = base_path / 'initvals.csv'
    initvals = pd.read_csv(csv_file, sep=';', skiprows=1)
    nc_file = base_path / 'initvals.nc'
    with h5netcdf.File(nc_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='dummy initial values',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment=''
        )
        # set dimensions with a dictionary
        f.dimensions = {'x': nrows, 'y': ncols}
        v = f.create_variable('x', ('x',), float)
        v.attrs['long_name'] = 'Zonal coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(f.dimensions["x"])
        v = f.create_variable('y', ('y',), float)
        v.attrs['long_name'] = 'Meridonial coordinate'
        v.attrs['units'] = 'meters'
        v[:] = onp.arange(f.dimensions["y"])
        for var_name in initvals.columns:
            v = f.create_variable(var_name, ('x', 'y'), float)
            v[:, :] = initvals[var_name].values.reshape((nrows, ncols)).astype(float)
