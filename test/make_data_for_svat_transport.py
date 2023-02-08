from pathlib import Path
import os
import shutil
import datetime
import glob
import h5netcdf


def make_data(float_type):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at
    from roger.core.surface import calc_parameters_surface_kernel
    from roger.tools.make_toy_data import make_toy_forcing
    import roger.lookuptables as lut
    import numpy as onp

    class SVATSetup(RogerSetup):
        """A SVAT model.
        """
        _base_path = Path(__file__).parent
        _input_dir = _base_path / "input"

        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj)

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['dt']
                return onp.sum(onp.array(var_obj))

        def _get_time_origin(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time'].attrs['time_origin']
                return str(var_obj)

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = "SVAT"

            settings.nx, settings.ny, settings.nz = 1, 1, 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')

            # spatial discretization (in meters)
            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = self._get_time_origin(self._input_dir, 'forcing.nc')

            settings.enable_macropore_lower_boundary_condition = False

        @roger_routine
        def set_grid(self, state):
            vs = state.variables

            # grid of model runs
            dx = allocate(state.dimensions, ("x"))
            dx = update(dx, at[:], 1)
            dy = allocate(state.dimensions, ("y"))
            dy = update(dy, at[:], 1)
            vs.x = update(vs.x, at[3:-2], npx.cumsum(dx[3:-2]))
            vs.y = update(vs.y, at[3:-2], npx.cumsum(dy[3:-2]))

        @roger_routine
        def set_look_up_tables(self, state):
            vs = state.variables

            vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
            vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
            vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
            vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
            vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine
        def set_parameters_setup(self, state):
            vs = state.variables

            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], 8)
            vs.sealing = update(vs.sealing, at[2:-2, 2:-2], 0)
            vs.S_dep_tot = update(vs.S_dep_tot, at[2:-2, 2:-2], 0)
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], 2000)
            vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], 50)
            vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], 50)
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], 0.1)
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], 0.1)
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2],  0.2)
            vs.ks = update(vs.ks, at[2:-2, 2:-2], 5)
            vs.kf = update(vs.kf, at[2:-2, 2:-2], 2500)

        @roger_routine
        def set_parameters(self, state):
            vs = state.variables

            if (vs.month[vs.tau] != vs.month[vs.taum1]) & (vs.itt > 1):
                vs.update(calc_parameters_surface_kernel(state))

        @roger_routine
        def set_initial_conditions_setup(self, state):
            pass

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables

            vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], 0.3)
            vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], 0.3)

        @roger_routine
        def set_boundary_conditions_setup(self, state):
            pass

        @roger_routine
        def set_boundary_conditions(self, state):
            pass

        @roger_routine
        def set_forcing_setup(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "dt_secs",
                "dt",
                "itt",
                "itt_forc",
                "itt_day",
                "prec",
                "ta",
                "pet",
                "pet_res",
                "event_id",
                "event_id_counter",
                "year",
                "month",
                "doy",
                "tau",
                "taum1",
                "q_snow",
                "swe",
                "swe_top",
                "time",
                "time_event0"
            ],
        )
        def set_forcing(self, state):
            vs = state.variables
            settings = state.settings

            vs.year = update(vs.year, at[1], self._read_var_from_nc("YEAR", self._input_dir, 'forcing.nc')[vs.itt_forc])
            vs.month = update(vs.month, at[1], self._read_var_from_nc("MONTH", self._input_dir, 'forcing.nc')[vs.itt_forc])
            vs.doy = update(vs.doy, at[1], self._read_var_from_nc("DOY", self._input_dir, 'forcing.nc')[vs.itt_forc])

            # adaptive time stepping
            condt = (vs.time % (24 * 60 * 60) == 0) & (vs.itt > 0)
            if condt:
                vs.itt_day = 0
                vs.itt_forc = vs.itt_forc + 6 * 24
            prec_day = self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc')[:, :, vs.itt_forc:vs.itt_forc+6*24]
            ta_day = self._read_var_from_nc("TA", self._input_dir, 'forcing.nc')[:, :, vs.itt_forc:vs.itt_forc+6*24]
            pet_day = self._read_var_from_nc("PET", self._input_dir, 'forcing.nc')[:, :, vs.itt_forc:vs.itt_forc+6*24]
            cond0 = (prec_day <= 0).all() & (vs.swe[2:-2, 2:-2, vs.tau] <= 0).all() & (vs.swe_top[2:-2, 2:-2, vs.tau] <= 0).all() & (ta_day > settings.ta_fm).all()
            cond00 = ((prec_day > 0) & (ta_day <= settings.ta_fm)).any() | ((prec_day <= 0) & (ta_day <= settings.ta_fm)).all()
            cond1 = (prec_day > settings.hpi).any() & (prec_day > 0).any() & (ta_day > settings.ta_fm).any()
            cond2 = (prec_day <= settings.hpi).all() & (prec_day > 0).any() & (ta_day > settings.ta_fm).any()
            cond3 = (prec_day > settings.hpi).any() & (prec_day > 0).any() & (((vs.swe[2:-2, 2:-2, vs.tau] > 0).any() | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0).any()) & (ta_day > settings.ta_fm).any())
            cond4 = (prec_day <= settings.hpi).all() & (prec_day > 0).any() & (((vs.swe[2:-2, 2:-2, vs.tau] > 0).any() | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0).any()) & (ta_day > settings.ta_fm).any())
            cond5 = (prec_day <= 0).all() & (((vs.swe[2:-2, 2:-2, vs.tau] > 0).any() | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0).any()) & (ta_day > settings.ta_fm).any())
            # no event or snowfall - daily time steps
            if cond0 or cond00:
                prec = npx.sum(prec_day, axis=-1)
                ta = npx.mean(ta_day, axis=-1)
                if (vs.time % (24 * 60 * 60) == 0):
                    vs.dt_secs = 24 * 60 * 60
                else:
                    vs.dt_secs = 60 * 60
            # rainfall/snow melt event - hourly time steps
            elif (cond2 or cond4 or cond5) and not cond1 and not cond3:
                prec_hour = prec_day[:, :, vs.itt_day:vs.itt_day+6]
                ta_hour = ta_day[:, :, vs.itt_day:vs.itt_day+6]
                prec = npx.sum(prec_hour, axis=-1)
                ta = npx.mean(ta_hour, axis=-1)
                vs.dt_secs = 60 * 60
            # heavy rainfall event - 10 minutes time steps
            elif (cond1 or cond3) and not cond2 and not cond4 and not cond5:
                prec = prec_day[:, :, vs.itt_day]
                ta = ta_day[:, :, vs.itt_day]
                vs.dt_secs = 10 * 60

            # determine end of event
            if ((prec > 0).any() | ((vs.swe[2:-2, 2:-2, vs.tau] > 0) | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0)).any() and (ta > settings.ta_fm)).any():
                vs.time_event0 = 0
            elif ((prec <= 0) & (ta > settings.ta_fm)).all() or ((prec > 0) & (ta <= settings.ta_fm)).all() or ((vs.swe[2:-2, 2:-2, vs.taum1] > 0).any() & (vs.swe[2:-2, 2:-2, vs.tau] <= 0).all()):
                vs.time_event0 = vs.time_event0 + vs.dt_secs

            # increase time stepping at end of event if either full hour
            # or full day, respectively
            if vs.time_event0 <= settings.end_event and (vs.dt_secs == 10 * 60):
                ta = ta_day[:, :, vs.itt_day]
                pet = pet_day[:, :, vs.itt_day]
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], vs.event_id_counter,
                )
                vs.dt = 1 / 6
                vs.itt_day = vs.itt_day + 1
            elif vs.time_event0 <= settings.end_event and (vs.dt_secs == 60 * 60):
                ta = npx.mean(ta_day[:, :, vs.itt_day:vs.itt_day+6], axis=-1)
                pet = npx.sum(pet_day[:, :, vs.itt_day:vs.itt_day+6], axis=-1)
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], vs.event_id_counter,
                )
                vs.dt = 1
                vs.itt_day = vs.itt_day + 6
            elif vs.time_event0 <= settings.end_event and (vs.dt_secs == 24 * 60 * 60):
                ta = npx.mean(ta_day[:, :, vs.itt_day:vs.itt_day+24*6], axis=-1)
                pet = npx.sum(pet_day[:, :, vs.itt_day:vs.itt_day+24*6], axis=-1)
                vs.dt = 24
                vs.itt_day = 0
            elif vs.time_event0 > settings.end_event and (vs.time % (60 * 60) != 0) and (vs.dt_secs == 10 * 60):
                vs.dt_secs = 10 * 60
                vs.dt = 1 / 6
                ta = ta_day[:, :, vs.itt_day]
                pet = pet_day[:, :, vs.itt_day]
                vs.itt_day = vs.itt_day + 1
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], 0,
                )
            elif vs.time_event0 > settings.end_event and (vs.time % (60 * 60) == 0) and ((vs.dt_secs == 10 * 60) or (vs.dt_secs == 60 * 60)):
                ta = npx.mean(ta_day[:, :, vs.itt_day:vs.itt_day+6], axis=-1)
                pet = npx.sum(pet_day[:, :, vs.itt_day:vs.itt_day+6], axis=-1)
                vs.dt_secs = 60 * 60
                vs.dt = 1
                vs.itt_day = vs.itt_day + 6
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], 0,
                )
            elif vs.time_event0 > settings.end_event and (vs.time % (24 * 60 * 60) == 0) and (vs.dt_secs == 24 * 60 * 60):
                ta = npx.mean(ta_day[:, :, vs.itt_day:vs.itt_day+24*6], axis=-1)
                pet = npx.sum(pet_day[:, :, vs.itt_day:vs.itt_day+24*6], axis=-1)
                vs.dt_secs = 24 * 60 * 60
                vs.dt = 24
                vs.itt_day = 0
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], 0,
                )

            # set event id for next event
            if (vs.event_id[vs.taum1] > 0) & (vs.event_id[vs.tau] == 0):
                vs.event_id_counter = vs.event_id_counter + 1

            # set forcing for current time step
            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], prec)
            vs.ta = update(vs.ta, at[2:-2, 2:-2, vs.tau], ta)
            vs.pet = update(vs.pet, at[2:-2, 2:-2], pet)
            vs.pet_res = update(vs.pet_res, at[2:-2, 2:-2], pet)

        @roger_routine
        def set_diagnostics(self, state):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = ["prec", "aet", "transp", "evap_soil", "inf_mat_rz", "inf_mp_rz", "inf_sc_rz", "inf_ss", "q_rz", "q_ss", "cpr_rz", "dS_s", "dS", "q_snow"]
            diagnostics["rate"].output_frequency = 24 * 60 * 60
            diagnostics["rate"].sampling_frequency = 1
            diagnostics["rate"].base_output_path = self._base_path

            diagnostics["collect"].output_variables = ["S_rz", "S_ss",
                                                       "S_pwp_rz", "S_fc_rz",
                                                       "S_sat_rz", "S_pwp_ss",
                                                       "S_fc_ss", "S_sat_ss",
                                                       "theta_rz", "theta_ss", "theta",
                                                       "S_snow"]
            diagnostics["collect"].output_frequency = 24 * 60 * 60
            diagnostics["collect"].sampling_frequency = 1
            diagnostics["collect"].base_output_path = self._base_path

            diagnostics["average"].output_variables = ["ta"]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            diagnostics["average"].base_output_path = self._base_path

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.ta = update(
            vs.ta,
            at[2:-2, 2:-2, vs.taum1], vs.ta[2:-2, 2:-2, vs.tau],
        )
        vs.z_root = update(
            vs.z_root,
            at[2:-2, 2:-2, vs.taum1], vs.z_root[2:-2, 2:-2, vs.tau],
        )
        vs.ground_cover = update(
            vs.ground_cover,
            at[2:-2, 2:-2, vs.taum1], vs.ground_cover[2:-2, 2:-2, vs.tau],
        )
        vs.S_sur = update(
            vs.S_sur,
            at[2:-2, 2:-2, vs.taum1], vs.S_sur[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_top = update(
            vs.S_int_top,
            at[2:-2, 2:-2, vs.taum1], vs.S_int_top[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_ground = update(
            vs.S_int_ground,
            at[2:-2, 2:-2, vs.taum1], vs.S_int_ground[2:-2, 2:-2, vs.tau],
        )
        vs.S_dep = update(
            vs.S_dep,
            at[2:-2, 2:-2, vs.taum1], vs.S_dep[2:-2, 2:-2, vs.tau],
        )
        vs.S_snow = update(
            vs.S_snow,
            at[2:-2, 2:-2, vs.taum1], vs.S_snow[2:-2, 2:-2, vs.tau],
        )
        vs.swe = update(
            vs.swe,
            at[2:-2, 2:-2, vs.taum1], vs.swe[2:-2, 2:-2, vs.tau],
        )
        vs.S_rz = update(
            vs.S_rz,
            at[2:-2, 2:-2, vs.taum1], vs.S_rz[2:-2, 2:-2, vs.tau],
        )
        vs.S_ss = update(
            vs.S_ss,
            at[2:-2, 2:-2, vs.taum1], vs.S_ss[2:-2, 2:-2, vs.tau],
        )
        vs.S_s = update(
            vs.S_s,
            at[2:-2, 2:-2, vs.taum1], vs.S_s[2:-2, 2:-2, vs.tau],
        )
        vs.S = update(
            vs.S,
            at[2:-2, 2:-2, vs.taum1], vs.S[2:-2, 2:-2, vs.tau],
        )
        vs.z_sat = update(
            vs.z_sat,
            at[2:-2, 2:-2, vs.taum1], vs.z_sat[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf = update(
            vs.z_wf,
            at[2:-2, 2:-2, vs.taum1], vs.z_wf[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t0 = update(
            vs.z_wf_t0,
            at[2:-2, 2:-2, vs.taum1], vs.z_wf_t0[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t1 = update(
            vs.z_wf_t1,
            at[2:-2, 2:-2, vs.taum1], vs.z_wf_t1[2:-2, 2:-2, vs.tau],
        )
        vs.y_mp = update(
            vs.y_mp,
            at[2:-2, 2:-2, vs.taum1], vs.y_mp[2:-2, 2:-2, vs.tau],
        )
        vs.y_sc = update(
            vs.y_sc,
            at[2:-2, 2:-2, vs.taum1], vs.y_sc[2:-2, 2:-2, vs.tau],
        )
        vs.theta_rz = update(
            vs.theta_rz,
            at[2:-2, 2:-2, vs.taum1], vs.theta_rz[2:-2, 2:-2, vs.tau],
        )
        vs.theta_ss = update(
            vs.theta_ss,
            at[2:-2, 2:-2, vs.taum1], vs.theta_ss[2:-2, 2:-2, vs.tau],
        )
        vs.theta = update(
            vs.theta,
            at[2:-2, 2:-2, vs.taum1], vs.theta[2:-2, 2:-2, vs.tau],
        )
        vs.k_rz = update(
            vs.k_rz,
            at[2:-2, 2:-2, vs.taum1], vs.k_rz[2:-2, 2:-2, vs.tau],
        )
        vs.k_ss = update(
            vs.k_ss,
            at[2:-2, 2:-2, vs.taum1], vs.k_ss[2:-2, 2:-2, vs.tau],
        )
        vs.k = update(
            vs.k,
            at[2:-2, 2:-2, vs.taum1], vs.k[2:-2, 2:-2, vs.tau],
        )
        vs.h_rz = update(
            vs.h_rz,
            at[2:-2, 2:-2, vs.taum1], vs.h_rz[2:-2, 2:-2, vs.tau],
        )
        vs.h_ss = update(
            vs.h_ss,
            at[2:-2, 2:-2, vs.taum1], vs.h_ss[2:-2, 2:-2, vs.tau],
        )
        vs.h = update(
            vs.h,
            at[2:-2, 2:-2, vs.taum1], vs.h[2:-2, 2:-2, vs.tau],
        )
        vs.z0 = update(
            vs.z0,
            at[2:-2, 2:-2, vs.taum1], vs.z0[2:-2, 2:-2, vs.tau],
        )
        # set to 0 for numerical errors
        vs.S_fp_rz = update(
            vs.S_fp_rz,
            at[2:-2, 2:-2], npx.where((vs.S_fp_rz > -1e-6) & (vs.S_fp_rz < 0), 0, vs.S_fp_rz)[2:-2, 2:-2],
        )
        vs.S_lp_rz = update(
            vs.S_lp_rz,
            at[2:-2, 2:-2], npx.where((vs.S_lp_rz > -1e-6) & (vs.S_lp_rz < 0), 0, vs.S_lp_rz)[2:-2, 2:-2],
        )
        vs.S_fp_ss = update(
            vs.S_fp_ss,
            at[2:-2, 2:-2], npx.where((vs.S_fp_ss > -1e-6) & (vs.S_fp_ss < 0), 0, vs.S_fp_ss)[2:-2, 2:-2],
        )
        vs.S_lp_ss = update(
            vs.S_lp_ss,
            at[2:-2, 2:-2], npx.where((vs.S_lp_ss > -1e-6) & (vs.S_lp_ss < 0), 0, vs.S_lp_ss)[2:-2, 2:-2],
        )
        vs.prec = update(
            vs.prec,
            at[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau],
        )
        vs.event_id = update(
            vs.event_id,
            at[vs.taum1], vs.event_id[vs.tau],
        )
        vs.year = update(
            vs.year,
            at[vs.taum1], vs.year[vs.tau],
        )
        vs.month = update(
            vs.month,
            at[vs.taum1], vs.month[vs.tau],
        )
        vs.doy = update(
            vs.doy,
            at[vs.taum1], vs.doy[vs.tau],
        )

        return KernelOutput(
            ta=vs.ta,
            z_root=vs.z_root,
            ground_cover=vs.ground_cover,
            S_sur=vs.S_sur,
            S_int_top=vs.S_int_top,
            S_int_ground=vs.S_int_ground,
            S_dep=vs.S_dep,
            S_snow=vs.S_snow,
            swe=vs.swe,
            S_rz=vs.S_rz,
            S_ss=vs.S_ss,
            S_s=vs.S_s,
            S=vs.S,
            z_sat=vs.z_sat,
            z_wf=vs.z_wf,
            z_wf_t0=vs.z_wf_t0,
            z_wf_t1=vs.z_wf_t1,
            y_mp=vs.y_mp,
            y_sc=vs.y_sc,
            theta_rz=vs.theta_rz,
            theta_ss=vs.theta_ss,
            theta=vs.theta,
            h_rz=vs.h_rz,
            h_ss=vs.h_ss,
            h=vs.h,
            k_rz=vs.k_rz,
            k_ss=vs.k_ss,
            k=vs.k,
            z0=vs.z0,
            prec=vs.prec,
            event_id=vs.event_id,
            year=vs.year,
            month=vs.month,
            doy=vs.doy,
            S_fp_rz=vs.S_fp_rz,
            S_lp_rz=vs.S_lp_rz,
            S_fp_ss=vs.S_fp_ss,
            S_lp_ss=vs.S_lp_ss,
        )

    sim = SVATSetup()
    make_toy_forcing(sim._base_path, event_type='rain', ndays=10,
                     float_type=float_type)
    print(sim._base_path)
    sim.setup()
    sim.run()
    # delete toy forcing
    files = sim._base_path / "input"
    shutil.rmtree(files, ignore_errors=True)
    # merge model output into single file
    path = str(sim._base_path / "SVAT.*.nc")
    diag_files = glob.glob(path)
    states_hm_file = sim._base_path / "states_hm.nc"
    with h5netcdf.File(states_hm_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title='Test data for SVAT transport models',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='SVAT model with free drainage'
        )
        # collect dimensions
        for dfs in diag_files:
            with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                if not dfs.split('/')[-1].split('.')[1] == 'constant':
                    dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time'])}
                    time = onp.array(df.variables.get('Time'))
        for dfs in diag_files:
            with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                if not f.dimensions:
                    f.dimensions = dict_dim
                    v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = 'model run'
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["x"])
                    v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                    v.attrs['long_name'] = ''
                    v.attrs['units'] = ''
                    v[:] = onp.arange(dict_dim["y"])
                    v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                    units=var_obj.attrs["units"])
                    v[:] = time
                for var_sim in list(df.variables.keys()):
                    var_obj = df.variables.get(var_sim)
                    if var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] > 1:
                        v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals.swapaxes(0, 2)
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] <= 2:
                        v = f.create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                        vals = onp.array(var_obj)
                        v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])

        for dfs in diag_files:
            os.remove(dfs)