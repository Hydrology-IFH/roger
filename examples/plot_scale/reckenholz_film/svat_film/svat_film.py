from pathlib import Path
import os
import h5netcdf
import numpy as onp
from roger.cli.roger_run_base import roger_base_cli


@roger_base_cli
def main():
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.utilities import _get_first_row_no, _get_last_row_no
    from roger.core.operators import numpy as npx, update, at
    import roger.lookuptables as lut
    from roger.core.numerics import calc_parameters_surface_kernel
    from roger.tools.setup import write_forcing

    class SVATFILMSetup(RogerSetup):
        """A SVAT model including gravity-driven infiltration and percolation.
        """
        _base_path = Path(__file__).parent
        _input_dir = None

        def _set_input_dir(self, path):
            if os.path.exists(path):
                self._input_dir = path
            else:
                self._input_dir = path
                if not os.path.exists(self._input_dir):
                    os.mkdir(self._input_dir)

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

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = "SVATFILM"

            settings.nx, settings.ny, settings.nz = 1, 1, 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')
            settings.nittevent_ff = 5 * 24 * 6
            settings.nittevent_ff_p1 = settings.nittevent_ff + 1
            settings.nevent_ff = 20

            settings.dx = 1
            settings.dy = 1
            settings.dz = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0

            settings.enable_film_flow = True
            settings.enable_macropore_lower_boundary_condition = False

            settings.ff_tc = 0.15
            settings.end_event = 21600

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "x",
                "y",
            ],
        )
        def set_grid(self, state):
            vs = state.variables

            # spatial grid
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
            vs.slope = update(vs.slope, at[2:-2, 2:-2], 0)
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], 1350)
            vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], 50)
            vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], 1000)
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], 0.09)
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], 0.11)
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], 0.15)
            vs.ks = update(vs.ks, at[2:-2, 2:-2], 25)
            vs.kf = update(vs.kf, at[2:-2, 2:-2], 2500)
            vs.a_ff = update(vs.a_ff, at[2:-2, 2:-2], 0.19)
            vs.c_ff = update(vs.c_ff, at[2:-2, 2:-2], 0.001)

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

            vs.S_int_top = update(vs.S_int_top, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.swe_top = update(vs.swe_top, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.S_int_ground = update(vs.S_int_ground, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.swe_ground = update(vs.swe_ground, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.S_dep = update(vs.S_dep, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.swe = update(vs.swe, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], 0.4)
            vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], 0.47)

        @roger_routine
        def set_boundary_conditions_setup(self, state):
            pass

        @roger_routine
        def set_boundary_conditions(self, state):
            pass

        @roger_routine
        def set_forcing_setup(self, state):
            pass

        @roger_routine
        def set_forcing(self, state):
            vs = state.variables
            settings = state.settings

            vs.year = update(vs.year, at[1], self._read_var_from_nc("YEAR", self._input_dir, 'forcing.nc')[vs.itt_forc])
            vs.month = update(vs.month, at[1], self._read_var_from_nc("MONTH", self._input_dir, 'forcing.nc')[vs.itt_forc])
            vs.doy = update(vs.doy, at[1], self._read_var_from_nc("DOY", self._input_dir, 'forcing.nc')[vs.itt_forc])

            N_prec = self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc').shape[-1]

            # adaptive time stepping
            condt = (vs.time % (24 * 60 * 60) == 0)
            if condt:
                vs.itt_event = 0
                itt_forc_start = vs.itt_forc
                itt_forc_end = vs.itt_forc + 6 * 24
                end_event = int(settings.end_event / 60 / 6)
                prec_events = self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc')[:, :, itt_forc_start:itt_forc_end]
                ta_events = self._read_var_from_nc("TA", self._input_dir, 'forcing.nc')[:, :, itt_forc_start:itt_forc_end]
                while (prec_events[:, :, itt_forc_end-end_event:itt_forc_end] > 0).any():
                    itt_forc_end = itt_forc_end + 6 * 24
                    prec_events = self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc')[:, :, itt_forc_start:itt_forc_end]
                    ta_events = self._read_var_from_nc("TA", self._input_dir, 'forcing.nc')[:, :, itt_forc_start:itt_forc_end]
                    if itt_forc_end >= N_prec:
                        break

                event = npx.zeros((prec_events.shape[-1]), dtype=int)
                break_counter = prec_events.shape[-1]
                event_counter = 1
                for i in range(0, prec_events.shape[-1]):
                    if (prec_events[:, :, vs.itt_forc+i] > 0) & (ta_events[:, :, vs.itt_forc+i] > 0):
                        event = update(event, at[i], event_counter)
                        break_counter = 0
                    elif ((prec_events[:, :, vs.itt_forc+i] <= 0) & (ta_events[:, :, vs.itt_forc+i] <= 0)).all() & (break_counter < settings.end_event / (60 * 10)):
                        event = update(event, at[i], event_counter)
                        break_counter = break_counter + 1
                    elif ((prec_events[:, :, vs.itt_forc+i] <= 0) & (ta_events[:, :, vs.itt_forc+i] <= 0)).all() & (break_counter < settings.end_event / (60 * 10)):
                        event = update(event, at[i], 0)
                        break_counter = break_counter + 1
                    elif (event[i-end_event:i] <= 0).all() & (i >= end_event):
                        event_counter = event_counter + 1

            prec_day = self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc')[:, :, vs.itt_forc:vs.itt_forc+24*6]
            ta_day = self._read_var_from_nc("TA", self._input_dir, 'forcing.nc')[:, :, vs.itt_forc:vs.itt_forc+24*6]
            pet_day = self._read_var_from_nc("PET", self._input_dir, 'forcing.nc')[:, :, vs.itt_forc:vs.itt_forc+24*6]

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
                prec_hour = prec_day[:, :, vs.itt_event:vs.itt_event+6]
                ta_hour = ta_day[:, :, vs.itt_event:vs.itt_event+6]
                prec = npx.sum(prec_hour, axis=-1)
                ta = npx.mean(ta_hour, axis=-1)
                vs.dt_secs = 60 * 60
            # heavy rainfall event - 10 minutes time steps
            elif (cond1 or cond3) and not cond2 and not cond4 and not cond5:
                prec = prec_day[:, :, vs.itt_event]
                ta = ta_day[:, :, vs.itt_event]
                vs.dt_secs = 10 * 60

            # determine end of event
            if ((prec > 0).any() | ((vs.swe[2:-2, 2:-2, vs.tau] > 0) | (vs.swe_top[2:-2, 2:-2, vs.tau] > 0)).any() and (ta > settings.ta_fm)).any():
                vs.time_event0 = 0
            elif ((prec <= 0) & (ta > settings.ta_fm)).all() or ((prec > 0) & (ta <= settings.ta_fm)).all() or ((vs.swe[2:-2, 2:-2, vs.taum1] > 0).any() & (vs.swe[2:-2, 2:-2, vs.tau] <= 0).all()):
                vs.time_event0 = vs.time_event0 + vs.dt_secs

            # increase time stepping at end of event if either full hour
            # or full day, respectively
            if vs.time_event0 <= settings.end_event and (vs.dt_secs == 10 * 60):
                ta = ta_day[:, :, vs.itt_event]
                pet = pet_day[:, :, vs.itt_event]
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], vs.event_id_counter,
                )
                vs.dt = 1 / 6
                vs.itt_event = vs.itt_event + 1
            elif vs.time_event0 <= settings.end_event and (vs.dt_secs == 60 * 60):
                ta = npx.mean(ta_day[:, :, vs.itt_event:vs.itt_event+6], axis=-1)
                pet = npx.sum(pet_day[:, :, vs.itt_event:vs.itt_event+6], axis=-1)
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], vs.event_id_counter,
                )
                vs.dt = 1
                vs.itt_event = vs.itt_event + 6
            elif vs.time_event0 <= settings.end_event and (vs.dt_secs == 24 * 60 * 60):
                ta = npx.mean(ta_day[:, :, vs.itt_event:vs.itt_event+24*6], axis=-1)
                pet = npx.sum(pet_day[:, :, vs.itt_event:vs.itt_event+24*6], axis=-1)
                vs.dt = 24
                vs.itt_event = 0
            elif vs.time_event0 > settings.end_event and (vs.time % (60 * 60) != 0) and (vs.dt_secs == 10 * 60):
                vs.dt_secs = 10 * 60
                vs.dt = 1 / 6
                vs.itt_event = vs.itt_event + 1
                ta = ta_day[:, :, vs.itt_event]
                pet = pet_day[:, :, vs.itt_event]
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], 0,
                )
            elif vs.time_event0 > settings.end_event and (vs.time % (60 * 60) == 0) and ((vs.dt_secs == 10 * 60) or (vs.dt_secs == 60 * 60)):
                ta = npx.mean(ta_day[:, :, vs.itt_event:vs.itt_event+6], axis=-1)
                pet = npx.sum(pet_day[:, :, vs.itt_event:vs.itt_event+6], axis=-1)
                vs.dt_secs = 60 * 60
                vs.dt = 1
                vs.itt_event = vs.itt_event + 6
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], 0,
                )
            elif vs.time_event0 > settings.end_event and (vs.time % (24 * 60 * 60) == 0) and (vs.dt_secs == 24 * 60 * 60):
                ta = npx.mean(ta_day[:, :, vs.itt_event:vs.itt_event+24*6], axis=-1)
                pet = npx.sum(pet_day[:, :, vs.itt_event:vs.itt_event+24*6], axis=-1)
                vs.dt_secs = 24 * 60 * 60
                vs.dt = 24
                vs.itt_event = 0
                vs.event_id = update(
                    vs.event_id,
                    at[vs.tau], 0,
                )

            cond_event = (event == vs.event_id_counter)
            t1 = _get_first_row_no(cond_event, vs.event_id[vs.tau])
            t2 = _get_last_row_no(cond_event, vs.event_id[vs.tau])
            prec_event = prec_events[:, :, t1:t2]
            ta_event = ta_events[:, :, t1:t2]

            # set event id for next event
            if (vs.event_id[vs.taum1] > 0) & (vs.event_id[vs.tau] == 0):
                vs.event_id_counter = vs.event_id_counter + 1

            # set forcing for current time step
            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], prec)
            vs.ta = update(vs.ta, at[2:-2, 2:-2, vs.tau], ta)
            vs.pet = update(vs.pet, at[2:-2, 2:-2], pet)
            vs.pet_res = update(vs.pet_res, at[2:-2, 2:-2], pet)

            vs.itt_forc = vs.itt_forc + 6 * 24

        @roger_routine
        def set_diagnostics(self, state):
            pass

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))
            vs.update(after_timestep_film_flow_kernel(state))

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
        vs.prec_day_sum = update(
            vs.prec_day_sum,
            at[2:-2, 2:-2, vs.taum1], vs.prec_day_sum[2:-2, 2:-2, vs.tau],
        )
        vs.t_event_sum = update(
            vs.t_event_sum,
            at[2:-2, 2:-2, vs.taum1], vs.t_event_sum[2:-2, 2:-2, vs.tau],
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
            t_event_sum=vs.t_event_sum,
            prec_day_sum=vs.prec_day_sum,
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
        )

    @roger_kernel
    def after_timestep_film_flow_kernel(state):
        vs = state.variables

        vs.z_wf_ff = update(
            vs.z_wf_ff,
            at[2:-2, 2:-2, :, vs.taum1], vs.z_wf_ff[2:-2, 2:-2, :, vs.tau],
        )
        vs.z_pf_ff = update(
            vs.z_pf_ff,
            at[2:-2, 2:-2, :, vs.taum1], vs.z_pf_ff[2:-2, 2:-2, :, vs.tau],
        )
        vs.theta_rz_ff = update(
            vs.theta_rz_ff,
            at[2:-2, 2:-2, vs.taum1], vs.theta_rz_ff[2:-2, 2:-2, vs.tau],
        )
        vs.theta_ss_ff = update(
            vs.theta_ss_ff,
            at[2:-2, 2:-2, vs.taum1], vs.theta_ss_ff[2:-2, 2:-2, vs.tau],
        )
        vs.theta_ff = update(
            vs.theta_ff,
            at[2:-2, 2:-2, vs.taum1], vs.theta_ff[2:-2, 2:-2, vs.tau],
        )
        vs.z_pf = update(
            vs.z_pf,
            at[2:-2, 2:-2, vs.taum1], vs.z_pf[2:-2, 2:-2, vs.tau],
        )

        return KernelOutput(
            z_wf_ff=vs.z_wf_ff,
            z_pf_ff=vs.z_pf_ff,
            theta_rz_ff=vs.theta_rz_ff,
            theta_ss_ff=vs.theta_ss_ff,
            theta_ff=vs.theta_ff,
            z_pf=vs.z_pf,
        )

    model = SVATFILMSetup()
    path_input = model._base_path / "input"
    model._set_input_dir(path_input)
    write_forcing(path_input)
    model.setup()
    model.run()
    return


if __name__ == "__main__":
    main()
