from pathlib import Path
import os
import h5netcdf
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-ns", "--nsamples", type=int, default=10000)
@click.option("-td", "--tmp-dir", type=str, default=None)
@roger_base_cli
def main(nsamples, tmp_dir):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, random_uniform
    from roger.core.numerics import calc_parameters_surface_kernel
    from roger.tools.setup import write_forcing
    import roger.lookuptables as lut
    import numpy as onp

    class SVATSetup(RogerSetup):
        """A SVAT model.
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
            settings.identifier = "SVAT"

            settings.nx, settings.ny, settings.nz = nsamples, 1, 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')

            # lysimeter surface 3.14 square meter (2m diameter)
            settings.dx = 2
            settings.dy = 2
            settings.dz = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = "1996-12-31 00:00:00"

            settings.enable_macropore_lower_boundary_condition = False

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "x",
                "y",
            ],
        )
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

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "lu_id",
                "z_soil",
                "dmpv",
                "lmpv",
                "theta_eff",
                "frac_lp",
                "frac_fp",
                "theta_ac",
                "theta_ufc",
                "theta_pwp",
                "ks",
                "kf",
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables

            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], 8)
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], 2200)
            vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], npx.array(random_uniform(1, 400, vs.dmpv.shape), dtype=int)[2:-2, 2:-2])
            vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], npx.array(random_uniform(1, 2000, vs.lmpv.shape), dtype=int)[2:-2, 2:-2])
            # effective porosity
            vs.theta_eff = update(vs.theta_eff, at[2:-2, 2:-2], random_uniform(0.15, 0.35, vs.theta_eff.shape)[2:-2, 2:-2])
            vs.frac_lp = update(vs.frac_lp, at[2:-2, 2:-2], random_uniform(0.1, 0.9, vs.theta_eff.shape)[2:-2, 2:-2])
            vs.frac_lp = update(vs.frac_lp, at[2:-2, 2:-2], 1 - vs.frac_lp[2:-2, 2:-2])
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], vs.theta_eff[2:-2, 2:-2] * vs.frac_lp)
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], vs.theta_eff[2:-2, 2:-2] * vs.frac_fp)
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], random_uniform(0.15, 0.35, vs.theta_pwp.shape)[2:-2, 2:-2])
            vs.ks = update(vs.ks, at[2:-2, 2:-2], random_uniform(1, 150, vs.ks.shape)[2:-2, 2:-2])
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

            vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], npx.where(0.46 > vs.theta_sat[2:-2, 2:-2, npx.newaxis], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis], 0.46))
            vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], npx.where(0.44 > vs.theta_sat[2:-2, 2:-2, npx.newaxis], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis], 0.44))

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
                vs.itt_day = vs.itt_day + 1
                ta = ta_day[:, :, vs.itt_day]
                pet = pet_day[:, :, vs.itt_day]
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
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["rates"].output_variables = ["prec", "aet", "transp", "evap_soil", "inf_mat_rz", "inf_mp_rz", "inf_sc_rz", "inf_ss", "q_rz", "q_ss", "cpr_rz", "dS_s", "dS", "q_snow"]
            diagnostics["rates"].output_frequency = 24 * 60 * 60
            diagnostics["rates"].sampling_frequency = 1
            if base_path:
                diagnostics["rates"].base_output_path = base_path

            diagnostics["collect"].output_variables = ["S_rz", "S_ss",
                                                       "S_pwp_rz", "S_fc_rz",
                                                       "S_sat_rz", "S_pwp_ss",
                                                       "S_fc_ss", "S_sat_ss",
                                                       "theta_rz", "theta_ss", "theta",
                                                       "S_snow"]
            diagnostics["collect"].output_frequency = 24 * 60 * 60
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            diagnostics["averages"].output_variables = ["ta"]
            diagnostics["averages"].output_frequency = 24 * 60 * 60
            diagnostics["averages"].sampling_frequency = 1
            if base_path:
                diagnostics["averages"].base_output_path = base_path

            diagnostics["constant"].output_variables = ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'theta_eff', 'frac_lp', 'frac_fp']
            diagnostics["constant"].output_frequency = 0
            diagnostics["constant"].sampling_frequency = 1
            if base_path:
                diagnostics["constant"].base_output_path = base_path

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

    model = SVATSetup()
    input_path = model._base_path / "input"
    model._set_input_dir(input_path)
    write_forcing(input_path)
    model.setup()
    model.run()
    return


if __name__ == "__main__":
    main()
