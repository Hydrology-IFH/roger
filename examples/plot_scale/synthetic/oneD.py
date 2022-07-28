from pathlib import Path
import os
import h5netcdf
import pandas as pd
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-ms", "--meteo-station", type=click.Choice(['breitnau', 'ihringen']), default='ihringen')
@roger_base_cli
def main(meteo_station):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at
    from roger.core.numerics import calc_parameters_surface_kernel
    from roger.tools.setup import write_forcing
    import roger.lookuptables as lut
    from roger.io_tools.csv import write_meteo_csv_from_dwd

    class ONEDSetup(RogerSetup):
        """A 1D model.
        """
        _base_path = Path(__file__).parent
        _input_dir = None
        _identifier = None

        def _set_input_dir(self, path):
            if os.path.exists(path):
                self._input_dir = path
            else:
                self._input_dir = path
                if not os.path.exists(self._input_dir):
                    os.mkdir(self._input_dir)

        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = self._input_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj)

        def _read_var_from_csv(self, var, path_dir, file):
            csv_file = path_dir / file
            infile = pd.read_csv(csv_file, sep=';', skiprows=1)
            var_obj = infile.loc[:, var]
            return npx.array(var_obj)[:, npx.newaxis]

        def _get_nx(self, path_dir, file):
            csv_file = path_dir / file
            infile = pd.read_csv(csv_file, sep=';', skiprows=1)
            return len(infile.index)

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return len(onp.array(var_obj))

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['dt']
                return onp.sum(onp.array(var_obj))

        def _set_identifier(self, identifier):
            self._identifier = identifier

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = self._identifier

            settings.nx, settings.ny, settings.nz = self._get_nx(self._base_path, 'parameter_grid.csv'), 1, 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')

            settings.dx = 1
            settings.dy = 1
            settings.dz = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = "2010-09-30 00:00:00"

            settings.enable_groundwater_boundary = False
            settings.enable_lateral_flow = True
            settings.enable_routing = False

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
            vs.lut_mlms = update(vs.lut_mlms, at[:, :], lut.ARR_MLMS)

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine
        def set_parameters_setup(self, state):
            vs = state.variables

            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], self._read_var_from_csv("lu_id", self._base_path,  "parameter_grid.csv"))
            vs.sealing = update(vs.sealing, at[2:-2, 2:-2], 0)
            vs.slope = update(vs.slope, at[2:-2, 2:-2], self._read_var_from_csv("slope", self._base_path,  "parameter_grid.csv"))
            vs.slope_per = update(vs.slope_per, at[2:-2, 2:-2], vs.slope[2:-2, 2:-2] * 100)
            vs.S_dep_tot = update(vs.S_dep_tot, at[2:-2, 2:-2], self._read_var_from_csv("S_dep_tot", self._base_path,  "parameter_grid.csv"))
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], self._read_var_from_csv("z_soil", self._base_path,  "parameter_grid.csv"))
            vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], self._read_var_from_csv("dmpv", self._base_path,  "parameter_grid.csv"))
            vs.dmph = update(vs.dmph, at[2:-2, 2:-2], self._read_var_from_csv("dmph", self._base_path,  "parameter_grid.csv"))
            vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], self._read_var_from_csv("lmpv", self._base_path,  "parameter_grid.csv"))
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], self._read_var_from_csv("theta_ac", self._base_path,  "parameter_grid.csv"))
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], self._read_var_from_csv("theta_ufc", self._base_path,  "parameter_grid.csv"))
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], self._read_var_from_csv("theta_pwp", self._base_path,  "parameter_grid.csv"))
            vs.ks = update(vs.ks, at[2:-2, 2:-2], self._read_var_from_csv("ks", self._base_path,  "parameter_grid.csv"))
            vs.kf = update(vs.kf, at[2:-2, 2:-2], self._read_var_from_csv("kf", self._base_path,  "parameter_grid.csv"))

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
            vs.z_sat = update(vs.z_sat, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_csv("theta", self._base_path,  "parameter_grid.csv")[:, :, npx.newaxis])
            vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_csv("theta", self._base_path,  "parameter_grid.csv")[:, :, npx.newaxis])

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
        def set_diagnostics(self, state):
            diagnostics = state.diagnostics

            diagnostics["rates"].output_variables = ["aet", "pet", "transp",
                                                     "evap_soil", "inf_mat",
                                                     "inf_mp", "inf_sc", "q_ss",
                                                     "q_sub", "q_sub_mp",
                                                     "q_sub_mat", "q_hof", "q_sof",
                                                     "prec", "rain", "snow",
                                                     "int_prec", "int_rain_top",
                                                     "int_rain_ground", "int_snow_top",
                                                     "int_snow_ground", "q_snow",
                                                     "evap_sur", "snow_top",
                                                     "snow_ground", "snow_melt_drip"]
            diagnostics["rates"].output_frequency = 24 * 60 * 60
            diagnostics["rates"].sampling_frequency = 1

            diagnostics["maximum"].output_variables = ["z_sat", "theta", "S_s", "S_int_top", "S_int_ground", "S_int_top_tot", "S_int_ground_tot", "S_snow", "swe", "swe_top", "swe_ground", "swe_top_tot"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1

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
            z0=vs.z0,
            prec=vs.prec,
            event_id=vs.event_id,
            year=vs.year,
            month=vs.month,
            doy=vs.doy,
            k_rz=vs.k_rz,
            k_ss=vs.k_ss,
            k=vs.k,
            S_fp_rz=vs.S_fp_rz,
            S_lp_rz=vs.S_lp_rz,
            S_fp_ss=vs.S_fp_ss,
            S_lp_ss=vs.S_lp_ss,
        )

    model = ONEDSetup()
    identifier = f'ONED_{meteo_station}'
    model._set_identifier(identifier)
    path_meteo_station = model._base_path / "input" / meteo_station
    model._set_input_dir(path_meteo_station)
    write_meteo_csv_from_dwd(path_meteo_station)
    write_forcing(path_meteo_station)
    model.setup()
    model.run()
    return


if __name__ == "__main__":
    main()
