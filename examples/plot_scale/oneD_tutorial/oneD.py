from pathlib import Path
import os
import h5netcdf
import pandas as pd
import numpy as onp
from roger.cli.roger_run_base import roger_base_cli

# --- set the model parameters ------------------------
# land use ID (see README for description)
LU_ID = 8
# degree of sealing (-)
SEALING = 0
# surface slope (-)
SLOPE = 0.05
# total surface depression storage (mm)
S_DEP_TOT = 0
# soil depth (mm)
Z_SOIL = 1000
# density of vertical macropores (1/m2)
DMPV = 50
# density of horizontal macropores (1/m2)
DMPH = 100
# total length of vertical macropores (mm)
LMPV = 500
# air capacity (-)
THETA_AC = 0.1
# usable field capacity (-)
THETA_UFC = 0.1
# permanent wilting point (-)
THETA_PWP = 0.2
# saturated hydraulic conductivity (mm/h)
KS = 9.2
# hydraulic conductivity of bedrock/saturated zone (mm/h)
KF = 5

# --- set the initial conditions -----------------------
# soil water content of root zone/upper soil layer (-)
THETA_RZ = 0.3
# soil water content of subsoil/lower soil layer (-)
THETA_SS = 0.3

# --- set the output variables -----------------------
# list with simulated fluxes (see variables for description)
OUTPUT_FLUXES = ["aet", "transp", "evap_soil", "inf_mat", "inf_mp", "inf_sc", "q_ss",
                 "q_sub", "q_sub_mp", "q_sub_mat",
                 "q_hof", "q_sof"]
# list with simulated storages (see variables for description)
OUTPUT_STORAGES = ["theta"]
# !!!Do not modify the script below!!!


@roger_base_cli
def main():
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at
    from roger.core.surface import calc_parameters_surface_kernel
    from roger.tools.setup import write_forcing
    import roger.lookuptables as lut

    class ONEDSetup(RogerSetup):
        """A 1D model.
        """
        # custom attributes required by helper functions
        _base_path = Path(__file__).parent
        _input_dir = None

        # custom helper functions
        def _set_input_dir(self, path):
            if os.path.exists(path):
                self._input_dir = path
            else:
                if not os.path.exists(path):
                    os.mkdir(self._input_dir)
                    self._input_dir = path

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
            settings.identifier = "ONED"

            # total grid numbers in x- and y-direction
            settings.nx, settings.ny = 1, 1
            # derive total number of time steps from forcing
            settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')

            # spatial discretization (in meters)
            settings.dx = 1
            settings.dy = 1

            # origin of spatial grid
            settings.x_origin = 0.0
            settings.y_origin = 0.0
            # origin of time steps (e.g. 01-01-2023)
            settings.time_origin = self._get_time_origin(self._input_dir, 'forcing.nc')

            # enable specific processes
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
            settings = state.settings

            # spatial grid
            dx = allocate(state.dimensions, ("x"))
            dx = update(dx, at[:], settings.dx)
            dy = allocate(state.dimensions, ("y"))
            dy = update(dy, at[:], settings.dy)
            # distance from origin
            vs.x = update(vs.x, at[3:-2], settings.x_origin + npx.cumsum(dx[3:-2]))
            vs.y = update(vs.y, at[3:-2], settings.y_origin + npx.cumsum(dy[3:-2]))

        @roger_routine
        def set_look_up_tables(self, state):
            vs = state.variables

            # land use-dependent interception storage
            vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
            # land use-dependent ground cover
            vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
            # land use-dependent maximum ground cover
            vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
            # land use-dependent maximum ground cover
            vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
            # land use-dependent rooting depth
            vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)
            # macropore flow velocities
            vs.lut_mlms = update(vs.lut_mlms, at[:, :], lut.ARR_MLMS)

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine
        def set_parameters_setup(self, state):
            vs = state.variables

            # land use ID (see README for description)
            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], LU_ID)
            # degree of sealing (-)
            vs.sealing = update(vs.sealing, at[2:-2, 2:-2], 0)
            # surface slope (-)
            vs.slope = update(vs.slope, at[2:-2, 2:-2], SLOPE)
            # convert slope to percentage
            vs.slope_per = update(vs.slope_per, at[2:-2, 2:-2], vs.slope[2:-2, 2:-2] * 100)
            # total surface depression storage (mm)
            vs.S_dep_tot = update(vs.S_dep_tot, at[2:-2, 2:-2], 0)
            # soil depth (mm)
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], Z_SOIL)
            # density of vertical macropores (1/m2)
            vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], DMPV)
            # density of horizontal macropores (1/m2)
            vs.dmph = update(vs.dmph, at[2:-2, 2:-2], DMPH)
            # total length of vertical macropores (mm)
            vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], LMPV)
            # air capacity (-)
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], THETA_AC)
            # usable field capacity (-)
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], THETA_UFC)
            # permanent wilting point (-)
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], THETA_PWP)
            # saturated hydraulic conductivity (-)
            vs.ks = update(vs.ks, at[2:-2, 2:-2], KS)
            # hydraulic conductivity of bedrock/saturated zone (-)
            vs.kf = update(vs.kf, at[2:-2, 2:-2], KF)

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

            # interception storage of upper surface layer (mm)
            vs.S_int_top = update(vs.S_int_top, at[2:-2, 2:-2, :vs.taup1], 0)
            # snow water equivalent stored in upper surface layer (mm)
            vs.swe_top = update(vs.swe_top, at[2:-2, 2:-2, :vs.taup1], 0)
            # interception storage of lower surface layer (mm)
            vs.S_int_ground = update(vs.S_int_ground, at[2:-2, 2:-2, :vs.taup1], 0)
            # snow water equivalent stored in lower surface layer (mm)
            vs.swe_ground = update(vs.swe_ground, at[2:-2, 2:-2, :vs.taup1], 0)
            # surface depression storage (mm)
            vs.S_dep = update(vs.S_dep, at[2:-2, 2:-2, :vs.taup1], 0)
            # snow cover storage (mm)
            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, :vs.taup1], 0)
            # snow water equivalent of snow cover (mm)
            vs.swe = update(vs.swe, at[2:-2, 2:-2, :vs.taup1], 0)
            # soil water content of root zone/upper soil layer (mm/h)
            vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], THETA_RZ)
            # soil water content of subsoil/lower soil layer (mm/h)
            vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], THETA_SS)

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

            # variables written to output files
            diagnostics["rates"].output_variables = OUTPUT_FLUXES
            # values are aggregated to daily
            diagnostics["rates"].output_frequency = 24 * 60 * 60  # in seconds
            diagnostics["rates"].sampling_frequency = 1

            diagnostics["collect"].output_variables = OUTPUT_STORAGES
            # values are aggregated to daily
            diagnostics["collect"].output_frequency = 24 * 60 * 60  # in seconds
            diagnostics["collect"].sampling_frequency = 1

            # maximum bias of deterministic/numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            # shift variables backwards
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
        )

    # initialize the model structure
    model = ONEDSetup()
    # set path to directory containing the input files
    path_input = model._base_path / "input"
    model._set_input_dir(path_input)
    # runs event classification and writes the forcing
    write_forcing(path_input)
    # runs the model setup
    model.setup()
    # iterates over time steps
    model.run()
    return


if __name__ == "__main__":
    main()
