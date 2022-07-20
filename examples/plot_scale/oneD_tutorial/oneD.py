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
LMPV = 300
# air capacity (-)
THETA_AC = 0.08
# usable field capacity (-)
THETA_UFC = 0.15
# permanent wilting point (-)
THETA_PWP = 0.17
# saturated hydraulic conductivity (-)
KS = 9.2
# hydraulic conductivity of bedrock/saturated zone (-)
KF = 5

# --- set the initial conditions -----------------------
# soil water content of root zone/upper soil layer (-)
THETA_RZ = 0.32
# soil water content of subsoil/lower soil layer (-)
THETA_SS = 0.32

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
    from roger.core.operators import numpy as npx, update, at, for_loop
    from roger.core.utilities import _get_row_no
    from roger.tools.setup import write_forcing
    import roger.lookuptables as lut

    class ONEDSetup(RogerSetup):
        """A 1D model.
        """
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

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return len(onp.array(var_obj))

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return onp.array(var_obj)[-1] * 60 * 60

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = "ONED"

            # total grid numbers in x-,y- and z-direction
            settings.nx, settings.ny, settings.nz = 1, 1, 1
            # derive total number of time steps from forcing
            settings.nitt = self._get_nitt(self._input_dir, 'forcing.nc')
            settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')

            # spatial discretization (in meters)
            settings.dx = 1
            settings.dy = 1
            settings.dz = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = "2010-09-30 23:00:00"

            # enable specific processes
            settings.enable_groundwater_boundary = False
            settings.enable_lateral_flow = True
            settings.enable_routing = False

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "DT_SECS",
                "DT",
                "YEAR",
                "MONTH",
                "DOY",
                "dt_secs",
                "dt",
                "year",
                "month",
                "doy",
                "t",
                "itt",
                "x",
                "y",
            ],
        )
        def set_grid(self, state):
            vs = state.variables

            # temporal grid
            vs.DT_SECS = update(vs.DT_SECS, at[:], self._read_var_from_nc("dt", self._input_dir, 'forcing.nc'))
            vs.DT = update(vs.DT, at[:], vs.DT_SECS / (60 * 60))
            vs.YEAR = update(vs.YEAR, at[:], self._read_var_from_nc("year", self._input_dir, 'forcing.nc'))
            vs.MONTH = update(vs.MONTH, at[:], self._read_var_from_nc("month", self._input_dir, 'forcing.nc'))
            vs.DOY = update(vs.DOY, at[:], self._read_var_from_nc("doy", self._input_dir, 'forcing.nc'))
            vs.dt_secs = vs.DT_SECS[vs.itt]
            vs.dt = vs.DT[vs.itt]
            vs.year = vs.YEAR[vs.itt]
            vs.month = vs.MONTH[vs.itt]
            vs.doy = vs.DOY[vs.itt]
            vs.t = update(vs.t, at[:], npx.cumsum(vs.DT))
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

            if (vs.MONTH[vs.itt] != vs.MONTH[vs.itt - 1]) & (vs.itt > 1):
                vs.update(set_parameters_monthly_kernel(state))

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
            # soil water content of root zone/upper soil layer (-)
            vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], THETA_RZ)
            # soil water content of subsoil/lower soil layer (-)
            vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], THETA_SS)

        @roger_routine
        def set_forcing_setup(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "DT_SECS",
                "DT",
                "dt_secs",
                "dt",
                "itt",
                "prec",
                "ta",
                "pet",
                "pet_res",
                "event_id",
                "YEAR",
                "MONTH",
                "DOY",
                "year",
                "month",
                "doy",
                "tau"
            ],
        )
        def set_forcing(self, state):
            vs = state.variables

            vs.prec = update(vs.prec, at[2:-2, 2:-2], self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc')[:, :, vs.itt])
            vs.ta = update(vs.ta, at[2:-2, 2:-2], self._read_var_from_nc("TA", self._input_dir, 'forcing.nc')[:, :, vs.itt])
            vs.pet = update(vs.pet, at[2:-2, 2:-2], self._read_var_from_nc("PET", self._input_dir, 'forcing.nc')[:, :, vs.itt])
            vs.pet_res = update(vs.pet_res, at[2:-2, 2:-2], vs.pet[2:-2, 2:-2])
            vs.event_id = update(vs.event_id, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("EVENT_ID", self._input_dir, 'forcing.nc')[vs.itt])

            vs.dt_secs = vs.DT_SECS[vs.itt]
            vs.dt = vs.DT[vs.itt]
            vs.year = vs.YEAR[vs.itt]
            vs.month = vs.MONTH[vs.itt]
            vs.doy = vs.DOY[vs.itt]

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

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            # shift variables backwards
            vs.update(after_timestep_kernel(state))

    @roger_kernel
    def set_parameters_monthly_kernel(state):
        vs = state.variables

        # land use dependent upper interception storage
        S_int_top_tot = allocate(state.dimensions, ("x", "y"))
        trees_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
        trees_cond = update(
            trees_cond,
            at[:, :], npx.isin(vs.lu_id, npx.array([10, 11, 12, 15])),
        )

        def loop_body_S_int_top_tot(i, S_int_top_tot):
            mask = (vs.lu_id == i) & trees_cond
            row_no = _get_row_no(vs.lut_ilu[:, 0], i)
            S_int_top_tot = update(
                S_int_top_tot,
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month], S_int_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
            )

            return S_int_top_tot

        S_int_top_tot = for_loop(10, 16, loop_body_S_int_top_tot, S_int_top_tot)

        vs.S_int_top_tot = update(
            vs.S_int_top_tot,
            at[2:-2, 2:-2], S_int_top_tot[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
        )

        # land use dependent lower interception storage
        S_int_ground_tot = allocate(state.dimensions, ("x", "y"))

        ground_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
        ground_cond = update(
            ground_cond,
            at[:, :], npx.isin(vs.lu_id, npx.array([0, 5, 6, 7, 8, 9, 13, 98, 31, 32, 33, 40, 41, 50, 98]))
        )

        def loop_body_S_int_ground_tot(i, S_int_ground_tot):
            mask = (vs.lu_id == i) & ground_cond
            row_no = _get_row_no(vs.lut_ilu[:, 0], i)
            S_int_ground_tot = update(
                S_int_ground_tot,
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month], S_int_ground_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
            )

            return S_int_ground_tot

        trees_ground_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
        trees_ground_cond = update(
            trees_ground_cond,
            at[:, :], npx.isin(vs.lu_id, npx.array([10, 11, 12, 15]))
        )

        def loop_body_S_int_ground_tot_trees(i, S_int_ground_tot):
            mask = (vs.lu_id == i) & trees_ground_cond
            S_int_ground_tot = update(
                S_int_ground_tot,
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 1, S_int_ground_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
            )

            return S_int_ground_tot

        S_int_ground_tot = for_loop(0, 51, loop_body_S_int_ground_tot, S_int_ground_tot)
        S_int_ground_tot = for_loop(10, 16, loop_body_S_int_ground_tot_trees, S_int_ground_tot)

        vs.S_int_ground_tot = update(
            vs.S_int_ground_tot,
            at[2:-2, 2:-2], S_int_ground_tot[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
        )

        # land use dependent ground cover (canopy cover)
        ground_cover = allocate(state.dimensions, ("x", "y"))

        cc_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
        cc_cond = update(
            cc_cond,
            at[:, :], npx.isin(vs.lu_id, npx.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 98, 31, 32, 33, 40, 41, 50, 98]))
        )

        def loop_body_ground_cover(i, ground_cover):
            mask = (vs.lu_id == i) & cc_cond
            row_no = _get_row_no(vs.lut_gc[:, 0], i)
            ground_cover = update(
                ground_cover,
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_gc[row_no, vs.month], ground_cover[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
            )

            return ground_cover

        ground_cover = for_loop(0, 51, loop_body_ground_cover, ground_cover)

        vs.ground_cover = update(
            vs.ground_cover,
            at[2:-2, 2:-2, vs.tau], ground_cover[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
        )

        # land use dependent transpiration coeffcient
        basal_transp_coeff = allocate(state.dimensions, ("x", "y"))

        def loop_body_basal_transp_coeff(i, basal_transp_coeff):
            mask = (vs.lu_id == i) & cc_cond
            row_no = _get_row_no(vs.lut_gc[:, 0], i)
            basal_transp_coeff = update(
                basal_transp_coeff,
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_gc[row_no, vs.month] / vs.lut_gcm[row_no, 1], basal_transp_coeff[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
            )

            return basal_transp_coeff

        basal_transp_coeff = for_loop(0, 51, loop_body_basal_transp_coeff, basal_transp_coeff)

        basal_transp_coeff = update(
            basal_transp_coeff,
            at[2:-2, 2:-2], npx.where(vs.maskRiver[2:-2, 2:-2] | vs.maskLake[2:-2, 2:-2], 0, basal_transp_coeff[2:-2, 2:-2]),
        )

        vs.basal_transp_coeff = update(
            vs.basal_transp_coeff,
            at[2:-2, 2:-2], basal_transp_coeff[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
        )

        # land use dependent evaporation coeffcient
        basal_evap_coeff = allocate(state.dimensions, ("x", "y"))

        def loop_body_basal_evap_coeff(i, basal_evap_coeff):
            mask = (vs.lu_id == i) & cc_cond
            row_no = _get_row_no(vs.lut_gc[:, 0], i)
            basal_evap_coeff = update(
                basal_evap_coeff,
                at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 1 - ((vs.lut_gc[row_no, vs.month] / vs.lut_gcm[row_no, 1]) * vs.lut_gcm[row_no, 1]), basal_evap_coeff[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
            )

            return basal_evap_coeff

        basal_evap_coeff = for_loop(0, 51, loop_body_basal_evap_coeff, basal_evap_coeff)

        basal_evap_coeff = update(
            basal_evap_coeff,
            at[2:-2, 2:-2], npx.where(vs.maskRiver[2:-2, 2:-2] | vs.maskLake[2:-2, 2:-2], 1, basal_evap_coeff[2:-2, 2:-2]),
        )

        vs.basal_evap_coeff = update(
            vs.basal_evap_coeff,
            at[2:-2, 2:-2], basal_evap_coeff[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
        )

        # maximum snow interception storage
        vs.swe_top_tot = update(
            vs.swe_top_tot,
            at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] >= -3) & (vs.ta[2:-2, 2:-2, vs.tau] <= -1) & (vs.lu_id[2:-2, 2:-2] == 10), 2.5 + 0.5 * vs.ta[2:-2, 2:-2, vs.tau] * 9, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.swe_top_tot = update(
            vs.swe_top_tot,
            at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] >= -3) & (vs.ta[2:-2, 2:-2, vs.tau] <= -1) & (vs.lu_id[2:-2, 2:-2] == 11), 2.5 + 0.5 * vs.ta[2:-2, 2:-2, vs.tau] * 15, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
        )
        vs.swe_top_tot = update(
            vs.swe_top_tot,
            at[2:-2, 2:-2], npx.where((vs.ta[2:-2, 2:-2, vs.tau] >= -3) & (vs.ta[2:-2, 2:-2, vs.tau] <= -1) & (vs.lu_id[2:-2, 2:-2] == 12), 2.5 + 0.5 * vs.ta[2:-2, 2:-2, vs.tau] * 25, vs.swe_top_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
        )

        vs.lai = update(
            vs.lai,
            at[2:-2, 2:-2], npx.log(1 / (1 - vs.ground_cover[2:-2, 2:-2, vs.tau])) / npx.log(1 / 0.7) * vs.maskCatch[2:-2, 2:-2]
        )

        vs.throughfall_coeff_top = update(
            vs.throughfall_coeff_top,
            at[2:-2, 2:-2], npx.where(npx.isin(vs.lu_id[2:-2, 2:-2], npx.array([10, 11, 12])), npx.where(vs.lai[2:-2, 2:-2] > 1, 0, 1 - vs.lai[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2]
        )

        vs.throughfall_coeff_ground = update(
            vs.throughfall_coeff_ground,
            at[2:-2, 2:-2], npx.where(npx.isin(vs.lu_id[2:-2, 2:-2], npx.arange(500, 598)), npx.where(vs.lai[2:-2, 2:-2] > 1, 0, 1 - vs.lai[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2]
        )

        return KernelOutput(
            S_int_top_tot=vs.S_int_top_tot,
            S_int_ground_tot=vs.S_int_ground_tot,
            ground_cover=vs.ground_cover,
            basal_transp_coeff=vs.basal_transp_coeff,
            basal_evap_coeff=vs.basal_evap_coeff,
            swe_top_tot=vs.swe_top_tot,
            lai=vs.lai,
            throughfall_coeff_top=vs.throughfall_coeff_top,
            throughfall_coeff_ground=vs.throughfall_coeff_ground,
        )
        
    @roger_kernel
    def set_forcing_kernel(state):
        vs = state.variables

        vs.prec = update(vs.prec, at[2:-2, 2:-2], vs.PREC[2:-2, 2:-2, vs.itt])
        vs.ta = update(vs.ta, at[2:-2, 2:-2, vs.tau], vs.TA[2:-2, 2:-2, vs.itt])
        vs.pet = update(vs.pet, at[2:-2, 2:-2], vs.PET[2:-2, 2:-2, vs.itt])
        vs.pet_res = update(vs.pet, at[2:-2, 2:-2], vs.PET[2:-2, 2:-2, vs.itt])

        vs.dt_secs = vs.DT_SECS[vs.itt]
        vs.dt = vs.DT[vs.itt]
        vs.year = vs.YEAR[vs.itt]
        vs.month = vs.MONTH[vs.itt]
        vs.doy = vs.DOY[vs.itt]

        return KernelOutput(
            prec=vs.prec,
            ta=vs.ta,
            pet=vs.pet,
            pet_res=vs.pet_res,
            dt=vs.dt,
            dt_secs=vs.dt_secs,
            year=vs.year,
            month=vs.month,
            doy=vs.doy,
        )

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
            at[2:-2, 2:-2, vs.taum1], vs.event_id[2:-2, 2:-2, vs.tau],
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
