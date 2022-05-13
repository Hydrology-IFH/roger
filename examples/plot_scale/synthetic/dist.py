from pathlib import Path
import datetime
import glob
import os
import h5netcdf
import pandas as pd
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at, for_loop, where
from roger.core.utilities import _get_row_no
from roger.tools.setup import write_forcing
import roger.lookuptables as lut
import numpy as onp
from roger.io_tools.csv import write_meteo_csv_from_dwd


class DISTSetup(RogerSetup):
    """A distributed model.
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
        nc_file = self._input_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables[var]
            return npx.array(var_obj)

    def _read_var_from_csv(self, var, path_dir, file):
        csv_file = self._input_dir / file
        with pd.read_csv(csv_file, sep=';', skiprows=1) as infile:
            var_obj = infile.loc[:, var]
            return npx.array(var_obj)

    def _get_nx(self, path_dir, file):
        csv_file = self._input_dir / file
        with pd.read_csv(csv_file, sep=';', skiprows=1) as infile:
            return len(infile.index)

    def _get_nittevent(self, path_dir, file):
        nc_file = self._input_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['nitt_event']
            return onp.int32(onp.array(var_obj)[0])

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
        settings.identifier = "DIST"

        settings.nx, settings.ny, settings.nz = self._get_nx(self._base_path, 'parameter_grid.csv'), 1, 1
        settings.nitt = self._get_nitt(self._input_dir, 'forcing.nc')
        settings.nittevent = self._get_nittevent(self._input_dir, 'forcing.nc')
        settings.nittevent_p1 = settings.nittevent + 1
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
        vs.slope_per = update(vs.slope_per, at[2:-2, 2:-2], vs.slope * 100)
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

        if (vs.MONTH[vs.itt] != vs.MONTH[vs.itt - 1]) & (vs.itt > 1):
            vs.update(set_parameters_monthly_kernel(state))

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
        vs.z_sat = update(vs.z_sat, at[2:-2, 2:-2, :vs.taup1], 250)

    @roger_routine
    def set_forcing_setup(self, state):
        vs = state.variables

        vs.PREC = update(vs.PREC, at[2:-2, 2:-2, :], self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc'))
        vs.TA = update(vs.TA, at[2:-2, 2:-2, :], self._read_var_from_nc("TA", self._input_dir, 'forcing.nc'))
        vs.PET = update(vs.PET, at[2:-2, 2:-2, :], self._read_var_from_nc("PET", self._input_dir, 'forcing.nc'))
        vs.EVENT_ID = update(vs.EVENT_ID, at[2:-2, 2:-2, :], self._read_var_from_nc("EVENT_ID", self._input_dir, 'forcing.nc'))

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables

        vs.update(set_forcing_kernel(state))

    @roger_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics

        diagnostics["rates"].output_variables = ["aet", "transp", "evap_soil", "inf_mat", "inf_mp", "inf_sc", "q_ss", "q_sub", "q_sub_mp", "q_sub_mat", "q_hof", "q_sof"]
        diagnostics["rates"].output_frequency = 24 * 60 * 60
        diagnostics["rates"].sampling_frequency = 1

    @roger_routine
    def after_timestep(self, state):
        vs = state.variables

        vs.update(after_timestep_kernel(state))


@roger_kernel
def set_parameters_monthly_kernel(state):
    vs = state.variables

    # land use dependent upper interception storage
    def loop_body_S_int_top_tot(i, S_int_top_tot):
        arr_i = allocate(state.dimensions, ("x", "y"))
        arr_i = update(
            arr_i,
            at[:, :], i * (vs.lu_id == i),
        )
        mask = (vs.lu_id == i) & npx.isin(arr_i, npx.array([10, 11, 12, 15, 16]))
        row_no = _get_row_no(vs.lut_ilu[:, 0], i)
        S_int_top_tot = update(
            S_int_top_tot,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month], 0),
        )

        return S_int_top_tot

    S_int_top_tot = allocate(state.dimensions, ("x", "y"))

    S_int_top_tot = for_loop(10, 17, loop_body_S_int_top_tot, S_int_top_tot)
    mask = npx.isin(vs.lu_id, npx.array([10, 11, 12, 15, 16]))
    vs.S_int_top_tot = update(
        vs.S_int_top_tot,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], S_int_top_tot[2:-2, 2:-2], vs.S_int_top_tot[2:-2, 2:-2]),
    )

    # land use dependent lower interception storage
    S_int_ground_tot = allocate(state.dimensions, ("x", "y"))

    def loop_body_S_int_ground_tot(i, S_int_ground_tot):
        arr_i = allocate(state.dimensions, ("x", "y"))
        arr_i = update(
            arr_i,
            at[:, :], i * vs.maskCatch,
        )
        mask = (vs.lu_id == i) & ~npx.isin(arr_i, npx.array([10, 11, 12, 15, 16]))
        row_no = _get_row_no(vs.lut_ilu[:, 0], i)
        S_int_ground_tot = update_add(
            S_int_ground_tot,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month], 0),
        )

        return S_int_ground_tot

    def loop_body_S_int_ground_tot_trees(i, S_int_ground_tot):
        arr_i = allocate(state.dimensions, ("x", "y"))
        arr_i = update(
            arr_i,
            at[:, :], i * vs.maskCatch,
        )
        mask = (vs.lu_id == i) & npx.isin(arr_i, npx.array([10, 11, 12, 15, 16]))
        S_int_ground_tot = update_add(
            S_int_ground_tot,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 1, 0),
        )

        return S_int_ground_tot

    S_int_ground_tot = update(
        S_int_ground_tot,
        at[:, :], for_loop(0, 51, loop_body_S_int_ground_tot, S_int_ground_tot),
    )
    S_int_ground_tot = update(
        S_int_ground_tot,
        at[:, :], for_loop(10, 17, loop_body_S_int_ground_tot_trees, S_int_ground_tot),
    )

    mask = npx.isin(vs.lu_id, npx.arange(0, 51, 1, dtype=int))
    vs.S_int_ground_tot = update(
        vs.S_int_ground_tot,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], S_int_ground_tot[2:-2, 2:-2], vs.S_int_ground_tot[2:-2, 2:-2]),
    )

    # land use dependent ground cover (canopy cover)
    ground_cover = allocate(state.dimensions, ("x", "y"))

    def loop_body_ground_cover(i, ground_cover):
        mask = (vs.lu_id == i)
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        ground_cover = update_add(
            ground_cover,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_gc[row_no, vs.month], 0),
        )

        return ground_cover

    ground_cover = for_loop(0, 51, loop_body_ground_cover, ground_cover)

    mask = npx.isin(vs.lu_id, npx.arange(0, 51, 1, dtype=int))
    vs.ground_cover = update(
        vs.ground_cover,
        at[2:-2, 2:-2, vs.tau], npx.where(mask[2:-2, 2:-2], ground_cover[2:-2, 2:-2], vs.ground_cover[2:-2, 2:-2, vs.tau]),
    )

    # land use dependent transpiration coeffcient
    basal_transp_coeff = allocate(state.dimensions, ("x", "y"))

    def loop_body_basal_transp_coeff(i, basal_transp_coeff):
        mask = (vs.lu_id == i)
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        basal_transp_coeff = update_add(
            basal_transp_coeff,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.lut_gc[row_no, vs.month] / vs.lut_gcm[row_no, 1], 0),
        )

        return basal_transp_coeff

    basal_transp_coeff = update(
        basal_transp_coeff,
        at[:, :], where(vs.maskRiver | vs.maskLake, 0, for_loop(0, 51, loop_body_basal_transp_coeff, basal_transp_coeff)),
    )

    mask = npx.isin(vs.lu_id, npx.arange(0, 51, 1, dtype=int))
    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], basal_transp_coeff[2:-2, 2:-2], vs.basal_transp_coeff[2:-2, 2:-2]),
    )

    # land use dependent evaporation coeffcient
    basal_evap_coeff = allocate(state.dimensions, ("x", "y"))

    def loop_body_basal_evap_coeff(i, basal_evap_coeff):
        mask = (vs.lu_id == i)
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        basal_evap_coeff = update_add(
            basal_evap_coeff,
            at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], 1 - ((vs.lut_gc[row_no, vs.month] / vs.lut_gcm[row_no, 1]) * vs.lut_gcm[row_no, 1]), 0),
        )

        return basal_evap_coeff

    basal_evap_coeff = for_loop(0, 51, loop_body_basal_evap_coeff, basal_evap_coeff)

    basal_evap_coeff = update(
        basal_evap_coeff,
        at[:, :], where(vs.maskRiver | vs.maskLake, 1, basal_evap_coeff),
    )

    mask = npx.isin(vs.lu_id, npx.arange(0, 51, 1, dtype=int))
    vs.basal_evap_coeff = update(
        vs.basal_evap_coeff,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], basal_evap_coeff[2:-2, 2:-2], vs.basal_evap_coeff[2:-2, 2:-2]),
    )

    return KernelOutput(
        S_int_top_tot=vs.S_int_top_tot,
        S_int_ground_tot=vs.S_int_ground_tot,
        ground_cover=vs.ground_cover,
        basal_transp_coeff=vs.basal_transp_coeff,
        basal_evap_coeff=vs.basal_evap_coeff
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
    vs.prec_event_sum = update(
        vs.prec_event_sum,
        at[2:-2, 2:-2, vs.taum1], vs.prec_event_sum[2:-2, 2:-2, vs.tau],
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
        prec_event_sum=vs.prec_event_sum,
        theta_rz=vs.theta_rz,
        theta_ss=vs.theta_ss,
        theta=vs.theta,
        h_rz=vs.h_rz,
        h_ss=vs.h_ss,
        h=vs.h,
        z0=vs.z0,
        k_rz=vs.k_rz,
        k_ss=vs.k_ss,
        k=vs.k,
    )


meteo_stations = ["breitnau", "ihringen"]

for meteo_station in meteo_stations:
    model = DISTSetup()
    path_meteo_station = model._base_path / "input" / meteo_station
    model._set_input_dir(path_meteo_station)
    write_meteo_csv_from_dwd(path_meteo_station)
    write_forcing(path_meteo_station)
    model.setup()
    model.run()

    # merge model output into single file
    path = str(model._base_path / "SVAT.*.nc")
    diag_files = glob.glob(path)
    states_hm_file = model._base_path / "states_hm.nc"
    with h5netcdf.File(states_hm_file, 'a', decode_vlen_strings=False) as ff:
        f = ff.create_group(meteo_station)
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title=f'RoGeR model results for parameter grid and input from DWD station {meteo_station}',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment='1D model with free drainage'
        )
        for dfs in diag_files:
            with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                if not f.dimensions:
                    dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time'])}
                    f.dimensions = dict_dim
                    v = f.create_variable('x', ('x',), float)
                    v.attrs['long_name'] = 'Zonal coordinate'
                    v.attrs['units'] = 'meters'
                    v[:] = onp.arange(dict_dim["x"])
                    v = f.create_variable('y', ('y',), float)
                    v.attrs['long_name'] = 'Meridonial coordinate'
                    v.attrs['units'] = 'meters'
                    v[:] = onp.arange(dict_dim["y"])
                    v = f.create_variable('Time', ('Time',), float)
                    var_obj = df.variables.get('Time')
                    v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                    units=var_obj.attrs["units"])
                    v[:] = onp.array(var_obj)
                for key in list(df.variables.keys()):
                    var_obj = df.variables.get(key)
                    if key not in list(f.dimensions.keys()) and var_obj.ndim == 3:
                        v = f.create_variable(key, ('x', 'y', 'Time'), float)
                        vals = onp.array(var_obj)
                        v[:, :, :] = vals.swapaxes(0, 2)
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                        units=var_obj.attrs["units"])
