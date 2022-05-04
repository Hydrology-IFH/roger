import shutil
import glob
import os
from pathlib import Path
import datetime
from cftime import num2date
import h5netcdf
import xarray as xr
import pandas as pd
from de import de
from SALib.sample import saltelli
from SALib.analyze import sobol
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import seaborn as sns

from roger import runtime_settings as rs
rs.backend = "numpy"
rs.force_overwrite = True
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at, for_loop, where
from roger.core.utilities import _get_row_no
from roger.tools.setup import write_forcing
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
import roger.lookuptables as lut
import numpy as onp
onp.random.seed(42)


class SVATSetup(RogerSetup):
    """A SVAT model.
    """
    _base_path = Path(__file__).parent
    # sampled parameters with Saltelli's extension of the Sobol' sequence
    _nsamples = 2**10
    _bounds = {
        'num_vars': 6,
        'names': ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks'],
        'bounds': [[1, 400],
                   [1, 1500],
                   [0.05, 0.33],
                   [0.05, 0.33],
                   [0.05, 0.33],
                   [0.1, 120]]
    }
    _params = saltelli.sample(_bounds, _nsamples, calc_second_order=False)
    _nrows = _params.shape[0]

    def _read_var_from_nc(self, var, file):
        nc_file = self._base_path / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables[var]
            return npx.array(var_obj)

    def _get_nittevent(self):
        nc_file = self._base_path / 'forcing.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['nitt_event']
            return onp.int32(onp.array(var_obj)[0])

    def _get_nitt(self):
        nc_file = self._base_path / 'forcing.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['time']
            return len(onp.array(var_obj))

    def _get_runlen(self):
        nc_file = self._base_path / 'forcing.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['time']
            return onp.array(var_obj)[-1] * 60 * 60 + 24 * 60 * 60

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "SVAT"

        settings.nx, settings.ny, settings.nz = self._nrows, 1, 1
        settings.nitt = self._get_nitt()
        settings.nittevent = self._get_nittevent()
        settings.nittevent_p1 = settings.nittevent + 1
        settings.runlen = self._get_runlen()

        # lysimeter surface 3.14 square meter (2m diameter)
        settings.dx = 2
        settings.dy = 2
        settings.dz = 1

        settings.x_origin = 0.0
        settings.y_origin = 0.0

        settings.enable_macropore_lower_boundary_condition = False

    @roger_routine
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        # temporal grid
        vs.DT_SECS = update(vs.DT_SECS, at[:], self._read_var_from_nc("dt", 'forcing.nc'))
        vs.DT = update(vs.DT, at[:], vs.DT_SECS / (60 * 60))
        vs.YEAR = update(vs.YEAR, at[:], self._read_var_from_nc("year", 'forcing.nc'))
        vs.MONTH = update(vs.MONTH, at[:], self._read_var_from_nc("month", 'forcing.nc'))
        vs.DOY = update(vs.DOY, at[:], self._read_var_from_nc("doy", 'forcing.nc'))
        vs.dt_secs = vs.DT_SECS[vs.itt]
        vs.dt = vs.DT[vs.itt]
        vs.year = vs.YEAR[vs.itt]
        vs.month = vs.MONTH[vs.itt]
        vs.doy = vs.DOY[vs.itt]
        vs.t = update(vs.t, at[:], npx.cumsum(vs.DT))
        # spatial grid
        vs.x = update(vs.x, at[2:-2], npx.arange(1, settings.nx + 1) * (settings.dx / 2))
        vs.y = update(vs.y, at[2:-2], npx.arange(1, settings.ny + 1) * (settings.dy / 2))

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
    def set_parameters(self, state):
        vs = state.variables

        if (vs.itt == 0):

            vs.lu_id = update(vs.lu_id, at[:, :], 8)
            vs.sealing = update(vs.sealing, at[:, :], 0)
            vs.S_dep_tot = update(vs.S_dep_tot, at[:, :], 0)
            vs.z_soil = update(vs.z_soil, at[:, :], 2200)
            vs.dmpv = update(vs.dmpv, at[2:-2, :], npx.array(npx.repeat(self._params[:, 0, npx.newaxis], vs.dmpv.shape[1], axis=-1), dtype=int))
            vs.dmpv = update(vs.dmpv, at[:2, :], 50)  # fill ghosts
            vs.dmpv = update(vs.dmpv, at[-2:, :], 50)
            vs.lmpv = update(vs.lmpv, at[2:-2, :], npx.array(npx.repeat(self._params[:, 1, npx.newaxis], vs.lmpv.shape[1], axis=-1), dtype=int))
            vs.lmpv = update(vs.lmpv, at[:2, :], 400)
            vs.lmpv = update(vs.lmpv, at[-2:, :], 400)
            vs.theta_ac = update(vs.theta_ac, at[2:-2, :], npx.repeat(self._params[:, 2, npx.newaxis], vs.theta_ac.shape[1], axis=-1))
            vs.theta_ac = update(vs.theta_ac, at[:2, :], 0.16)
            vs.theta_ac = update(vs.theta_ac, at[-2:, :], 0.16)
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, :], npx.repeat(self._params[:, 3, npx.newaxis], vs.theta_ufc.shape[1], axis=-1))
            vs.theta_ufc = update(vs.theta_ufc, at[:2, :], 0.2)
            vs.theta_ufc = update(vs.theta_ufc, at[-2:, :], 0.2)
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, :], npx.repeat(self._params[:, 4, npx.newaxis], vs.theta_pwp.shape[1], axis=-1))
            vs.theta_pwp = update(vs.theta_pwp, at[:2, :], 0.24)
            vs.theta_pwp = update(vs.theta_pwp, at[-2:, :], 0.24)
            vs.ks = update(vs.ks, at[2:-2, :], npx.repeat(self._params[:, 5, npx.newaxis], vs.ks.shape[1], axis=-1))
            vs.ks = update(vs.ks, at[:2, :], 9)
            vs.ks = update(vs.ks, at[-2:, :], 9)
            vs.kf = update(vs.kf, at[:, :], 2500)

        if (vs.MONTH[vs.itt] != vs.MONTH[vs.itt - 1]) & (vs.itt > 1):
            vs.update(set_parameters_monthly_kernel(state))

    @roger_routine
    def set_initial_conditions(self, state):
        vs = state.variables

        vs.S_int_top = update(vs.S_int_top, at[:, :, :vs.taup1], 0)
        vs.swe_top = update(vs.swe_top, at[:, :, :vs.taup1], 0)
        vs.S_int_ground = update(vs.S_int_ground, at[:, :, :vs.taup1], 0)
        vs.swe_ground = update(vs.swe_ground, at[:, :, :vs.taup1], 0)
        vs.S_dep = update(vs.S_dep, at[:, :, :vs.taup1], 0)
        vs.S_snow = update(vs.S_snow, at[:, :, :vs.taup1], 0)
        vs.swe = update(vs.swe, at[:, :, :vs.taup1], 0)
        vs.theta_rz = update(vs.theta_rz, at[:, :, :vs.taup1], npx.where(0.46 > vs.theta_sat[:, :, npx.newaxis], vs.theta_sat[:, :, npx.newaxis], 0.46))
        vs.theta_ss = update(vs.theta_ss, at[:, :, :vs.taup1], npx.where(0.44 > vs.theta_sat[:, :, npx.newaxis], vs.theta_sat[:, :, npx.newaxis], 0.44))

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables

        if (vs.itt == 0):
            vs.PREC = update(vs.PREC, at[:, :, :], self._read_var_from_nc("PREC", 'forcing.nc'))
            vs.TA = update(vs.TA, at[:, :, :], self._read_var_from_nc("TA", 'forcing.nc'))
            vs.PET = update(vs.PET, at[:, :, :], self._read_var_from_nc("PET", 'forcing.nc'))
            vs.EVENT_ID = update(vs.EVENT_ID, at[:, :, :], self._read_var_from_nc("EVENT_ID", 'forcing.nc'))

        vs.update(set_forcing_kernel(state))

    @roger_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics
        settings = state.settings

        diagnostics["rates"].output_variables = ["prec", "aet", "transp", "evap_soil", "inf_mat_rz", "inf_mp_rz", "inf_sc_rz", "inf_ss", "q_rz", "q_ss", "cpr_rz", "dS_s", "dS"]
        if settings.enable_groundwater_boundary:
            diagnostics["rates"].output_variables += ["cpr_ss"]
        diagnostics["rates"].output_frequency = 24 * 60 * 60
        diagnostics["rates"].sampling_frequency = 1

        diagnostics["collect"].output_variables = ["S_rz", "S_ss",
                                                   "S_pwp_rz", "S_fc_rz",
                                                   "S_sat_rz", "S_pwp_ss",
                                                   "S_fc_ss", "S_sat_ss",
                                                   "theta_rz", "theta_ss", "theta"]
        if settings.enable_crop_phenology:
            diagnostics["collect"].output_variables += ["re_rg", "re_rl", "z_root", "ground_cover"]
        diagnostics["collect"].output_frequency = 24 * 60 * 60
        diagnostics["collect"].sampling_frequency = 1

        diagnostics["averages"].output_variables = ["ta"]
        diagnostics["averages"].output_frequency = 24 * 60 * 60
        diagnostics["averages"].sampling_frequency = 1

        diagnostics["constant"].output_variables = ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks']
        diagnostics["constant"].output_frequency = 0
        diagnostics["constant"].sampling_frequency = 1

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
            at[:, :], npx.where(mask, vs.lut_ilu[row_no, vs.month], 0),
        )

        return S_int_top_tot

    S_int_top_tot = allocate(state.dimensions, ("x", "y"))

    S_int_top_tot = for_loop(10, 17, loop_body_S_int_top_tot, S_int_top_tot)
    mask = npx.isin(vs.lu_id, npx.array([10, 11, 12, 15, 16]))
    vs.S_int_top_tot = update(
        vs.S_int_top_tot,
        at[:, :], npx.where(mask, S_int_top_tot, vs.S_int_top_tot),
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
            at[:, :], npx.where(mask, vs.lut_ilu[row_no, vs.month], 0),
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
            at[:, :], npx.where(mask, 1, 0),
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
        at[:, :], npx.where(mask, S_int_ground_tot, vs.S_int_ground_tot),
    )

    # land use dependent ground cover (canopy cover)
    ground_cover = allocate(state.dimensions, ("x", "y"))

    def loop_body_ground_cover(i, ground_cover):
        mask = (vs.lu_id == i)
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        ground_cover = update_add(
            ground_cover,
            at[:, :], npx.where(mask, vs.lut_gc[row_no, vs.month], 0),
        )

        return ground_cover

    ground_cover = for_loop(0, 51, loop_body_ground_cover, ground_cover)

    mask = npx.isin(vs.lu_id, npx.arange(0, 51, 1, dtype=int))
    vs.ground_cover = update(
        vs.ground_cover,
        at[:, :, vs.tau], npx.where(mask, ground_cover, vs.ground_cover[:, :, vs.tau]),
    )

    # land use dependent transpiration coeffcient
    basal_transp_coeff = allocate(state.dimensions, ("x", "y"))

    def loop_body_basal_transp_coeff(i, basal_transp_coeff):
        mask = (vs.lu_id == i)
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        basal_transp_coeff = update_add(
            basal_transp_coeff,
            at[:, :], npx.where(mask, vs.lut_gc[row_no, vs.month] / vs.lut_gcm[row_no, 1], 0),
        )

        return basal_transp_coeff

    basal_transp_coeff = update(
        basal_transp_coeff,
        at[:, :], where(vs.maskRiver | vs.maskLake, 0, for_loop(0, 51, loop_body_basal_transp_coeff, basal_transp_coeff)),
    )

    mask = npx.isin(vs.lu_id, npx.arange(0, 51, 1, dtype=int))
    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff,
        at[:, :], npx.where(mask, basal_transp_coeff, vs.basal_transp_coeff),
    )

    # land use dependent evaporation coeffcient
    basal_evap_coeff = allocate(state.dimensions, ("x", "y"))

    def loop_body_basal_evap_coeff(i, basal_evap_coeff):
        mask = (vs.lu_id == i)
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        basal_evap_coeff = update_add(
            basal_evap_coeff,
            at[:, :], npx.where(mask, 1 - ((vs.lut_gc[row_no, vs.month] / vs.lut_gcm[row_no, 1]) * vs.lut_gcm[row_no, 1]), 0),
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
        at[:, :], npx.where(mask, basal_evap_coeff, vs.basal_evap_coeff),
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

    vs.prec = update(vs.prec, at[:, :], vs.PREC[:, :, vs.itt])
    vs.ta = update(vs.ta, at[:, :, vs.tau], vs.TA[:, :, vs.itt])
    vs.pet = update(vs.pet, at[:, :], vs.PET[:, :, vs.itt])
    vs.pet_res = update(vs.pet, at[:, :], vs.PET[:, :, vs.itt])

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
        at[:, :, vs.taum1], vs.ta[:, :, vs.tau],
    )
    vs.z_root = update(
        vs.z_root,
        at[:, :, vs.taum1], vs.z_root[:, :, vs.tau],
    )
    vs.ground_cover = update(
        vs.ground_cover,
        at[:, :, vs.taum1], vs.ground_cover[:, :, vs.tau],
    )
    vs.S_sur = update(
        vs.S_sur,
        at[:, :, vs.taum1], vs.S_sur[:, :, vs.tau],
    )
    vs.S_int_top = update(
        vs.S_int_top,
        at[:, :, vs.taum1], vs.S_int_top[:, :, vs.tau],
    )
    vs.S_int_ground = update(
        vs.S_int_ground,
        at[:, :, vs.taum1], vs.S_int_ground[:, :, vs.tau],
    )
    vs.S_dep = update(
        vs.S_dep,
        at[:, :, vs.taum1], vs.S_dep[:, :, vs.tau],
    )
    vs.S_snow = update(
        vs.S_snow,
        at[:, :, vs.taum1], vs.S_snow[:, :, vs.tau],
    )
    vs.swe = update(
        vs.swe,
        at[:, :, vs.taum1], vs.swe[:, :, vs.tau],
    )
    vs.S_rz = update(
        vs.S_rz,
        at[:, :, vs.taum1], vs.S_rz[:, :, vs.tau],
    )
    vs.S_ss = update(
        vs.S_ss,
        at[:, :, vs.taum1], vs.S_ss[:, :, vs.tau],
    )
    vs.S_s = update(
        vs.S_s,
        at[:, :, vs.taum1], vs.S_s[:, :, vs.tau],
    )
    vs.S = update(
        vs.S,
        at[:, :, vs.taum1], vs.S[:, :, vs.tau],
    )
    vs.z_sat = update(
        vs.z_sat,
        at[:, :, vs.taum1], vs.z_sat[:, :, vs.tau],
    )
    vs.z_wf = update(
        vs.z_wf,
        at[:, :, vs.taum1], vs.z_wf[:, :, vs.tau],
    )
    vs.z_wf_t0 = update(
        vs.z_wf_t0,
        at[:, :, vs.taum1], vs.z_wf_t0[:, :, vs.tau],
    )
    vs.z_wf_t1 = update(
        vs.z_wf_t1,
        at[:, :, vs.taum1], vs.z_wf_t1[:, :, vs.tau],
    )
    vs.y_mp = update(
        vs.y_mp,
        at[:, :, vs.taum1], vs.y_mp[:, :, vs.tau],
    )
    vs.y_sc = update(
        vs.y_sc,
        at[:, :, vs.taum1], vs.y_sc[:, :, vs.tau],
    )
    vs.prec_event_sum = update(
        vs.prec_event_sum,
        at[:, :, vs.taum1], vs.prec_event_sum[:, :, vs.tau],
    )
    vs.t_event_sum = update(
        vs.t_event_sum,
        at[:, :, vs.taum1], vs.t_event_sum[:, :, vs.tau],
    )
    vs.theta_rz = update(
        vs.theta_rz,
        at[:, :, vs.taum1], vs.theta_rz[:, :, vs.tau],
    )
    vs.theta_ss = update(
        vs.theta_ss,
        at[:, :, vs.taum1], vs.theta_ss[:, :, vs.tau],
    )
    vs.theta = update(
        vs.theta,
        at[:, :, vs.taum1], vs.theta[:, :, vs.tau],
    )
    vs.k_rz = update(
        vs.k_rz,
        at[:, :, vs.taum1], vs.k_rz[:, :, vs.tau],
    )
    vs.k_ss = update(
        vs.k_ss,
        at[:, :, vs.taum1], vs.k_ss[:, :, vs.tau],
    )
    vs.k = update(
        vs.k,
        at[:, :, vs.taum1], vs.k[:, :, vs.tau],
    )
    vs.h_rz = update(
        vs.h_rz,
        at[:, :, vs.taum1], vs.h_rz[:, :, vs.tau],
    )
    vs.h_ss = update(
        vs.h_ss,
        at[:, :, vs.taum1], vs.h_ss[:, :, vs.tau],
    )
    vs.h = update(
        vs.h,
        at[:, :, vs.taum1], vs.h[:, :, vs.tau],
    )
    vs.z0 = update(
        vs.z0,
        at[:, :, vs.taum1], vs.z0[:, :, vs.tau],
    )
    # set to 0 for numerical errors
    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[:, :], npx.where((vs.S_fp_rz > -1e-6) & (vs.S_fp_rz < 0), 0, vs.S_fp_rz),
    )
    vs.S_lp_rz = update(
        vs.S_lp_rz,
        at[:, :], npx.where((vs.S_lp_rz > -1e-6) & (vs.S_lp_rz < 0), 0, vs.S_lp_rz),
    )
    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[:, :], npx.where((vs.S_fp_ss > -1e-6) & (vs.S_fp_ss < 0), 0, vs.S_fp_ss),
    )
    vs.S_lp_ss = update(
        vs.S_lp_ss,
        at[:, :], npx.where((vs.S_lp_ss > -1e-6) & (vs.S_lp_ss < 0), 0, vs.S_lp_ss),
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
        k_rz=vs.k_rz,
        k_ss=vs.k_ss,
        k=vs.k,
        z0=vs.z0,
        S_fp_rz=vs.S_fp_rz,
        S_lp_rz=vs.S_lp_rz,
        S_fp_ss=vs.S_fp_ss,
        S_lp_ss=vs.S_lp_ss,
    )


model = SVATSetup()
input_path = model._base_path / "input"
write_forcing(input_path)
model.setup()
model.run()

# directory of results
base_path_results = model._base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)

# merge model output into single file
path = str(model._base_path / f"{model.state.settings.identifier}.*.nc")
diag_files = glob.glob(path)
states_hm_si_file = model._base_path / "states_hm_sensitivity.nc"
with h5netcdf.File(states_hm_si_file, 'w', decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title='RoGeR saltelli results at Rietholzbach Lysimeter site',
        institution='University of Freiburg, Chair of Hydrology',
        references='',
        comment='SVAT model with free drainage'
    )
    for dfs in diag_files:
        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
            # set dimensions with a dictionary
            if not f.dimensions:
                f.dimensions = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time'])}
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Number of model run'
                v.attrs['units'] = ''
                v[:] = npx.arange(f.dimensions["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = ''
                v.attrs['units'] = ''
                v[:] = npx.arange(f.dimensions["y"])
                v = f.create_variable('Time', ('Time',), float)
                var_obj = df.variables.get('Time')
                with h5netcdf.File(model._base_path / 'forcing.nc', "r", decode_vlen_strings=False) as infile:
                    time_origin = infile.variables['time'].attrs['time_origin']
                v.attrs.update(time_origin=time_origin,
                               units=var_obj.attrs["units"])
                v[:] = npx.array(var_obj)
            for var_sim in list(df.variables.keys()):
                var_obj = df.variables.get(var_sim)
                if var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] > 1:
                    v = f.create_variable(var_sim, ('x', 'y', 'Time'), float)
                    vals = npx.array(var_obj)
                    v[:, :, :] = vals.swapaxes(0, 2)
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])
                elif var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3 and var_obj.shape[0] == 1:
                    v = f.create_variable(var_sim, ('x', 'y'), float)
                    vals = npx.array(var_obj)
                    v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])

# move hydrologic states to directories of transport model
base_path_tm = model._base_path.parent / "svat_transport_sensitivity_reverse"
states_hm_si_file1 = base_path_tm / "states_hm_sensitivity.nc"
shutil.copy(states_hm_si_file, states_hm_si_file1)

# load simulation
ds_sim = xr.open_dataset(states_hm_si_file, engine="h5netcdf")

# load observations (measured data)
path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/rietholzbach/rietholzbach_lysimeter.nc")
ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")

# assign date
days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_sim = num2date(days_sim, units=f"days since {ds_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_sim = ds_sim.assign_coords(date=("Time", date_sim))
ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

# DataFrame with sampled model parameters and the corresponding metrics
nx = model.state.settings.nx  # number of rows
ny = model.state.settings.ny  # number of columns
df_params_eff = pd.DataFrame(index=range(nx * ny))
# sampled model parameters
df_params_eff.loc[:, 'dmpv'] = ds_sim["dmpv"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'lmpv'] = ds_sim["lmpv"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'theta_ac'] = ds_sim["theta_ac"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'theta_ufc'] = ds_sim["theta_ufc"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'theta_pwp'] = ds_sim["theta_pwp"].isel(y=0).values.flatten()
df_params_eff.loc[:, 'ks'] = ds_sim["ks"].isel(y=0).values.flatten()
# calculate metrics
vars_sim = ['aet', 'q_ss', 'theta', 'dS_s', 'dS']
vars_obs = ['AET', 'PERC', 'THETA', 'dWEIGHT', 'dWEIGHT']
for var_sim, var_obs in zip(vars_sim, vars_obs):
    if var_sim == 'theta':
        obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values, axis=0)
    elif var_sim == 'theta_rz':
        obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values[:5, :], axis=0)
    elif var_sim == 'theta_ss':
        obs_vals = onp.mean(ds_obs['THETA'].isel(x=0, y=0).values[5:, :], axis=0)
    else:
        obs_vals = ds_obs[var_obs].isel(x=0, y=0).values
    df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
    df_obs.loc[:, 'obs'] = obs_vals
    for nrow in range(nx * ny):
        sim_vals = ds_sim[var_sim].isel(x=nrow, y=0).values
        # join observations on simulations
        df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)
        df_eval = df_eval.dropna()

        if var_sim in ['theta_rz', 'theta_ss', 'theta']:
            Ni = len(df_eval.index)
            obs_vals = df_eval.loc[:, 'obs'].values
            sim_vals = df_eval.loc[:, 'sim'].values
            Nz = len(obs_vals)
            eff_swc = eval_utils.calc_kge(obs_vals, sim_vals)
            key_kge = 'KGE_' + var_sim
            df_params_eff.loc[nrow, key_kge] = (Nz / Ni) * eff_swc
        elif var_sim in ['dS', 'dS_s']:
            obs_vals = df_eval.loc[:, 'obs'].values
            sim_vals = df_eval.loc[:, 'sim'].values
            key_r = 'r_' + var_sim
            df_params_eff.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
        else:
            obs_vals = df_eval.loc[:, 'obs'].values
            sim_vals = df_eval.loc[:, 'sim'].values
            key_kge = 'KGE_' + var_sim
            df_params_eff.loc[nrow, key_kge] = eval_utils.calc_kge(obs_vals, sim_vals)
            key_kge_alpha = 'KGE_alpha_' + var_sim
            df_params_eff.loc[nrow, key_kge_alpha] = eval_utils.calc_kge_alpha(obs_vals, sim_vals)
            key_kge_beta = 'KGE_beta_' + var_sim
            df_params_eff.loc[nrow, key_kge_beta] = eval_utils.calc_kge_beta(obs_vals, sim_vals)
            key_r = 'r_' + var_sim
            df_params_eff.loc[nrow, key_r] = eval_utils.calc_temp_cor(obs_vals, sim_vals)
            cond0 = (df_eval['obs'] == 0)
            if cond0.any():
                # number of data points
                N_obs = len(df_eval.index)
                # simulations and observations for which observed
                # values are exclusively zero
                df_obs0_sim = df_eval.loc[cond0, :]
                N_obs0 = (df_obs0_sim['obs'] == 0).sum()
                N_sim0 = (df_obs0_sim['sim'] == 0).sum()
                # share of observations with zero values
                key_p0 = 'p0_' + var_sim
                df_params_eff.loc[nrow, key_p0] = N_obs0 / N_obs
                # agreement of zero values
                N_obs0 = (df_obs0_sim['obs'] == 0).sum()
                N_sim0 = (df_obs0_sim['sim'] == 0).sum()
                ioa0 = 1 - (N_sim0 / N_obs0)
                key_ioa0 = 'ioa0_' + var_sim
                df_params_eff.loc[nrow, key_ioa0] = ioa0
                # mean absolute error from observations with zero values
                obs0_vals = df_obs0_sim.loc[:, 'obs'].values
                sim0_vals = df_obs0_sim.loc[:, 'sim'].values
                key_mae0 = 'MAE0_' + var_sim
                df_params_eff.loc[nrow, key_mae0] = eval_utils.calc_mae(obs0_vals,
                                                                   sim0_vals)
                # peak difference from observations with zero values
                key_pdiff0 = 'PDIFF0_' + var_sim
                df_params_eff.loc[nrow, key_pdiff0] = onp.max(sim0_vals)
                # simulations and observations with non-zero values
                cond_no0 = (df_eval['obs'] > 0)
                df_obs_sim_no0 = df_eval.loc[cond_no0, :]
                obs_vals_no0 = df_obs_sim_no0.loc[:, 'obs'].values
                sim_vals_no0 = df_obs_sim_no0.loc[:, 'sim'].values
                # number of data with non-zero observations
                N_no0 = len(df_obs_sim_no0.index)
                # mean absolute relative error
                key_mare = 'MARE_' + var_sim
                df_params_eff.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals_no0, sim_vals_no0)
                # mean relative bias
                key_brel_mean = 'brel_mean_' + var_sim
                brel_mean = de.calc_brel_mean(obs_vals_no0, sim_vals_no0)
                df_params_eff.loc[nrow, key_brel_mean] = brel_mean
                # residual relative bias
                brel_res = de.calc_brel_res(obs_vals_no0, sim_vals_no0)
                # area of relative residual bias
                key_b_area = 'b_area_' + var_sim
                b_area = de.calc_bias_area(brel_res)
                df_params_eff.loc[nrow, key_b_area] = b_area
                # temporal correlation
                key_temp_cor = 'temp_cor_' + var_sim
                temp_cor = de.calc_temp_cor(obs_vals_no0, sim_vals_no0)
                df_params_eff.loc[nrow, key_temp_cor] = temp_cor
                # diagnostic efficiency
                key_de = 'DE_' + var_sim
                df_params_eff.loc[nrow, key_de] = de.calc_de(obs_vals_no0, sim_vals_no0)
                # relative bias
                brel = de.calc_brel(obs_vals, sim_vals)
                # total bias
                key_b_tot = 'b_tot_' + var_sim
                b_tot = de.calc_bias_tot(brel)
                df_params_eff.loc[nrow, key_b_tot] = b_tot
                # bias of lower exceedance probability
                key_b_hf = 'b_hf_' + var_sim
                b_hf = de.calc_bias_hf(brel)
                df_params_eff.loc[nrow, key_b_hf] = b_hf
                # error contribution of higher exceedance probability
                key_err_hf = 'err_hf_' + var_sim
                err_hf = de.calc_err_hf(b_hf, b_tot)
                df_params_eff.loc[nrow, key_err_hf] = err_hf
                # bias of higher exceedance probability
                key_b_lf = 'b_lf_' + var_sim
                b_lf = de.calc_bias_lf(brel)
                df_params_eff.loc[nrow, key_b_lf] = b_lf
                # error contribution of lower exceedance probability
                key_err_lf = 'err_lf_' + var_sim
                err_lf = de.calc_err_hf(b_lf, b_tot)
                df_params_eff.loc[nrow, key_err_lf] = err_lf
                # direction of bias
                key_b_dir = 'b_dir_' + var_sim
                b_dir = de.calc_bias_dir(brel_res)
                df_params_eff.loc[nrow, key_b_dir] = b_dir
                # slope of bias
                key_b_slope = 'b_slope_' + var_sim
                b_slope = de.calc_bias_slope(b_area, b_dir)
                df_params_eff.loc[nrow, key_b_slope] = b_slope
                # (y, x) trigonometric inverse tangent
                key_phi = 'phi_' + var_sim
                df_params_eff.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)
                # combined diagnostic efficiency
                key_de0 = 'DE0_' + var_sim
                df_params_eff.loc[nrow, key_de0] = (N_no0 / N_obs) * df_params_eff.loc[nrow, key_de] + (N_obs0 / N_obs) * ioa0
            else:
                # share of observations with zero values
                key_p0 = 'p0_' + var_sim
                df_params_eff.loc[nrow, key_p0] = 0
                # mean absolute relative error
                key_mare = 'MARE_' + var_sim
                df_params_eff.loc[nrow, key_mare] = eval_utils.calc_mare(obs_vals, sim_vals)
                # mean relative bias
                key_brel_mean = 'brel_mean_' + var_sim
                brel_mean = de.calc_brel_mean(obs_vals, sim_vals)
                df_params_eff.loc[nrow, key_brel_mean] = brel_mean
                # residual relative bias
                brel_res = de.calc_brel_res(obs_vals, sim_vals)
                # area of relative residual bias
                key_b_area = 'b_area_' + var_sim
                b_area = de.calc_bias_area(brel_res)
                df_params_eff.loc[nrow, key_b_area] = b_area
                # temporal correlation
                key_temp_cor = 'temp_cor_' + var_sim
                temp_cor = de.calc_temp_cor(obs_vals, sim_vals)
                df_params_eff.loc[nrow, key_temp_cor] = temp_cor
                # diagnostic efficiency
                key_de = 'DE_' + var_sim
                df_params_eff.loc[nrow, key_de] = de.calc_de(obs_vals, sim_vals)
                # relative bias
                brel = de.calc_brel(obs_vals, sim_vals)
                # total bias
                key_b_tot = 'b_tot_' + var_sim
                b_tot = de.calc_bias_tot(brel)
                df_params_eff.loc[nrow, key_b_tot] = b_tot
                # bias of lower exceedance probability
                key_b_hf = 'b_hf_' + var_sim
                b_hf = de.calc_bias_hf(brel)
                df_params_eff.loc[nrow, key_b_hf] = b_hf
                # error contribution of higher exceedance probability
                key_err_hf = 'err_hf_' + var_sim
                err_hf = de.calc_err_hf(b_hf, b_tot)
                df_params_eff.loc[nrow, key_err_hf] = err_hf
                # bias of higher exceedance probability
                key_b_lf = 'b_lf_' + var_sim
                b_lf = de.calc_bias_lf(brel)
                df_params_eff.loc[nrow, key_b_lf] = b_lf
                # error contribution of lower exceedance probability
                key_err_lf = 'err_lf_' + var_sim
                err_lf = de.calc_err_hf(b_lf, b_tot)
                df_params_eff.loc[nrow, key_err_lf] = err_lf
                # direction of bias
                key_b_dir = 'b_dir_' + var_sim
                b_dir = de.calc_bias_dir(brel_res)
                df_params_eff.loc[nrow, key_b_dir] = b_dir
                # slope of bias
                key_b_slope = 'b_slope_' + var_sim
                b_slope = de.calc_bias_slope(b_area, b_dir)
                df_params_eff.loc[nrow, key_b_slope] = b_slope
                # (y, x) trigonometric inverse tangent
                key_phi = 'phi_' + var_sim
                df_params_eff.loc[nrow, key_phi] = de.calc_phi(brel_mean, b_slope)

# Calculate multi-objective metric
df_params_eff.loc[:, 'E_multi'] = 1/3 * df_params_eff.loc[:, 'r_dS'] + 1/3 * df_params_eff.loc[:, 'KGE_aet'] + 1/3 * df_params_eff.loc[:, 'KGE_q_ss']

# write .txt-file
file = base_path_results / "params_eff.txt"
df_params_eff.to_csv(file, header=True, index=False, sep="\t")

# perform sensitivity analysis
df_params = df_params_eff.loc[:, model._bounds['names']]
df_eff = df_params_eff.loc[:, ['KGE_aet', 'KGE_q_ss', 'r_dS', 'E_multi']]
dict_si = {}
for name in df_eff.columns:
    Y = df_eff[name].values
    Si = sobol.analyze(model._bounds, Y)
    Si_filter = {k: Si[k] for k in ['ST', 'ST_conf', 'S1', 'S1_conf']}
    dict_si[name] = pd.DataFrame(Si_filter, index=model._bounds['names'])

base_path_figs = model._base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# plot sobol indices
_LABS = {'KGE_aet': 'evapotranspiration',
         'KGE_q_ss': 'percolation',
         'r_dS': 'storage change',
         'E_multi': 'multi-objective metric',
         }
ncol = len(df_eff.columns)
xaxis_labels = [labs._LABS[k].split(' ')[0] for k in model._bounds['names']]
cmap = cm.get_cmap('Greys')
norm = Normalize(vmin=0, vmax=2)
colors = cmap(norm([0.5, 1.5]))
fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(14, 5))
for i, name in enumerate(df_eff.columns):
    indices = dict_si[name][['S1', 'ST']]
    err = dict_si[name][['S1_conf', 'ST_conf']]
    indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors)
    ax[i].set_xticklabels(xaxis_labels)
    ax[i].set_title(_LABS[name])
    ax[i].legend(["First-order", "Total"], frameon=False)
ax[-1].legend().set_visible(False)
ax[-2].legend().set_visible(False)
ax[-3].legend().set_visible(False)
ax[0].set_ylabel('Sobol index [-]')
fig.tight_layout()
file = base_path_figs / "sobol_indices.png"
fig.savefig(file, dpi=250)

# make dotty plots
nrow = len(df_eff.columns)
ncol = model._bounds['num_vars']
fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(14, 7))
for i in range(nrow):
    for j in range(ncol):
        y = df_eff.iloc[:, i]
        x = df_params.iloc[:, j]
        sns.regplot(x=x, y=y, ax=ax[i, j], ci=None, color='k',
                    scatter_kws={'alpha': 0.2, 's': 4, 'color': 'grey'})
        ax[i, j].set_xlabel('')
        ax[i, j].set_ylabel('')

for j in range(ncol):
    xlabel = labs._AXS_LABS[model._bounds['names'][j]]
    ax[-1, j].set_xlabel(xlabel)

ax[0, 0].set_ylabel('$KGE_{ET}$ [-]')
ax[1, 0].set_ylabel('$KGE_{PERC}$ [-]')
ax[2, 0].set_ylabel(r'$r_{\Delta S}$ [-]')
ax[3, 0].set_ylabel('$E_{multi}$\n [-]')

fig.subplots_adjust(wspace=0.2, hspace=0.3)
file = base_path_figs / "dotty_plots.png"
fig.savefig(file, dpi=250)
