import shutil
import glob
from pathlib import Path
import datetime
from cftime import num2date
import h5netcdf
import xarray as xr
import pandas as pd

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
from roger.io_tools import yml
import numpy as onp


class SVATSetup(RogerSetup):
    """A SVAT model.
    """
    _base_path = Path(__file__).parent

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

    def _read_config(self):
        config_file = self._base_path / "config.yml"
        config = yml.Config(config_file)

        return config

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "SVAT"

        settings.nx, settings.ny, settings.nz = 1, 1, 1
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
            vs.dmpv = update(vs.dmpv, at[:, :], 50)
            vs.lmpv = update(vs.lmpv, at[:, :], 1300)
            vs.theta_ac = update(vs.theta_ac, at[:, :], 0.16)
            vs.theta_ufc = update(vs.theta_ufc, at[:, :], 0.2)
            vs.theta_pwp = update(vs.theta_pwp, at[:, :], 0.24)
            vs.ks = update(vs.ks, at[:, :], 9)
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
        vs.theta_rz = update(vs.theta_rz, at[:, :, :vs.taup1], 0.46)
        vs.theta_ss = update(vs.theta_ss, at[:, :, :vs.taup1], 0.44)

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

# merge model output into single file
path = str(model._base_path / f"{model.state.settings.identifier}.*.nc")
diag_files = glob.glob(path)
states_hm_file = model._base_path / "states_hm.nc"
with h5netcdf.File(states_hm_file, 'w', decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title='RoGeR model results at Rietholzbach Lysimeter site',
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
            for key in list(df.variables.keys()):
                var_obj = df.variables.get(key)
                if key not in list(f.dimensions.keys()) and var_obj.ndim == 3:
                    v = f.create_variable(key, ('x', 'y', 'Time'), float)
                    vals = npx.array(var_obj)
                    v[:, :, :] = vals.swapaxes(0, 2)
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])
