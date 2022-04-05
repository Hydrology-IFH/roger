import shutil
from pathlib import Path
import glob
import datetime
from cftime import num2date
import h5netcdf
import xarray as xr
import matplotlib.pyplot as plt

import roger
roger.runtime_settings.backend = 'numpy'
roger.runtime_settings.force_overwrite = 'True'
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at, for_loop, where
from roger.core.utilities import _get_row_no
import roger.lookuptables as lut
from roger.setups.make_dummy_setup import make_setup
from roger.io_tools import yml
import numpy as onp

# generate dummy setup
BASE_PATH = Path(__file__).parent
make_setup(BASE_PATH, event_type='heavyrain', ndays=10,
           enable_groundwater_boundary=False,
           enable_film_flow=True,
           enable_crop_phenology=False,
           enable_crop_rotation=False,
           enable_lateral_flow=False,
           enable_groundwater=False,
           enable_offline_transport=False,
           enable_bromide=False,
           enable_chloride=False,
           enable_deuterium=False,
           enable_oxygen18=False,
           enable_nitrate=False,
           tm_structure=None,
           z_soil=1350,
           a=0.19,
           rain_sum_ff=100,
           max_dur=72,
           z_soil_max=5000)

# read config file
CONFIG_FILE = BASE_PATH / "config.yml"
config = yml.Config(CONFIG_FILE)


class SVATFILMSetup(RogerSetup):
    """A SVAT model including film flow.
    """
    def _read_var_from_nc(self, var, file):
        nc_file = BASE_PATH / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables[var]
            return npx.array(var_obj)

    def _get_nittevent(self):
        nc_file = BASE_PATH / 'forcing.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['nitt_event']
            return onp.int32(onp.array(var_obj)[0])

    def _get_nevent_ff(self):
        nc_file = BASE_PATH / 'forcing.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['nevent_ff']
            return onp.int32(onp.array(var_obj)[0])

    def _get_nitt(self):
        nc_file = BASE_PATH / 'forcing.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['time']
            return len(onp.array(var_obj))

    def _get_runlen(self):
        nc_file = BASE_PATH / 'forcing.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['time']
            return onp.array(var_obj)[-1] * 60 * 60 + 24 * 60 * 60

    def _get_ncr(self):
        nc_file = BASE_PATH / 'crop_rotation.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['year_season']
            return len(onp.array(var_obj))

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "SVATFILM"

        settings.nx, settings.ny, settings.nz = config.nrows, config.ncols, 1
        settings.nitt = self._get_nitt()
        settings.nittevent = self._get_nittevent()
        settings.nittevent_p1 = settings.nittevent + 1
        settings.runlen = self._get_runlen()
        settings.nittevent_ff = 5 * 24 * 6
        settings.nittevent_ff_p1 = settings.nittevent_ff + 1
        settings.nevent_ff = self._get_nevent_ff()

        settings.dx = config.cell_width
        settings.dy = config.cell_width
        settings.dz = 1

        settings.x_origin = 0.0
        settings.y_origin = 0.0

        settings.enable_groundwater_boundary = config.enable_groundwater_boundary
        settings.enable_film_flow = config.enable_film_flow
        settings.enable_crop_phenology = config.enable_crop_phenology
        settings.enable_crop_rotation = config.enable_crop_rotation
        settings.enable_lateral_flow = config.enable_lateral_flow
        settings.enable_groundwater = config.enable_groundwater
        settings.enable_offline_transport = config.enable_offline_transport
        settings.enable_bromide = config.enable_bromide
        settings.enable_chloride = config.enable_chloride
        settings.enable_deuterium = config.enable_deuterium
        settings.enable_oxygen18 = config.enable_oxygen18
        settings.enable_nitrate = config.enable_nitrate
        settings.enable_macropore_lower_boundary_condition = False

        settings.ff_tc = 0.15

        if settings.enable_crop_rotation:
            settings.ncrops = 3
            settings.ncr = self._get_ncr()

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
        settings = state.settings

        vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
        vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
        vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
        vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
        vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)
        if settings.enable_crop_phenology:
            vs.lut_crops = update(vs.lut_crops, at[:, :], lut.ARR_CP)

    @roger_routine
    def set_topography(self, state):
        pass

    @roger_routine
    def set_parameters(self, state):
        vs = state.variables
        settings = state.settings

        if (vs.itt == 0):

            vs.lu_id = update(vs.lu_id, at[:, :], 8)
            vs.sealing = update(vs.sealing, at[:, :], 0)
            vs.slope = update(vs.slope, at[:, :], 0)
            vs.slope_per = update(vs.slope_per, at[:, :], vs.slope * 100)
            vs.S_dep_tot = update(vs.S_dep_tot, at[:, :], 0)
            vs.z_soil = update(vs.z_soil, at[:, :], 1350)
            vs.dmpv = update(vs.dmpv, at[:, :], 100)
            vs.lmpv = update(vs.lmpv, at[:, :], 1000)
            vs.theta_ac = update(vs.theta_ac, at[:, :], 0.13)
            vs.theta_ufc = update(vs.theta_ufc, at[:, :], 0.24)
            vs.theta_pwp = update(vs.theta_pwp, at[:, :], 0.23)
            vs.ks = update(vs.ks, at[:, :], 25)
            vs.kf = update(vs.kf, at[:, :], 2500)
            vs.a_ff = update(vs.a_ff, at[:, :], 0.19)
            vs.c_ff = update(vs.c_ff, at[:, :], 0.001)

            if settings.enable_crop_phenology and settings.enable_crop_rotation:
                vs.CROP_TYPE = update(vs.CROP_TYPE, at[:, :, :], self._read_var_from_nc("crop", 'crop_rotation.nc'))
                vs.crop_type = update(vs.crop_type, at[:, :, 0], vs.CROP_TYPE[:, :, 1])
                vs.crop_type = update(vs.crop_type, at[:, :, 1], vs.CROP_TYPE[:, :, 2])
                vs.crop_type = update(vs.crop_type, at[:, :, 2], vs.CROP_TYPE[:, :, 3])

            if settings.enable_crop_phenology and not settings.enable_crop_rotation:
                mask = npx.isin(vs.lu_id[:, :, npx.newaxis], npx.arange(500, 600, 1, dtype=int))
                vs.crop_type = update(vs.crop_type, at[:, :, 0], npx.where(mask, vs.lu_id, 598))

        if (vs.MONTH[vs.itt] != vs.MONTH[vs.itt - 1]) & (vs.itt > 1):
            vs.update(set_parameters_monthly_kernel(state))

    @roger_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        vs.S_int_top = update(vs.S_int_top, at[:, :, :vs.taup1], 0)
        vs.swe_top = update(vs.swe_top, at[:, :, :vs.taup1], 0)
        vs.S_int_ground = update(vs.S_int_ground, at[:, :, :vs.taup1], 0)
        vs.swe_ground = update(vs.swe_ground, at[:, :, :vs.taup1], 0)
        vs.S_dep = update(vs.S_dep, at[:, :, :vs.taup1], 0)
        vs.S_snow = update(vs.S_snow, at[:, :, :vs.taup1], 0)
        vs.swe = update(vs.swe, at[:, :, :vs.taup1], 0)
        vs.theta_rz = update(vs.theta_rz, at[:, :, :vs.taup1], 0.4)
        vs.theta_ss = update(vs.theta_ss, at[:, :, :vs.taup1], 0.47)

        if settings.enable_crop_phenology:
            vs.z_root = update(vs.z_root, at[:, :, :vs.taup1], 0)
            vs.z_root_crop = update(vs.z_root_crop, at[:, :, :vs.taup1, 0], 0)
            vs.update(set_initial_conditions_crops_kernel(state))

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables
        settings = state.settings

        if (vs.itt == 0):
            vs.PREC = update(vs.PREC, at[:, :, :], self._read_var_from_nc("PREC", 'forcing.nc'))
            vs.TA = update(vs.TA, at[:, :, :], self._read_var_from_nc("TA", 'forcing.nc'))
            vs.PET = update(vs.PET, at[:, :, :], self._read_var_from_nc("PET", 'forcing.nc'))
            vs.EVENT_ID = update(vs.EVENT_ID, at[:, :, :], self._read_var_from_nc("EVENT_ID", 'forcing.nc'))
            vs.EVENT_ID_FF = update(vs.EVENT_ID_FF, at[:, :, :], self._read_var_from_nc("EVENT_ID_FF", 'forcing.nc'))
            if settings.enable_crop_phenology:
                vs.TA_MIN = update(vs.TA_MIN, at[:, :, :], self._read_var_from_nc("TA_min", 'forcing.nc'))
                vs.TA_MAX = update(vs.TA_MAX, at[:, :, :], self._read_var_from_nc("TA_max", 'forcing.nc'))

        vs.itt_event_ff = update(
            vs.itt_event_ff,
            at[:], vs.itt - vs.event_start_ff,
        )
        arr_itt = allocate(state.dimensions, ("x", "y"))
        arr_itt = update(
            arr_itt,
            at[:, :], vs.itt,
        )
        cond1 = ((vs.EVENT_ID_FF[:, :, vs.itt-1] == 0) & (vs.EVENT_ID_FF[:, :, vs.itt] >= 1) & (arr_itt >= 1))
        if cond1.any():
            vs.event_no_ff = vs.event_no_ff + 1
            # number of event
            vs.event_id_ff = npx.max(vs.EVENT_ID_FF[:, :, vs.itt])
            # iteration at event start
            vs.event_start_ff = update(vs.event_start_ff, at[vs.event_no_ff - 1], vs.itt)
            # iteration at event end
            vs.event_end_ff = update(vs.event_end_ff, at[vs.event_no_ff - 1], vs.itt + settings.nittevent_ff)
            if (vs.event_end_ff[vs.event_no_ff - 1] >= settings.nitt):
                vs.event_end_ff = update(vs.event_end_ff, at[vs.event_no_ff - 1], settings.nitt)
            vs.rain_event = update(
                vs.rain_event,
                at[:, :, :], 0,
            )
            vs.rain_event = update(
                vs.rain_event,
                at[:, :, 0:vs.event_end_ff[vs.event_no_ff - 1]-vs.event_start_ff[vs.event_no_ff - 1]], npx.where(vs.EVENT_ID_FF == vs.event_id_ff, vs.PREC, 0)[:, :, vs.event_start_ff[vs.event_no_ff - 1]:vs.event_end_ff[vs.event_no_ff - 1]],
            )
            vs.rain_event_sum = update(
                vs.rain_event_sum,
                at[:, :], npx.sum(vs.rain_event, axis=-1),
            )
            vs.rain_event_csum = update(
                vs.rain_event_csum,
                at[:, :, :], npx.cumsum(vs.rain_event, axis=-1),
            )
            # subtract interception storage
            vs.rain_event = update(
                vs.rain_event,
                at[:, :, 0], vs.rain_event[:, :, 0] - (vs.S_int_top_tot - vs.S_int_top[:, :, vs.tau]) + (vs.S_int_ground_tot - vs.S_int_ground[:, :, vs.tau]),
            )
            vs.rain_event = update(
                vs.rain_event,
                at[:, :, 1:], npx.diff(npx.where(npx.cumsum(vs.rain_event, axis=-1) < 0, 0, npx.cumsum(vs.rain_event, axis=-1)), axis=-1),
            )
            vs.rain_event = update(
                vs.rain_event,
                at[:, :, 0], npx.where(vs.rain_event[:, :, 0] < 0, 0, vs.rain_event[:, :, 0]),
            )

        vs.update(set_forcing_kernel(state))
        if settings.enable_crop_phenology:
            vs.ta_min = update(vs.ta_min, at[:, :, vs.tau], vs.TA_MIN[:, :, vs.itt])
            vs.ta_max = update(vs.ta_max, at[:, :, vs.tau], vs.TA_MAX[:, :, vs.itt])

    @roger_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics
        settings = state.settings

        diagnostics["rates"].output_variables = ["ff_drain", "ff_abs_rz", "ff_abs_ss"]
        if settings.enable_groundwater_boundary:
            diagnostics["rates"].output_variables += ["cpr_ss"]
        diagnostics["rates"].output_frequency = 10 * 60
        diagnostics["rates"].sampling_frequency = 1

        diagnostics["collect"].output_variables = ["z_wf", "z_pf", "theta_rz", "theta_rz_ff", "theta_ss", "theta_ss_ff"]
        diagnostics["collect"].output_frequency = 10 * 60
        diagnostics["collect"].sampling_frequency = 1

        # diagnostics["rates"].output_variables = ["prec", "transp", "evap_soil", "inf_mat_rz", "inf_mp_rz", "inf_sc_rz", "inf_ss", "q_rz", "q_ss", "cpr_rz"]
        # if settings.enable_groundwater_boundary:
        #     diagnostics["rates"].output_variables += ["cpr_ss"]
        # diagnostics["rates"].output_frequency = 24 * 60 * 60
        # diagnostics["rates"].sampling_frequency = 1

        # diagnostics["collect"].output_variables = ["S_rz", "S_ss",
        #                                            "S_pwp_rz", "S_fc_rz",
        #                                            "S_sat_rz", "S_pwp_ss",
        #                                            "S_fc_ss", "S_sat_ss"]
        # if settings.enable_crop_phenology:
        #     diagnostics["collect"].output_variables += ["re_rg", "re_rl", "z_root", "ground_cover"]
        # diagnostics["collect"].output_frequency = 24 * 60 * 60
        # diagnostics["collect"].sampling_frequency = 1
        #
        # diagnostics["averages"].output_variables = ["ta"]
        # diagnostics["averages"].output_frequency = 24 * 60 * 60
        # diagnostics["averages"].sampling_frequency = 1

        # diagnostics["rates"].output_variables = ["prec", "rain_ground", "snow_ground", "q_sur", "inf_rz", "inf_mat_rz", "inf_mp_rz", "inf_sc_rz", "inf_ss", "q_rz", "q_ss", "transp", "evap_soil", "cpr_rz"]
        # if settings.enable_groundwater_boundary:
        #     diagnostics["rates"].output_variables += ["cpr_ss"]
        # diagnostics["rates"].output_frequency = 24 * 60 * 60
        # diagnostics["rates"].sampling_frequency = 1

        # diagnostics["collect"].output_variables = ["S_snow", "S_int_ground", "S_rz", "S_ss",
        #                                            "theta_rz", "theta_ss", "theta",
        #                                            "z_wf"]
        # if settings.enable_crop_phenology:
        #     diagnostics["collect"].output_variables += ["re_rg", "re_rl"]
        # diagnostics["collect"].output_frequency = 24 * 60 * 60
        # diagnostics["collect"].sampling_frequency = 1

        # diagnostics["averages"].output_variables = ["ta"]
        # diagnostics["averages"].output_frequency = 24 * 60 * 60
        # diagnostics["averages"].sampling_frequency = 1

    @roger_routine
    def after_timestep(self, state):
        vs = state.variables
        settings = state.settings

        vs.update(after_timestep_kernel(state))
        if settings.enable_film_flow:
            vs.update(after_timestep_film_flow_kernel(state))
        if settings.enable_crop_phenology:
            vs.update(after_timestep_crops_kernel(state))


@roger_kernel
def set_initial_conditions_crops_kernel(state):
    vs = state.variables

    # calculate time since growing
    t_grow = allocate(state.dimensions, ("x", "y", "crops"))
    t_grow = update(
        t_grow,
        at[:, :, :], npx.where(vs.z_root_crop[:, :, vs.taum1, :] > 0, (-1 / vs.root_growth_rate) * npx.log(1 / ((vs.z_root_crop[:, :, vs.taum1, :] / 1000 - vs.z_root_crop_max / 1000) * (-1 / (vs.z_root_crop_max / 1000 - vs.z_evap[:, :, npx.newaxis] / 1000)))), 0)
    )

    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[:, :, :2, :], t_grow[:, :, npx.newaxis, :]
    )

    vs.t_grow_root = update(
        vs.t_grow_root,
        at[:, :, :2, :], t_grow[:, :, npx.newaxis, :]
    )

    return KernelOutput(
        t_grow_cc=vs.t_grow_cc,
        t_grow_root=vs.t_grow_root,
    )


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

    # update precipitation with available interception storage while film flow event
    mask_noff = (vs.EVENT_ID_FF[:, :, vs.itt] == 0)
    vs.prec = update(vs.prec, at[:, :], 0)
    vs.prec = update_add(vs.prec, at[:, :],
                         npx.where((vs.EVENT_ID_FF[:, :, vs.itt] >= 1) & ((vs.S_int_top_tot - vs.S_int_top[:, :, vs.tau]) + (vs.S_int_ground_tot - vs.S_int_ground[:, :, vs.tau]) > 0),
                         npx.where(vs.PREC[:, :, vs.itt] > (vs.S_int_top_tot - vs.S_int_top[:, :, vs.tau]) + (vs.S_int_ground_tot - vs.S_int_ground[:, :, vs.tau]), (vs.S_int_top_tot - vs.S_int_top[:, :, vs.tau]) + (vs.S_int_ground_tot - vs.S_int_ground[:, :, vs.tau]), vs.PREC[:, :, vs.itt]), 0))
    vs.prec = update(vs.prec, at[:, :], npx.where(mask_noff, vs.PREC[:, :, vs.itt], 0))
    vs.ta = update(vs.ta, at[:, :, vs.tau], vs.TA[:, :, vs.itt])
    vs.pet = update(vs.pet, at[:, :], vs.PET[:, :, vs.itt])
    vs.pet_res = update(vs.pet, at[:, :], vs.PET[:, :, vs.itt])

    vs.dt_secs = vs.DT_SECS[vs.itt]
    vs.dt = vs.DT[vs.itt]
    vs.year = vs.YEAR[vs.itt]
    vs.month = vs.MONTH[vs.itt]
    vs.doy = vs.DOY[vs.itt]

    # reset fluxes at beginning of time step
    q_sur = allocate(state.dimensions, ("x", "y"))
    vs.q_sur = update(
        vs.q_sur,
        at[:, :], q_sur,
    )

    q_hof = allocate(state.dimensions, ("x", "y"))
    vs.q_hof = update(
        vs.q_hof,
        at[:, :], q_hof,
    )

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
        q_sur=vs.q_sur,
        q_hof=vs.q_hof,
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
    )


@roger_kernel
def after_timestep_film_flow_kernel(state):
    vs = state.variables

    vs.z_wf_ff = update(
        vs.z_wf_ff,
        at[:, :, :, vs.taum1], vs.z_wf_ff[:, :, :, vs.tau],
    )
    vs.z_pf_ff = update(
        vs.z_pf_ff,
        at[:, :, :, vs.taum1], vs.z_pf_ff[:, :, :, vs.tau],
    )
    vs.theta_rz_ff = update(
        vs.theta_rz_ff,
        at[:, :, vs.taum1], vs.theta_rz_ff[:, :, vs.tau],
    )
    vs.theta_ss_ff = update(
        vs.theta_ss_ff,
        at[:, :, vs.taum1], vs.theta_ss_ff[:, :, vs.tau],
    )
    vs.theta_ff = update(
        vs.theta_ff,
        at[:, :, vs.taum1], vs.theta_ff[:, :, vs.tau],
    )
    vs.z_pf = update(
        vs.z_pf,
        at[:, :, vs.taum1], vs.z_pf[:, :, vs.tau],
    )

    return KernelOutput(
        z_wf_ff=vs.z_wf_ff,
        z_pf_ff=vs.z_pf_ff,
        theta_rz_ff=vs.theta_rz_ff,
        theta_ss_ff=vs.theta_ss_ff,
        theta_ff=vs.theta_ff,
        z_pf=vs.z_pf,
    )


@roger_kernel
def after_timestep_crops_kernel(state):
    vs = state.variables

    vs.ta_min = update(vs.ta_min, at[:, :, vs.taum1], vs.ta_min[:, :, vs.tau])
    vs.ta_max = update(vs.ta_max, at[:, :, vs.taum1], vs.ta_max[:, :, vs.tau])
    vs.gdd_sum = update(vs.gdd_sum, at[:, :, vs.taum1, :], vs.gdd_sum[:, :, vs.tau, :])
    vs.t_grow_cc = update(vs.t_grow_cc, at[:, :, vs.taum1, :], vs.t_grow_cc[:, :, vs.tau, :])
    vs.t_grow_root = update(vs.t_grow_root, at[:, :, vs.taum1, :], vs.t_grow_root[:, :, vs.tau, :])
    vs.ccc = update(vs.ccc, at[:, :, vs.taum1, :], vs.ccc[:, :, vs.tau, :])
    vs.z_root_crop = update(vs.z_root_crop, at[:, :, vs.taum1, :], vs.z_root_crop[:, :, vs.tau, :])

    return KernelOutput(
        ta_min=vs.ta_min,
        ta_max=vs.ta_max,
        gdd_sum=vs.gdd_sum,
        t_grow_cc=vs.t_grow_cc,
        t_grow_root=vs.t_grow_root,
        ccc=vs.ccc,
        z_root_crop=vs.z_root_crop,
    )


model = DUMMYFFSetup()
model.setup()
model.run()

path = str(BASE_PATH / f"{model.state.settings.identifier}.*.nc")
diag_files = glob.glob(path)
states_hm_file = BASE_PATH / "states_hm.nc"
with h5netcdf.File(states_hm_file, 'w', decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title='test results',
        institution='University of Freiburg, Chair of Hydrology',
        references='',
        comment=''
    )
    for dfs in diag_files:
        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
            # set dimensions with a dictionary
            if not f.dimensions:
                f.dimensions = {'x': df.dimensions["x"], 'y': df.dimensions["y"], 'Time': len(df.variables['Time'])}
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Zonal coordinate'
                v.attrs['units'] = 'meters'
                v[:] = npx.arange(f.dimensions["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = 'Meridonial coordinate'
                v.attrs['units'] = 'meters'
                v[:] = npx.arange(f.dimensions["y"])
                v = f.create_variable('Time', ('Time',), float)
                var_obj = df.variables.get('Time')
                with h5netcdf.File(BASE_PATH / 'forcing.nc', "r", decode_vlen_strings=False) as infile:
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
BASE_PATH_TM = BASE_PATH.parent / "dummytm"
states_hm_file1 = BASE_PATH_TM / "states_hm.nc"
shutil.copy(states_hm_file, states_hm_file1)

# with xr.open_dataset(states_hm_file, engine="h5netcdf") as ds:
#     time_origin = ds['Time'].attrs['time_origin']
#     time = (ds['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
#     date = num2date(time, units=f"days since {ds['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
#     ds = ds.assign_coords(date=("Time", date))
#     keys = list(ds.data_vars.keys())
#     for key in keys:
#         fig, ax = plt.subplots()
#         ds[key].isel(x=0, y=0).plot(x="date", ax=ax)
#         fig.tight_layout()
