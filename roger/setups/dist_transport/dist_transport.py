from pathlib import Path
import glob
import datetime
import h5netcdf
import xarray as xr
import matplotlib.pyplot as plt

from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at, where, scipy_stats as sstx
from roger.setups.make_dummy_setup import make_setup
from roger.io_tools import yml
import numpy as onp

# generate dummy setup
BASE_PATH = Path(__file__).parent
make_setup(BASE_PATH, event_type='', ndays=365,
           enable_groundwater_boundary=False,
           enable_film_flow=False,
           enable_crop_phenology=False,
           enable_crop_rotation=False,
           enable_lateral_flow=False,
           enable_groundwater=False,
           enable_offline_transport=True,
           enable_bromide=False,
           enable_chloride=False,
           enable_deuterium=False,
           enable_oxygen18=False,
           enable_nitrate=True,
           tm_structure='complete-mixing')

# read config file
CONFIG_FILE = BASE_PATH / "config.yml"
config = yml.Config(CONFIG_FILE)


class DISTTRANSPORTSetup(RogerSetup):
    """A distributed transport model.
    """
    def _read_var_from_nc(self, var, file):
        nc_file = BASE_PATH / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables[var]
            return npx.array(var_obj)

    def _get_nitt(self):
        nc_file = BASE_PATH / 'states_hm.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['Time']
            return len(onp.array(var_obj))

    def _get_runlen(self):
        nc_file = BASE_PATH / 'states_hm.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['Time']
            return len(onp.array(var_obj)) * 60 * 60 * 24

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "DISTTRANSPORT"

        settings.nx, settings.ny, settings.nz = config.nrows, config.ncols, 1
        settings.nitt = self._get_nitt()
        settings.ages = settings.nitt
        settings.nages = settings.nitt + 1
        settings.runlen = self._get_runlen()

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
        settings.tm_structure = config.tm_structure

    @roger_routine
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        # temporal grid
        vs.dt_secs = 60 * 60 * 24
        vs.dt = 60 * 60 * 24 / (60 * 60)
        vs.DT_SECS = update(vs.DT_SECS, at[:], vs.dt_secs)
        vs.DT = update(vs.DT, at[:], vs.dt)
        vs.t = update(vs.t, at[:], npx.cumsum(vs.DT))
        vs.ages = update(vs.ages, at[:], npx.arange(1, settings.nages))
        vs.nages = update(vs.nages, at[:], npx.arange(settings.nages))
        # spatial grid
        vs.x = update(vs.x, at[2:-2], npx.arange(1, settings.nx + 1) * (settings.dx / 2))
        vs.y = update(vs.y, at[2:-2], npx.arange(1, settings.ny + 1) * (settings.dy / 2))

    @roger_routine
    def set_look_up_tables(self, state):
        pass

    @roger_routine
    def set_topography(self, state):
        pass

    @roger_routine
    def set_parameters(self, state):
        vs = state.variables
        settings = state.settings

        if (vs.itt == 0):
            vs.S_PWP_RZ = update(vs.S_PWP_RZ, at[:, :, :], self._read_var_from_nc("S_pwp_rz", 'states_hm.nc'))
            vs.S_PWP_SS = update(vs.S_PWP_SS, at[:, :, :], self._read_var_from_nc("S_pwp_ss", 'states_hm.nc'))
            vs.S_SAT_RZ = update(vs.S_SAT_RZ, at[:, :, :], self._read_var_from_nc("S_sat_rz", 'states_hm.nc'))
            vs.S_SAT_SS = update(vs.S_SAT_SS, at[:, :, :], self._read_var_from_nc("S_sat_ss", 'states_hm.nc'))

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[:, :], vs.S_PWP_RZ[:, :, 0])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[:, :], vs.S_PWP_SS[:, :, 0])
            vs.S_sat_rz = update(vs.S_sat_rz, at[:, :], vs.S_SAT_RZ[:, :, 0])
            vs.S_sat_ss = update(vs.S_sat_ss, at[:, :], vs.S_SAT_SS[:, :, 0])

            if (settings.enable_bromide | settings.enable_chloride):
                vs.alpha_transp = update(vs.alpha_transp, at[:, :], self._read_var_from_nc("alpha_transp", 'parameters.nc'))
                vs.alpha_q = update(vs.alpha_q, at[:, :], self._read_var_from_nc("alpha_q", 'parameters.nc'))
            if settings.enable_nitrate:
                vs.alpha_transp = update(vs.alpha_transp, at[:, :], self._read_var_from_nc("alpha_transp", 'parameters.nc'))
                vs.alpha_q = update(vs.alpha_q, at[:, :], self._read_var_from_nc("alpha_q", 'parameters.nc'))
                vs.km_denit_rz = update(vs.km_denit_rz, at[:, :], self._read_var_from_nc("km_denit_rz", 'parameters.nc'))
                vs.km_denit_ss = update(vs.km_denit_ss, at[:, :], self._read_var_from_nc("km_denit_ss", 'parameters.nc'))
                vs.dmax_denit_rz = update(vs.dmax_denit_rz, at[:, :], self._read_var_from_nc("dmax_denit_rz", 'parameters.nc'))
                vs.dmax_denit_ss = update(vs.dmax_denit_ss, at[:, :], self._read_var_from_nc("dmax_denit_ss", 'parameters.nc'))
                vs.km_nit_rz = update(vs.km_nit_rz, at[:, :], self._read_var_from_nc("km_nit_rz", 'parameters.nc'))
                vs.km_nit_ss = update(vs.km_nit_ss, at[:, :], self._read_var_from_nc("km_nit_ss", 'parameters.nc'))
                vs.dmax_nit_rz = update(vs.dmax_nit_rz, at[:, :], self._read_var_from_nc("dmax_nit_rz", 'parameters.nc'))
                vs.dmax_nit_ss = update(vs.dmax_nit_ss, at[:, :], self._read_var_from_nc("dmax_nit_ss", 'parameters.nc'))
                vs.kmin_rz = update(vs.kmin_rz, at[:, :], self._read_var_from_nc("kmin_rz", 'parameters.nc'))
                vs.kmin_ss = update(vs.kmin_ss, at[:, :], self._read_var_from_nc("kmin_ss", 'parameters.nc'))

        if settings.tm_structure == "complete-mixing":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 1)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 1)
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 0], 1)
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 0], 1)
        elif settings.tm_structure == "piston":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 21)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 22)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 22)
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 0], 22)
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 0], 22)
        elif settings.tm_structure in ["advection-dispersion", "preferential"]:
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_transp.nc'))
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 1], self._read_var_from_nc("a", 'sas_parameters_transp.nc'))
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 2], self._read_var_from_nc("b", 'sas_parameters_transp.nc'))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_rz.nc'))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 1], self._read_var_from_nc("a", 'sas_parameters_q_rz.nc'))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 2], self._read_var_from_nc("b", 'sas_parameters_q_rz.nc'))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_ss.nc'))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 1], self._read_var_from_nc("a", 'sas_parameters_q_ss.nc'))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 2], self._read_var_from_nc("b", 'sas_parameters_q_ss.nc'))
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_sub_rz.nc'))
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 1], self._read_var_from_nc("a", 'sas_parameters_q_sub_rz.nc'))
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 2], self._read_var_from_nc("b", 'sas_parameters_q_sub_rz.nc'))
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_sub_ss.nc'))
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 1], self._read_var_from_nc("a", 'sas_parameters_q_sub_ss.nc'))
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 2], self._read_var_from_nc("b", 'sas_parameters_q_sub_ss.nc'))
        elif settings.tm_structure == "complete-mixing + advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_rz.nc'))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 1], self._read_var_from_nc("a", 'sas_parameters_q_rz.nc'))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 2], self._read_var_from_nc("b", 'sas_parameters_q_rz.nc'))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_ss.nc'))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 1], self._read_var_from_nc("a", 'sas_parameters_q_ss.nc'))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 2], self._read_var_from_nc("b", 'sas_parameters_q_ss.nc'))
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_sub_rz.nc'))
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 1], self._read_var_from_nc("a", 'sas_parameters_q_sub_rz.nc'))
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 2], self._read_var_from_nc("b", 'sas_parameters_q_sub_rz.nc'))
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_sub_ss.nc'))
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 1], self._read_var_from_nc("a", 'sas_parameters_q_sub_ss.nc'))
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 2], self._read_var_from_nc("b", 'sas_parameters_q_sub_ss.nc'))
        elif settings.tm_structure in ["time-variant advection-disperison", "time-variant preferential"]:
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_transp.nc'))
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 3], self._read_var_from_nc("lower_limit", 'sas_parameters_transp.nc'))
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 4], self._read_var_from_nc("upper_limit", 'sas_parameters_transp.nc'))
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 6], vs.S_sat_rz - vs.S_pwp_rz)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_rz.nc'))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 3], self._read_var_from_nc("lower_limit", 'sas_parameters_q_rz.nc'))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 4], self._read_var_from_nc("upper_limit", 'sas_parameters_q_rz.nc'))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 6], vs.S_sat_rz - vs.S_pwp_rz)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_ss.nc'))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 3], self._read_var_from_nc("lower_limit", 'sas_parameters_q_ss.nc'))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 4], self._read_var_from_nc("upper_limit", 'sas_parameters_q_ss.nc'))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 6], vs.S_sat_ss - vs.S_pwp_ss)
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_sub_rz.nc'))
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 3], self._read_var_from_nc("lower_limit", 'sas_parameters_q_sub_rz.nc'))
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 4], self._read_var_from_nc("upper_limit", 'sas_parameters_q_sub_rz.nc'))
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 5], 0)
            vs.sas_params_q_sub_rz = update(vs.sas_params_q_sub_rz, at[:, :, 6], vs.S_sat_rz - vs.S_pwp_rz)
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 0], self._read_var_from_nc("sas_function", 'sas_parameters_q_sub_ss.nc'))
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 3], self._read_var_from_nc("lower_limit", 'sas_parameters_q_sub_ss.nc'))
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 4], self._read_var_from_nc("upper_limit", 'sas_parameters_q_sub_ss.nc'))
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 5], 0)
            vs.sas_params_q_sub_ss = update(vs.sas_params_q_sub_ss, at[:, :, 6], vs.S_sat_ss - vs.S_pwp_ss)

    @roger_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        vs.S_RZ = update(vs.S_RZ, at[:, :, :], self._read_var_from_nc("S_rz", 'states_hm.nc'))
        vs.S_SS = update(vs.S_SS, at[:, :, :], self._read_var_from_nc("S_ss", 'states_hm.nc'))
        vs.S_S = update(vs.S_S, at[:, :, :], vs.S_RZ + vs.S_SS)

        vs.S_rz = update(vs.S_rz, at[:, :, :2], vs.S_RZ[:, :, 0, npx.newaxis] - vs.S_pwp_rz[:, :, npx.newaxis])
        vs.S_ss = update(vs.S_ss, at[:, :, :2], vs.S_SS[:, :, 0, npx.newaxis] - vs.S_pwp_ss[:, :, npx.newaxis])
        vs.S_s = update(vs.S_s, at[:, :, :2], vs.S_S[:, :, 0, npx.newaxis] - (vs.S_pwp_rz[:, :, npx.newaxis] + vs.S_pwp_ss[:, :, npx.newaxis]))

        arr0 = allocate(state.dimensions, ("x", "y"))
        vs.sa_rz = update(
            vs.sa_rz,
            at[:, :, :2, 1:], npx.diff(npx.linspace(arr0, vs.S_rz[:, :, vs.tau], settings.ages, axis=-1), axis=-1)[:, :, npx.newaxis, :],
        )
        vs.sa_ss = update(
            vs.sa_ss,
            at[:, :, :2, 1:], npx.diff(npx.linspace(arr0, vs.S_ss[:, :, vs.tau], settings.ages, axis=-1), axis=-1)[:, :, npx.newaxis, :],
        )

        vs.SA_rz = update(
            vs.SA_rz,
            at[:, :, :, 1:], npx.cumsum(vs.sa_rz, axis=3),
        )

        vs.SA_ss = update(
            vs.SA_ss,
            at[:, :, :, 1:], npx.cumsum(vs.sa_rz, axis=3),
        )

        vs.sa_s = update(
            vs.sa_s,
            at[:, :, :, :], vs.sa_rz + vs.sa_ss,
        )
        vs.SA_s = update(
            vs.SA_s,
            at[:, :, :, 1:], npx.cumsum(vs.sa_s, axis=3),
        )

        if (settings.enable_bromide | settings.enable_chloride):
            vs.C_rz = update(vs.C_rz, at[:, :, :2], self._read_var_from_nc("C_rz", 'initvals.nc')[:, :, npx.newaxis])
            vs.C_ss = update(vs.C_ss, at[:, :, :2], self._read_var_from_nc("C_ss", 'initvals.nc')[:, :, npx.newaxis])
            vs.msa_rz = update(
                vs.msa_rz,
                at[:, :, :2, :], vs.C_rz[:, :, :2, npx.newaxis] * vs.sa_rz[:, :, :2, :],
            )
            vs.msa_ss = update(
                vs.msa_ss,
                at[:, :, :2, :], vs.C_ss[:, :, :2, npx.newaxis] * vs.sa_ss[:, :, :2, :],
            )
            vs.msa_s = update(
                vs.msa_s,
                at[:, :, :, :], vs.msa_rz + vs.msa_ss,
            )
            vs.M_rz = update(
                vs.M_rz,
                at[:, :, :], npx.sum(vs.msa_rz, axis=-1),
            )
            vs.M_ss = update(
                vs.M_ss,
                at[:, :, :], npx.sum(vs.msa_ss, axis=-1),
            )
            vs.M_s = update(
                vs.M_s,
                at[:, :, :], npx.sum(vs.msa_s, axis=-1),
            )

        elif (settings.enable_oxygen18 | settings.enable_deuterium):
            vs.C_rz = update(vs.C_rz, at[:, :, :2], self._read_var_from_nc("C_rz", 'initvals.nc')[:, :, npx.newaxis])
            vs.C_ss = update(vs.C_ss, at[:, :, :2], self._read_var_from_nc("C_ss", 'initvals.nc')[:, :, npx.newaxis])
            vs.msa_rz = update(
                vs.msa_rz,
                at[:, :, :2, :], vs.C_rz[:, :, :2, npx.newaxis],
            )
            vs.msa_rz = update(
                vs.msa_rz,
                at[:, :, :2, 0], npx.NaN,
            )
            vs.msa_ss = update(
                vs.msa_ss,
                at[:, :, :2, :], vs.C_ss[:, :, :2, npx.newaxis],
            )
            vs.msa_ss = update(
                vs.msa_ss,
                at[:, :, :2, 0], npx.NaN,
            )
            iso_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
            iso_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
            iso_rz = update(
                iso_rz,
                at[:, :, :, :], npx.where(npx.isnan(vs.msa_rz), 0, vs.msa_rz),
            )
            iso_ss = update(
                iso_ss,
                at[:, :, :, :], npx.where(npx.isnan(vs.msa_ss), 0, vs.msa_ss),
            )
            vs.msa_s = update(
                vs.msa_s,
                at[:, :, :, :], (vs.sa_rz / vs.sa_s) * iso_rz + (vs.sa_ss / vs.sa_s) * iso_ss,
            )

            vs.C_s = update(
                vs.C_s,
                at[:, :, vs.tau], calc_conc_iso_storage(state, vs.sa_s, vs.msa_s) * vs.maskCatch,
            )

            vs.C_s = update(
                vs.C_s,
                at[:, :, vs.taum1], vs.C_s[:, :, vs.tau] * vs.maskCatch,
            )

        elif settings.enable_nitrate:
            vs.C_rz = update(vs.C_rz, at[:, :, :2], self._read_var_from_nc("C_rz", 'initvals.nc')[:, :, npx.newaxis])
            vs.C_ss = update(vs.C_ss, at[:, :, :2], self._read_var_from_nc("C_ss", 'initvals.nc')[:, :, npx.newaxis])
            p_dec = allocate(state.dimensions, ("x", "y", 2, "ages"))
            p_dec = update(p_dec, at[:, :, :2, :], sstx.expon.pdf(npx.linspace(sstx.expon.ppf(0.001), sstx.expon.ppf(0.999), settings.ages))[npx.newaxis, npx.newaxis, npx.newaxis, :])
            vs.Nmin_rz = update(vs.Nmin_rz, at[:, :, :2, :], self._read_var_from_nc("Nmin_rz", 'initvals.nc')[:, :, npx.newaxis, npx.newaxis] * p_dec * settings.dx * settings.dy * 100)
            vs.Nmin_ss = update(vs.Nmin_ss, at[:, :, :2, :], self._read_var_from_nc("Nmin_ss", 'initvals.nc')[:, :, npx.newaxis, npx.newaxis] * p_dec * settings.dx * settings.dy * 100)
            vs.msa_rz = update(
                vs.msa_rz,
                at[:, :, :2, :], vs.C_rz[:, :, :2, npx.newaxis] * vs.sa_rz[:, :, :2, :],
            )
            vs.msa_ss = update(
                vs.msa_ss,
                at[:, :, :2, :], vs.C_ss[:, :, :2, npx.newaxis] * vs.sa_ss[:, :, :2, :],
            )
            vs.msa_s = update(
                vs.msa_s,
                at[:, :, :, :], vs.msa_rz + vs.msa_ss,
            )
            vs.M_rz = update(
                vs.M_rz,
                at[:, :, :], npx.sum(vs.msa_rz, axis=-1),
            )
            vs.M_ss = update(
                vs.M_ss,
                at[:, :, :], npx.sum(vs.msa_ss, axis=-1),
            )
            vs.M_s = update(
                vs.M_s,
                at[:, :, :], npx.sum(vs.msa_s, axis=-1),
            )

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables
        settings = state.settings

        if (vs.itt == 0):
            vs.TA = update(vs.TA, at[:, :, :], self._read_var_from_nc("ta", 'states_hm.nc'))
            vs.PREC = update(vs.PREC, at[:, :, :], self._read_var_from_nc("prec", 'states_hm.nc'))
            vs.INF_MAT_RZ = update(vs.INF_MAT_RZ, at[:, :, :], self._read_var_from_nc("inf_mat_rz", 'states_hm.nc'))
            vs.INF_PF_RZ = update(vs.INF_PF_RZ, at[:, :, :], self._read_var_from_nc("inf_mp_rz", 'states_hm.nc') + self._read_var_from_nc("inf_sc_rz", 'states_hm.nc'))
            vs.INF_PF_SS = update(vs.INF_PF_SS, at[:, :, :], self._read_var_from_nc("inf_ss", 'states_hm.nc'))
            vs.TRANSP = update(vs.TRANSP, at[:, :, :], self._read_var_from_nc("transp", 'states_hm.nc'))
            vs.EVAP_SOIL = update(vs.EVAP_SOIL, at[:, :, :], self._read_var_from_nc("evap_soil", 'states_hm.nc'))
            vs.CPR_RZ = update(vs.CPR_RZ, at[:, :, :], self._read_var_from_nc("cpr_rz", 'states_hm.nc'))
            vs.Q_RZ = update(vs.Q_RZ, at[:, :, :], self._read_var_from_nc("q_rz", 'states_hm.nc'))
            vs.Q_SS = update(vs.Q_SS, at[:, :, :], self._read_var_from_nc("q_ss", 'states_hm.nc'))
            vs.Q_SUB_RZ = update(vs.Q_SUB_RZ, at[:, :, :], self._read_var_from_nc("q_sub_rz", 'states_hm.nc'))
            vs.Q_SUB_SS = update(vs.Q_SUB_SS, at[:, :, :], self._read_var_from_nc("q_sub_ss", 'states_hm.nc'))

            if settings.enable_bromide:
                vs.M_IN = update(vs.M_IN, at[:, :, :], self._read_var_from_nc("Br", 'tracer_input.nc'))

            if settings.enable_chloride:
                vs.C_IN = update(vs.C_IN, at[:, :, :], self._read_var_from_nc("Cl", 'tracer_input.nc'))

            if settings.enable_deuterium:
                vs.C_IN = update(vs.C_IN, at[:, :, :], self._read_var_from_nc("d2H", 'tracer_input.nc'))

            if settings.enable_oxygen18:
                vs.C_IN = update(vs.C_IN, at[:, :, :], self._read_var_from_nc("d18O", 'tracer_input.nc'))

            if settings.enable_nitrate:
                # convert kg N/ha to mg/square meter
                vs.NMIN_IN = update(vs.NMIN_IN, at[:, :, :], self._read_var_from_nc("Nmin", 'tracer_input.nc') * 100 * settings.dx * settings.dy)
                vs.NORG_IN = update(vs.NORG_IN, at[:, :, :], self._read_var_from_nc("Norg", 'tracer_input.nc') * 100 * settings.dx * settings.dy)

            if settings.enable_bromide:
                mask_rain = (vs.PREC > 0) & (vs.TA > 0)
                mask_sol = (vs.M_IN > 0)
                nn_rain = npx.int64(npx.sum(npx.any(mask_rain, axis=(0, 1))))
                nn_sol = npx.int64(npx.sum(npx.any(mask_sol, axis=(0, 1))))
                vs.update(set_bromide_input_kernel(state, nn_rain, nn_sol))

            if settings.enable_deuterium or settings.enable_oxygen18:
                vs.update(set_iso_input_kernel(state))

            if settings.enable_nitrate:
                mask_rain = (vs.PREC > 0) & (vs.TA > 0)
                mask_sol = (vs.NMIN_IN > 0)
                nn_rain = npx.int64(npx.sum(npx.any(mask_rain, axis=(0, 1))))
                nn_sol = npx.int64(npx.sum(npx.any(mask_sol, axis=(0, 1))))
                vs.update(set_nitrate_input_kernel(state, nn_rain, nn_sol))

        vs.update(set_states_kernel(state))
        if settings.enable_bromide:
            vs.update(set_forcing_bromide_kernel(state))

        if settings.enable_chloride:
            vs.update(set_forcing_chloride_kernel(state))

        if settings.enable_deuterium or settings.enable_oxygen18:
            vs.update(set_forcing_iso_kernel(state))

        if settings.enable_nitrate:
            vs.update(set_forcing_nitrate_kernel(state))

    @roger_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics

        diagnostics["rates"].output_variables = ["M_transp"]
        diagnostics["rates"].output_frequency = 24 * 60 * 60
        diagnostics["rates"].sampling_frequency = 1

        diagnostics["collect"].output_variables = ["TT_transp"]
        diagnostics["collect"].output_frequency = 24 * 60 * 60
        diagnostics["collect"].sampling_frequency = 1

        diagnostics["averages"].output_variables = ["C_rz", "C_ss"]
        diagnostics["averages"].output_frequency = 24 * 60 * 60
        diagnostics["averages"].sampling_frequency = 1

    @roger_routine
    def after_timestep(self, state):
        vs = state.variables
        settings = state.settings

        vs.update(after_timestep_kernel(state))
        if settings.enable_nitrate:
            vs.update(after_timestep_nitrate_kernel(state))


@roger_kernel(static_args=("nn_rain", "nn_sol"))
def set_bromide_input_kernel(state, nn_rain, nn_sol):
    vs = state.variables

    M_IN = allocate(state.dimensions, ("x", "y", "t"))

    mask_rain = (vs.PREC > 0) & (vs.TA > 0)
    mask_sol = (vs.M_IN > 0)
    sol_idx = npx.zeros((nn_sol,), dtype=int)
    sol_idx = update(sol_idx, at[:], where(npx.any(mask_sol, axis=(0, 1)), size=nn_sol, fill_value=0)[0])
    rain_idx = npx.zeros((nn_rain,), dtype=int)
    rain_idx = update(rain_idx, at[:], where(npx.any(mask_rain, axis=(0, 1)), size=nn_rain, fill_value=0)[0])
    end_rain = npx.zeros((1,), dtype=int)

    # join solute input on closest rainfall event
    for i in range(nn_sol):
        rain_sum = allocate(state.dimensions, ("x", "y"))
        nn_end = allocate(state.dimensions, ("x", "y"))
        input_itt = npx.nanargmin(npx.where(rain_idx - sol_idx[i] < 0, npx.NaN, rain_idx - sol_idx[i]))
        start_rain = rain_idx[input_itt]
        rain_sum = update(
            rain_sum,
            at[:, :], npx.max(npx.where(npx.cumsum(vs.PREC[:, :, start_rain:], axis=-1) <= 20, npx.max(npx.cumsum(vs.PREC[:, :, start_rain:], axis=-1), axis=-1), 0), axis=-1),
        )
        nn_end = npx.max(npx.where(npx.cumsum(vs.PREC[:, :, start_rain:]) <= 20, npx.max(npx.arange(npx.shape(vs.PREC)[2])[npx.newaxis, npx.newaxis, npx.shape(vs.PREC)[2]-start_rain], axis=-1), 0))
        end_rain = update(end_rain, at[:], start_rain + nn_end)
        end_rain = update(end_rain, at[:], npx.where(end_rain > npx.shape(vs.PREC)[2], npx.shape(vs.PREC)[2], end_rain))

        # proportions for redistribution
        M_IN = update(
            M_IN,
            at[:, :, start_rain:end_rain[0]], vs.M_IN[:, :, sol_idx[i], npx.newaxis] * (vs.PREC[:, :, start_rain:end_rain[0]] / rain_sum[:, :, npx.newaxis]),
        )

    # solute input concentration
    vs.M_IN = update(
        vs.M_IN,
        at[:, :, :], M_IN,
    )
    vs.C_IN = update(
        vs.C_IN,
        at[:, :, :], npx.where(vs.PREC > 0, vs.M_IN / vs.PREC, 0),
    )

    return KernelOutput(
        M_IN=vs.M_IN,
        C_IN=vs.C_IN,
    )


@roger_kernel
def set_forcing_bromide_kernel(state):
    vs = state.variables

    vs.M_in = update(
        vs.M_in,
        at[:, :], vs.M_IN[:, :, vs.itt] * vs.maskCatch,
    )
    vs.C_in = update(
        vs.C_in,
        at[:, :], vs.C_IN[:, :, vs.itt] * vs.maskCatch,
    )

    return KernelOutput(
        M_in=vs.M_in,
        C_in=vs.C_in,
    )


@roger_kernel
def set_forcing_chloride_kernel(state):
    vs = state.variables

    vs.C_in = update(
        vs.C_in,
        at[:, :], npx.where((vs.PREC[:, :, vs.itt] > 0) & (vs.TA[:, :, vs.itt] > 0), vs.C_IN[:, :, vs.itt], 0) * vs.maskCatch,
    )

    vs.M_in = update(
        vs.M_in,
        at[:, :], vs.C_in * vs.PREC[:, :, vs.itt] * vs.maskCatch,
    )

    return KernelOutput(
        M_in=vs.M_in,
        C_in=vs.C_in,
    )


@roger_kernel
def set_iso_input_kernel(state):
    vs = state.variables

    vs.C_IN = update(vs.C_IN, at[:, :, :], _ffill_3d(state, vs.C_IN))

    return KernelOutput(
        C_IN=vs.C_IN,
    )


@roger_kernel
def set_forcing_iso_kernel(state):
    vs = state.variables

    vs.C_in = update(
        vs.C_in,
        at[:, :], npx.where(vs.PREC[:, :, vs.itt] > 0, vs.C_IN[:, :, vs.itt], npx.NaN) * vs.maskCatch,
    )

    vs.M_in = update(
        vs.M_in,
        at[:, :], vs.C_in * vs.PREC[:, :, vs.itt] * vs.maskCatch,
    )

    return KernelOutput(
        M_in=vs.M_in,
        C_in=vs.C_in,
    )


@roger_kernel(static_args=("nn_rain", "nn_sol"))
def set_nitrate_input_kernel(state, nn_rain, nn_sol):
    vs = state.variables

    NMIN_IN = allocate(state.dimensions, ("x", "y", "t"))

    mask_rain = (vs.PREC > 0) & (vs.TA > 0)
    mask_sol = (vs.NMIN_IN > 0)
    sol_idx = npx.zeros((nn_sol,), dtype=int)
    sol_idx = update(sol_idx, at[:], where(npx.any(mask_sol, axis=(0, 1)), size=nn_sol, fill_value=0)[0])
    rain_idx = npx.zeros((nn_rain,), dtype=int)
    rain_idx = update(rain_idx, at[:], where(npx.any(mask_rain, axis=(0, 1)), size=nn_rain, fill_value=0)[0])
    end_rain = npx.zeros((1,), dtype=int)

    # join solute input on closest rainfall event
    for i in range(nn_sol):
        rain_sum = allocate(state.dimensions, ("x", "y"))
        nn_end = allocate(state.dimensions, ("x", "y"))
        input_itt = npx.nanargmin(npx.where(rain_idx - sol_idx[i] < 0, npx.NaN, rain_idx - sol_idx[i]))
        start_rain = rain_idx[input_itt]
        rain_sum = update(
            rain_sum,
            at[:, :], npx.max(npx.where(npx.cumsum(vs.PREC[:, :, start_rain:], axis=-1) <= 20, npx.max(npx.cumsum(vs.PREC[:, :, start_rain:], axis=-1), axis=-1), 0), axis=-1),
        )
        nn_end = npx.max(npx.where(npx.cumsum(vs.PREC[:, :, start_rain:]) <= 20, npx.max(npx.arange(npx.shape(vs.PREC)[2])[npx.newaxis, npx.newaxis, npx.shape(vs.PREC)[2]-start_rain], axis=-1), 0))
        end_rain = update(end_rain, at[:], start_rain + nn_end)
        end_rain = update(end_rain, at[:], npx.where(end_rain > npx.shape(vs.PREC)[2], npx.shape(vs.PREC)[2], end_rain))

        # proportions for redistribution
        NMIN_IN = update(
            NMIN_IN,
            at[:, :, start_rain:end_rain[0]], vs.M_IN[:, :, sol_idx[i], npx.newaxis] * (vs.PREC[:, :, start_rain:end_rain[0]] / rain_sum[:, :, npx.newaxis]),
        )

    # solute input concentration
    vs.M_IN = update(
        vs.M_IN,
        at[:, :, :], NMIN_IN * 0.3,
    )
    vs.C_IN = update(
        vs.C_IN,
        at[:, :, :], npx.where(vs.PREC > 0, vs.M_IN / vs.PREC, 0),
    )
    vs.NMIN_IN = update(
        vs.NMIN_IN,
        at[:, :, :], NMIN_IN * 0.7,
    )

    return KernelOutput(
        M_IN=vs.M_IN,
        C_IN=vs.C_IN,
        NMIN_IN=vs.NMIN_IN,
    )


@roger_kernel
def set_forcing_nitrate_kernel(state):
    vs = state.variables

    vs.C_in = update(
        vs.C_in,
        at[:, :], vs.C_IN[:, :, vs.itt] * vs.maskCatch,
    )

    vs.M_in = update(
        vs.M_in,
        at[:, :], vs.M_IN[:, :, vs.itt] * vs.maskCatch,
    )

    vs.Nmin_in = update(
        vs.Nmin_in,
        at[:, :], vs.NMIN_IN[:, :, vs.itt] * vs.maskCatch,
    )

    vs.Norg_in = update(
        vs.Norg_in,
        at[:, :], vs.NORG_IN[:, :, vs.itt] * vs.maskCatch,
    )

    return KernelOutput(
        M_in=vs.M_in,
        C_in=vs.C_in,
        Nmin_in=vs.Nmin_in,
        Norg_in=vs.Norg_in,
    )


@roger_kernel
def set_states_kernel(state):
    vs = state.variables

    vs.inf_mat_rz = update(vs.inf_mat_rz, at[:, :], vs.INF_MAT_RZ[:, :, vs.itt])
    vs.inf_pf_rz = update(vs.inf_pf_rz, at[:, :], vs.INF_PF_RZ[:, :, vs.itt])
    vs.inf_pf_ss = update(vs.inf_pf_ss, at[:, :], vs.INF_PF_SS[:, :, vs.itt])
    vs.transp = update(vs.transp, at[:, :], vs.TRANSP[:, :, vs.itt])
    vs.evap_soil = update(vs.evap_soil, at[:, :], vs.EVAP_SOIL[:, :, vs.itt])
    vs.q_rz = update(vs.q_rz, at[:, :], vs.Q_RZ[:, :, vs.itt])
    vs.q_ss = update(vs.q_ss, at[:, :], vs.Q_SS[:, :, vs.itt])
    vs.cpr_rz = update(vs.cpr_ss, at[:, :], vs.CPR_RZ[:, :, vs.itt])

    vs.S_rz = update(vs.S_rz, at[:, :, vs.tau], vs.S_RZ[:, :, vs.itt])
    vs.S_ss = update(vs.S_ss, at[:, :, vs.tau], vs.S_SS[:, :, vs.itt])
    vs.S_s = update(vs.S_s, at[:, :, vs.tau], vs.S_S[:, :, vs.itt])

    return KernelOutput(
        inf_mat_rz=vs.inf_mat_rz,
        inf_pf_rz=vs.inf_pf_rz,
        inf_pf_ss=vs.inf_pf_ss,
        transp=vs.transp,
        evap_soil=vs.evap_soil,
        q_rz=vs.q_rz,
        q_ss=vs.q_ss,
        cpr_rz=vs.cpr_rz,
        S_rz=vs.S_rz,
        S_ss=vs.S_ss,
    )


@roger_kernel
def after_timestep_kernel(state):
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[:, :, vs.taum1, :], vs.SA_rz[:, :, vs.tau, :],
    )
    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, vs.taum1, :], vs.sa_rz[:, :, vs.tau, :],
    )
    vs.MSA_rz = update(
        vs.MSA_rz,
        at[:, :, vs.taum1, :], vs.MSA_rz[:, :, vs.tau, :],
    )
    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, vs.taum1, :], vs.msa_rz[:, :, vs.tau, :],
    )
    vs.M_rz = update(
        vs.M_rz,
        at[:, :, vs.taum1], vs.M_rz[:, :, vs.tau],
    )
    vs.C_rz = update(
        vs.C_rz,
        at[:, :, vs.taum1], vs.C_rz[:, :, vs.tau],
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[:, :, vs.taum1, :], vs.SA_ss[:, :, vs.tau, :],
    )
    vs.sa_ss = update(
        vs.sa_ss,
        at[:, :, vs.taum1, :], vs.sa_ss[:, :, vs.tau, :],
    )
    vs.MSA_ss = update(
        vs.MSA_ss,
        at[:, :, vs.taum1, :], vs.MSA_ss[:, :, vs.tau, :],
    )
    vs.msa_ss = update(
        vs.msa_ss,
        at[:, :, vs.taum1, :], vs.msa_ss[:, :, vs.tau, :],
    )
    vs.M_ss = update(
        vs.M_ss,
        at[:, :, vs.taum1], vs.M_ss[:, :, vs.tau],
    )
    vs.C_ss = update(
        vs.C_ss,
        at[:, :, vs.taum1], vs.C_ss[:, :, vs.tau],
    )
    vs.SA_s = update(
        vs.SA_s,
        at[:, :, vs.taum1, :], vs.SA_s[:, :, vs.tau, :],
    )
    vs.sa_s = update(
        vs.sa_s,
        at[:, :, vs.taum1, :], vs.sa_s[:, :, vs.tau, :],
    )
    vs.MSA_s = update(
        vs.MSA_s,
        at[:, :, vs.taum1, :], vs.MSA_s[:, :, vs.tau, :],
    )
    vs.msa_s = update(
        vs.msa_s,
        at[:, :, vs.taum1, :], vs.msa_s[:, :, vs.tau, :],
    )
    vs.M_s = update(
        vs.M_s,
        at[:, :, vs.taum1], vs.M_s[:, :, vs.tau],
    )
    vs.C_s = update(
        vs.C_s,
        at[:, :, vs.taum1], vs.C_s[:, :, vs.tau],
    )

    return KernelOutput(
        SA_rz=vs.SA_rz,
        sa_rz=vs.sa_rz,
        MSA_rz=vs.MSA_rz,
        msa_rz=vs.msa_rz,
        M_rz=vs.M_rz,
        C_rz=vs.C_rz,
        SA_ss=vs.SA_ss,
        sa_ss=vs.sa_ss,
        MSA_ss=vs.MSA_ss,
        msa_ss=vs.msa_ss,
        M_ss=vs.M_ss,
        C_ss=vs.C_ss,
        SA_s=vs.SA_s,
        sa_s=vs.sa_s,
        MSA_s=vs.MSA_s,
        msa_s=vs.msa_s,
        M_s=vs.M_s,
        C_s=vs.C_s,
        )


@roger_kernel
def after_timestep_nitrate_kernel(state):
    vs = state.variables

    vs.Nmin_rz = update(
        vs.Nmin_rz,
        at[:, :, vs.taum1, :], vs.Nmin_rz[:, :, vs.tau, :],
        )

    vs.Nmin_ss = update(
        vs.Nmin_ss,
        at[:, :, vs.taum1, :], vs.Nmin_ss[:, :, vs.tau, :],
        )

    vs.Nmin_s = update(
        vs.Nmin_s,
        at[:, :, vs.taum1, :], vs.Nmin_s[:, :, vs.tau, :],
        )

    return KernelOutput(
        Nmin_rz=vs.Nmin_rz,
        Nmin_ss=vs.Nmin_ss,
        Nmin_s=vs.Nmin_s
        )


@roger_kernel
def calc_conc_iso_storage(state, sa, msa):
    """Calculates isotope signal of storage.
    """
    vs = state.variables

    mask = npx.isfinite(msa[:, :, vs.tau, :])
    vals = allocate(state.dimensions, ("x", "y", "ages"))
    weights = allocate(state.dimensions, ("x", "y", "ages"))
    vals = update(
        vals,
        at[:, :, :], npx.where(mask, msa[:, :, vs.tau, :], 0),
    )
    weights = update(
        weights,
        at[:, :, :], npx.where(sa[:, :, vs.tau, :] * mask > 0, sa[:, :, vs.tau, :] / npx.sum(sa[:, :, vs.tau, :] * mask, axis=-1)[:, :, npx.newaxis], 0),
    )
    conc = allocate(state.dimensions, ("x", "y"))
    # calculate weighted average
    conc = update(
        conc,
        at[:, :], npx.sum(vals * weights, axis=-1),
    )
    conc = update(
        conc,
        at[:, :], npx.where(conc != 0, conc, npx.NaN),
    )

    return conc


@roger_kernel
def _ffill_3d(state, arr):
    idx_shape = tuple([slice(None)] + [npx.newaxis] * (3 - 2 - 1))
    idx = allocate(state.dimensions, ("x", "y", "t"), dtype=int)
    arr1 = allocate(state.dimensions, ("x", 1, 1), dtype=int)
    arr2 = allocate(state.dimensions, (1, "y", 1), dtype=int)
    arr3 = allocate(state.dimensions, ("x", "y", "t"), dtype=int)
    arr_fill = allocate(state.dimensions, ("x", "y", "t"))
    idx = update(
        idx,
        at[:, :, :], npx.where(npx.isfinite(arr), npx.arange(npx.shape(arr)[2])[idx_shape], 0),
    )
    idx = update(
        idx,
        at[:, :, :], npx.maximum.accumulate(idx, axis=-1),
    )
    arr1 = update(
        arr1,
        at[:, 0, 0], npx.arange(npx.shape(arr)[0]),
    )
    arr2 = update(
        arr2,
        at[0, :, 0], npx.arange(npx.shape(arr)[1]),
    )
    arr3 = update(
        arr3,
        at[:, :, :], idx,
    )
    arr_fill = update(
        arr_fill,
        at[:, :, :], arr[arr1, arr2, arr3],
    )

    return arr_fill


model = DISTTRANSPORTSetup()
model.setup()
model.warmup()
model.run()
