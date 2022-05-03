from pathlib import Path
import glob
import os
import datetime
import h5netcdf
import matplotlib.pyplot as plt
import xarray as xr
from cftime import num2date
import pandas as pd
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
from roger.core.operators import numpy as npx, update, at
from roger.tools.setup import write_forcing_tracer
import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs
from roger.io_tools import yml
import numpy as onp


class SVATTRANSPORTSetup(RogerSetup):
    """A SVAT transport model.
    """
    _base_path = Path(__file__).parent
    _bounds = None
    _params = None
    _nrows = None
    _tm_structure = None

    def _read_var_from_nc(self, var, file):
        nc_file = self._base_path / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables[var]
            return npx.array(var_obj)

    def _get_nitt(self):
        nc_file = self._base_path / 'states_hm.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['Time']
            return len(onp.array(var_obj))

    def _get_nx(self):
        nc_file = self._base_path / 'states_hm.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['x']
            return len(onp.array(var_obj))

    def _get_runlen(self):
        nc_file = self._base_path / 'states_hm.nc'
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['Time']
            return len(onp.array(var_obj)) * 60 * 60 * 24

    def _read_config(self):
        config_file = self._base_path / "config.yml"
        config = yml.Config(config_file)
        return config

    def _set_tm_structure(self, tm_structure):
        self._tm_structure = tm_structure

    def _make_params(self, nsamples):
        if self._tm_structure == "preferential":
            self._bounds = {
                'num_vars': 3,
                'names': ['b_transp', 'b_q_rz', 'b_q_ss'],
                'bounds': [[1, 100],
                           [1, 100],
                           [1, 100]]
            }
            self._params = saltelli.sample(self._bounds, nsamples)
            self._nrows = self._params.shape[0]

        elif self._tm_structure == "advection-dispersion":
            self._bounds = {
                'num_vars': 3,
                'names': ['b_transp', 'a_q_rz', 'a_q_ss'],
                'bounds': [[1, 100],
                           [1, 100],
                           [1, 100]]
            }
            self._params = saltelli.sample(self._bounds, nsamples)
            self._nrows = self._params.shape[0]

        elif self._tm_structure == "complete-mixing + advection-dispersion":
            self._bounds = {
                'num_vars': 2,
                'names': ['a_q_rz', 'a_q_ss'],
                'bounds': [[1, 100],
                           [1, 100]]
            }
            self._params = saltelli.sample(self._bounds, nsamples)
            self._nrows = self._params.shape[0]

        elif self._tm_structure == "time-variant preferential":
            self._bounds = {
                'num_vars': 3,
                'names': ['b_transp', 'b_q_rz', 'b_q_ss'],
                'bounds': [[1, 100],
                           [1, 100],
                           [1, 100]]
            }
            self._params = saltelli.sample(self._bounds, nsamples)
            self._nrows = self._params.shape[0]

        elif self._tm_structure == "time-variant advection-dispersion":
            self._bounds = {
                'num_vars': 3,
                'names': ['b_transp', 'a_q_rz', 'a_q_ss'],
                'bounds': [[1, 100],
                           [1, 100],
                           [1, 100]]
            }
            self._params = saltelli.sample(self._bounds, nsamples)
            self._nrows = self._params.shape[0]

        elif self._tm_structure == "time-variant":
            self._bounds = {
                'num_vars': 6,
                'names': ['a_transp', 'b_transp', 'a_q_rz', 'b_q_rz', 'a_q_ss', 'b_q_ss'],
                'bounds': [[1, 100],
                           [1, 100],
                           [1, 100],
                           [1, 100],
                           [1, 100],
                           [1, 100]]
            }
            self._params = saltelli.sample(self._bounds, nsamples)
            self._nrows = self._params.shape[0]

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "SVATTRANSPORT"

        settings.nx, settings.ny, settings.nz = self._get_nx(), 1, 1
        settings.nitt = self._get_nitt()
        settings.ages = settings.nitt
        settings.nages = settings.nitt + 1
        settings.runlen = self._get_runlen()

        # lysimeter surface 3.14 square meter (2m diameter)
        settings.dx = 2
        settings.dy = 2
        settings.dz = 1

        settings.x_origin = 0.0
        settings.y_origin = 0.0

        settings.enable_offline_transport = True
        settings.enable_oxygen18 = True
        settings.tm_structure = self._tm_structure

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

        if settings.tm_structure == "complete-mixing":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 1)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 1)
        elif settings.tm_structure == "piston":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 21)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 22)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 22)
        elif settings.tm_structure == "preferential":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 1], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, :, 2], npx.repeat(self._params[:, 0, npx.newaxis], vs.sas_params_transp.shape[1], axis=-1))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 1], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, :, 2], npx.repeat(self._params[:, 1, npx.newaxis], vs.sas_params_q_rz.shape[1], axis=-1))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 1], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, :, 2], npx.repeat(self._params[:, 2, npx.newaxis], vs.sas_params_q_ss.shape[1], axis=-1))
        elif settings.tm_structure == "advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 1], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, :, 2], npx.repeat(self._params[:, 0, npx.newaxis], vs.sas_params_transp.shape[1], axis=-1))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 1], npx.repeat(self._params[:, 1, npx.newaxis], vs.sas_params_q_rz.shape[1], axis=-1))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 2], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 1], npx.repeat(self._params[:, 2, npx.newaxis], vs.sas_params_q_ss.shape[1], axis=-1))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 2], 1)
        elif settings.tm_structure == "complete-mixing + advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 1], npx.repeat(self._params[:, 0, npx.newaxis], vs.sas_params_q_rz.shape[1], axis=-1))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 2], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 1], npx.repeat(self._params[:, 1, npx.newaxis], vs.sas_params_q_ss.shape[1], axis=-1))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 2], 1)
        elif settings.tm_structure == "time-variant advection-disperison":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 31)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, :, 4], npx.repeat(self._params[:, 0, npx.newaxis], vs.sas_params_transp.shape[1], axis=-1))
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 6], vs.S_sat_rz - vs.S_pwp_rz)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 32)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 4], npx.repeat(self._params[:, 1, npx.newaxis], vs.sas_params_q_rz.shape[1], axis=-1))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 6], vs.S_sat_rz - vs.S_pwp_rz)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 32)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 4], npx.repeat(self._params[:, 2, npx.newaxis], vs.sas_params_q_ss.shape[1], axis=-1))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 6], vs.S_sat_ss - vs.S_pwp_ss)
        elif settings.tm_structure == "time-variant preferential":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 31)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, :, 4], npx.repeat(self._params[:, 0, npx.newaxis], vs.sas_params_transp.shape[1], axis=-1))
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 6], vs.S_sat_rz - vs.S_pwp_rz)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 31)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 4], npx.repeat(self._params[:, 1, npx.newaxis], vs.sas_params_q_rz.shape[1], axis=-1))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 6], vs.S_sat_rz - vs.S_pwp_rz)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 31)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 4], npx.repeat(self._params[:, 2, npx.newaxis], vs.sas_params_q_ss.shape[1], axis=-1))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 6], vs.S_sat_ss - vs.S_pwp_ss)
        elif settings.tm_structure == "time-variant":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[:, :, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[:, :, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 0], 35)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, :, 4], npx.repeat(self._params[:, 0, npx.newaxis], vs.sas_params_transp.shape[1], axis=-1))
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[:, :, 6], vs.S_sat_rz - vs.S_pwp_rz)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 0], 35)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 4], npx.repeat(self._params[:, 1, npx.newaxis], vs.sas_params_q_rz.shape[1], axis=-1))
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[:, :, 6], vs.S_sat_rz - vs.S_pwp_rz)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 0], 35)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 4], npx.repeat(self._params[:, 2, npx.newaxis], vs.sas_params_q_ss.shape[1], axis=-1))
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[:, :, 6], vs.S_sat_ss - vs.S_pwp_ss)

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

        if (settings.enable_oxygen18 | settings.enable_deuterium):
            vs.C_rz = update(vs.C_rz, at[:, :, :2], -13)
            vs.C_ss = update(vs.C_ss, at[:, :, :2], -7)
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
            vs.Q_RZ = update(vs.Q_RZ, at[:, :, :], self._read_var_from_nc("q_rz", 'states_hm.nc'))
            vs.Q_SS = update(vs.Q_SS, at[:, :, :], self._read_var_from_nc("q_ss", 'states_hm.nc'))

            if settings.enable_deuterium:
                vs.C_IN = update(vs.C_IN, at[:, :, :], self._read_var_from_nc("d2H", 'forcing_tracer.nc'))

            if settings.enable_oxygen18:
                vs.C_IN = update(vs.C_IN, at[:, :, :], self._read_var_from_nc("d18O", 'forcing_tracer.nc'))

            if settings.enable_deuterium or settings.enable_oxygen18:
                vs.update(set_iso_input_kernel(state))

        vs.update(set_states_kernel(state))

        if settings.enable_deuterium or settings.enable_oxygen18:
            vs.update(set_forcing_iso_kernel(state))

    @roger_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics

        diagnostics["averages"].output_variables = ["C_q_ss"]
        diagnostics["averages"].output_frequency = 24 * 60 * 60
        diagnostics["averages"].sampling_frequency = 1

        diagnostics["constant"].output_variables = ["sas_params_transp", "sas_params_q_rz", "sas_params_q_ss"]
        diagnostics["constant"].output_frequency = 0
        diagnostics["constant"].sampling_frequency = 1

    @roger_routine
    def after_timestep(self, state):
        vs = state.variables

        vs.update(after_timestep_kernel(state))


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


dict_params_eff = {}
tm_structures = ['preferential', 'advection-dispersion',
                 'complete-mixing advection-dispersion',
                 'time-variant preferential',
                 'time-variant advection-dispersion']
for tm_structure in tm_structures:
    tms = tm_structure.replace(" ", "_")
    model = SVATTRANSPORTSetup()
    input_path = model._base_path / "input"
    write_forcing_tracer(input_path, 'd18O', nrows=model._get_nx(), ncols=5)
    model._set_tm_structure(tm_structure)
    model.setup()
    model.warmup()
    model.run()

    # directory of results
    base_path_results = model._base_path / "results"
    if not os.path.exists(base_path_results):
        os.mkdir(base_path_results)

    # merge model output into single file
    path = str(model._base_path / f"{model.state.settings.identifier}.*.nc")
    diag_files = glob.glob(path)
    states_tm_file = model._base_path / f"states_tm_{tms}_sensitivity.nc"
    with h5netcdf.File(states_tm_file, 'w', decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title=f'RoGeR {tm_structure} transport model saltelli results at Rietholzbach Lysimeter site',
            institution='University of Freiburg, Chair of Hydrology',
            references='',
            comment=f'SVAT {tm_structure} transport model with free drainage'
        )
        for dfs in diag_files:
            with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                # set dimensions with a dictionary
                if not f.dimensions:
                    f.dimensions = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
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
                    with h5netcdf.File(model._base_path / 'forcing_tracer.nc', "r", decode_vlen_strings=False) as infile:
                        time_origin = infile.variables['time'].attrs['time_origin']
                    v.attrs.update(time_origin=time_origin,
                                   units=var_obj.attrs["units"])
                    v[:] = npx.array(var_obj)
                    v = f.create_variable('ages', ('ages',), float)
                    v.attrs['long_name'] = 'Water ages'
                    v.attrs['units'] = 'days'
                    v[:] = npx.arange(1, f.dimensions["ages"]+1)
                    v = f.create_variable('nages', ('nages',), float)
                    v.attrs['long_name'] = 'Water ages (cumulated)'
                    v.attrs['units'] = 'days'
                    v[:] = npx.arange(0, f.dimensions["nages"])
                    v = f.create_variable('n_sas_params', ('n_sas_params',), float)
                    v.attrs['long_name'] = 'Number of SAS parameters'
                    v.attrs['units'] = ''
                    v[:] = npx.arange(0, f.dimensions["n_sas_params"])
                for var_sim in list(df.variables.keys()):
                    var_obj = df.variables.get(var_sim)
                    if var_sim not in list(f.dimensions.keys()) and "Time" in list(var_obj.dimensions.keys()):
                        v = f.create_variable(var_sim, ('x', 'y', 'Time'), float)
                        vals = npx.array(var_obj)
                        v[:, :, :] = vals.swapaxes(0, 2)
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and var_obj.shape[-1] == f.dimensions["n_sas_params"]:
                        v = f.create_variable(var_sim, ('x', 'y', 'n_sas_params'), float)
                        vals = npx.array(var_obj)
                        v = f.create_variable(var_sim, ('x', 'y', 'n_sas_params'), float)
                        vals = npx.array(var_obj)
                        vals = vals.swapaxes(0, 3)
                        vals = vals.swapaxes(1, 2)
                        vals = vals.swapaxes(2, 3)
                        v[:, :, :] = vals[0, :, :, :]
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and "ages" in var_obj.dimensions:
                        v = f.create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float)
                        vals = npx.array(var_obj)
                        vals = vals.swapaxes(0, 3)
                        vals = vals.swapaxes(1, 2)
                        vals = vals.swapaxes(2, 3)
                        v[:, :, :, :] = vals
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])
                    elif var_sim not in list(f.dimensions.keys()) and "nages" in var_obj.dimensions:
                        v = f.create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float)
                        vals = npx.array(var_obj)
                        vals = vals.swapaxes(0, 3)
                        vals = vals.swapaxes(1, 2)
                        vals = vals.swapaxes(2, 3)
                        v[:, :, :, :] = vals
                        v.attrs.update(long_name=var_obj.attrs["long_name"],
                                       units=var_obj.attrs["units"])

    # load simulation
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    states_hm_file = model._base_path / "states_hm.nc"
    ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")

    # load observations (measured data)
    path_obs = Path("/Users/robinschwemmle/Desktop/PhD/data/plot/rietholzbach/rietholzbach_lysimeter.nc")
    ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")

    # assign date
    time_origin = ds_sim_tm['Time'].attrs['time_origin']
    days = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim = num2date(days, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    date_obs = num2date(days, units=f"days since {ds_obs['time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim))
    ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim))
    ds_obs = ds_obs.assign_coords(date=("Time", date_obs))

    # DataFrame with sampled model parameters and the corresponding metrics
    nx = model.state.settings.nx  # number of rows
    ny = model.state.settings.ny  # number of columns
    df_params_eff = pd.DataFrame(index=range(nx * ny))
    # sampled model parameters
    vs = model.state.variables
    if tm_structure == "preferential":
        df_params_eff.loc[:, 'b_transp'] = vs.sas_params_transp[2:-2, 2:-2, 2].flatten()
        df_params_eff.loc[:, 'b_q_rz'] = vs.sas_params_q_rz[2:-2, 2:-2, 2].flatten()
        df_params_eff.loc[:, 'b_q_ss'] = vs.sas_params_q_rz[2:-2, 2:-2, 2].flatten()
    elif tm_structure == "advection-dispersion":
        df_params_eff.loc[:, 'b_transp'] = vs.sas_params_transp[2:-2, 2:-2, 2].flatten()
        df_params_eff.loc[:, 'a_q_rz'] = vs.sas_params_q_rz[2:-2, 2:-2, 1].flatten()
        df_params_eff.loc[:, 'a_q_ss'] = vs.sas_params_q_rz[2:-2, 2:-2, 1].flatten()
    elif tm_structure == "complete-mixing advection-dispersion":
        df_params_eff.loc[:, 'a_q_rz'] = vs.sas_params_q_rz[2:-2, 2:-2, 1].flatten()
        df_params_eff.loc[:, 'a_q_ss'] = vs.sas_params_q_rz[2:-2, 2:-2, 1].flatten()
    elif tm_structure == "time-variant advection-dispersion":
        df_params_eff.loc[:, 'b_transp'] = vs.sas_params_transp[2:-2, 2:-2, 4].flatten()
        df_params_eff.loc[:, 'a_q_rz'] = vs.sas_params_q_rz[2:-2, 2:-2, 4].flatten()
        df_params_eff.loc[:, 'a_q_ss'] = vs.sas_params_q_rz[2:-2, 2:-2, 4].flatten()
    elif tm_structure == "time-variant preferential":
        df_params_eff.loc[:, 'b_transp'] = vs.sas_params_transp[2:-2, 2:-2, 4].flatten()
        df_params_eff.loc[:, 'a_q_rz'] = vs.sas_params_q_rz[2:-2, 2:-2, 4].flatten()
        df_params_eff.loc[:, 'a_q_ss'] = vs.sas_params_q_rz[2:-2, 2:-2, 4].flatten()
    elif tm_structure == "time-variant":
        df_params_eff.loc[:, 'b_transp'] = vs.sas_params_transp[2:-2, 2:-2, 4].flatten()
        df_params_eff.loc[:, 'a_q_rz'] = vs.sas_params_q_rz[2:-2, 2:-2, 4].flatten()
        df_params_eff.loc[:, 'a_q_ss'] = vs.sas_params_q_rz[2:-2, 2:-2, 4].flatten()

    # compare observations and simulations
    ncol = 0
    idx = ds_sim_tm.Time  # time index
    for nrow in range(nx):
        # calculate simulated oxygen-18 composite sample
        df_perc_18O_obs = pd.DataFrame(index=idx, columns=['perc_obs', 'd18O_perc_obs'])
        df_perc_18O_obs.loc[:, 'perc_obs'] = ds_obs['PERC'].isel(x=nrow, y=ncol).values
        df_perc_18O_obs.loc[:, 'd18O_perc_obs'] = ds_obs['d18O_perc'].isel(x=nrow, y=ncol).values
        sample_no = pd.DataFrame(index=df_perc_18O_obs.dropna().index, columns=['sample_no'])
        sample_no = sample_no.loc['1997':'2007']
        sample_no['sample_no'] = range(len(sample_no.index))
        df_perc_18O_sim = pd.DataFrame(index=idx, columns=['perc_sim', 'd18O_perc_sim'])
        df_perc_18O_sim['perc_sim'] = ds_sim_hm['q_ss'].isel(x=nrow, y=ncol).values
        df_perc_18O_sim['d18O_perc_sim'] = ds_sim_tm['C_q_ss'].isel(x=nrow, y=ncol).values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no)
        df_perc_18O_sim.loc[:, 'sample_no'] = df_perc_18O_sim.loc[:, 'sample_no'].fillna(method='bfill', limit=14)
        perc_sum = df_perc_18O_sim.groupby(['sample_no']).sum().loc[:, 'perc_sim']
        sample_no['perc_sum'] = perc_sum.values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_sum'])
        df_perc_18O_sim.loc[:, 'perc_sum'] = df_perc_18O_sim.loc[:, 'perc_sum'].fillna(method='bfill', limit=14)
        df_perc_18O_sim['weight'] = df_perc_18O_sim['perc_sim'] / df_perc_18O_sim['perc_sum']
        df_perc_18O_sim['d18O_weight'] = df_perc_18O_sim['d18O_perc_sim'] * df_perc_18O_sim['weight']
        d18O_sample = df_perc_18O_sim.groupby(['sample_no']).sum().loc[:, 'd18O_weight']
        sample_no['d18O_sample'] = d18O_sample.values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no['d18O_sample'])
        df_perc_18O_sim.loc[:, 'd18O_sample'] = df_perc_18O_sim.loc[:, 'd18O_sample'].fillna(method='bfill', limit=14)
        cond = (df_perc_18O_sim['d18O_sample'] == 0)
        df_perc_18O_sim.loc[cond, 'd18O_sample'] = onp.NaN
        d18O_perc_cs = onp.zeros((1, 1, len(idx)))
        d18O_perc_cs[nrow, ncol, :] = df_perc_18O_sim.loc[:, 'd18O_sample'].values
        ds_sim_tm.assign(d18O_perc_cs=d18O_perc_cs)
        # calculate observed oxygen-18 composite sample
        df_perc_18O_obs.loc[:, 'd18O_perc_cs'] = df_perc_18O_obs['d18O_perc'].fillna(method='bfill', limit=14)

        perc_sample_sum_obs = df_perc_18O_sim.join(df_perc_18O_obs).groupby(['sample_no']).sum().loc[:, 'perc_obs']
        sample_no['perc_obs_sum'] = perc_sample_sum_obs.values
        df_perc_18O_sim = df_perc_18O_sim.join(sample_no['perc_obs_sum'])
        df_perc_18O_sim.loc[:, 'perc_obs_sum'] = df_perc_18O_sim.loc[:, 'perc_obs_sum'].fillna(method='bfill', limit=14)

        # join observations on simulations
        obs_vals = ds_obs['d18O_perc'].isel(x=nrow, y=ncol).values
        sim_vals = ds_sim_tm['d18O_perc_cs'].isel(x=nrow, y=ncol).values
        df_obs = pd.DataFrame(index=date_obs, columns=['obs'])
        df_obs.loc[:, 'obs'] = ds_obs['d18O_perc'].isel(x=nrow, y=ncol).values
        df_eval = eval_utils.join_obs_on_sim(date_sim, sim_vals, df_obs)
        df_eval = df_eval.dropna()

        # calculate metrics
        var_sim = 'C_q_ss'
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

    # write to .txt
    file = base_path_results / f"params_eff_{tm_structure}.txt"
    df_params_eff.to_csv(file, header=True, index=False, sep="\t")
    dict_params_eff[tm_structure] = df_params_eff

    # perform sensitivity analysis
    df_params = df_params_eff.loc[:, model._bounds['names']]
    n_params = len(model._bounds['names'])
    df_eff = df_params_eff.iloc[:, n_params:]
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
    _LABS = {'KGE_C_q_ss': 'KGE',
             'KGE_beta_C_q_ss': '$\beta$',
             'KGE_alpha_C_q_ss': '$\alpha$',
             'r_C_q_ss': 'r',
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
    file = base_path_figs / f"sobol_indices_{tms}.png"
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
        xlabel = labs._LABS[model._bounds['names'][j].split(' ')[0]]
        ax[-1, j].set_xlabel(xlabel)

    ax[0, 0].set_ylabel(r'$KGE$ [-]')
    ax[1, 0].set_ylabel(r'$\alpha$ [-]')
    ax[2, 0].set_ylabel(r'$\beta$ [-]')
    ax[3, 0].set_ylabel(r'r [-]')

    fig.subplots_adjust(wspace=0.2, hspace=0.3)
    file = base_path_figs / f"dotty_plots_{tms}.png"
    fig.savefig(file, dpi=250)
