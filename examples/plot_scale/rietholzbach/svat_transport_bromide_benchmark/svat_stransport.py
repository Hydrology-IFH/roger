from pathlib import Path
import glob
import os
import datetime
import h5netcdf
import xarray as xr
from cftime import num2date
import pandas as pd

from roger import runtime_settings as rs
rs.backend = "numpy"
rs.force_overwrite = True
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at, where
from roger.tools.setup import write_forcing_tracer
import numpy as onp


class SVATTRANSPORTSetup(RogerSetup):
    """A SVAT transport model.
    """
    _base_path = Path(__file__).parent
    _year = None
    _tm_structure = None
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
        with xr.open_dataset(nc_file, engine="h5netcdf") as ds:
            days = (ds['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date = num2date(days, units=f"days since {ds['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds = ds.assign_coords(Time=("Time", date))
            ds_year = ds.sel(Time=slice(f'{self._year}-01-01', f'{self._year + 1}-12-31'))
            vals = ds_year[var].values

        return vals

    def _get_nitt(self, path_dir, file):
        nc_file = path_dir / file
        with xr.open_dataset(nc_file, engine="h5netcdf") as ds:
            days = (ds['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date = num2date(days, units=f"days since {ds['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds = ds.assign_coords(Time=("Time", date))
            ds_year = ds.sel(Time=slice(f'{self._year}-01-01', f'{self._year + 1}-12-31'))
            return len(ds_year['Time'].values)

    def _get_runlen(self, path_dir, file):
        nc_file = path_dir / file
        with xr.open_dataset(nc_file, engine="h5netcdf") as ds:
            days = (ds['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
            date = num2date(days, units=f"days since {ds['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
            ds = ds.assign_coords(Time=("Time", date))
            ds_year = ds.sel(Time=slice(f'{self._year}-01-01', f'{self._year + 1}-12-31'))
            return len(ds_year['Time'].values) * 60 * 60 * 24

    def _set_year(self, year):
        self._year = year

    def _set_tm_structure(self, tm_structure):
        self._tm_structure = tm_structure

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "SVATTRANSPORT"

        settings.nx, settings.ny, settings.nz = 1, 1, 1
        settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
        settings.ages = settings.nitt
        settings.nages = settings.nitt + 1
        settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')

        # lysimeter surface 3.14 square meter (2m diameter)
        settings.dx = 2
        settings.dy = 2
        settings.dz = 1

        settings.x_origin = 0.0
        settings.y_origin = 0.0
        settings.time_origin = f"{self._year - 1}-12-31 00:00:00"

        settings.enable_offline_transport = True
        settings.enable_bromide = True
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
        # grid of model runs
        dx = allocate(state.dimensions, ("x"))
        dx = update(dx, at[:], 1)
        dy = allocate(state.dimensions, ("y"))
        dy = update(dy, at[:], 1)
        vs.x = update(vs.x, at[3:-2], npx.cumsum(dx[3:-2]))
        vs.y = update(vs.y, at[3:-2], npx.cumsum(dy[3:-2]))

    @roger_routine
    def set_look_up_tables(self, state):
        pass

    @roger_routine
    def set_topography(self, state):
        pass

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "S_PWP_RZ",
            "S_PWP_SS",
            "S_SAT_RZ",
            "S_SAT_SS",
            "S_pwp_rz",
            "S_pwp_ss",
            "S_sat_rz",
            "S_sat_ss",
            "alpha_transp",
            "alpha_q",
            "sas_params_evap_soil",
            "sas_params_cpr_rz",
            "sas_params_transp",
            "sas_params_q_rz",
            "sas_params_q_ss",
        ],
    )
    def set_parameters_setup(self, state):
        vs = state.variables
        settings = state.settings

        vs.S_PWP_RZ = update(vs.S_PWP_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc'))
        vs.S_PWP_SS = update(vs.S_PWP_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc'))
        vs.S_SAT_RZ = update(vs.S_SAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.nc'))
        vs.S_SAT_SS = update(vs.S_SAT_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.nc'))

        vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], vs.S_PWP_RZ[2:-2, 2:-2, 0])
        vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], vs.S_PWP_SS[2:-2, 2:-2, 0])
        vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], vs.S_SAT_RZ[2:-2, 2:-2, 0])
        vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], vs.S_SAT_SS[2:-2, 2:-2, 0])

        vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], 0.5)
        vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], 0.3)

        if settings.tm_structure == "complete-mixing":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 1)
        elif settings.tm_structure == "piston":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 22)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 22)
        elif settings.tm_structure == "preferential":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], 30)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 2)
        elif settings.tm_structure == "advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], 30)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 2)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
        elif settings.tm_structure == "complete-mixing + advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 2)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
        elif settings.tm_structure == "time-variant advection-disperison":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], 30)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 32)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], 2)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 32)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
        elif settings.tm_structure == "time-variant preferential":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], 30)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], 2)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
        elif settings.tm_structure == "time-variant":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 35)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], 30)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 35)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 35)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], 2)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])

    @roger_routine
    def set_parameters(self, state):
        pass

    @roger_routine
    def set_initial_conditions_setup(self, state):
        vs = state.variables

        vs.S_RZ = update(vs.S_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc'))
        vs.S_SS = update(vs.S_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc'))

    @roger_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
        vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], vs.S_RZ[2:-2, 2:-2, 0, npx.newaxis] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
        vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], vs.S_SS[2:-2, 2:-2, 0, npx.newaxis] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
        vs.S_s = update(vs.S_s, at[2:-2, 2:-2, :vs.taup1], vs.S_S[2:-2, 2:-2, 0, npx.newaxis] - (vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis] + vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis]))

        arr0 = allocate(state.dimensions, ("x", "y"))
        vs.sa_rz = update(
            vs.sa_rz,
            at[2:-2, 2:-2, :vs.taup1, 1:], npx.diff(npx.linspace(arr0[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, vs.tau], settings.ages, axis=-1), axis=-1)[:, :, npx.newaxis, :],
        )
        vs.sa_ss = update(
            vs.sa_ss,
            at[2:-2, 2:-2, :vs.taup1, 1:], npx.diff(npx.linspace(arr0[2:-2, 2:-2], vs.S_ss[2:-2, 2:-2, vs.tau], settings.ages, axis=-1), axis=-1)[:, :, npx.newaxis, :],
        )

        vs.SA_rz = update(
            vs.SA_rz,
            at[2:-2, 2:-2, :, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, :, :], axis=-1),
        )

        vs.SA_ss = update(
            vs.SA_ss,
            at[2:-2, 2:-2, :, 1:], npx.cumsum(vs.sa_rz[2:-2, 2:-2, :, :], axis=-1),
        )

        vs.sa_s = update(
            vs.sa_s,
            at[2:-2, 2:-2, :, :], vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :],
        )
        vs.SA_s = update(
            vs.SA_s,
            at[2:-2, 2:-2, :, 1:], npx.cumsum(vs.sa_s[2:-2, 2:-2, :, :], axis=-1),
        )

    @roger_routine
    def set_forcing_setup(self, state):
        vs = state.variables

        vs.TA = update(vs.TA, at[2:-2, 2:-2, :], self._read_var_from_nc("ta", self._base_path, 'states_hm.nc'))
        vs.PREC = update(vs.PREC, at[2:-2, 2:-2, :], self._read_var_from_nc("prec", self._base_path, 'states_hm.nc'))
        vs.INF_MAT_RZ = update(vs.INF_MAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mat_rz", self._base_path, 'states_hm.nc'))
        vs.INF_PF_RZ = update(vs.INF_PF_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mp_rz", self._base_path, 'states_hm.nc') + self._read_var_from_nc("inf_sc_rz", self._base_path, 'states_hm.nc'))
        vs.INF_PF_SS = update(vs.INF_PF_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_ss", self._base_path, 'states_hm.nc'))
        vs.TRANSP = update(vs.TRANSP, at[2:-2, 2:-2, :], self._read_var_from_nc("transp", self._base_path, 'states_hm.nc'))
        vs.EVAP_SOIL = update(vs.EVAP_SOIL, at[2:-2, 2:-2, :], self._read_var_from_nc("evap_soil", self._base_path, 'states_hm.nc'))
        vs.CPR_RZ = update(vs.CPR_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("cpr_rz", self._base_path, 'states_hm.nc'))
        vs.Q_RZ = update(vs.Q_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("q_rz", self._base_path, 'states_hm.nc'))
        vs.Q_SS = update(vs.Q_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("q_ss", self._base_path, 'states_hm.nc'))

        vs.M_IN = update(vs.M_IN, at[2:-2, 2:-2, :], self._read_var_from_nc("Br", self._input_dir, 'forcing_tracer.nc'))

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables

        vs.update(set_states_kernel(state))
        vs.update(set_forcing_bromide_kernel(state))

    @roger_routine
    def set_diagnostics(self, state):
        diagnostics = state.diagnostics

        diagnostics["rates"].output_variables = ["M_q_ss"]
        diagnostics["rates"].output_frequency = 24 * 60 * 60
        diagnostics["rates"].sampling_frequency = 1

        diagnostics["averages"].output_variables = ["C_rz", "C_ss", "C_s", "C_q_ss"]
        diagnostics["averages"].output_frequency = 24 * 60 * 60
        diagnostics["averages"].sampling_frequency = 1

    @roger_routine
    def after_timestep(self, state):
        vs = state.variables

        vs.update(after_timestep_kernel(state))


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
        at[2:-2, 2:-2, :], M_IN[2:-2, 2:-2, :],
    )
    vs.C_IN = update(
        vs.C_IN,
        at[2:-2, 2:-2, :], npx.where(vs.PREC[2:-2, 2:-2, :] > 0, vs.M_IN[2:-2, 2:-2, :] / vs.PREC[2:-2, 2:-2, :], 0),
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
        at[2:-2, 2:-2], vs.M_IN[2:-2, 2:-2, vs.itt] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_in = update(
        vs.C_in,
        at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        M_in=vs.M_in,
        C_in=vs.C_in,
    )


@roger_kernel
def set_states_kernel(state):
    vs = state.variables

    vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], vs.INF_MAT_RZ[2:-2, 2:-2, vs.itt])
    vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], vs.INF_PF_RZ[2:-2, 2:-2, vs.itt])
    vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], vs.INF_PF_SS[2:-2, 2:-2, vs.itt])
    vs.transp = update(vs.transp, at[2:-2, 2:-2], vs.TRANSP[2:-2, 2:-2, vs.itt])
    vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], vs.EVAP_SOIL[2:-2, 2:-2, vs.itt])
    vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], vs.Q_RZ[2:-2, 2:-2, vs.itt])
    vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], vs.Q_SS[2:-2, 2:-2, vs.itt])
    vs.cpr_rz = update(vs.cpr_ss, at[2:-2, 2:-2], vs.CPR_RZ[2:-2, 2:-2, vs.itt])

    vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], vs.S_RZ[2:-2, 2:-2, vs.itt])
    vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], vs.S_SS[2:-2, 2:-2, vs.itt])
    vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_S[2:-2, 2:-2, vs.itt])

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
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.MSA_rz = update(
        vs.MSA_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.MSA_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_rz[2:-2, 2:-2, vs.tau, :],
    )
    vs.M_rz = update(
        vs.M_rz,
        at[2:-2, 2:-2, vs.taum1], vs.M_rz[2:-2, 2:-2, vs.tau],
    )
    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, vs.taum1], vs.C_rz[2:-2, 2:-2, vs.tau],
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.MSA_ss = update(
        vs.MSA_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.MSA_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_ss[2:-2, 2:-2, vs.tau, :],
    )
    vs.M_ss = update(
        vs.M_ss,
        at[2:-2, 2:-2, vs.taum1], vs.M_ss[2:-2, 2:-2, vs.tau],
    )
    vs.C_ss = update(
        vs.C_ss,
        at[2:-2, 2:-2, vs.taum1], vs.C_ss[2:-2, 2:-2, vs.tau],
    )
    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.SA_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.sa_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.MSA_s = update(
        vs.MSA_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.MSA_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, vs.taum1, :], vs.msa_s[2:-2, 2:-2, vs.tau, :],
    )
    vs.M_s = update(
        vs.M_s,
        at[2:-2, 2:-2, vs.taum1], vs.M_s[2:-2, 2:-2, vs.tau],
    )
    vs.C_s = update(
        vs.C_s,
        at[2:-2, 2:-2, vs.taum1], vs.C_s[2:-2, 2:-2, vs.tau],
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


tm_structures = ['complete-mixing', 'piston',
                 'preferential', 'advection-dispersion',
                 'time-variant preferential',
                 'time-variant advection-dispersion']
years = onp.arange(1997, 2007).tolist()
for tm_structure in tm_structures:
    for year in years:
        model = SVATTRANSPORTSetup()
        # set transport model structure
        model._set_tm_structure(tm_structure)
        # set year
        model._set_year(year)
        tms = model._tm_structure.replace(" ", "_")
        input_path = model._base_path / "input" / str(year)
        model._set_input_dir(input_path)
        forcing_path = model._input_dir / "forcing_tracer.nc"
        if not os.path.exists(forcing_path):
            # export bromide data to .txt
            idx_start = '%s-01-01' % (year)
            year_end = year + 1
            idx_end = '%s-12-31' % (year_end)
            idx = pd.date_range(start=idx_start,
                                end=idx_end, freq='D')
            df_Br = pd.DataFrame(index=idx, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'Br'])
            df_Br['YYYY'] = df_Br.index.year.values
            df_Br['MM'] = df_Br.index.month.values
            df_Br['DD'] = df_Br.index.day.values
            df_Br['hh'] = df_Br.index.hour.values
            df_Br['mm'] = 0
            injection_date = '%s-11-12' % (year)
            df_Br.loc[injection_date, 'Br'] = 79900  # bromide mass in mg
            path_txt = input_path / "Br.txt"
            df_Br.to_csv(path_txt, header=True, index=False, sep="\t")
            write_forcing_tracer(input_path, 'Br')
        model.setup()
        model.warmup()
        model.run()

        # merge model output into single file
        path = str(model._base_path / f"{model.state.settings.identifier}.*.nc")
        diag_files = glob.glob(path)
        states_tm_file = model._base_path / f"states_tm_{tms}_{year}.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            f.create_group(f"{tm_structure}-{year}")
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title=f'RoGeR {model._tm_structure} ({year} - {year + 1}) transport model results for bromide benchmark at Rietholzbach Lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='SVAT transport model with free drainage'
            )
            for dfs in diag_files:
                with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                    # set dimensions with a dictionary
                    if not f.dimensions:
                        dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages'])}
                        f.dimensions = dict_dim
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('x', ('x',), float)
                        v.attrs['long_name'] = 'Number of model run'
                        v.attrs['units'] = ''
                        v[:] = npx.arange(dict_dim["x"])
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('y', ('y',), float)
                        v.attrs['long_name'] = ''
                        v.attrs['units'] = ''
                        v[:] = npx.arange(dict_dim["y"])
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('Time', ('Time',), float)
                        var_obj = df.variables.get('Time')
                        v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                        units=var_obj.attrs["units"])
                        v[:] = npx.array(var_obj)
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('ages', ('ages',), float)
                        v.attrs['long_name'] = 'Water ages'
                        v.attrs['units'] = 'days'
                        v[:] = npx.arange(1, dict_dim["ages"]+1)
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('nages', ('nages',), float)
                        v.attrs['long_name'] = 'Water ages (cumulated)'
                        v.attrs['units'] = 'days'
                        v[:] = npx.arange(0, dict_dim["nages"])
                    for var_sim in list(df.variables.keys()):
                        var_obj = df.variables.get(var_sim)
                        if var_sim not in list(f.dimensions.keys()) and var_obj.ndim == 3:
                            v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'Time'), float)
                            vals = npx.array(var_obj)
                            v[:, :, :] = vals.swapaxes(0, 2)
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(f.dimensions.keys()) and "ages" in var_obj.dimensions:
                            v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float)
                            vals = npx.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(f.dimensions.keys()) and "nages" in var_obj.dimensions:
                            v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float)
                            vals = npx.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
