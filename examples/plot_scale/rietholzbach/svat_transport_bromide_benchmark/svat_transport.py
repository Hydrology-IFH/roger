from pathlib import Path
import pandas as pd
import xarray as xr
from cftime import num2date
import os
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'piston', 'preferential', 'advection-dispersion', 'time-variant_preferential', 'time-variant_advection-dispersion', 'time-variant']), default='complete-mixing')
@click.option("-ss", "--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@roger_base_cli
def main(transport_model_structure, sas_solver):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, where
    from roger.tools.setup import write_forcing_tracer

    class SVATTRANSPORTSetup(RogerSetup):
        """A SVAT transport model.
        """
        _base_path = Path(__file__).parent
        _year = None
        _tm_structure = None
        _input_dir = None
        _identifier = None
        _sas_solver = None

        def _set_input_dir(self, path):
            if os.path.exists(path):
                self._input_dir = path
            else:
                self._input_dir = path
                if not os.path.exists(self._input_dir):
                    os.mkdir(self._input_dir)

        def _read_var_from_nc(self, var, path_dir, file, group=None):
            nc_file = path_dir / file
            with xr.open_dataset(nc_file, engine="h5netcdf", group=group) as ds:
                days = (ds['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
                date = num2date(days, units=f"days since {ds['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
                ds = ds.assign_coords(Time=("Time", date))
                ds_year = ds.sel(Time=slice(f'{self._year - 1}-12-31', f'{self._year + 1}-12-31'))
                vals = ds_year[var].values

            return vals

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with xr.open_dataset(nc_file, engine="h5netcdf") as ds:
                days = (ds['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
                date = num2date(days, units=f"days since {ds['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
                ds = ds.assign_coords(Time=("Time", date))
                ds_year = ds.sel(Time=slice(f'{self._year}-01-01', f'{self._year + 1}-12-31'))
                return len(ds_year['Time'].values) + 1

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

        def _set_sas_solver(self, sas_solver):
            self._sas_solver = sas_solver

        def _set_identifier(self, identifier):
            self._identifier = identifier

        def _set_bromide_input(self, state, nn_rain, nn_sol, prec, ta):
            vs = state.variables

            M_IN = allocate(state.dimensions, ("x", "y", "t"))

            mask_rain = (prec > 0) & (ta > 0)
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
                input_itt = npx.nanargmin(npx.where(rain_idx - sol_idx[i] < 0, npx.nan, rain_idx - sol_idx[i]))
                start_rain = rain_idx[input_itt]
                rain_sum = update(
                    rain_sum,
                    at[:, :], npx.max(npx.where(npx.cumsum(prec[:, :, start_rain:], axis=-1) <= 20, npx.max(npx.cumsum(prec[:, :, start_rain:], axis=-1), axis=-1)[:, :, npx.newaxis], 0), axis=-1),
                )
                nn_end = npx.max(npx.where(npx.cumsum(prec[:, :, start_rain:]) <= 20, npx.max(npx.arange(npx.shape(prec)[2])[npx.newaxis, npx.newaxis, npx.shape(prec)[2]-start_rain], axis=-1), 0))
                end_rain = update(end_rain, at[:], start_rain + nn_end)
                end_rain = update(end_rain, at[:], npx.where(end_rain > npx.shape(prec)[2], npx.shape(prec)[2], end_rain))

                # proportions for redistribution
                M_IN = update(
                    M_IN,
                    at[:, :, start_rain:end_rain[0]], vs.M_IN[:, :, sol_idx[i], npx.newaxis] * (prec[:, :, start_rain:end_rain[0]] / rain_sum[:, :, npx.newaxis]),
                )

            C_IN = npx.where(prec > 0, M_IN / prec, 0)

            return M_IN, C_IN

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = self._identifier
            settings.sas_solver = self._sas_solver

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
                "itt"
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt])

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
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'sas_params.nc', group=settings.tm_structure))
            elif settings.tm_structure == "advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'sas_params.nc', group=settings.tm_structure))
            elif settings.tm_structure == "complete-mixing + advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'sas_params.nc', group=settings.tm_structure))
            elif settings.tm_structure == "time-variant advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'sas_params.nc', group=settings.tm_structure))
            elif settings.tm_structure == "time-variant preferential":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'sas_params.nc', group=settings.tm_structure))
            elif settings.tm_structure == "time-variant":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'sas_params.nc', group=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'sas_params.nc', group=settings.tm_structure))

        @roger_routine
        def set_parameters(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "S_pwp_rz",
                "S_pwp_ss",
                "S_rz",
                "S_rz_init",
                "S_ss",
                "S_ss_init",
                "S_s",
                "itt",
                "taup1"
            ],
        )
        def set_initial_conditions_setup(self, state):
            vs = state.variables

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, :vs.taup1], vs.S_rz[2:-2, 2:-2, :vs.taup1] + vs.S_ss[2:-2, 2:-2, :vs.taup1])
            vs.S_rz_init = update(vs.S_rz_init, at[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, 0])
            vs.S_ss_init = update(vs.S_ss_init, at[2:-2, 2:-2], vs.S_ss[2:-2, 2:-2, 0])

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

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
        def set_boundary_conditions_setup(self, state):
            pass

        @roger_routine
        def set_boundary_conditions(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "M_IN",
                "C_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables

            TA = self._read_var_from_nc("ta", self._base_path, 'states_hm.nc', group=self._lys)
            PREC = self._read_var_from_nc("prec", self._base_path, 'states_hm.nc', group=self._lys)

            vs.M_IN = update(vs.M_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("Br", self._input_dir, 'forcing_tracer.nc'))

            mask_rain = (PREC > 0) & (TA > 0)
            mask_sol = (vs.M_IN > 0)
            nn_rain = npx.int64(npx.sum(npx.any(mask_rain, axis=(0, 1))))
            nn_sol = npx.int64(npx.sum(npx.any(mask_sol, axis=(0, 1))))
            M_IN, C_IN = self._set_bromide_input(state, nn_rain, nn_sol, PREC, TA)
            vs.M_IN = update(vs.M_IN, at[:, :, :], M_IN)
            vs.C_IN = update(vs.C_IN, at[:, :, :], C_IN)

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "ta",
                "prec",
                "inf_mat_rz",
                "inf_pf_rz",
                "inf_pf_ss",
                "transp",
                "evap_soil",
                "cpr_rz",
                "q_rz",
                "q_ss",
                "re_rg",
                "re_rl",
                "S_pwp_rz",
                "S_rz",
                "S_pwp_ss",
                "S_ss",
                "S_s",
                "S_snow",
                "tau",
                "taum1",
                "itt",
                "C_in",
                "C_IN",
                "M_in"

            ],
        )
        def set_forcing(self, state):
            vs = state.variables

            vs.ta = update(vs.ta, at[2:-2, 2:-2], self._read_var_from_nc("ta", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("prec", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mat_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mp_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt] + self._read_var_from_nc("inf_sc_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], self._read_var_from_nc("inf_ss", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.transp = update(vs.transp, at[2:-2, 2:-2], self._read_var_from_nc("transp", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], self._read_var_from_nc("evap_soil", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.cpr_rz = update(vs.cpr_rz, at[2:-2, 2:-2], self._read_var_from_nc("cpr_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], self._read_var_from_nc("q_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], self._read_var_from_nc("q_ss", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.re_rg = update(vs.re_rg, at[2:-2, 2:-2], self._read_var_from_nc("re_rg", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.re_rl = update(vs.re_rl, at[2:-2, 2:-2], self._read_var_from_nc("re_rl", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])

            vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt])
            vs.M_in = update(
                vs.M_in,
                at[2:-2, 2:-2], vs.C_in[2:-2, 2:-2] * vs.prec[2:-2, 2:-2, vs.tau],
            )

        @roger_routine
        def set_diagnostics(self, state, base_path=None):
            diagnostics = state.diagnostics

            diagnostics["rates"].output_variables = ["M_q_ss"]
            diagnostics["rates"].output_frequency = 24 * 60 * 60
            diagnostics["rates"].sampling_frequency = 1
            if base_path:
                diagnostics["rates"].base_output_path = base_path

            diagnostics["averages"].output_variables = ["C_rz", "C_ss", "C_s", "C_q_ss"]
            diagnostics["averages"].output_frequency = 24 * 60 * 60
            diagnostics["averages"].sampling_frequency = 1
            if base_path:
                diagnostics["averages"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))

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

    years = onp.arange(1997, 2007).tolist()
    tms = transport_model_structure.replace("_", " ")
    for year in years:
        model = SVATTRANSPORTSetup()
        # set transport model structure
        model._set_sas_solver(sas_solver)
        model._set_tm_structure(tms)
        if sas_solver:
            identifier = f'SVATTRANSPORT_{tms}_{year}_{sas_solver}'
        else:
            identifier = f'SVATTRANSPORT_{tms}_{year}'
        model._set_identifier(identifier)
        # set year
        model._set_year(year)
        tms = model._tm_structure.replace(" ", "_")
        input_path = model._base_path / "input" / str(year)
        model._set_input_dir(input_path)
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
    return


if __name__ == "__main__":
    main()
