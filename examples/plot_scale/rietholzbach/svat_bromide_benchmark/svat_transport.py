import os
from pathlib import Path
import pandas as pd
import h5netcdf
import xarray as xr
from cftime import num2date, date2num
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-y", "--year", type=click.Choice(['1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006']), default='2001')
@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'piston', 'advection-dispersion-power', 'time-variant_advection-dispersion-power']), default='advection-dispersion-power')
@click.option("-ss", "--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-td", "--tmp-dir", type=str, default=None)
@roger_base_cli
def main(year, transport_model_structure, sas_solver, tmp_dir):
    from roger import RogerSetup, roger_routine
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, where
    from roger.tools.setup import write_forcing_tracer

    class SVATTRANSPORTSetup(RogerSetup):
        """A SVAT transport model for bromide.
        """
        _base_path = Path(__file__).parent
        _year = int(year)
        _tm_structure = transport_model_structure.replace("_", " ")
        _input_dir = _base_path / "input" / str(year)
        _states_hm_file = f'states_hm_best_for_{transport_model_structure}.nc'

        def _set_input_dir(self, path):
            if os.path.exists(path):
                self._input_dir = path
            else:
                self._input_dir = path
                if not os.path.exists(self._input_dir):
                    os.mkdir(self._input_dir)

        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj)

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return len(onp.array(var_obj)) + 1

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return len(onp.array(var_obj)) * 60 * 60 * 24

        def _set_bromide_input(self, state, nn_rain, nn_sol, inf):
            vs = state.variables

            C_IN = allocate(state.dimensions, ("x", "y", "t"))
            M_IN = allocate(state.dimensions, ("x", "y", "t"))
            idx = allocate(state.dimensions, ("x", "y", "t"))
            idx = update(
                idx,
                at[2:-2, 2:-2, :], npx.arange(idx.shape[-1])[npx.newaxis, npx.newaxis, :],
            )

            mask_sol = (vs.M_IN > 0)
            sol_idx = npx.zeros((nn_sol,), dtype=int)
            sol_idx = update(sol_idx, at[:], where(npx.any(mask_sol, axis=(0, 1)), size=nn_sol, fill_value=0)[0])
            inf_idx = npx.where((inf > 0), idx, 0)

            # join solute input on closest rainfall event
            for i in range(nn_sol):
                input_itt = npx.nanargmin(npx.where(inf_idx[2:-2, 2:-2, :] - sol_idx[i] < 0, npx.nan, inf_idx[2:-2, 2:-2, :] - sol_idx[i]), axis=-1)
                for x in range(input_itt.shape[0]):
                    for y in range(input_itt.shape[1]):
                        start_inf = input_itt[x, y]
                        end_inf = npx.max(npx.where(npx.cumsum(inf[x+2, y+2, start_inf:]) <= 40, npx.arange(inf.shape[-1])[start_inf:], 0)) + 1
                        if npx.sum(inf[x+2, y+2, start_inf:end_inf]) <= 0:
                            end_inf = end_inf + 1

                        # proportions for redistribution
                        M_IN = update(
                            M_IN,
                            at[x+2, y+2, start_inf:end_inf], vs.M_IN[x+2, y+2, sol_idx[i]] * (inf[x+2, y+2, start_inf:end_inf] / npx.sum(inf[x+2, y+2, start_inf:end_inf])),
                        )

            C_IN = update(
                C_IN,
                at[2:-2, 2:-2, :], npx.where(inf[2:-2, 2:-2, :] > 0, M_IN[2:-2, 2:-2, :] / inf[2:-2, 2:-2, :], 0),
            )

            return M_IN, C_IN

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = f'SVATTRANSPORT_{transport_model_structure}_{year}_{sas_solver}'
            settings.sas_solver = sas_solver
            settings.sas_solver_substeps = 6
            if settings.sas_solver in ['RK4', 'Euler']:
                settings.h = 1 / settings.sas_solver_substeps

            settings.nx, settings.ny = 100, 1
            settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
            settings.ages = 1500
            settings.nages = settings.ages + 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')
            settings.runlen_warmup = settings.runlen

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = f"{self._year - 1}-12-31 00:00:00"

            settings.enable_offline_transport = True
            settings.enable_bromide = True
            settings.tm_structure = self._tm_structure
            settings.enable_age_statistics = True

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
                "sas_params_q_ss"
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._input_dir, self._states_hm_file)[npx.newaxis, :, 0])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._input_dir, self._states_hm_file)[npx.newaxis, :, 0])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._input_dir, self._states_hm_file)[npx.newaxis, :, 0])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._input_dir, self._states_hm_file)[npx.newaxis, :, 0])

            alpha = npx.linspace(0.1, 1, num=10).tolist()
            params = npx.array(onp.meshgrid(alpha, alpha)).T.reshape(-1, 2)
            vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], params[:, 0, npx.newaxis])
            vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], params[:, 1, npx.newaxis])

            if settings.tm_structure == "complete-mixing":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 1)
            elif settings.tm_structure == "piston":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 100)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 100)
            elif settings.tm_structure == "advection-dispersion-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 0.49)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 4.16)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 4.11)
            elif settings.tm_structure == "time-variant advection-dispersion-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 62)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 0.41)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1.85)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 61)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1.83)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], 3.83)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 61)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1.82)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], 2.99)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])

        @roger_routine
        def set_parameters(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
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

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, 0], self._read_var_from_nc("S_rz", self._input_dir, self._states_hm_file)[npx.newaxis, :, vs.itt])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, 0], self._read_var_from_nc("S_ss", self._input_dir, self._states_hm_file)[npx.newaxis, :, vs.itt])
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, 1], self._read_var_from_nc("S_rz", self._input_dir, self._states_hm_file)[npx.newaxis, :, vs.itt])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, 1], self._read_var_from_nc("S_ss", self._input_dir, self._states_hm_file)[npx.newaxis, :, vs.itt])
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
                "PREC_DIST_DAILY",
                "INF_MAT_RZ",
                "INF_PF_RZ",
                "INF_PF_SS",
                "TRANSP",
                "EVAP_SOIL",
                "CPR_RZ",
                "Q_RZ",
                "Q_SS",
                "S_RZ",
                "S_SS",
                "S_S",
                "M_IN",
                "C_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC_DIST_DAILY = update(vs.PREC_DIST_DAILY, at[2:-2, 2:-2, :], self._read_var_from_nc("prec", self._input_dir, self._states_hm_file))
            vs.INF_MAT_RZ = update(vs.INF_MAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mat_rz", self._input_dir, self._states_hm_file))
            vs.INF_PF_RZ = update(vs.INF_PF_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mp_rz", self._input_dir, self._states_hm_file) + self._read_var_from_nc("inf_sc_rz", self._input_dir, self._states_hm_file))
            vs.INF_PF_SS = update(vs.INF_PF_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_ss", self._input_dir, self._states_hm_file))
            vs.TRANSP = update(vs.TRANSP, at[2:-2, 2:-2, :], self._read_var_from_nc("transp", self._input_dir, self._states_hm_file))
            vs.EVAP_SOIL = update(vs.EVAP_SOIL, at[2:-2, 2:-2, :], self._read_var_from_nc("evap_soil", self._input_dir, self._states_hm_file))
            vs.CPR_RZ = update(vs.CPR_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("cpr_rz", self._input_dir, self._states_hm_file))
            vs.Q_RZ = update(vs.Q_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("q_rz", self._input_dir, self._states_hm_file))
            vs.Q_SS = update(vs.Q_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("q_ss", self._input_dir, self._states_hm_file))
            vs.S_RZ = update(vs.S_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_rz", self._input_dir, self._states_hm_file))
            vs.S_SS = update(vs.S_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_ss", self._input_dir, self._states_hm_file))
            vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
            TA = allocate(state.dimensions, ("x", "y", "t"))
            TA = update(TA, at[2:-2, 2:-2, :], self._read_var_from_nc("ta", self._input_dir, self._states_hm_file)[npx.newaxis, :, :])

            vs.M_IN = update(vs.M_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("Br", self._input_dir, 'forcing_tracer.nc'))
            mask_rain = (vs.PREC_DIST_DAILY > 0) & (TA > 0)
            mask_sol = (vs.M_IN > 0)
            nn_rain = npx.int64(npx.sum(npx.any(mask_rain, axis=(0, 1))))
            nn_sol = npx.int64(npx.sum(npx.any(mask_sol, axis=(0, 1))))
            INF = vs.INF_MAT_RZ + vs.INF_PF_RZ + vs.INF_PF_SS
            M_IN, C_IN = self._set_bromide_input(state, nn_rain, nn_sol, INF)
            vs.M_IN = update(vs.M_IN, at[:, :, :], M_IN)
            vs.C_IN = update(vs.C_IN, at[:, :, :], C_IN)

        @roger_routine
        def set_forcing(self, state):
            vs = state.variables

            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], vs.PREC_DIST_DAILY[2:-2, 2:-2, vs.itt])
            vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], vs.INF_MAT_RZ[2:-2, 2:-2, vs.itt])
            vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], vs.INF_PF_RZ[2:-2, 2:-2, vs.itt])
            vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], vs.INF_PF_SS[2:-2, 2:-2, vs.itt])
            vs.transp = update(vs.transp, at[2:-2, 2:-2], vs.TRANSP[2:-2, 2:-2, vs.itt])
            vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], vs.EVAP_SOIL[2:-2, 2:-2, vs.itt])
            vs.cpr_rz = update(vs.cpr_rz, at[2:-2, 2:-2], vs.CPR_RZ[2:-2, 2:-2, vs.itt])
            vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], vs.Q_RZ[2:-2, 2:-2, vs.itt])
            vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], vs.Q_SS[2:-2, 2:-2, vs.itt])
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], vs.S_RZ[2:-2, 2:-2, vs.itt])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], vs.S_SS[2:-2, 2:-2, vs.itt])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])

            vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt])
            vs.M_in = update(
                vs.M_in,
                at[2:-2, 2:-2], vs.C_in[2:-2, 2:-2] * (vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2]),
            )

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = ["M_q_ss", "M_transp"]
            diagnostics["rate"].output_frequency = 24 * 60 * 60
            diagnostics["rate"].sampling_frequency = 1
            if base_path:
                diagnostics["rate"].base_output_path = base_path

            diagnostics["average"].output_variables = ["C_in", "C_s", "C_rz", "C_ss", "C_transp", "C_q_ss", "ttavg_q_ss", "tt50_q_ss"]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            diagnostics["collect"].output_variables = ["M_in", "M_s"]
            diagnostics["collect"].output_frequency = 24 * 60 * 60
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            diagnostics["constant"].output_variables = ["alpha_transp", "alpha_q"]
            diagnostics["constant"].output_frequency = 24 * 60 * 60
            diagnostics["constant"].sampling_frequency = 1
            if base_path:
                diagnostics["constant"].base_output_path = base_path

            # maximum bias of numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error", "dC_num_error"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            pass

    model = SVATTRANSPORTSetup()
    # write bromide data to .txt
    idx_start = '%s-01-01' % (year)
    year_end = int(year) + 1
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
    df_Br.loc[injection_date, 'Br'] = 79900/3.14  # bromide mass in mg per m2
    path_txt = model._input_dir / "Br.txt"
    df_Br.to_csv(path_txt, header=True, index=False, sep="\t")
    write_forcing_tracer(model._input_dir, 'Br')
    nc_file = model._base_path / model._states_hm_file
    with xr.open_dataset(nc_file, engine="h5netcdf") as ds:
        days = (ds['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date = num2date(days, units=f"days since {ds['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds = ds.assign_coords(Time=("Time", date))
        ds_year = ds.sel(Time=slice(f'{int(year) - 1}-12-31', f'{int(year) + 1}-12-31'))
        days_year = date2num(ds_year["Time"].values.astype('M8[ms]').astype('O'), units=f"days since {int(year) - 1}-12-31", calendar='standard')
        ds_year = ds_year.assign_coords(Time=("Time", days_year))
        ds_year.Time.attrs['units'] = "days"
        ds_year.Time.attrs['time_origin'] = f"{int(year) - 1}-12-31"
        nc_file_year = model._base_path / "input" / str(year) / model._states_hm_file
        ds_year.to_netcdf(nc_file_year, engine="h5netcdf")
        ds_year = ds_year.load()
        ds_year = ds_year.close()
        del ds_year
    model.setup()
    model.warmup()
    model.run()
    return


if __name__ == "__main__":
    main()
