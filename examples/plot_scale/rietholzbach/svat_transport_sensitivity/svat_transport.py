from pathlib import Path
import os
import h5netcdf
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-tms", "--transport-model-structure", type=click.Choice(['advection-dispersion', 'time-variant_advection-dispersion']), default='advection-dispersion')
@click.option("-ss", "--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("--x1", type=int, default=None)
@click.option("--x2", type=int, default=None)
@click.option("-td", "--tmp-dir", type=str, default=None)
@roger_base_cli
def main(transport_model_structure, sas_solver, x1, x2, tmp_dir):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, for_loop
    from roger.tools.setup import write_forcing_tracer
    from roger.core.transport import delta_to_conc, conc_to_delta

    class SVATTRANSPORTSetup(RogerSetup):
        """A SVAT transport model.
        """
        _base_path = Path(__file__).parent
        _nrows = None
        _tm_structure = None
        _identifier = None
        _sas_solver = None
        _input_dir = None

        def _set_input_dir(self, path):
            if os.path.exists(path):
                self._input_dir = path
            else:
                self._input_dir = path
                if not os.path.exists(self._input_dir):
                    os.mkdir(self._input_dir)

        def _read_var_from_nc(self, var, path_dir, file, group=None):
            nc_file = path_dir / file
            if group:
                with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                    var_obj = infile.groups[group].variables[var]
                    return npx.array(var_obj)
            else:
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

        def _set_tm_structure(self, tm_structure):
            self._tm_structure = tm_structure

        def _set_sas_solver(self, sas_solver):
            self._sas_solver = sas_solver

        def _set_identifier(self, identifier):
            self._identifier = identifier

        def _bfill_3d(self, state, arr):
            idx_shape = tuple([slice(None)] + [npx.newaxis] * (3 - 2 - 1))
            idx = allocate(state.dimensions, ("x", "y", "t"), dtype=int)
            arr1 = allocate(state.dimensions, ("x", 1, 1), dtype=int)
            arr2 = allocate(state.dimensions, (1, "y", 1), dtype=int)
            arr3 = allocate(state.dimensions, ("x", "y", "t"), dtype=int)
            arr_fill = allocate(state.dimensions, ("x", "y", "t"))
            idx = update(
                idx,
                at[2:-2, 2:-2, :], npx.where(npx.isfinite(arr), npx.arange(npx.shape(arr)[2])[idx_shape], 0)[2:-2, 2:-2, :],
            )
            idx = update(
                idx,
                at[2:-2, 2:-2, :], _bfill(idx)[2:-2, 2:-2, :],
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

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = self._identifier
            settings.sas_solver = self._sas_solver
            settings.sas_solver_substeps = 6
            if settings.sas_solver in ['RK4', 'Euler']:
                settings.h = 1 / settings.sas_solver_substeps

            settings.nx, settings.ny = x2 - x1, 1
            settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
            settings.ages = 1500
            settings.nages = settings.ages + 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = "1996-12-31 00:00:00"

            settings.enable_offline_transport = True
            settings.enable_oxygen18 = True
            settings.tm_structure = self._tm_structure
            settings.enable_age_statistics = True

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "dt_secs",
                "dt",
                "ages",
                "nages",
                "x",
                "y",
            ],
        )
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
                "S_fc_ss",
                "S_sat_rz",
                "S_sat_ss",
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

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.S_fc_ss = update(vs.S_fc_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_fc_ss", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])

            if settings.tm_structure == "advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, :, 2], self._read_var_from_nc("b_transp", self._base_path, 'params_saltelli.nc', group=self._tm_structure)[x1:x2, :])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], self._read_var_from_nc("a_q_rz", self._base_path, 'params_saltelli.nc', group=self._tm_structure)[x1:x2, :])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], self._read_var_from_nc("a_q_ss", self._base_path, 'params_saltelli.nc', group=self._tm_structure)[x1:x2, :])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
            elif settings.tm_structure == "time-variant advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 31)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], self._read_var_from_nc("c1_transp", self._base_path, 'params_saltelli.nc', group=self._tm_structure)[x1:x2, :])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, :, 4], self._read_var_from_nc("c2_transp", self._base_path, 'params_saltelli.nc', group=self._tm_structure)[x1:x2, :])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 32)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], self._read_var_from_nc("c1_q_rz", self._base_path, 'params_saltelli.nc', group=self._tm_structure)[x1:x2, :])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], self._read_var_from_nc("c2_q_rz", self._base_path, 'params_saltelli.nc', group=self._tm_structure)[x1:x2, :])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 32)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], self._read_var_from_nc("c1_q_ss", self._base_path, 'params_saltelli.nc', group=self._tm_structure)[x1:x2, :])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], self._read_var_from_nc("c2_q_ss", self._base_path, 'params_saltelli.nc', group=self._tm_structure)[x1:x2, :])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_fc_ss[2:-2, 2:-2])

        @roger_routine
        def set_parameters(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "S_pwp_rz",
                "S_pwp_ss",
                "S_snow",
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

            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_snow", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_rz", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_ss", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
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

            if settings.enable_oxygen18:
                vs.C_iso_snow = update(vs.C_iso_snow, at[2:-2, 2:-2, :vs.taup1], npx.nan)
                vs.C_iso_rz = update(vs.C_iso_rz, at[2:-2, 2:-2, :vs.taup1], -13)
                vs.C_iso_ss = update(vs.C_iso_ss, at[2:-2, 2:-2, :vs.taup1], -7)
                vs.C_rz = update(
                    vs.C_rz,
                    at[2:-2, 2:-2, :vs.taup1], delta_to_conc(state, vs.C_iso_rz[2:-2, 2:-2, vs.tau, npx.newaxis]),
                )
                vs.msa_rz = update(
                    vs.msa_rz,
                    at[2:-2, 2:-2, :vs.taup1, :], vs.C_rz[2:-2, 2:-2, :vs.taup1, npx.newaxis],
                )
                vs.msa_rz = update(
                    vs.msa_rz,
                    at[2:-2, 2:-2, :vs.taup1, 0], 0,
                )
                vs.C_ss = update(
                    vs.C_ss,
                    at[2:-2, 2:-2, :vs.taup1], delta_to_conc(state, vs.C_iso_ss[2:-2, 2:-2, vs.tau, npx.newaxis]),
                )
                vs.msa_ss = update(
                    vs.msa_ss,
                    at[2:-2, 2:-2, :vs.taup1, :], vs.C_ss[2:-2, 2:-2, :vs.taup1, npx.newaxis],
                )
                vs.msa_ss = update(
                    vs.msa_ss,
                    at[2:-2, 2:-2, :vs.taup1, 0], 0,
                )
                vs.msa_s = update(
                    vs.msa_s,
                    at[2:-2, 2:-2, :, :], npx.where(vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] > 0, vs.msa_rz[2:-2, 2:-2, :, :] * (vs.sa_rz[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :])) + vs.msa_ss[2:-2, 2:-2, :, :] * (vs.sa_ss[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :])), 0),
                )
                vs.msa_s = update(
                    vs.msa_s,
                    at[2:-2, 2:-2, :vs.taup1, 0], 0,
                )
                vs.C_s = update(
                    vs.C_s,
                    at[2:-2, 2:-2, vs.tau], npx.sum(npx.where(vs.sa_s[2:-2, 2:-2, vs.tau, :] > 0, vs.msa_s[2:-2, 2:-2, vs.tau, :] * (vs.sa_s[2:-2, 2:-2, vs.tau, :] / npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis]), 0), axis=-1),
                )
                vs.C_s = update(
                    vs.C_s,
                    at[2:-2, 2:-2, vs.taum1], vs.C_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
                )
                vs.C_iso_s = update(
                    vs.C_iso_s,
                    at[2:-2, 2:-2, vs.taum1], conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
                )
                vs.C_iso_s = update(
                    vs.C_iso_s,
                    at[2:-2, 2:-2, vs.tau], conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
                )
                vs.csa_rz = update(
                    vs.csa_rz,
                    at[2:-2, 2:-2, vs.tau, :], conc_to_delta(state, vs.msa_rz[2:-2, 2:-2, vs.tau, :]),
                )
                vs.csa_ss = update(
                    vs.csa_ss,
                    at[2:-2, 2:-2, vs.tau, :], conc_to_delta(state, vs.msa_ss[2:-2, 2:-2, vs.tau, :]),
                )
                vs.csa_s = update(
                    vs.csa_s,
                    at[2:-2, 2:-2, vs.tau, :], conc_to_delta(state, vs.msa_s[2:-2, 2:-2, vs.tau, :]),
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
                "C_ISO_IN",
                "C_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables
            settings = state.settings

            if settings.enable_oxygen18:
                vs.C_ISO_IN = update(vs.C_ISO_IN, at[2:-2, 2:-2, 0], npx.nan)
                vs.C_ISO_IN = update(vs.C_ISO_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("d18O", self._input_dir, 'forcing_tracer.nc'))
                vs.C_ISO_IN = update(vs.C_ISO_IN, at[2:-2, 2:-2, :], self._bfill_3d(state, vs.C_ISO_IN)[2:-2, 2:-2, :])
                vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, :], delta_to_conc(state, vs.C_ISO_IN)[2:-2, 2:-2, :])

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
                "C_iso_in",
                "C_IN",
                "C_snow",
                "C_iso_snow",
            ],
        )
        def set_forcing(self, state):
            vs = state.variables

            vs.ta = update(vs.ta, at[2:-2, 2:-2], self._read_var_from_nc("ta", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("prec", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mat_rz", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mp_rz", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt] + self._read_var_from_nc("inf_sc_rz", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], self._read_var_from_nc("inf_ss", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.transp = update(vs.transp, at[2:-2, 2:-2], self._read_var_from_nc("transp", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], self._read_var_from_nc("evap_soil", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.cpr_rz = update(vs.cpr_rz, at[2:-2, 2:-2], self._read_var_from_nc("cpr_rz", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], self._read_var_from_nc("q_rz", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])
            vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], self._read_var_from_nc("q_ss", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_rz", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_ss", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])
            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_snow", self._base_path, f'states_hm_for_{transport_model_structure}.nc')[npx.newaxis, :, vs.itt])

            vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt])
            # mixing of isotopes while snow accumulation
            vs.C_snow = update(
                vs.C_snow,
                at[2:-2, 2:-2, vs.tau], npx.where(vs.S_snow[2:-2, 2:-2, vs.tau] > 0, npx.where(npx.isnan(vs.C_snow[2:-2, 2:-2, vs.tau]), vs.C_in[2:-2, 2:-2], (vs.prec[2:-2, 2:-2, vs.tau] / (vs.prec[2:-2, 2:-2, vs.tau] + vs.S_snow[2:-2, 2:-2, vs.tau])) * vs.C_in[2:-2, 2:-2] + (vs.S_snow[2:-2, 2:-2, vs.tau] / (vs.prec[2:-2, 2:-2, vs.tau] + vs.S_snow[2:-2, 2:-2, vs.tau])) * vs.C_snow[2:-2, 2:-2, vs.taum1]), npx.nan),
            )
            vs.C_snow = update(
                vs.C_snow,
                at[2:-2, 2:-2, vs.tau], npx.where(vs.S_snow[2:-2, 2:-2, vs.tau] <= 0, npx.nan, vs.C_snow[2:-2, 2:-2, vs.tau]),
            )
            vs.C_iso_snow = update(
                vs.C_iso_snow,
                at[2:-2, 2:-2, vs.tau], conc_to_delta(state, vs.C_snow[2:-2, 2:-2, vs.tau]),
            )

            # mix isotopes from snow melt and rainfall
            vs.C_in = update(
                vs.C_in,
                at[2:-2, 2:-2], npx.where(npx.isfinite(vs.C_snow[2:-2, 2:-2, vs.taum1]), vs.C_snow[2:-2, 2:-2, vs.taum1], npx.where(vs.prec[2:-2, 2:-2, vs.tau] > 0, vs.C_IN[2:-2, 2:-2, vs.itt], 0)),
            )
            vs.C_iso_in = update(vs.C_iso_in, at[2:-2, 2:-2], conc_to_delta(state, vs.C_in[2:-2, 2:-2]))

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["average"].output_variables = ["C_iso_transp", "C_iso_q_ss", "C_iso_rz", "C_iso_ss", "C_iso_s",
                                                       "tt25_transp", "tt50_transp", "tt75_transp",  "ttavg_transp",
                                                       "tt25_q_ss", "tt50_q_ss", "tt75_q_ss",  "ttavg_q_ss",
                                                       "rt25_rz", "rt50_rz", "rt75_rz",  "rtavg_rz",
                                                       "rt25_ss", "rt50_ss", "rt75_ss",  "rtavg_ss",
                                                       "rt25_s", "rt50_s", "rt75_s",  "rtavg_s"]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            diagnostics["constant"].output_variables = ["sas_params_transp", "sas_params_q_rz", "sas_params_q_ss"]
            diagnostics["constant"].output_frequency = 0
            diagnostics["constant"].sampling_frequency = 1
            if base_path:
                diagnostics["constant"].base_output_path = base_path

            # maximum bias of deterministic/numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error", "dC_num_error"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.S_snow = update(
            vs.S_snow,
            at[2:-2, 2:-2, vs.taum1], vs.S_snow[2:-2, 2:-2, vs.tau],
        )
        vs.C_snow = update(
            vs.C_snow,
            at[2:-2, 2:-2, vs.taum1], vs.C_snow[2:-2, 2:-2, vs.tau],
        )
        vs.prec = update(
            vs.prec,
            at[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau],
        )

        return KernelOutput(
            prec=vs.prec,
            C_snow=vs.C_snow,
            S_snow=vs.S_snow,
            )

    @roger_kernel
    def _bfill(loop_arr):
        def loop_body(i, loop_arr):
            j = loop_arr.shape[2] - i
            loop_arr = update(
                loop_arr,
                at[:, :, j-1], npx.where(loop_arr[:, :, j-1] == 0, loop_arr[:, :, j], loop_arr[:, :, j-1]),
            )

            return loop_arr

        loop_arr = for_loop(1, loop_arr.shape[2], loop_body, loop_arr)
        loop_arr = update(
            loop_arr,
            at[:, :, -1], npx.where(loop_arr[:, :, -1] == 0, loop_arr[:, :, -2], loop_arr[:, :, -1]),
        )

        return loop_arr

    tms = transport_model_structure.replace("_", " ")
    model = SVATTRANSPORTSetup()
    model._set_sas_solver(sas_solver)
    model._set_tm_structure(tms)
    identifier = f'SVATTRANSPORT_{transport_model_structure}_{sas_solver}_{x1}_{x2}'
    model._set_identifier(identifier)
    input_path = model._base_path / "input"
    model._set_input_dir(input_path)
    write_forcing_tracer(input_path, 'd18O')
    model.setup()
    model.warmup()
    model.run()
    return


if __name__ == "__main__":
    main()
