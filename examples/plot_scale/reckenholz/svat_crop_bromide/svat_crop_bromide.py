from pathlib import Path
import os
import h5netcdf
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-lys", "--lys-experiment", type=click.Choice(["lys2_bromide", "lys8_bromide", "lys9_bromide"]), default="lys2_bromide")
@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'advection-dispersion-power', 'time-variant_advection-dispersion-power']), default='complete-mixing')
@click.option("-ecp", "--crop-partitioning", is_flag=True)
@roger_base_cli
def main(lys_experiment, transport_model_structure, crop_partitioning):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, where
    from roger.tools.setup import write_forcing_tracer
    from roger.core.crop import update_alpha_transp

    class SVATCROPTRANSPORTSetup(RogerSetup):
        """A SVAT transport model for bromide including
        crop phenology/crop rotation.
        """
        _base_path = Path(__file__).parent
        _lys = None
        _tm_structure = None
        _input_dir = None
        _identifier = None

        def _set_input_dir(self, path):
            if os.path.exists(path):
                self._input_dir = path
            else:
                self._input_dir = path
                if not os.path.exists(self._input_dir):
                    os.mkdir(self._input_dir)

        def _read_var_from_nc(self, var, path_dir, file, group=None, subgroup=None):
            nc_file = path_dir / file
            if group and not subgroup:
                with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                    var_obj = infile.groups[group].variables[var]
                    return npx.array(var_obj)
            elif group and subgroup:
                with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                    var_obj = infile.groups[group].groups[subgroup].variables[var]
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

        def _get_time_origin(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                date = infile.variables['Time'].attrs['time_origin'].split(" ")[0]
                return f"{date} 00:00:00"

        def _set_lys(self, lys):
            self._lys = lys

        def _set_tm_structure(self, tm_structure):
            self._tm_structure = tm_structure

        def _set_identifier(self, identifier):
            self._identifier = identifier

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
            settings.identifier = "SVATCROPTRANSPORT"

            settings.nx, settings.ny = 1, 1
            settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
            settings.ages = settings.nitt
            settings.nages = settings.nitt + 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = self._get_time_origin(self._input_dir, 'forcing_tracer.nc')

            settings.enable_crop_phenology = True
            settings.enable_crop_rotation = True
            settings.enable_offline_transport = True
            settings.enable_bromide = True
            settings.tm_structure = self._tm_structure

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
            # spatial grid
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
                "sas_params_re_rg",
                "sas_params_re_rl",
                "itt",
                "lu_id",
                "lut_crops",
                "lut_crop_scale"

            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])

            if settings.enable_crop_partitioning:
                vs.update(update_alpha_transp(state))
            else:
                vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], 0.8)
            vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], 0.8)

            if settings.tm_structure == "complete-mixing":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 1)
            elif settings.tm_structure == "advection-dispersion-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 2)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 2)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 1], 100)
            elif settings.tm_structure == "time-variant advection-dispersion-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 62)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 61)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 61)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 1], 100)

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

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.mc')[:, :, vs.itt, npx.newaxis])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.mc')[:, :, vs.itt, npx.newaxis])
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

            vs.C_rz = update(vs.C_rz, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.C_ss = update(vs.C_ss, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.msa_rz = update(
                vs.msa_rz,
                at[2:-2, 2:-2, :vs.taup1, :], vs.C_rz[2:-2, 2:-2, :vs.taup1, npx.newaxis] * vs.sa_rz[2:-2, 2:-2, :vs.taup1, :],
            )
            vs.msa_ss = update(
                vs.msa_ss,
                at[2:-2, 2:-2, :vs.taup1, :], vs.C_ss[2:-2, 2:-2, :vs.taup1, npx.newaxis] * vs.sa_ss[2:-2, 2:-2, :vs.taup1, :],
            )
            vs.msa_s = update(
                vs.msa_s,
                at[2:-2, 2:-2, :, :], vs.msa_rz[2:-2, 2:-2, :, :] + vs.msa_ss[2:-2, 2:-2, :, :],
            )
            vs.M_rz = update(
                vs.M_rz,
                at[2:-2, 2:-2, :], npx.sum(vs.msa_rz[2:-2, 2:-2, :, :], axis=-1),
            )
            vs.M_ss = update(
                vs.M_ss,
                at[2:-2, 2:-2, :], npx.sum(vs.msa_ss[2:-2, 2:-2, :, :], axis=-1),
            )
            vs.M_s = update(
                vs.M_s,
                at[2:-2, 2:-2, :], npx.sum(vs.msa_s[2:-2, 2:-2, :, :], axis=-1),
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
                "RE_RG",
                "RE_RL",
                "S_RZ",
                "S_PWP_RZ",
                "S_SAT_RZ",
                "S_SS",
                "S_PWP_SS",
                "S_SAT_SS",
                "S_S",
                "M_IN",
                "C_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC_DIST_DAILY = update(vs.PREC_DIST_DAILY, at[2:-2, 2:-2, :], self._read_var_from_nc("prec", self._base_path, 'states_hm.nc'))
            vs.INF_MAT_RZ = update(vs.INF_MAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mat_rz", self._base_path, 'states_hm.nc'))
            vs.INF_PF_RZ = update(vs.INF_PF_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mp_rz", self._base_path, 'states_hm.nc') + self._read_var_from_nc("inf_sc_rz", self._base_path, 'states_hm.nc'))
            vs.INF_PF_SS = update(vs.INF_PF_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_ss", self._base_path, 'states_hm.nc'))
            vs.TRANSP = update(vs.TRANSP, at[2:-2, 2:-2, :], self._read_var_from_nc("transp", self._base_path, 'states_hm.nc'))
            vs.EVAP_SOIL = update(vs.EVAP_SOIL, at[2:-2, 2:-2, :], self._read_var_from_nc("evap_soil", self._base_path, 'states_hm.nc'))
            vs.CPR_RZ = update(vs.CPR_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("cpr_rz", self._base_path, 'states_hm.nc'))
            vs.Q_RZ = update(vs.Q_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("q_rz", self._base_path, 'states_hm.nc'))
            vs.Q_SS = update(vs.Q_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("q_ss", self._base_path, 'states_hm.nc'))
            vs.RE_RG = update(vs.RE_RG, at[2:-2, 2:-2, :], self._read_var_from_nc("re_rg", self._base_path, 'states_hm.nc'))
            vs.RE_RL = update(vs.RE_RL, at[2:-2, 2:-2, :], self._read_var_from_nc("re_rl", self._base_path, 'states_hm.nc'))
            vs.S_RZ = update(vs.S_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc'))
            vs.S_PWP_RZ = update(vs.S_PWP_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc'))
            vs.S_SAT_RZ = update(vs.S_SAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.nc'))
            vs.S_SS = update(vs.S_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc'))
            vs.S_PWP_SS = update(vs.S_PWP_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc'))
            vs.S_SAT_SS = update(vs.S_SAT_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.nc'))
            vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
            TA = allocate(state.dimensions, ("x", "y", "t"))
            TA = update(TA, at[2:-2, 2:-2, :], self._read_var_from_nc("ta", self._input_dir, 'states_hm.nc')[npx.newaxis, :, :])

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
            vs.re_rg = update(vs.re_rg, at[2:-2, 2:-2], vs.RE_RG[2:-2, 2:-2, vs.itt])
            vs.re_rl = update(vs.re_rl, at[2:-2, 2:-2], vs.RE_RL[2:-2, 2:-2, vs.itt])
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], vs.S_RZ[2:-2, 2:-2, vs.itt])
            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2, vs.tau], vs.S_PWP_RZ[2:-2, 2:-2, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2, vs.tau], vs.S_SAT_RZ[2:-2, 2:-2, vs.itt])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], vs.S_SS[2:-2, 2:-2, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2, vs.tau], vs.S_PWP_SS[2:-2, 2:-2, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2, vs.tau], vs.S_SAT_SS[2:-2, 2:-2, vs.itt])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])

            vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt])
            vs.M_in = update(
                vs.M_in,
                at[2:-2, 2:-2], vs.C_in[2:-2, 2:-2] * (vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2]),
            )

        @roger_routine
        def set_diagnostics(self, state):
            pass

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

    tms = transport_model_structure.replace("_", " ")
    model = SVATCROPTRANSPORTSetup()
    model._set_lys(lys_experiment)
    model._set_tm_structure(tms)
    identifier = f'SVATCROPTRANSPORT_{transport_model_structure}_{lys_experiment}_bromide'
    model._set_identifier(identifier)
    input_path = model._base_path / "input" / lys_experiment
    model._set_input_dir(input_path)
    write_forcing_tracer(input_path, 'Br')
    model.setup()
    model.warmup()
    model.run()
    return


if __name__ == "__main__":
    main()
