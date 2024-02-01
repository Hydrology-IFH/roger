from pathlib import Path
import h5netcdf
import numpy as onp
import pandas as pd
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-lys", "--lys-experiment", type=click.Choice(["lys2", "lys3", "lys4", "lys8", "lys9"]), default="lys3")
@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'advection-dispersion-power', 'time-variant_advection-dispersion-power']), default='complete-mixing')
@click.option("-ss", "--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-ecp", "--crop-partitioning", is_flag=True)
@click.option("-td", "--tmp-dir", type=str, default=None)
@roger_base_cli
def main(lys_experiment, transport_model_structure, sas_solver, crop_partitioning, tmp_dir):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, update_add, at, where, scipy_stats as sstx
    from roger.tools.setup import write_forcing_tracer
    import roger.lookuptables as lut
    from roger.core.crop import update_alpha_transp

    class SVATCROPNITRATESetup(RogerSetup):
        """A SVAT transport model for nitrate including
        crop phenology/crop rotation.
        """
        _base_path = Path(__file__).parent
        if tmp_dir:
            # read fluxes and states from local SSD on cluster node
            _input_dir1 = Path(tmp_dir)
        else:
            _input_dir1 = _base_path.parent / "output" / "svat_crop_monte_carlo"
        _input_dir2 = _base_path.parent / "input" / f"{lys_experiment}"

        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                return npx.array(var_obj, dtype=npx.float32)

        def _read_var_from_csv(self, var, path_dir, file):
            csv_file = path_dir / file
            infile = pd.read_csv(csv_file, sep=';', skiprows=1)
            var_obj = infile.loc[:, var]
            return npx.array(var_obj, dtype=npx.float32)[:, npx.newaxis]
        
        def _get_nsamples(self, path_dir, file):
            csv_file = path_dir / file
            infile = pd.read_csv(csv_file, sep=';', skiprows=1)
            return len(infile.index)

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

        def _set_nitrate_input(self, state, nn_rain, nn_sol, inf):
            vs = state.variables

            NMIN_IN = allocate(state.dimensions, ("x", "y", "t"))

            mask_rain = (inf > 0)
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
                input_itt = npx.nanargmin(npx.where(rain_idx - sol_idx[i] < 0, npx.nan, rain_idx - sol_idx[i]))
                start_rain = rain_idx[input_itt]
                rain_sum = update(
                    rain_sum,
                    at[:, :], npx.max(npx.where(npx.cumsum(inf[:, :, start_rain:], axis=-1) <= 20, npx.max(npx.cumsum(inf[:, :, start_rain:], axis=-1), axis=-1)[:, :, npx.newaxis], 0), axis=-1),
                )
                nn_end = npx.max(npx.where(npx.cumsum(inf[:, :, start_rain:]) <= 20, npx.max(npx.arange(npx.shape(inf)[2])[npx.newaxis, npx.newaxis, npx.shape(inf)[2]-start_rain], axis=-1), 0))
                end_rain = update(end_rain, at[:], start_rain + nn_end)
                end_rain = update(end_rain, at[:], npx.where(end_rain > npx.shape(inf)[2], npx.shape(inf)[2], end_rain))

                # proportions for redistribution
                NMIN_IN = update(
                    NMIN_IN,
                    at[:, :, start_rain:end_rain[0]], vs.M_IN[:, :, sol_idx[i], npx.newaxis] * (inf[:, :, start_rain:end_rain[0]] / rain_sum[:, :, npx.newaxis]),
                )

            # solute input concentration
            M_IN = NMIN_IN * 0.3
            C_IN = npx.where(inf > 0, M_IN / inf, 0)

            NMIN_IN1 = NMIN_IN * 0.7

            return M_IN, C_IN, NMIN_IN1

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = f"SVATCROPNITRATE_{transport_model_structure}_{lys_experiment}"
            settings.sas_solver = sas_solver
            settings.sas_solver_substeps = 8

            settings.nx, settings.ny = self._get_nsamples(self._base_path, f'parameters_for_{transport_model_structure}.csv'), 1
            settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
            settings.ages = 1000
            settings.nages = settings.ages + 1
            settings.runlen_warmup = 2 * 365 * 24 * 60 * 60
            settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = self._get_time_origin(self._input_dir, 'forcing_tracer.nc')

            settings.enable_crop_phenology = True
            settings.enable_crop_rotation = True
            settings.enable_offline_transport = True
            settings.enable_nitrate = True
            settings.enable_age_statistics = False
            settings.tm_structure = transport_model_structure
            settings.enable_crop_partitioning = crop_partitioning

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

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "lut_crops",
                "lut_crop_scale",
            ],
        )
        def set_look_up_tables(self, state):
            vs = state.variables

            vs.lut_crops = update(vs.lut_crops, at[:, :], lut.ARR_CP)
            # scale partition coefficient of crop solute uptake
            for i in range(vs.lut_crop_scale.shape[-1]):
                vs.lut_crop_scale = update(vs.lut_crop_scale, at[2:-2, 2:-2, i], 1)

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "S_PWP_RZ",
                "S_SAT_RZ",
                "S_PWP_SS",
                "S_SAT_SS",
                "S_pwp_rz",
                "S_pwp_ss",
                "S_sat_rz",
                "S_sat_ss",
                "sas_params_evap_soil",
                "sas_params_cpr_rz",
                "sas_params_transp",
                "sas_params_q_rz",
                "sas_params_q_ss",
                "sas_params_re_rg",
                "sas_params_re_rl",
                "alpha_transp",
                "alpha_q",
                "km_denit_rz",
                "km_denit_ss",
                "dmax_denit_rz",
                "dmax_denit_ss",
                "km_nit_rz",
                "km_nit_ss",
                "dmax_nit_rz",
                "dmax_nit_ss",
                "kmin_rz",
                "kmin_ss",
                "kfix_rz",
                "z_soil",
                "phi_soil_temp",
                "damp_soil_temp",
                "YEAR",
                "DOY",
                "LU_ID",
                "Z_ROOT",
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.S_PWP_RZ = update(vs.S_PWP_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.S_SAT_RZ = update(vs.S_SAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.S_PWP_SS = update(vs.S_PWP_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_ss", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.S_SAT_SS = update(vs.S_SAT_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_ss", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc')[:, :, 0])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._input_dir1, f'SVATCROP_{lys_experiment}.nc')[:, :, 0])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc')[:, :, 0])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._input_dir1, f'SVATCROP_{lys_experiment}.nc')[:, :, 0])

            if settings.enable_crop_partitioning:
                vs.update(update_alpha_transp(state))
            else:
                vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], self._read_var_from_csv("alpha_transp", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], self._read_var_from_csv("alpha_q", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            _c = 0.1
            vs.km_denit_rz = update(vs.km_denit_rz, at[2:-2, 2:-2], self._read_var_from_csv("km_denit", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.km_denit_ss = update(vs.km_denit_ss, at[2:-2, 2:-2], _c * self._read_var_from_csv("km_denit", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.dmax_denit_rz = update(vs.dmax_denit_rz, at[2:-2, 2:-2], self._read_var_from_csv("dmax_denit", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.dmax_denit_ss = update(vs.dmax_denit_ss, at[2:-2, 2:-2], _c * self._read_var_from_csv("dmax_denit", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.km_nit_rz = update(vs.km_nit_rz, at[2:-2, 2:-2], self._read_var_from_csv("km_nit", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.km_nit_ss = update(vs.km_nit_ss, at[2:-2, 2:-2], _c * self._read_var_from_csv("km_nit", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.dmax_nit_rz = update(vs.dmax_nit_rz, at[2:-2, 2:-2], self._read_var_from_csv("dmax_nit", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.dmax_nit_ss = update(vs.dmax_nit_ss, at[2:-2, 2:-2], _c * self._read_var_from_csv("dmax_nit", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.kmin_rz = update(vs.kmin_rz, at[2:-2, 2:-2], self._read_var_from_csv("kmin", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
            vs.kmin_ss = update(vs.kmin_ss, at[2:-2, 2:-2], _c * self._read_var_from_csv("kmin", self._base_path, f"parameters_for_{transport_model_structure}.csv"))

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
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], self._read_var_from_csv("k_transp", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], self._read_var_from_csv("k_q_rz", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], self._read_var_from_csv("k_q_ss", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 1], 0.5)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 1], 10)
            elif settings.tm_structure == "time-variant advection-dispersion-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 62)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], self._read_var_from_csv("c1_transp", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], self._read_var_from_csv("c2_transp", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 61)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], self._read_var_from_csv("c1_q_rz", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], self._read_var_from_csv("c2_q_rz", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 61)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], self._read_var_from_csv("c1_q_ss", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], self._read_var_from_csv("c2_q_ss", self._base_path, f"parameters_for_{transport_model_structure}.csv"))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 1], 0.5)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 1], 10)

        @roger_routine
        def set_parameters(self, state):
            vs = state.variables
            settings = state.settings

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], vs.S_PWP_RZ[2:-2, 2:-2, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], vs.S_SAT_RZ[2:-2, 2:-2, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], vs.S_PWP_SS[2:-2, 2:-2, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], vs.S_SAT_SS[2:-2, 2:-2, vs.itt])

            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], vs.S_pwp_ss[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2])

            if settings.enable_crop_partitioning:
                vs.update(update_alpha_transp(state))

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

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc')[:, :, vs.itt, npx.newaxis])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_ss", self._input_dir1, f'SVATCROP_{lys_experiment}.nc')[:, :, vs.itt, npx.newaxis])
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

            # initial nitrate concentration (in mg/l)
            vs.C_rz = update(vs.C_rz, at[2:-2, 2:-2, :vs.taup1], 5)
            vs.C_ss = update(vs.C_ss, at[2:-2, 2:-2, :vs.taup1], 5)
            # exponential distribution of mineral soil nitrogen
            # mineral soil nitrogen is decreasing with increasing age
            # p_dec = allocate(state.dimensions, ("x", "y", 2, "ages"))
            # p_dec1 = sstx.expon.pdf(npx.linspace(sstx.expon.ppf(0.001), sstx.expon.ppf(0.999), settings.ages))
            # p_dec2 = npx.sum(p_dec1)
            # p_dec3 = p_dec1 / p_dec2
            # p_dec = update(p_dec, at[:, :, :vs.taup1, :], p_dec3[npx.newaxis, npx.newaxis, npx.newaxis, :])
            # vs.Nmin_rz = update(vs.Nmin_rz, at[2:-2, 2:-2, :vs.taup1, :], 100 * p_dec[2:-2, 2:-2, :, :] * settings.dx * settings.dy * 100)
            # vs.Nmin_ss = update(vs.Nmin_ss, at[2:-2, 2:-2, :vs.taup1, :], 100 * p_dec[2:-2, 2:-2, :, :] * settings.dx * settings.dy * 100)
            vs.Nmin_rz = update(vs.Nmin_rz, at[2:-2, 2:-2, :vs.taup1, :], (100 / settings.ages) * settings.dx * settings.dy * 100)
            vs.Nmin_ss = update(vs.Nmin_ss, at[2:-2, 2:-2, :vs.taup1, :], (100 / settings.ages) * settings.dx * settings.dy * 100)
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
                "S_SS",
                "S_S",
                "TA",
                "ta_year",
                "NMIN_IN",
                "NORG_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.PREC_DIST_DAILY = update(vs.PREC_DIST_DAILY, at[2:-2, 2:-2, :], self._read_var_from_nc("prec", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.INF_MAT_RZ = update(vs.INF_MAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mat_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.INF_PF_RZ = update(vs.INF_PF_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mp_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc') + self._read_var_from_nc("inf_sc_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.INF_PF_SS = update(vs.INF_PF_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_ss", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.TRANSP = update(vs.TRANSP, at[2:-2, 2:-2, :], self._read_var_from_nc("transp", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.EVAP_SOIL = update(vs.EVAP_SOIL, at[2:-2, 2:-2, :], self._read_var_from_nc("evap_soil", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.CPR_RZ = update(vs.CPR_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("cpr_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.Q_RZ = update(vs.Q_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("q_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.Q_SS = update(vs.Q_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("q_ss", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.RE_RG = update(vs.RE_RG, at[2:-2, 2:-2, :], self._read_var_from_nc("re_rg", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.RE_RL = update(vs.RE_RL, at[2:-2, 2:-2, :], self._read_var_from_nc("re_rl", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.S_RZ = update(vs.S_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_rz", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.S_SS = update(vs.S_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_ss", self._input_dir1, f'SVATCROP_{lys_experiment}.nc'))
            vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
            vs.TA = update(
                vs.TA,
                at[:],
                self._read_var_from_nc(
                    "ta", self._input_dir, f'SVATCROP_{lys_experiment}.nc'
                )[0, 0, :],
            )
            vs.ta_year = update(
                vs.ta_year,
                at[2:-2, 2:-2],
                npx.mean(vs.TA[:365])[npx.newaxis, npx.newaxis],
            )

            # convert kg N/ha to mg/square meter
            vs.NMIN_IN = update(vs.NMIN_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("Nmin", self._input_dir2, 'forcing_tracer.nc') * 100 * settings.dx * settings.dy)
            vs.NORG_IN = update(vs.NORG_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("Norg", self._input_dir2, 'forcing_tracer.nc') * 100 * settings.dx * settings.dy)

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
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], vs.S_SS[2:-2, 2:-2, vs.itt])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])

            # apply nitrogen fertilizer
            vs.update(apply_fertilizer_kernel(state))

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = ["M_in", "M_q_ss", "M_transp"]
            diagnostics["rate"].output_frequency = 24 * 60 * 60
            diagnostics["rate"].sampling_frequency = 1
            if base_path:
                diagnostics["rate"].base_output_path = base_path

            diagnostics["average"].output_variables = [
                "C_q_ss"
            ]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            diagnostics["collect"].output_variables = ["M_s"]
            diagnostics["collect"].output_frequency = 24 * 60 * 60
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            # maximum bias of deterministic/numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error", "dC_num_error"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            pass

    @roger_kernel
    def apply_fertilizer_kernel(state):
        vs = state.variables
        settings = state.settings

        # apply nitrogen fertilizer
        vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], npx.where(vs.NMIN_IN[2:-2, 2:-2, vs.itt] > 0, vs.NMIN_IN[2:-2, 2:-2, vs.itt] * settings.dx * settings.dy * 100, vs.Nmin_in[2:-2, 2:-2]))
        # nitrogen deposition (10 kg N/ha/yr)
        vs.Nmin_in = update_add(vs.Nmin_in, at[2:-2, 2:-2], (10/365) * settings.dx * settings.dy * 100)              
        inf = vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2]
        vs.inf_in_tracer = update(vs.inf_in_tracer, at[2:-2, 2:-2], npx.where((vs.doy_dist[2:-2, 2:-2] == vs.doy_fert1[2:-2, 2:-2]) | (vs.doy_dist[2:-2, 2:-2] == vs.doy_fert2[2:-2, 2:-2]) | (vs.doy_dist[2:-2, 2:-2] == vs.doy_fert3[2:-2, 2:-2]), 0, vs.inf_in_tracer[2:-2, 2:-2]))
        vs.inf_in_tracer = update_add(vs.inf_in_tracer, at[2:-2, 2:-2], inf)
        inf_ratio = npx.where((inf/settings.cum_inf_for_N_input) < 1, inf/settings.cum_inf_for_N_input, 1)
        # dissolved nitrogen input
        vs.M_in = update(vs.M_in, at[2:-2, 2:-2], npx.where(vs.inf_in_tracer[2:-2, 2:-2] > 0, vs.Nmin_in[2:-2, 2:-2] * inf_ratio * 0.3, 0))
        vs.C_in = update(vs.C_in, at[2:-2, 2:-2], npx.where(vs.inf_in_tracer[2:-2, 2:-2] > 0, vs.M_in[2:-2, 2:-2]/inf, 0))
        # undissolved nitrogen input
        vs.Nmin_rz = update_add(
            vs.Nmin_rz,
            at[2:-2, 2:-2, vs.tau, 0],
            npx.where(vs.inf_in_tracer[2:-2, 2:-2] > 0, vs.Nmin_in[2:-2, 2:-2] * inf_ratio * 0.7, 0),
        )
        vs.Nmin_in = update_add(vs.Nmin_in, at[2:-2, 2:-2], -vs.Nmin_in[2:-2, 2:-2] * inf_ratio)
        vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], npx.where((vs.Nmin_in[2:-2, 2:-2] < 0), 0, vs.Nmin_in[2:-2, 2:-2]))
        vs.inf_in_tracer = update(vs.inf_in_tracer, at[2:-2, 2:-2], npx.where((vs.inf_in_tracer[2:-2, 2:-2] > settings.cum_inf_for_N_input), 0, vs.inf_in_tracer[2:-2, 2:-2]))

        return KernelOutput(
            Nmin_in=vs.Nmin_in,
            inf_in_tracer=vs.inf_in_tracer,
            M_in=vs.M_in,
            C_in=vs.C_in,
            Nmin_rz=vs.Nmin_rz,
        )

    model = SVATCROPNITRATESetup()
    write_forcing_tracer(model._input_dir, 'NO3')
    model.setup()
    model.warmup()
    model.run()
    return


if __name__ == "__main__":
    main()
