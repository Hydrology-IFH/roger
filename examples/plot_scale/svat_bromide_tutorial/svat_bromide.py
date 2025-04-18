from pathlib import Path
import h5netcdf
import numpy as onp
import yaml
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-td", "--tmp-dir", type=str, default=Path(__file__).parent / "output")
@roger_base_cli
def main(tmp_dir):
    from roger import RogerSetup, roger_routine
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, where
    from roger.tools.setup import write_forcing_tracer

    class SVATBROMIDESetup(RogerSetup):
        """A SVAT bromide transport model.
        """
        # custom attributes required by helper functions
        _base_path = Path(__file__).parent
        _input_dir = _base_path / "input"
        # load configuration file
        _file_config = _base_path / "config.yml"
        with open(_file_config, 'r') as file:
            _config = yaml.safe_load(file)
        _tm_structure = _config["TRANSPORT_MODEL_STRUCTURE"].replace("_", " ")

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

        def _get_time_origin(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                date = infile.variables['Time'].attrs['time_origin'].split(" ")[0]
                return f"{date} 00:00:00"

        def _set_bromide_input(self, state, nn_rain, nn_sol, prec, ta):
            vs = state.variables

            C_IN = allocate(state.dimensions, ("x", "y", "t"))
            M_IN = allocate(state.dimensions, ("x", "y", "t"))
            idx = allocate(state.dimensions, ("x", "y", "t"))
            idx = update(
                idx,
                at[2:-2, 2:-2, :], npx.arange(idx.shape[-1])[npx.newaxis, npx.newaxis, :],
            )

            mask_rain = (prec > 0) & (ta > 0)
            mask_sol = (vs.M_IN > 0)
            sol_idx = npx.zeros((nn_sol,), dtype=int)
            sol_idx = update(sol_idx, at[:], where(npx.any(mask_sol, axis=(0, 1)), size=nn_sol, fill_value=0)[0])
            rain_idx = npx.where(mask_rain, idx, 0)

            # join solute input on closest rainfall event
            for i in range(nn_sol):
                input_itt = npx.nanargmin(npx.where(rain_idx[2:-2, 2:-2, :] - sol_idx[i] < 0, npx.nan, rain_idx[2:-2, 2:-2, :] - sol_idx[i]), axis=-1)
                for x in range(input_itt.shape[0]):
                    for y in range(input_itt.shape[1]):
                        start_rain = input_itt[x, y]
                        end_rain = npx.max(npx.where(npx.cumsum(prec[x, y, start_rain:]) <= 20, npx.arange(prec.shape[-1])[start_rain:], 0))
                        if npx.sum(prec[x, y, start_rain:end_rain]) <= 0:
                            end_rain = end_rain + 1
                        # proportions for redistribution
                        M_IN = update(
                            M_IN,
                            at[x+2, y+2, start_rain:end_rain], vs.M_IN[x+2, y+2, sol_idx[i]] * (prec[x, y, start_rain:end_rain] / npx.sum(prec[x, y, start_rain:end_rain])),
                        )

            C_IN = update(
                C_IN,
                at[2:-2, 2:-2, :], npx.where(prec > 0, M_IN[2:-2, 2:-2, :] / prec, 0),
            )

            return M_IN, C_IN

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = self._config["identifier"]
            settings.sas_solver = self._config["SAS_SOLVER"]
            # number of substeps
            settings.sas_solver_substeps = 6
            if settings.sas_solver in ['RK4', 'Euler']:
                settings.h = 1 / settings.sas_solver_substeps

            # total grid numbers in x- and y-direction
            settings.nx, settings.ny = 1, 1
            # length of simulation (in seconds)
            settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')
            # length of warmup simulation (in seconds)
            settings.runlen_warmup = settings.runlen
            # total number of iterations
            settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
            # maximum water age (in days)
            settings.ages = settings.nitt
            settings.nages = settings.ages + 1

            # spatial discretization (in meters)
            settings.dx = 1
            settings.dy = 1

            # origin of spatial grid
            settings.x_origin = 0.0
            settings.y_origin = 0.0
            # origin of time steps (e.g. 01-01-2023)
            settings.time_origin = self._get_time_origin(self._input_dir, 'forcing_tracer.nc')

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
            # spatial grid
            dx = allocate(state.dimensions, ("x"))
            dx = update(dx, at[:], settings.dx)
            dy = allocate(state.dimensions, ("y"))
            dy = update(dy, at[:], settings.dy)
            # distance from origin
            vs.x = update(vs.x, at[3:-2], settings.x_origin + npx.cumsum(dx[3:-2]))
            vs.y = update(vs.y, at[3:-2], settings.y_origin + npx.cumsum(dy[3:-2]))

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

            # storage volumes at permanent wilting point and at saturation
            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])

            # partition coefficients
            vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], 0.8)
            vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], 0.8)

            # SAS parameterization
            if settings.tm_structure == "complete-mixing":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 1)
            elif settings.tm_structure == "piston":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 100)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 100)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
            elif settings.tm_structure == "kumaraswamy":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 2], 100)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], 25)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 3)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
            elif settings.tm_structure == "power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 0.2)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 2)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 3)

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

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
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

            TA = self._read_var_from_nc("ta", self._base_path, 'states_hm.nc')[npx.newaxis, :, :]
            PREC = self._read_var_from_nc("prec", self._base_path, 'states_hm.nc')[npx.newaxis, :, :]

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
                "S_rz",
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

            vs.ta = update(vs.ta, at[2:-2, 2:-2], self._read_var_from_nc("ta", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("prec", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mat_rz", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mp_rz", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt] + self._read_var_from_nc("inf_sc_rz", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], self._read_var_from_nc("inf_ss", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.transp = update(vs.transp, at[2:-2, 2:-2], self._read_var_from_nc("transp", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], self._read_var_from_nc("evap_soil", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.cpr_rz = update(vs.cpr_rz, at[2:-2, 2:-2], self._read_var_from_nc("cpr_rz", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], self._read_var_from_nc("q_rz", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], self._read_var_from_nc("q_ss", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc')[npx.newaxis, :, vs.itt])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])

            vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt])
            vs.M_in = update(
                vs.M_in,
                at[2:-2, 2:-2], vs.C_in[2:-2, 2:-2] * (vs.inf_mat_rz[2:-2, 2:-2] + vs.inf_pf_rz[2:-2, 2:-2] + vs.inf_pf_ss[2:-2, 2:-2]),
            )

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = self._config["OUTPUT_RATE"]
            diagnostics["rate"].output_frequency = 24 * 60 * 60
            diagnostics["rate"].sampling_frequency = 1
            if base_path:
                diagnostics["rate"].base_output_path = base_path

            diagnostics["average"].output_variables = self._config["OUTPUT_AVERAGE"]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            diagnostics["collect"].output_variables = self._config["OUTPUT_COLLECT"]
            diagnostics["collect"].output_frequency = 24 * 60 * 60
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            # maximum bias of numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error", "dC_num_error"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            pass

    # initializes the model structure
    model = SVATBROMIDESetup()
    # writes the forcing data to netcdf
    write_forcing_tracer(model._input_dir, 'Br')
    # runs the model setup
    model.setup()
    # runs the model warmup
    model.warmup()
    # iterate over time steps
    model.run()
    return


if __name__ == "__main__":
    main()
