from pathlib import Path
import os
import h5netcdf
import pandas as pd
import numpy as onp
from roger.cli.roger_run_base import roger_base_cli


@roger_base_cli
def main():
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at
    import roger.lookuptables as lut
    from roger.tools.make_toy_data import make_toy_forcing_event

    class ONEDEVENTSetup(RogerSetup):
        """A 1D model for a single event.
        """
        _base_path = Path(__file__).parent
        _input_dir = None
        _identifier = None
        _initial_conditions = None

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

        def _read_var_from_csv(self, var, path_dir, file):
            csv_file = path_dir / file
            infile = pd.read_csv(csv_file, sep=';', skiprows=1)
            var_obj = infile.loc[:, var]
            return npx.array(var_obj)[:, npx.newaxis]

        def _get_nx(self, path_dir, file):
            csv_file = path_dir / file
            infile = pd.read_csv(csv_file, sep=';', skiprows=1)
            return len(infile.index)

        def _get_nitt(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['Time']
                return len(onp.array(var_obj))

        def _set_identifier(self, identifier):
            self._identifier = identifier
            self._initial_conditions = identifier.split('_')[-1]

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = self._identifier

            settings.nx, settings.ny = self._get_nx(self._base_path, 'parameter_grid.csv'), 1
            settings.nitt = self._get_nitt(self._input_dir, 'forcing.nc')
            settings.runlen = settings.nitt * 10 * 60

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0

            settings.enable_groundwater_boundary = False
            settings.enable_lateral_flow = True
            settings.enable_routing = False

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "dt_secs",
                "dt",
                "x",
                "y",
            ],
        )
        def set_grid(self, state):
            vs = state.variables

            # temporal grid
            vs.dt_secs = 10 * 60
            vs.dt = 1 / 6
            # spatial grid
            dx = allocate(state.dimensions, ("x"))
            dx = update(dx, at[:], 1)
            dy = allocate(state.dimensions, ("y"))
            dy = update(dy, at[:], 1)
            vs.x = update(vs.x, at[3:-2], npx.cumsum(dx[3:-2]))
            vs.y = update(vs.y, at[3:-2], npx.cumsum(dy[3:-2]))

        @roger_routine
        def set_look_up_tables(self, state):
            vs = state.variables

            vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
            vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
            vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
            vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
            vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)
            vs.lut_mlms = update(vs.lut_mlms, at[:, :], lut.ARR_MLMS)

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine
        def set_parameters_setup(self, state):
            vs = state.variables

            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], self._read_var_from_csv("lu_id", self._base_path,  "parameter_grid.csv"))
            vs.sealing = update(vs.sealing, at[2:-2, 2:-2], 0)
            vs.slope = update(vs.slope, at[2:-2, 2:-2], self._read_var_from_csv("slope", self._base_path,  "parameter_grid.csv"))
            vs.slope_per = update(vs.slope_per, at[2:-2, 2:-2], vs.slope[2:-2, 2:-2] * 100)
            vs.S_dep_tot = update(vs.S_dep_tot, at[2:-2, 2:-2], self._read_var_from_csv("S_dep_tot", self._base_path,  "parameter_grid.csv"))
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], self._read_var_from_csv("z_soil", self._base_path,  "parameter_grid.csv"))
            vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], self._read_var_from_csv("dmpv", self._base_path,  "parameter_grid.csv"))
            vs.dmph = update(vs.dmph, at[2:-2, 2:-2], self._read_var_from_csv("dmph", self._base_path,  "parameter_grid.csv"))
            vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], self._read_var_from_csv("lmpv", self._base_path,  "parameter_grid.csv"))
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], self._read_var_from_csv("theta_ac", self._base_path,  "parameter_grid.csv"))
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], self._read_var_from_csv("theta_ufc", self._base_path,  "parameter_grid.csv"))
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], self._read_var_from_csv("theta_pwp", self._base_path,  "parameter_grid.csv"))
            vs.ks = update(vs.ks, at[2:-2, 2:-2], self._read_var_from_csv("ks", self._base_path,  "parameter_grid.csv"))
            vs.kf = update(vs.kf, at[2:-2, 2:-2], self._read_var_from_csv("kf", self._base_path,  "parameter_grid.csv"))

        @roger_routine
        def set_parameters(self, state):
            pass

        @roger_routine
        def set_initial_conditions_setup(self, state):
            pass

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables

            vs.S_int_top = update(vs.S_int_top, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.swe_top = update(vs.swe_top, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.S_int_ground = update(vs.S_int_ground, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.swe_ground = update(vs.swe_ground, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.S_dep = update(vs.S_dep, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, :vs.taup1], 0)
            vs.swe = update(vs.swe, at[2:-2, 2:-2, :vs.taup1], 0)
            if self._initial_conditions == "dry":
                vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + 0.1 * (vs.theta_ufc[2:-2, 2:-2, npx.newaxis] + vs.theta_ac[2:-2, 2:-2, npx.newaxis]))
                vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + 0.1 * (vs.theta_ufc[2:-2, 2:-2, npx.newaxis] + vs.theta_ac[2:-2, 2:-2, npx.newaxis]))
            elif self._initial_conditions == "normal":
                vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis])
                vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis])
            elif self._initial_conditions == "wet":
                vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + 0.9 * (vs.theta_ufc[2:-2, 2:-2, npx.newaxis] + vs.theta_ac[2:-2, 2:-2, npx.newaxis]))
                vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + 0.9 * (vs.theta_ufc[2:-2, 2:-2, npx.newaxis] + vs.theta_ac[2:-2, 2:-2, npx.newaxis]))
            else:
                vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_csv("theta", self._base_path,  "parameter_grid.csv")[:, :, npx.newaxis])
                vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_csv("theta", self._base_path,  "parameter_grid.csv")[:, :, npx.newaxis])
            vs.z_sat = update(vs.z_sat, at[2:-2, 2:-2, vs.taum1], (vs.theta_ss[2:-2, 2:-2, vs.tau] - (vs.theta_pwp[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2])) / vs.theta_ac[2:-2, 2:-2])
            vs.z_sat = update(vs.z_sat, at[2:-2, 2:-2, vs.tau], (vs.theta_ss[2:-2, 2:-2, vs.tau] - (vs.theta_pwp[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2])) / vs.theta_ac[2:-2, 2:-2])

        @roger_routine
        def set_boundary_conditions_setup(self, state):
            pass

        @roger_routine
        def set_boundary_conditions(self, state):
            pass

        @roger_routine
        def set_forcing_setup(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "itt",
                "prec",
                "ta",
                "event_id",
                "tau",
            ],
        )
        def set_forcing(self, state):
            vs = state.variables

            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], 0)
            vs.ta = update(vs.ta, at[2:-2, 2:-2], self._read_var_from_nc("TA", self._input_dir, 'forcing.nc')[:, :, vs.itt])
            vs.event_id = update(vs.event_id, at[vs.tau], 1)

        @roger_routine
        def set_diagnostics(self, state):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = ["prec", "inf_mat", "inf_mp", "inf_sc", "q_ss", "q_sub", "q_sub_mp", "q_sub_mat", "q_hof", "q_sof", "int_rain_top", "int_rain_ground", "int_prec"]
            diagnostics["rate"].output_frequency = 10 * 60
            diagnostics["rate"].sampling_frequency = 1

            diagnostics["collect"].output_variables = ["theta", "z0", "S_s"]
            diagnostics["collect"].output_frequency = 10 * 60
            diagnostics["collect"].sampling_frequency = 1

            # maximum bias of deterministic/numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error", "z_sat"]
            diagnostics["maximum"].output_frequency = 24 * 60 * 60
            diagnostics["maximum"].sampling_frequency = 1

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.ta = update(
            vs.ta,
            at[2:-2, 2:-2, vs.taum1], vs.ta[2:-2, 2:-2, vs.tau],
        )
        vs.S_sur = update(
            vs.S_sur,
            at[2:-2, 2:-2, vs.taum1], vs.S_sur[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_top = update(
            vs.S_int_top,
            at[2:-2, 2:-2, vs.taum1], vs.S_int_top[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_ground = update(
            vs.S_int_ground,
            at[2:-2, 2:-2, vs.taum1], vs.S_int_ground[2:-2, 2:-2, vs.tau],
        )
        vs.S_dep = update(
            vs.S_dep,
            at[2:-2, 2:-2, vs.taum1], vs.S_dep[2:-2, 2:-2, vs.tau],
        )
        vs.S_snow = update(
            vs.S_snow,
            at[2:-2, 2:-2, vs.taum1], vs.S_snow[2:-2, 2:-2, vs.tau],
        )
        vs.swe = update(
            vs.swe,
            at[2:-2, 2:-2, vs.taum1], vs.swe[2:-2, 2:-2, vs.tau],
        )
        vs.S_rz = update(
            vs.S_rz,
            at[2:-2, 2:-2, vs.taum1], vs.S_rz[2:-2, 2:-2, vs.tau],
        )
        vs.S_ss = update(
            vs.S_ss,
            at[2:-2, 2:-2, vs.taum1], vs.S_ss[2:-2, 2:-2, vs.tau],
        )
        vs.S_s = update(
            vs.S_s,
            at[2:-2, 2:-2, vs.taum1], vs.S_s[2:-2, 2:-2, vs.tau],
        )
        vs.S = update(
            vs.S,
            at[2:-2, 2:-2, vs.taum1], vs.S[2:-2, 2:-2, vs.tau],
        )
        vs.z_sat = update(
            vs.z_sat,
            at[2:-2, 2:-2, vs.taum1], vs.z_sat[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf = update(
            vs.z_wf,
            at[2:-2, 2:-2, vs.taum1], vs.z_wf[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t0 = update(
            vs.z_wf_t0,
            at[2:-2, 2:-2, vs.taum1], vs.z_wf_t0[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t1 = update(
            vs.z_wf_t1,
            at[2:-2, 2:-2, vs.taum1], vs.z_wf_t1[2:-2, 2:-2, vs.tau],
        )
        vs.y_mp = update(
            vs.y_mp,
            at[2:-2, 2:-2, vs.taum1], vs.y_mp[2:-2, 2:-2, vs.tau],
        )
        vs.y_sc = update(
            vs.y_sc,
            at[2:-2, 2:-2, vs.taum1], vs.y_sc[2:-2, 2:-2, vs.tau],
        )
        vs.theta_rz = update(
            vs.theta_rz,
            at[2:-2, 2:-2, vs.taum1], vs.theta_rz[2:-2, 2:-2, vs.tau],
        )
        vs.theta_ss = update(
            vs.theta_ss,
            at[2:-2, 2:-2, vs.taum1], vs.theta_ss[2:-2, 2:-2, vs.tau],
        )
        vs.theta = update(
            vs.theta,
            at[2:-2, 2:-2, vs.taum1], vs.theta[2:-2, 2:-2, vs.tau],
        )
        vs.k_rz = update(
            vs.k_rz,
            at[2:-2, 2:-2, vs.taum1], vs.k_rz[2:-2, 2:-2, vs.tau],
        )
        vs.k_ss = update(
            vs.k_ss,
            at[2:-2, 2:-2, vs.taum1], vs.k_ss[2:-2, 2:-2, vs.tau],
        )
        vs.k = update(
            vs.k,
            at[2:-2, 2:-2, vs.taum1], vs.k[2:-2, 2:-2, vs.tau],
        )
        vs.h_rz = update(
            vs.h_rz,
            at[2:-2, 2:-2, vs.taum1], vs.h_rz[2:-2, 2:-2, vs.tau],
        )
        vs.h_ss = update(
            vs.h_ss,
            at[2:-2, 2:-2, vs.taum1], vs.h_ss[2:-2, 2:-2, vs.tau],
        )
        vs.h = update(
            vs.h,
            at[2:-2, 2:-2, vs.taum1], vs.h[2:-2, 2:-2, vs.tau],
        )
        vs.prec = update(
            vs.prec,
            at[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau],
        )

        return KernelOutput(
            ta=vs.ta,
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
            theta_rz=vs.theta_rz,
            theta_ss=vs.theta_ss,
            theta=vs.theta,
            h_rz=vs.h_rz,
            h_ss=vs.h_ss,
            h=vs.h,
            k_rz=vs.k_rz,
            k_ss=vs.k_ss,
            k=vs.k,
            prec=vs.prec,
        )

    initial_conditions = ["dry", "normal", "wet", "75"]
    rainfall_scenarios = ["rain", "block-rain", "rain-with-break", "heavyrain",
                          "heavyrain-normal", "heavyrain-gamma",
                          "heavyrain-gamma-reverse", "block-heavyrain"]
    for initial_condition in initial_conditions:
        for rainfall_scenario in rainfall_scenarios:
            model = ONEDEVENTSetup()
            identifier = f'ONEDEVENT_{rainfall_scenario}_{initial_condition}'
            model._set_identifier(identifier)
            path_input = model._base_path / "input"
            path_scenario = model._base_path / "input" / rainfall_scenario
            model._set_input_dir(path_scenario)
            if not os.path.exists(path_input):
                os.mkdir(path_input)
            if not os.path.exists(path_scenario):
                os.mkdir(path_scenario)
            make_toy_forcing_event(path_scenario, nhours=5, event_type=rainfall_scenario)
            model.setup()
            model.run()
    return


if __name__ == "__main__":
    main()
