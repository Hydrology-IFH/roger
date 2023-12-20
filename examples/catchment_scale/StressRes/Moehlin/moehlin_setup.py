from pathlib import Path
import h5netcdf
import pandas as pd
import numpy as onp
import yaml
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-td", "--temp-dir", type=str, default=Path(__file__).parent / "output")
@roger_base_cli
def main(temp_dir):
    from roger import roger_routine, roger_kernel, KernelOutput
    from roger.models.oneD import ONEDSetup
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at
    from roger.core.surface import calc_parameters_surface_kernel
    from roger.tools.setup import write_forcing
    import roger.lookuptables as lut

    class MoehlinSetup(ONEDSetup):
        """A 1D model."""

        # custom attributes required by helper functions
        _base_path = Path(__file__).parent
        _input_dir = _base_path / "input"
        # load configuration file
        _file_config = _base_path / "config.yml"
        with open(_file_config, "r") as file:
            _config = yaml.safe_load(file)

        # custom helper functions
        def _read_var_from_nc(self, var, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables[var]
                if var in ["lu_id"]:
                    vals = npx.array(var_obj)
                    vals1 = npx.where(vals <= -9999, 999, vals)
                elif var in ["dmph", "dmpv", "lmpv", "z_soil", "sealing", "prec_weight", "pet_weight", "ta_offset"]:
                    vals = npx.array(var_obj)
                    vals1 = npx.where(vals <= -9999, 0, vals)
                else:
                    vals = npx.array(var_obj)
                    vals1 = npx.where(vals <= -9999, npx.nan, vals)
                return vals1

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = self._config["identifier"]

            # total grid numbers in x- and y-direction
            settings.nx, settings.ny = self._config["nx"], self._config["ny"]

            # spatial discretization (in meters)
            settings.dx = self._config["dx"]
            settings.dy = self._config["dy"]

            # origin of spatial grid
            settings.x_origin = self._config["x_origin"]
            settings.y_origin = self._config["y_origin"]

            # derive total number of time steps from forcing
            settings.runlen = self._get_runlen(self._input_dir, "forcing.nc")
            settings.nitt_forc = len(self._read_var_from_nc("Time", self._input_dir, "forcing.nc"))

            # origin of time steps (e.g. 01-01-2023)
            settings.time_origin = self._get_time_origin(self._input_dir, "forcing.nc")

            # enable specific processes
            settings.enable_lateral_flow = True
            settings.enable_macropore_lower_boundary_condition = False
            settings.enable_adaptive_time_stepping = True

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "x",
                "y",
            ],
        )
        def set_grid(self, state):
            vs = state.variables
            settings = state.settings

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
            vs = state.variables

            # land use-dependent interception storage
            vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
            # land use-dependent ground cover
            vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
            # land use-dependent maximum ground cover
            vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
            # land use-dependent maximum ground cover
            vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
            # land use-dependent rooting depth
            vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)
            # macropore flow velocities
            vs.lut_mlms = update(vs.lut_mlms, at[:, :], lut.ARR_MLMS)

        @roger_routine
        def set_topography(self, state):
            vs = state.variables

            # catchment mask (bool)
            dgm = self._read_var_from_nc("dgm", self._input_dir, "parameters.nc")
            vs.maskCatch = update(
                vs.maskCatch,
                at[2:-2, 2:-2],
                onp.isfinite(dgm),
            )
            # surface slope (-)
            vs.slope = update(
                vs.slope_per,
                at[2:-2, 2:-2],
                self._read_var_from_nc("slope", self._input_dir, "parameters.nc")/100,
            )
            # slope in percentage
            vs.slope_per = update(
                vs.slope_per,
                at[2:-2, 2:-2],
                self._read_var_from_nc("slope", self._base_path, "parameters.nc"),
            )
            # surface elevation (m)
            vs.elev = update(
                vs.dgm,
                at[2:-2, 2:-2],
                self._read_var_from_nc("dgm", self._base_path, "parameters.nc"),
            )

        @roger_routine
        def set_parameters_setup(self, state):
            vs = state.variables

            # land use ID (see README for description)
            vs.lu_id = update(
                vs.lu_id,
                at[2:-2, 2:-2],
                self._read_var_from_nc("lanu", self._base_path, "parameters.nc")
            )
            # degree of sealing (-)
            vs.sealing = update(
                vs.sealing,
                at[2:-2, 2:-2],
                self._read_var_from_nc("vers", self._base_path, "parameters.nc")/100) # % to -
            # total surface depression storage (mm)
            vs.S_dep_tot = update(
                vs.S_dep_tot,
                at[2:-2, 2:-2],
                self._read_var_from_nc("MULDE", self._base_path, "parameters.nc"))
            # soil depth (mm)
            vs.z_soil = update(
                vs.z_soil,
                at[2:-2, 2:-2],
                self._read_var_from_nc("GRUND", self._base_path, "parameters.nc"),
            )
            # groundwater table depth
            vs.z_gw = update(
                vs.z_gw,
                at[2:-2, 2:-2],
                self._read_var_from_nc("gwfa_gew", self._base_path, "parameters.nc")
            )
            # density of vertical macropores (1/m2)
            vs.dmpv = update(
                vs.dmpv,
                at[2:-2, 2:-2],
                self._read_var_from_nc("MPD_V", self._base_path, "parameters.nc"),
            )
            # length of vertical macropores (1/m2)
            vs.lmpv = update(
                vs.lmpv,
                at[2:-2, 2:-2],
                self._read_var_from_nc("MPL_V", self._base_path, "parameters.nc"),
            )
            # density of horizontal macropores (1/m2)
            vs.dmph = update(
                vs.dmph,
                at[2:-2, 2:-2],
                self._read_var_from_nc("MPD_H", self._base_path, "parameters.nc"),
            )
            # air capacity (-)
            vs.theta_ac = update(
                vs.theta_ac,
                at[2:-2, 2:-2],
                self._read_var_from_nc("LK", self._base_path, "parameters.nc")/100,
            )
            # usable field capacity (-)
            vs.theta_ufc = update(
                vs.theta_ufc,
                at[2:-2, 2:-2],
                self._read_var_from_nc("NFK", self._base_path, "parameters.nc")/100
            )
            # permanent wilting point (-)
            vs.theta_pwp = update(
                vs.theta_pwp,
                at[2:-2, 2:-2],
                self._read_var_from_nc("PWP", self._base_path, "parameters.nc")/100
            )
            # saturated hydraulic conductivity (mm/h)
            vs.ks = update(
                vs.ks,
                at[2:-2, 2:-2],
                self._read_var_from_nc("KS", self._base_path, "parameters.nc")
            )
            # hydraulic conductivity of bedrock/saturated zone (mm/h)
            vs.kf = update(
                vs.kf,
                at[2:-2, 2:-2],
                self._read_var_from_nc("TP", self._base_path, "parameters.nc")
            )
            # weight factor of precipitation (-)
            vs.prec_weight = update(
                vs.prec_weight,
                at[2:-2, 2:-2],
                self._read_var_from_nc("F_n_h_y", self._base_path, "parameters.nc")/100
            )
            # weight factor of air temperature (-)
            vs.ta_offset = update(
                vs.ta_offset,
                at[2:-2, 2:-2],
                self._read_var_from_nc("F_t", self._base_path, "parameters.nc")
            )
            # weight factor of potential evapotranspiration (-)
            vs.pet_weight = update(
                vs.pet_weight,
                at[2:-2, 2:-2],
                self._read_var_from_nc("F_et", self._base_path, "parameters.nc")/100
            )

        @roger_routine
        def set_parameters(self, state):
            vs = state.variables

            if (vs.month[vs.tau] != vs.month[vs.taum1]) & (vs.itt > 1):
                vs.update(calc_parameters_surface_kernel(state))

        @roger_routine
        def set_initial_conditions_setup(self, state):
            pass

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables

            # interception storage of upper surface layer (mm)
            vs.S_int_top = update(vs.S_int_top, at[2:-2, 2:-2, : vs.taup1], 0)
            # snow water equivalent stored in upper surface layer (mm)
            vs.swe_top = update(vs.swe_top, at[2:-2, 2:-2, : vs.taup1], 0)
            # interception storage of lower surface layer (mm)
            vs.S_int_ground = update(vs.S_int_ground, at[2:-2, 2:-2, : vs.taup1], 0)
            # snow water equivalent stored in lower surface layer (mm)
            vs.swe_ground = update(vs.swe_ground, at[2:-2, 2:-2, : vs.taup1], 0)
            # surface depression storage (mm)
            vs.S_dep = update(vs.S_dep, at[2:-2, 2:-2, : vs.taup1], 0)
            # snow cover storage (mm)
            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, : vs.taup1], 0)
            # snow water equivalent of snow cover (mm)
            vs.swe = update(vs.swe, at[2:-2, 2:-2, : vs.taup1], 0)
            # soil water content of root zone/upper soil layer (mm/h)
            vs.theta_rz = update(
                vs.theta_rz,
                at[2:-2, 2:-2, : vs.taup1],
                vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis],
            )
            # soil water content of subsoil/lower soil layer (mm/h)
            vs.theta_ss = update(
                vs.theta_ss,
                at[2:-2, 2:-2, : vs.taup1],
                vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis],
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
                "PREC",
                "TA",
                "PET",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC = update(vs.PREC, at[:], self._read_var_from_nc("PREC", self._input_dir, "forcing.nc")[0, 0, :])
            vs.TA = update(vs.TA, at[:], self._read_var_from_nc("TA", self._input_dir, "forcing.nc")[0, 0, :])
            vs.PET = update(vs.PET, at[:], self._read_var_from_nc("PET", self._input_dir, "forcing.nc")[0, 0, :])

        @roger_routine
        def set_forcing(self, state):
            vs = state.variables

            condt = vs.time % (24 * 60 * 60) == 0
            if condt:
                vs.itt_day = 0
                vs.year = update(
                    vs.year, at[1], self._read_var_from_nc("YEAR", self._input_dir, "forcing.nc")[vs.itt_forc]
                )
                vs.month = update(
                    vs.month, at[1], self._read_var_from_nc("MONTH", self._input_dir, "forcing.nc")[vs.itt_forc]
                )
                vs.doy = update(
                    vs.doy, at[1], self._read_var_from_nc("DOY", self._input_dir, "forcing.nc")[vs.itt_forc]
                )
                vs.prec_day = update(
                    vs.prec_day, at[:, :, :], vs.PREC[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24] * vs.prec_weight[:, :, npx.newaxis]
                )
                vs.ta_day = update(
                    vs.ta_day, at[:, :, :], vs.TA[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24] + vs.ta_offset[:, :, npx.newaxis]
                )
                vs.pet_day = update(
                    vs.pet_day, at[:, :, :], vs.PET[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24] * vs.pet_weight[:, :, npx.newaxis]
                )
                vs.itt_forc = vs.itt_forc + 6 * 24

        @roger_routine
        def set_diagnostics(self, state, base_path=temp_dir):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = self._config["OUTPUT_RATE"]
            # values are aggregated to daily
            diagnostics["rate"].output_frequency = self._config["OUTPUT_FREQUENCY"] # in seconds
            diagnostics["rate"].sampling_frequency = 1
            if base_path:
                diagnostics["rate"].base_output_path = base_path

            diagnostics["collect"].output_variables = self._config["OUTPUT_COLLECT"]
            # values are aggregated to daily
            diagnostics["collect"].output_frequency = self._config["OUTPUT_FREQUENCY"] # in seconds
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            # maximum bias of deterministic/numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error"]
            diagnostics["maximum"].output_frequency = self._config["OUTPUT_FREQUENCY"]
            diagnostics["maximum"].sampling_frequency = 1
            if base_path:
                diagnostics["maximum"].base_output_path = base_path

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            # shift variables backwards
            vs.update(after_timestep_kernel(state))

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.ta = update(
            vs.ta,
            at[2:-2, 2:-2, vs.taum1],
            vs.ta[2:-2, 2:-2, vs.tau],
        )
        vs.z_root = update(
            vs.z_root,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_root[2:-2, 2:-2, vs.tau],
        )
        vs.ground_cover = update(
            vs.ground_cover,
            at[2:-2, 2:-2, vs.taum1],
            vs.ground_cover[2:-2, 2:-2, vs.tau],
        )
        vs.S_sur = update(
            vs.S_sur,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_sur[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_top = update(
            vs.S_int_top,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_int_top[2:-2, 2:-2, vs.tau],
        )
        vs.S_int_ground = update(
            vs.S_int_ground,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_int_ground[2:-2, 2:-2, vs.tau],
        )
        vs.S_dep = update(
            vs.S_dep,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_dep[2:-2, 2:-2, vs.tau],
        )
        vs.S_snow = update(
            vs.S_snow,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_snow[2:-2, 2:-2, vs.tau],
        )
        vs.swe = update(
            vs.swe,
            at[2:-2, 2:-2, vs.taum1],
            vs.swe[2:-2, 2:-2, vs.tau],
        )
        vs.S_rz = update(
            vs.S_rz,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_rz[2:-2, 2:-2, vs.tau],
        )
        vs.S_ss = update(
            vs.S_ss,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_ss[2:-2, 2:-2, vs.tau],
        )
        vs.S_s = update(
            vs.S_s,
            at[2:-2, 2:-2, vs.taum1],
            vs.S_s[2:-2, 2:-2, vs.tau],
        )
        vs.S = update(
            vs.S,
            at[2:-2, 2:-2, vs.taum1],
            vs.S[2:-2, 2:-2, vs.tau],
        )
        vs.z_sat = update(
            vs.z_sat,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_sat[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf = update(
            vs.z_wf,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_wf[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t0 = update(
            vs.z_wf_t0,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_wf_t0[2:-2, 2:-2, vs.tau],
        )
        vs.z_wf_t1 = update(
            vs.z_wf_t1,
            at[2:-2, 2:-2, vs.taum1],
            vs.z_wf_t1[2:-2, 2:-2, vs.tau],
        )
        vs.y_mp = update(
            vs.y_mp,
            at[2:-2, 2:-2, vs.taum1],
            vs.y_mp[2:-2, 2:-2, vs.tau],
        )
        vs.y_sc = update(
            vs.y_sc,
            at[2:-2, 2:-2, vs.taum1],
            vs.y_sc[2:-2, 2:-2, vs.tau],
        )
        vs.theta_rz = update(
            vs.theta_rz,
            at[2:-2, 2:-2, vs.taum1],
            vs.theta_rz[2:-2, 2:-2, vs.tau],
        )
        vs.theta_ss = update(
            vs.theta_ss,
            at[2:-2, 2:-2, vs.taum1],
            vs.theta_ss[2:-2, 2:-2, vs.tau],
        )
        vs.theta = update(
            vs.theta,
            at[2:-2, 2:-2, vs.taum1],
            vs.theta[2:-2, 2:-2, vs.tau],
        )
        vs.k_rz = update(
            vs.k_rz,
            at[2:-2, 2:-2, vs.taum1],
            vs.k_rz[2:-2, 2:-2, vs.tau],
        )
        vs.k_ss = update(
            vs.k_ss,
            at[2:-2, 2:-2, vs.taum1],
            vs.k_ss[2:-2, 2:-2, vs.tau],
        )
        vs.k = update(
            vs.k,
            at[2:-2, 2:-2, vs.taum1],
            vs.k[2:-2, 2:-2, vs.tau],
        )
        vs.h_rz = update(
            vs.h_rz,
            at[2:-2, 2:-2, vs.taum1],
            vs.h_rz[2:-2, 2:-2, vs.tau],
        )
        vs.h_ss = update(
            vs.h_ss,
            at[2:-2, 2:-2, vs.taum1],
            vs.h_ss[2:-2, 2:-2, vs.tau],
        )
        vs.h = update(
            vs.h,
            at[2:-2, 2:-2, vs.taum1],
            vs.h[2:-2, 2:-2, vs.tau],
        )
        vs.z0 = update(
            vs.z0,
            at[2:-2, 2:-2, vs.taum1],
            vs.z0[2:-2, 2:-2, vs.tau],
        )
        vs.prec = update(
            vs.prec,
            at[2:-2, 2:-2, vs.taum1],
            vs.prec[2:-2, 2:-2, vs.tau],
        )
        vs.event_id = update(
            vs.event_id,
            at[vs.taum1],
            vs.event_id[vs.tau],
        )
        vs.year = update(
            vs.year,
            at[vs.taum1],
            vs.year[vs.tau],
        )
        vs.month = update(
            vs.month,
            at[vs.taum1],
            vs.month[vs.tau],
        )
        vs.doy = update(
            vs.doy,
            at[vs.taum1],
            vs.doy[vs.tau],
        )

        return KernelOutput(
            ta=vs.ta,
            z_root=vs.z_root,
            ground_cover=vs.ground_cover,
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
            z0=vs.z0,
            prec=vs.prec,
            event_id=vs.event_id,
            year=vs.year,
            month=vs.month,
            doy=vs.doy,
            k_rz=vs.k_rz,
            k_ss=vs.k_ss,
            k=vs.k,
        )

    # initializes the model structure
    model = MoehlinSetup()
    # writes the forcing data to netcdf
    write_forcing(model._input_dir)
    # runs the model setup
    model.setup()
    # iterate over time steps
    model.run()
    return


if __name__ == "__main__":
    main()
