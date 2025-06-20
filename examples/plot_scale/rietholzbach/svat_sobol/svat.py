from pathlib import Path
import h5netcdf
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option(
    "-tms",
    "--transport-model-structure",
    type=click.Choice(
        ["complete-mixing", "piston", "advection-dispersion-power", "time-variant_advection-dispersion-power"]
    ),
    default="piston",
)
@click.option("-td", "--tmp-dir", type=str, default=None)
@roger_base_cli
def main(transport_model_structure, tmp_dir):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at
    from roger.core.surface import calc_parameters_surface_kernel
    from roger.tools.setup import write_forcing
    import roger.lookuptables as lut

    class SVATSetup(RogerSetup):
        """A SVAT model."""

        _base_path = Path(__file__).parent
        _input_dir = _base_path / "input"
        if tmp_dir:
            _tmp_dir = tmp_dir / "output"
        else:
            _tmp_dir = _base_path / "output"

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

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables["dt"]
                return onp.sum(onp.array(var_obj))

        def _get_nx(self, path_dir, file, group=None):
            nc_file = path_dir / file
            if group:
                with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                    var_obj = infile.groups[group].dimensions["x"].size
                    return int(var_obj)
            else:
                with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                    var_obj = infile.dimensions["x"].size
                    return int(var_obj)

        def _calc_pet_with_makkink(self, rs, ta, z, c1=0.63, c2=-0.05):
            """Calculate potential evapotranspiration according to Makkink.

            Args
            ----------
            rs : np.ndarray
                solar radiation (in MJ m-2)

            ta : np.ndarray
                air temperature (in celsius)

            z : float
                elevation above sea level (in m)

            c1 : float, optional
                Makkink coefficient (-)

            c2 : float, optional
                Makkink coefficient (-)

            Reference
            ----------
            Makkink, G. F., Testing the Penman formula by means of lysimeters,
            J. Inst. Wat. Engrs, 11, 277-288, 1957.

            Returns
            ----------
            pet : np.ndarray
                potential evapotranspiration
            """
            # slope of saturation vapour pressure curve (in kPa celsius-1)
            svpc = 4098 * (0.6108 * npx.exp((17.27 * ta) / (ta + 237.3))) / (ta + 237.3) ** 2

            # atmospheric pressure (in kPa)
            p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26

            # psychometric constant (in kPa celsius-1)
            gam = 0.665 * 1e-3 * p

            # special heat of evaporation (in MJ m-2 mm-1)
            lam = 0.0864 * (28.4 - 0.028 * ta)

            # potential evapotranspiration (in mm)
            pet = (svpc / (svpc + gam)) * ((c1 * rs / lam) + c2)

            return npx.where(pet < 0, 0, pet)

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = f"SVAT_for_{transport_model_structure}"

            settings.nx, settings.ny = (
                self._get_nx(self._base_path, "params_sobol.nc", group=transport_model_structure.replace("_", " ")),
                1,
            )
            settings.runlen = self._get_runlen(self._input_dir, "forcing.nc")
            settings.nitt_forc = len(self._read_var_from_nc("Time", self._input_dir, "forcing.nc"))

            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = "1996-12-31 00:00:00"

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

            # grid of model runs
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

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "lu_id",
                "z_soil",
                "c1_mak",
                "c2_mak",
                "dmpv",
                "lmpv",
                "theta_eff",
                "frac_lp",
                "frac_fp",
                "theta_ac",
                "theta_ufc",
                "theta_pwp",
                "ks",
                "kf",
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables

            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], 8)
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], 2200)
            vs.c1_mak = update(
                vs.c1_mak,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "c1_mak", self._base_path, "params_sobol.nc", group=transport_model_structure.replace("_", " ")
                ),
            )
            vs.c2_mak = update(
                vs.c2_mak,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "c2_mak", self._base_path, "params_sobol.nc", group=transport_model_structure.replace("_", " ")
                ),
            )
            vs.dmpv = update(
                vs.dmpv,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "dmpv", self._base_path, "params_sobol.nc", group=transport_model_structure.replace("_", " ")
                ),
            )
            vs.lmpv = update(
                vs.lmpv,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "lmpv", self._base_path, "params_sobol.nc", group=transport_model_structure.replace("_", " ")
                ),
            )
            vs.theta_eff = update(
                vs.theta_eff,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "theta_eff",
                    self._base_path,
                    "params_sobol.nc",
                    group=transport_model_structure.replace("_", " "),
                ),
            )
            vs.frac_lp = update(
                vs.frac_lp,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "frac_lp", self._base_path, "params_sobol.nc", group=transport_model_structure.replace("_", " ")
                ),
            )
            vs.frac_fp = update(vs.frac_fp, at[2:-2, 2:-2], 1 - vs.frac_lp[2:-2, 2:-2])
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], vs.theta_eff[2:-2, 2:-2] * vs.frac_lp[2:-2, 2:-2])
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], vs.theta_eff[2:-2, 2:-2] * vs.frac_fp[2:-2, 2:-2])
            vs.theta_pwp = update(
                vs.theta_pwp,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "theta_pwp",
                    self._base_path,
                    "params_sobol.nc",
                    group=transport_model_structure.replace("_", " "),
                ),
            )
            vs.ks = update(
                vs.ks,
                at[2:-2, 2:-2],
                self._read_var_from_nc(
                    "ks", self._base_path, "params_sobol.nc", group=transport_model_structure.replace("_", " ")
                ),
            )
            vs.kf = update(vs.kf, at[2:-2, 2:-2], 2500)

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

            vs.theta_rz = update(
                vs.theta_rz,
                at[2:-2, 2:-2, : vs.taup1],
                vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis],
            )
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
                "RS",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables

            vs.PREC = update(vs.PREC, at[:], self._read_var_from_nc("PREC", self._input_dir, "forcing.nc")[0, 0, :])
            vs.TA = update(vs.TA, at[:], self._read_var_from_nc("TA", self._input_dir, "forcing.nc")[0, 0, :])
            vs.RS = update(vs.RS, at[:], self._read_var_from_nc("RS", self._input_dir, "forcing.nc")[0, 0, :])

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
                    vs.prec_day, at[:, :, :], vs.PREC[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24]
                )
                vs.ta_day = update(
                    vs.ta_day, at[:, :, :], vs.TA[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24]
                )
                vs.rs_day = update(
                    vs.rs_day, at[:, :, :], vs.RS[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24]
                )
                vs.pet_day = update(
                    vs.pet_day,
                    at[2:-2, 2:-2, :],
                    self._calc_pet_with_makkink(
                        npx.mean(vs.rs_day[2:-2, 2:-2, :], axis=-1),
                        npx.mean(vs.ta_day[2:-2, 2:-2, :], axis=-1),
                        755,
                        c1=vs.c1_mak[2:-2, 2:-2],
                        c2=vs.c2_mak[2:-2, 2:-2],
                    )[:, :, npx.newaxis]
                    / (6 * 24),
                )
                vs.itt_forc = vs.itt_forc + 6 * 24

        @roger_routine
        def set_diagnostics(self, state, base_path=_tmp_dir):
            diagnostics = state.diagnostics
            settings = state.settings

            diagnostics["rate"].output_variables = [
                "prec",
                "pet",
                "aet",
                "transp",
                "evap_soil",
                "inf_mat_rz",
                "inf_mp_rz",
                "inf_sc_rz",
                "inf_ss",
                "q_rz",
                "q_ss",
                "cpr_rz",
                "dS_s",
                "dS",
            ]
            if settings.enable_groundwater_boundary:
                diagnostics["rate"].output_variables += ["cpr_ss"]
            diagnostics["rate"].output_frequency = 24 * 60 * 60
            diagnostics["rate"].sampling_frequency = 1
            if base_path:
                diagnostics["rate"].base_output_path = base_path

            diagnostics["collect"].output_variables = [
                "S_rz",
                "S_ss",
                "S_pwp_rz",
                "S_fc_rz",
                "S_sat_rz",
                "S_pwp_ss",
                "S_fc_ss",
                "S_sat_ss",
                "theta_rz",
                "theta_ss",
                "theta",
                "S_snow",
            ]
            diagnostics["collect"].output_frequency = 24 * 60 * 60
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            diagnostics["average"].output_variables = ["ta"]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            diagnostics["constant"].output_variables = [
                "c1_mak",
                "c2_mak",
                "dmpv",
                "lmpv",
                "theta_ac",
                "theta_ufc",
                "theta_pwp",
                "ks",
                "theta_eff",
                "frac_lp",
                "frac_fp",
            ]
            diagnostics["constant"].output_frequency = 0
            diagnostics["constant"].sampling_frequency = 1
            if base_path:
                diagnostics["constant"].base_output_path = base_path

            # maximum bias of deterministic/numerical solution at time step t
            diagnostics["maximum"].output_variables = ["dS_num_error"]
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
        # set to 0 for numerical errors
        vs.S_fp_rz = update(
            vs.S_fp_rz,
            at[2:-2, 2:-2],
            npx.where((vs.S_fp_rz > -1e-6) & (vs.S_fp_rz < 0), 0, vs.S_fp_rz)[2:-2, 2:-2],
        )
        vs.S_lp_rz = update(
            vs.S_lp_rz,
            at[2:-2, 2:-2],
            npx.where((vs.S_lp_rz > -1e-6) & (vs.S_lp_rz < 0), 0, vs.S_lp_rz)[2:-2, 2:-2],
        )
        vs.S_fp_ss = update(
            vs.S_fp_ss,
            at[2:-2, 2:-2],
            npx.where((vs.S_fp_ss > -1e-6) & (vs.S_fp_ss < 0), 0, vs.S_fp_ss)[2:-2, 2:-2],
        )
        vs.S_lp_ss = update(
            vs.S_lp_ss,
            at[2:-2, 2:-2],
            npx.where((vs.S_lp_ss > -1e-6) & (vs.S_lp_ss < 0), 0, vs.S_lp_ss)[2:-2, 2:-2],
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
            k_rz=vs.k_rz,
            k_ss=vs.k_ss,
            k=vs.k,
            z0=vs.z0,
            prec=vs.prec,
            event_id=vs.event_id,
            year=vs.year,
            month=vs.month,
            doy=vs.doy,
            S_fp_rz=vs.S_fp_rz,
            S_lp_rz=vs.S_lp_rz,
            S_fp_ss=vs.S_fp_ss,
            S_lp_ss=vs.S_lp_ss,
        )

    model = SVATSetup()
    write_forcing(model._input_dir)
    model.setup()
    model.run()
    return


if __name__ == "__main__":
    main()
