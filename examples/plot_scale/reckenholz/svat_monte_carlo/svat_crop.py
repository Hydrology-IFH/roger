from pathlib import Path
import os
import h5netcdf
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-lys", "--lys-experiment", type=click.Choice(["lys1", "lys2", "lys3", "lys4", "lys8", "lys9", "lys2_bromide", "lys8_bromide", "lys9_bromide"]), default="lys8")
@click.option("-ns", "--nsamples", type=int, default=10000)
@click.option("-td", "--tmp-dir", type=str, default=None)
@roger_base_cli
def main(nsamples, lys_experiment, tmp_dir):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, random_uniform
    from roger.core.surface import calc_parameters_surface_kernel
    from roger.tools.setup import write_forcing, write_crop_rotation
    import roger.lookuptables as lut

    class SVATCROPSetup(RogerSetup):
        """A SVAT model including crop phenology/crop rotation.
        """
        _base_path = Path(__file__).parent
        _input_dir = None
        _identifier = None
        _lys = None

        def _set_lys(self, lys):
            self._lys = lys

        def _set_identifier(self, identifier):
            self._identifier = identifier

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
                return len(onp.array(var_obj))

        def _get_runlen(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['dt']
                return onp.sum(onp.array(var_obj))

        def _get_time_origin(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                date = infile.variables['Time'].attrs['time_origin'].split(" ")[0]
                return f"{date} 00:00:00"

        def _get_ncr(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['year_season']
                return len(onp.array(var_obj))

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = self._identifier

            settings.nx, settings.ny = nsamples, 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')

            # lysimeter surface 1 square meter (1m diameter)
            settings.dx = 1
            settings.dy = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = self._get_time_origin(self._input_dir, 'forcing.nc')

            settings.enable_crop_water_stress = True
            settings.enable_crop_phenology = True
            settings.enable_crop_rotation = True
            settings.enable_macropore_lower_boundary_condition = False
            settings.enable_adaptive_time_stepping = True

            if settings.enable_crop_rotation:
                settings.ncrops = 3
                settings.ncr = self._get_ncr(self._input_dir, 'crop_rotation.nc')

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

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "lut_ilu",
                "lut_gc",
                "lut_gcm",
                "lut_is",
                "lut_rdlu",
                "lut_crops",
                "lut_crop_scale",
            ],
        )
        def set_look_up_tables(self, state):
            vs = state.variables

            vs.lut_ilu = update(vs.lut_ilu, at[:, :], lut.ARR_ILU)
            vs.lut_gc = update(vs.lut_gc, at[:, :], lut.ARR_GC)
            vs.lut_gcm = update(vs.lut_gcm, at[:, :], lut.ARR_GCM)
            vs.lut_is = update(vs.lut_is, at[:, :], lut.ARR_IS)
            vs.lut_rdlu = update(vs.lut_rdlu, at[:, :], lut.ARR_RDLU)
            vs.lut_crops = update(vs.lut_crops, at[:, :], lut.ARR_CP)
            # scale basal crop coeffcient with factor
            for i in range(vs.lut_crop_scale.shape[-1]):
                vs.lut_crop_scale = update(vs.lut_crop_scale, at[2:-2, 2:-2, i], random_uniform(0.5, 1.5, vs.lut_crop_scale.shape[:-1])[2:-2, 2:-2])

        @roger_routine
        def set_topography(self, state):
            pass

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "lu_id",
                "z_soil",
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
                "crop_type",
                "z_root",
                "z_root_crop",
            ],
        )
        def set_parameters_setup(self, state):
            vs = state.variables

            vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], 8)
            vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], 1350)
            vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], npx.array(random_uniform(1, 400, vs.dmpv.shape), dtype=int)[2:-2, 2:-2])
            vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], npx.array(random_uniform(1, 1200, vs.lmpv.shape), dtype=int)[2:-2, 2:-2])
            vs.theta_eff = update(vs.theta_eff, at[2:-2, 2:-2], random_uniform(0.15, 0.35, vs.theta_eff.shape)[2:-2, 2:-2])
            vs.frac_lp = update(vs.frac_lp, at[2:-2, 2:-2], random_uniform(0.1, 0.9, vs.theta_eff.shape)[2:-2, 2:-2])
            vs.frac_fp = update(vs.frac_fp, at[2:-2, 2:-2], 1 - vs.frac_lp[2:-2, 2:-2])
            vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], vs.theta_eff[2:-2, 2:-2] * vs.frac_lp[2:-2, 2:-2])
            vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], vs.theta_eff[2:-2, 2:-2] * vs.frac_fp[2:-2, 2:-2])
            vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], random_uniform(0.15, 0.35, vs.theta_pwp.shape)[2:-2, 2:-2])
            vs.ks = update(vs.ks, at[2:-2, 2:-2], random_uniform(0.1, 120, vs.ks.shape)[2:-2, 2:-2])
            vs.kf = update(vs.kf, at[2:-2, 2:-2], 2500)

            vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 0], self._read_var_from_nc("crop", self._input_dir, 'crop_rotation.nc')[:, :, 1])
            vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 1], self._read_var_from_nc("crop", self._input_dir, 'crop_rotation.nc')[:, :, 2])
            vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 2], self._read_var_from_nc("crop", self._input_dir, 'crop_rotation.nc')[:, :, 3])

            z_root = self._read_var_from_nc("z_root", self._base_path, 'initvals.nc', group=self._lys)
            vs.z_root = update(vs.z_root, at[2:-2, 2:-2, :2], z_root)
            vs.z_root_crop = update(
                vs.z_root_crop,
                at[2:-2, 2:-2, :2, 0], z_root
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

            theta_rz = self._read_var_from_nc("theta_rz", self._base_path, 'initvals.nc', group=self._lys)
            vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], npx.where(theta_rz > vs.theta_sat[2:-2, 2:-2, npx.newaxis], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis], theta_rz))
            theta_ss = self._read_var_from_nc("theta_ss", self._base_path, 'initvals.nc', group=self._lys)
            vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], npx.where(theta_ss > vs.theta_sat[2:-2, 2:-2, npx.newaxis], vs.theta_pwp[2:-2, 2:-2, npx.newaxis] + vs.theta_ufc[2:-2, 2:-2, npx.newaxis], theta_ss))

            vs.update(set_initial_conditions_crops_kernel(state))

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
                "time",
                "itt_day",
                "itt_forc",
                "prec_day",
                "ta_day",
                "pet_day",
                "year",
                "month",
                "doy",
                "itt",
                "itt_cr",
                "crop_type",
            ],
        )
        def set_forcing(self, state):
            vs = state.variables

            condt = (vs.time % (24 * 60 * 60) == 0)
            if condt:
                vs.itt_day = 0
                vs.year = update(vs.year, at[1], self._read_var_from_nc("YEAR", self._input_dir, 'forcing.nc')[vs.itt_forc])
                vs.month = update(vs.month, at[1], self._read_var_from_nc("MONTH", self._input_dir, 'forcing.nc')[vs.itt_forc])
                vs.doy = update(vs.doy, at[1], self._read_var_from_nc("DOY", self._input_dir, 'forcing.nc')[vs.itt_forc])
                vs.prec_day = update(vs.prec_day, at[2:-2, 2:-2, :], self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc')[:, :, vs.itt_forc:vs.itt_forc+6*24])
                vs.ta_day = update(vs.ta_day, at[2:-2, 2:-2, :], self._read_var_from_nc("TA", self._input_dir, 'forcing.nc')[:, :, vs.itt_forc:vs.itt_forc+6*24])
                vs.pet_day = update(vs.pet_day, at[2:-2, 2:-2, :], self._read_var_from_nc("PET", self._input_dir, 'forcing.nc')[:, :, vs.itt_forc:vs.itt_forc+6*24])
                vs.itt_forc = vs.itt_forc + 6 * 24

            if (vs.year[1] != vs.year[0]) & (vs.itt > 1):
                vs.itt_cr = vs.itt_cr + 2
                vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 0], vs.crop_type[2:-2, 2:-2, 2])
                vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 1], self._read_var_from_nc("crop", self._input_dir, 'crop_rotation.nc')[:, :, vs.itt_cr])
                vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 2], self._read_var_from_nc("crop", self._input_dir, 'crop_rotation.nc')[:, :, vs.itt_cr + 1])

        @roger_routine
        def set_diagnostics(self, state, base_path=tmp_dir):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = ["prec", "transp", "evap_soil", "inf_mat_rz", "inf_mp_rz", "inf_sc_rz", "inf_ss", "q_rz", "q_ss", "cpr_rz", "re_rg", "re_rl"]
            diagnostics["rate"].output_frequency = 24 * 60 * 60
            diagnostics["rate"].sampling_frequency = 1
            if base_path:
                diagnostics["rate"].base_output_path = base_path

            diagnostics["collect"].output_variables = ["S_rz", "S_ss", "S_s", "S",
                                                       "S_pwp_rz", "S_fc_rz",
                                                       "S_sat_rz", "S_pwp_ss",
                                                       "S_fc_ss", "S_sat_ss",
                                                       "theta",
                                                       "z_root", "ground_cover", "lu_id"]
            diagnostics["collect"].output_frequency = 24 * 60 * 60
            diagnostics["collect"].sampling_frequency = 1
            if base_path:
                diagnostics["collect"].base_output_path = base_path

            diagnostics["average"].output_variables = ["ta"]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1
            if base_path:
                diagnostics["average"].base_output_path = base_path

            diagnostics["constant"].output_variables = ['dmpv', 'lmpv', 'theta_ac', 'theta_ufc', 'theta_pwp', 'ks', 'lut_crop_scale', 'theta_eff', 'frac_lp', 'frac_fp']
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
            vs.update(after_timestep_crops_kernel(state))

    @roger_kernel
    def set_initial_conditions_crops_kernel(state):
        vs = state.variables

        # calculate time since growing
        t_grow = allocate(state.dimensions, ("x", "y", "crops"))
        t_grow = update(
            t_grow,
            at[2:-2, 2:-2, :], npx.where(vs.z_root_crop[2:-2, 2:-2, vs.taum1, :] > 0, (-1 / vs.root_growth_rate[2:-2, 2:-2, :]) * npx.log(1 / ((vs.z_root_crop[2:-2, 2:-2, vs.taum1, :] / 1000 - vs.z_root_crop_max[2:-2, 2:-2, :] / 1000) * (-1 / (vs.z_root_crop_max[2:-2, 2:-2, :] / 1000 - vs.z_evap[2:-2, 2:-2, npx.newaxis] / 1000)))), 0)
        )

        vs.t_grow_cc = update(
            vs.t_grow_cc,
            at[2:-2, 2:-2, :2, :], t_grow[2:-2, 2:-2, npx.newaxis, :]
        )

        vs.t_grow_root = update(
            vs.t_grow_root,
            at[2:-2, 2:-2, :2, :], t_grow[2:-2, 2:-2, npx.newaxis, :]
        )

        return KernelOutput(
            t_grow_cc=vs.t_grow_cc,
            t_grow_root=vs.t_grow_root,
        )

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        vs.ta = update(
            vs.ta,
            at[2:-2, 2:-2, vs.taum1], vs.ta[2:-2, 2:-2, vs.tau],
        )
        vs.z_root = update(
            vs.z_root,
            at[2:-2, 2:-2, vs.taum1], vs.z_root[2:-2, 2:-2, vs.tau],
        )
        vs.ground_cover = update(
            vs.ground_cover,
            at[2:-2, 2:-2, vs.taum1], vs.ground_cover[2:-2, 2:-2, vs.tau],
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
        vs.z0 = update(
            vs.z0,
            at[2:-2, 2:-2, vs.taum1], vs.z0[2:-2, 2:-2, vs.tau],
        )
        # set to 0 for numerical errors
        vs.S_fp_rz = update(
            vs.S_fp_rz,
            at[2:-2, 2:-2], npx.where((vs.S_fp_rz[2:-2, 2:-2] > -1e-6) & (vs.S_fp_rz[2:-2, 2:-2] < 0), 0, vs.S_fp_rz[2:-2, 2:-2]),
        )
        vs.S_lp_rz = update(
            vs.S_lp_rz,
            at[2:-2, 2:-2], npx.where((vs.S_lp_rz[2:-2, 2:-2] > -1e-6) & (vs.S_lp_rz[2:-2, 2:-2] < 0), 0, vs.S_lp_rz[2:-2, 2:-2]),
        )
        vs.S_fp_ss = update(
            vs.S_fp_ss,
            at[2:-2, 2:-2], npx.where((vs.S_fp_ss[2:-2, 2:-2] > -1e-6) & (vs.S_fp_ss[2:-2, 2:-2] < 0), 0, vs.S_fp_ss[2:-2, 2:-2]),
        )
        vs.S_lp_ss = update(
            vs.S_lp_ss,
            at[2:-2, 2:-2], npx.where((vs.S_lp_ss[2:-2, 2:-2] > -1e-6) & (vs.S_lp_ss[2:-2, 2:-2] < 0), 0, vs.S_lp_ss[2:-2, 2:-2]),
        )
        vs.prec = update(
            vs.prec,
            at[2:-2, 2:-2, vs.taum1], vs.prec[2:-2, 2:-2, vs.tau],
        )
        vs.event_id = update(
            vs.event_id,
            at[vs.taum1], vs.event_id[vs.tau],
        )
        vs.year = update(
            vs.year,
            at[vs.taum1], vs.year[vs.tau],
        )
        vs.month = update(
            vs.month,
            at[vs.taum1], vs.month[vs.tau],
        )
        vs.doy = update(
            vs.doy,
            at[vs.taum1], vs.doy[vs.tau],
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

    @roger_kernel
    def after_timestep_crops_kernel(state):
        vs = state.variables

        vs.ta_min = update(vs.ta_min, at[2:-2, 2:-2, vs.taum1], vs.ta_min[2:-2, 2:-2, vs.tau])
        vs.ta_max = update(vs.ta_max, at[2:-2, 2:-2, vs.taum1], vs.ta_max[2:-2, 2:-2, vs.tau])
        vs.gdd_sum = update(vs.gdd_sum, at[2:-2, 2:-2, vs.taum1, :], vs.gdd_sum[2:-2, 2:-2, vs.tau, :])
        vs.t_grow_cc = update(vs.t_grow_cc, at[2:-2, 2:-2, vs.taum1, :], vs.t_grow_cc[2:-2, 2:-2, vs.tau, :])
        vs.t_grow_root = update(vs.t_grow_root, at[2:-2, 2:-2, vs.taum1, :], vs.t_grow_root[2:-2, 2:-2, vs.tau, :])
        vs.ccc = update(vs.ccc, at[2:-2, 2:-2, vs.taum1, :], vs.ccc[2:-2, 2:-2, vs.tau, :])
        vs.z_root_crop = update(vs.z_root_crop, at[2:-2, 2:-2, vs.taum1, :], vs.z_root_crop[2:-2, 2:-2, vs.tau, :])

        return KernelOutput(
            ta_min=vs.ta_min,
            ta_max=vs.ta_max,
            gdd_sum=vs.gdd_sum,
            t_grow_cc=vs.t_grow_cc,
            t_grow_root=vs.t_grow_root,
            ccc=vs.ccc,
            z_root_crop=vs.z_root_crop,
        )

    model = SVATCROPSetup()
    model._set_lys(lys_experiment)
    input_path = model._base_path / "input" / lys_experiment
    model._set_input_dir(input_path)
    identifier = f'SVATCROP_{lys_experiment}'
    model._set_identifier(identifier)
    write_forcing(input_path, enable_crop_phenology=True)
    write_crop_rotation(input_path)
    model.setup()
    model.run()
    return


if __name__ == "__main__":
    main()
