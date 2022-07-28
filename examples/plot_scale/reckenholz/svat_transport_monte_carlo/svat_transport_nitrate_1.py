from pathlib import Path
import os
import h5netcdf
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-ns", "--nsamples", type=int, default=10000)
@click.option("-lys", "--lys-experiment", type=click.Choice(["lys2", "lys3", "lys4", "lys8", "lys9"]), default="lys2")
@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'piston', 'preferential', 'advection-dispersion', 'time-variant_preferential', 'time-variant_advection-dispersion']), default='complete-mixing')
@click.option("-ecp", "--crop-partitioning", is_flag=True)
@roger_base_cli
def main(nsamples, lys_experiment, transport_model_structure, crop_partitioning):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, where, random_uniform, scipy_stats as sstx
    from roger.tools.setup import write_forcing_tracer
    import roger.lookuptables as lut
    from roger.core.crop import update_alpha_transp

    class SVATCROPTRANSPORTSetup(RogerSetup):
        """A SVAT transport model for nitrate including
        crop phenology/crop rotation.
        """
        _base_path = Path(__file__).parent
        _lys = None
        _tm_structure = None
        _input_dir = None
        _identifier = None
        _nsamples = 1

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

        def _set_nsamples(self, nsamples):
            self._nsamples = nsamples

        def _set_nitrate_input(self, state, nn_rain, nn_sol, prec, ta):
            vs = state.variables

            NMIN_IN = allocate(state.dimensions, ("x", "y", "t"))

            mask_rain = (prec > 0) & (ta > 0)
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
                    at[:, :], npx.max(npx.where(npx.cumsum(prec[:, :, start_rain:], axis=-1) <= 20, npx.max(npx.cumsum(prec[:, :, start_rain:], axis=-1), axis=-1)[:, :, npx.newaxis], 0), axis=-1),
                )
                nn_end = npx.max(npx.where(npx.cumsum(prec[:, :, start_rain:]) <= 20, npx.max(npx.arange(npx.shape(prec)[2])[npx.newaxis, npx.newaxis, npx.shape(prec)[2]-start_rain], axis=-1), 0))
                end_rain = update(end_rain, at[:], start_rain + nn_end)
                end_rain = update(end_rain, at[:], npx.where(end_rain > npx.shape(prec)[2], npx.shape(prec)[2], end_rain))

                # proportions for redistribution
                NMIN_IN = update(
                    NMIN_IN,
                    at[:, :, start_rain:end_rain[0]], vs.M_IN[:, :, sol_idx[i], npx.newaxis] * (prec[:, :, start_rain:end_rain[0]] / rain_sum[:, :, npx.newaxis]),
                )

            # solute input concentration
            M_IN = NMIN_IN * 0.3
            C_IN = npx.where(prec > 0, M_IN / prec, 0)

            NMIN_IN1 = NMIN_IN * 0.7

            return M_IN, C_IN, NMIN_IN1

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = self._identifier

            settings.nx, settings.ny, settings.nz = self._nsamples, 1, 1
            settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
            settings.ages = settings.nitt
            settings.nages = settings.nitt + 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')

            settings.dx = 1
            settings.dy = 1
            settings.dz = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = self._get_time_origin(self._input_dir, 'forcing_tracer.nc')

            settings.enable_crop_phenology = True
            settings.enable_crop_rotation = True
            settings.enable_offline_transport = True
            settings.enable_nitrate = True
            settings.tm_structure = self._tm_structure
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
                vs.lut_crop_scale = update(vs.lut_crop_scale, at[2:-2, 2:-2, i], random_uniform(0.5, 1.5, vs.lut_crop_scale.shape[:-1])[2:-2, 2:-2])

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
                vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], random_uniform(0.01, 1.5, tuple((vs.alpha_transp.shape[0], vs.alpha_transp.shape[1])))[2:-2, 2:-2])
            vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], random_uniform(0.01, 1.0, tuple((vs.alpha_q.shape[0], vs.alpha_q.shape[1])))[2:-2, 2:-2])
            vs.km_denit_rz = update(vs.km_denit_rz, at[2:-2, 2:-2], random_uniform(1.0, 20.0, tuple((vs.km_denit_rz.shape[0], vs.km_denit_rz.shape[1])))[2:-2, 2:-2])
            vs.km_denit_ss = update(vs.km_denit_ss, at[2:-2, 2:-2], random_uniform(1.0, 20.0, tuple((vs.km_denit_ss.shape[0], vs.km_denit_ss.shape[1])))[2:-2, 2:-2])
            vs.dmax_denit_rz = update(vs.dmax_denit_rz, at[2:-2, 2:-2], random_uniform(1.0, 100.0, tuple((vs.dmax_denit_rz.shape[0], vs.dmax_denit_rz.shape[1])))[2:-2, 2:-2])
            vs.dmax_denit_ss = update(vs.dmax_denit_ss, at[2:-2, 2:-2], random_uniform(1.0, 100.0, tuple((vs.dmax_denit_ss.shape[0], vs.dmax_denit_ss.shape[1])))[2:-2, 2:-2])
            vs.km_nit_rz = update(vs.km_nit_rz, at[2:-2, 2:-2], random_uniform(1.0, 20.0, tuple((vs.km_nit_rz.shape[0], vs.km_nit_rz.shape[1])))[2:-2, 2:-2])
            vs.km_nit_ss = update(vs.km_nit_ss, at[2:-2, 2:-2], random_uniform(1.0, 20.0, tuple((vs.km_nit_ss.shape[0], vs.km_nit_ss.shape[1])))[2:-2, 2:-2])
            vs.dmax_nit_rz = update(vs.dmax_nit_rz, at[2:-2, 2:-2], random_uniform(1.0, 100.0, tuple((vs.dmax_nit_rz.shape[0], vs.dmax_nit_rz.shape[1])))[2:-2, 2:-2])
            vs.dmax_nit_ss = update(vs.dmax_nit_ss, at[2:-2, 2:-2], random_uniform(1.0, 100.0, tuple((vs.dmax_nit_ss.shape[0], vs.dmax_nit_ss.shape[1])))[2:-2, 2:-2])
            vs.kmin_rz = update(vs.kmin_rz, at[2:-2, 2:-2], random_uniform(1.0, 100.0, tuple((vs.kmin_rz.shape[0], vs.kmin_rz.shape[1])))[2:-2, 2:-2])
            vs.kmin_ss = update(vs.kmin_ss, at[2:-2, 2:-2], random_uniform(1.0, 100.0, tuple((vs.kmin_ss.shape[0], vs.kmin_ss.shape[1])))[2:-2, 2:-2])

            if self._lys in ['lys2', 'lys3']:
                _lys = 'lys2_bromide'
            elif self._lys in ['lys8']:
                _lys = 'lys8_bromide'
            elif self._lys in ['lys4', 'lys9']:
                _lys = 'lys9_bromide'
            if settings.tm_structure == "complete-mixing":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
            elif settings.tm_structure == "piston":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 22)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 22)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
            elif settings.tm_structure == "preferential":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
            elif settings.tm_structure == "advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
            elif settings.tm_structure == "complete-mixing + advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
            elif settings.tm_structure == "time-variant complete-mixing + advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
            elif settings.tm_structure == "time-variant advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
            elif settings.tm_structure == "time-variant preferential":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
            elif settings.tm_structure == "time-variant":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)

        @roger_routine
        def set_parameters(self, state):
            vs = state.variables
            settings = state.settings

            if settings.tm_structure == "time-variant complete-mixing + advection-dispersion":
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
            elif settings.tm_structure == "time-variant advection-dispersion":
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
            elif settings.tm_structure == "time-variant preferential":
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
            elif settings.tm_structure == "time-variant":
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])

            if settings.enable_crop_partitioning:
                vs.update(update_alpha_transp(state))

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "S_pwp_rz",
                "S_pwp_ss",
                "S_rz",
                "S_ss",
                "S_s",
                "itt"
            ],
        )
        def set_initial_conditions_setup(self, state):
            vs = state.variables

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :2], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :2], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, :2], vs.S_rz[2:-2, 2:-2, :2] + vs.S_ss[2:-2, 2:-2, :2])

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
            vs.C_rz = update(vs.C_rz, at[2:-2, 2:-2, :vs.taup1], 30)
            vs.C_ss = update(vs.C_ss, at[2:-2, 2:-2, :vs.taup1], 30)
            # exponential distribution of mineral soil nitrogen
            # mineral soil nitrogen is decreasing with increasing age
            p_dec = allocate(state.dimensions, ("x", "y", 2, "ages"))
            p_dec = update(p_dec, at[:, :, :vs.taup1, :], sstx.expon.pdf(npx.linspace(sstx.expon.ppf(0.001), sstx.expon.ppf(0.999), settings.ages))[npx.newaxis, npx.newaxis, npx.newaxis, :])
            vs.Nmin_rz = update(vs.Nmin_rz, at[2:-2, 2:-2, :vs.taup1, :], 100 * p_dec[2:-2, 2:-2, :, :] * settings.dx * settings.dy * 100)
            vs.Nmin_ss = update(vs.Nmin_ss, at[2:-2, 2:-2, :vs.taup1, :], 100 * p_dec[2:-2, 2:-2, :, :] * settings.dx * settings.dy * 100)
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
                "NMIN_IN",
                "NORG_IN",
                "M_IN",
                "C_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables
            settings = state.settings

            TA = self._read_var_from_nc("ta", self._base_path, 'states_hm.nc', group=self._lys)
            PREC = self._read_var_from_nc("prec", self._base_path, 'states_hm.nc', group=self._lys)

            # convert kg N/ha to mg/square meter
            vs.NMIN_IN = update(vs.NMIN_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("Nmin", self._input_dir, 'forcing_tracer.nc') * 100 * settings.dx * settings.dy)
            vs.NORG_IN = update(vs.NORG_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("Norg", self._input_dir, 'forcing_tracer.nc') * 100 * settings.dx * settings.dy)

            mask_rain = (PREC > 0) & (TA > 0)
            mask_sol = (vs.NMIN_IN > 0)
            nn_rain = npx.int64(npx.sum(npx.any(mask_rain, axis=(0, 1))))
            nn_sol = npx.int64(npx.sum(npx.any(mask_sol, axis=(0, 1))))
            M_IN, C_IN, NMIN_IN = self._set_nitrate_input(state, nn_rain, nn_sol, PREC, TA)
            vs.M_IN = update(vs.M_IN, at[:, :, :], M_IN)
            vs.C_IN = update(vs.C_IN, at[:, :, :], C_IN)
            vs.NMIN_IN = update(vs.NMIN_IN, at[:, :, :], NMIN_IN)

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
                "M_in",
                "NMIN_IN",
                "NORG_IN",
                "Nmin_in",
                "Norg_in"

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

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt])
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc', group=self._lys)[:, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])

            vs.C_in = update(vs.C_in, at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt])
            vs.M_in = update(
                vs.M_in,
                at[2:-2, 2:-2], vs.C_in[2:-2, 2:-2] * vs.prec[2:-2, 2:-2, vs.tau],
            )
            vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], vs.NMIN_IN[2:-2, 2:-2, vs.itt])
            vs.Norg_in = update(vs.Norg_in, at[2:-2, 2:-2], vs.NORG_IN[2:-2, 2:-2, vs.itt])

        @roger_routine
        def set_diagnostics(self, state):
            diagnostics = state.diagnostics

            diagnostics["rates"].output_variables = ["q_ss", "M_q_ss", "transp", "M_transp"]
            diagnostics["rates"].output_frequency = 24 * 60 * 60
            diagnostics["rates"].sampling_frequency = 1

            diagnostics["averages"].output_variables = ["C_q_ss"]
            diagnostics["averages"].output_frequency = 24 * 60 * 60
            diagnostics["averages"].sampling_frequency = 1

            diagnostics["constant"].output_variables = ["sas_params_transp", "sas_params_q_rz",
                                                        "sas_params_q_ss", "alpha_transp",
                                                        "alpha_q", "km_denit_rz",
                                                        "km_denit_ss", "dmax_denit_rz",
                                                        "dmax_denit_ss", "km_nit_rz",
                                                        "km_nit_ss", "dmax_nit_rz",
                                                        "dmax_nit_ss", "kmin_rz",
                                                        "kmin_ss"]
            diagnostics["constant"].output_frequency = 0
            diagnostics["constant"].sampling_frequency = 1

        @roger_routine
        def after_timestep(self, state):
            vs = state.variables

            vs.update(after_timestep_kernel(state))
            vs.update(after_timestep_nitrate_kernel(state))

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

    @roger_kernel
    def after_timestep_nitrate_kernel(state):
        vs = state.variables

        vs.Nmin_rz = update(
            vs.Nmin_rz,
            at[2:-2, 2:-2, vs.taum1, :], vs.Nmin_rz[2:-2, 2:-2, vs.tau, :],
            )

        vs.Nmin_ss = update(
            vs.Nmin_ss,
            at[2:-2, 2:-2, vs.taum1, :], vs.Nmin_ss[2:-2, 2:-2, vs.tau, :],
            )

        vs.Nmin_s = update(
            vs.Nmin_s,
            at[2:-2, 2:-2, vs.taum1, :], vs.Nmin_s[2:-2, 2:-2, vs.tau, :],
            )

        return KernelOutput(
            Nmin_rz=vs.Nmin_rz,
            Nmin_ss=vs.Nmin_ss,
            Nmin_s=vs.Nmin_s
            )

    tms = transport_model_structure.replace("_", " ")
    model = SVATCROPTRANSPORTSetup()
    model._set_nsamples(nsamples)
    model._set_lys(lys_experiment)
    model._set_tm_structure(tms)
    identifier = f'SVATCROPTRANSPORT_{transport_model_structure}_{lys_experiment}_nitrate'
    model._set_identifier(identifier)
    input_path = model._base_path / "input" / lys_experiment
    model._set_input_dir(input_path)
    model._set_crop_types(model._input_dir, "crop_rotation.nc")
    write_forcing_tracer(input_path, 'NO3')
    model.setup()
    model.warmup()
    model.run()
    return


if __name__ == "__main__":
    main()
