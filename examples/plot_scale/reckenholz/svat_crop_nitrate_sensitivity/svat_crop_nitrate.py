from pathlib import Path
import yaml
import os
import h5netcdf
from SALib.sample import saltelli
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-ns", "--nsamples", type=int, default=1000)
@click.option("-lys", "--lys-experiment", type=click.Choice(["lys2", "lys3", "lys4", "lys8", "lys9"]), default="lys2")
@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'power', 'time-variant power']), default='complete-mixing')
@click.option("-ss", "--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("-ecp", "--crop-partitioning", is_flag=True)
@roger_base_cli
def main(nsamples, lys_experiment, transport_model_structure, sas_solver, crop_partitioning):
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, where, scipy_stats as sstx
    from roger.tools.setup import write_forcing_tracer
    from roger.core.utilities import _get_row_no
    import roger.lookuptables as lut
    from roger.core.crop import update_alpha_transp

    class SVATCROPTRANSPORTSetup(RogerSetup):
        """A SVAT transport model for nitrate including
        crop phenology/crop rotation.
        """
        _base_path = Path(__file__).parent
        _crop_types = None
        _bounds = None
        _params = None
        _nrows = None
        _lys = None
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

        def _get_nx(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['x']
                return len(onp.array(var_obj))

        def _set_lys(self, lys):
            self._lys = lys

        def _set_tm_structure(self, tm_structure):
            self._tm_structure = tm_structure

        def _set_sas_solver(self, sas_solver):
            self._sas_solver = sas_solver

        def _set_identifier(self, identifier):
            self._identifier = identifier

        def _set_nsamples(self, nsamples):
            self._nsamples = nsamples

        def _set_crop_types(self, path_dir, file):
            nc_file = path_dir / file
            with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
                var_obj = infile.variables['crop']
                crop_types1 = onp.unique(onp.array(var_obj)[0, 0, :]).tolist()
                crop_types = []
                for ct in crop_types1:
                    if ct in onp.arange(500, 598, dtype=int).tolist():
                        crop_types.append(ct)

                self._crop_types = crop_types

        def _sample_params(self, nsamples, enable_crop_partitioning=crop_partitioning):
            if enable_crop_partitioning:
                _param_names = []
                _param_bounds = []
                if self._crop_types:
                    for ct in self._crop_types:
                        _param_names.append(f"crop_scale_{ct}")
                        _param_bounds.append([0.5, 1.5])
                _param_names.extend(['alpha_q', 'km_denit_rz', 'km_denit_ss',
                                     'dmax_denit_rz', 'dmax_denit_ss',
                                     'km_nit_rz', 'km_nit_ss', 'dmax_nit_rz',
                                     'dmax_nit_ss', 'kmin_rz', 'kmin_ss'])
                _param_bounds.extend([0.01, 1.0],
                                     [1.0, 20.0],
                                     [1.0, 20.0],
                                     [1.0, 100.0],
                                     [1.0, 100.0],
                                     [1.0, 20.0],
                                     [1.0, 20.0],
                                     [1.0, 100.0],
                                     [1.0, 100.0],
                                     [1.0, 100.0],
                                     [1.0, 100.0])
                self._bounds = {
                    'num_vars': len(_param_names),
                    'names': _param_names,
                    'bounds': _param_bounds
                }
                self._params = saltelli.sample(self._bounds, nsamples, calc_second_order=False)
                self._nrows = self._params.shape[0]

            else:
                self._bounds = {
                    'num_vars': 12,
                    'names': ['alpha_transp', 'alpha_q',
                              'km_denit_rz', 'km_denit_ss', 'dmax_denit_rz',
                              'dmax_denit_ss', 'km_nit_rz', 'km_nit_ss',
                              'dmax_nit_rz', 'dmax_nit_ss', 'kmin_rz', 'kmin_ss'],
                    'bounds': [[0.01, 1.5],
                               [0.01, 1.0],
                               [1.0, 20.0],
                               [1.0, 20.0],
                               [1.0, 100.0],
                               [1.0, 100.0],
                               [1.0, 20.0],
                               [1.0, 20.0],
                               [1.0, 100.0],
                               [1.0, 100.0],
                               [1.0, 100.0],
                               [1.0, 100.0]]
                }
                self._params = saltelli.sample(self._bounds, nsamples, calc_second_order=False)
                self._nrows = self._params.shape[0]

            # write sampled boundaries to .yml
            file_path = self._base_path / "param_bounds_svat_crop_nitrate.yml"
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    _bounds_yml = yaml.safe_load(file)
                if lys_experiment not in list(_bounds_yml.keys()):
                    _bounds_yml[lys_experiment] = {}
                _bounds_yml[lys_experiment][self._tm_structure] = self._bounds
                with open(file_path, 'w') as file:
                    yaml.dump(_bounds_yml, file)
            else:
                _bounds_yml = {}
                if lys_experiment not in list(_bounds_yml.keys()):
                    _bounds_yml[lys_experiment] = {}
                _bounds_yml[lys_experiment][self._tm_structure] = self._bounds
                with open(file_path, 'w') as file:
                    yaml.dump(_bounds_yml, file)

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
            settings.identifier = self._identifier

            settings.nx, settings.ny = self._nrows, 1
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
            # scale basal crop coeffcient with factor
            offset = len(self._param_names) - len(self._crop_types) - 11
            for i, crop_type in enumerate(self._crop_types):
                row_no = _get_row_no(vs.lut_crops[:, 0], crop_type)
                j = offset + i
                vs.lut_crop_scale = update(vs.lut_crop_scale, at[2:-2, 2:-2, row_no], self._params[:, j, npx.newaxis, npx.newaxis])

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
                vs.alpha_transp = update(vs.alpha_transp, at[2:-2, 2:-2], self._params[:, -12, npx.newaxis])
            vs.alpha_q = update(vs.alpha_q, at[2:-2, 2:-2], self._params[:, -11, npx.newaxis])
            vs.km_denit_rz = update(vs.km_denit_rz, at[2:-2, 2:-2], self._params[:, -10, npx.newaxis])
            vs.km_denit_ss = update(vs.km_denit_ss, at[2:-2, 2:-2], self._params[:, -9, npx.newaxis])
            vs.dmax_denit_rz = update(vs.dmax_denit_rz, at[2:-2, 2:-2], self._params[:, -8, npx.newaxis])
            vs.dmax_denit_ss = update(vs.dmax_denit_ss, at[2:-2, 2:-2], self._params[:, -7, npx.newaxis])
            vs.km_nit_rz = update(vs.km_nit_rz, at[2:-2, 2:-2], self._params[:, -6, npx.newaxis])
            vs.km_nit_ss = update(vs.km_nit_ss, at[2:-2, 2:-2], self._params[:, -5, npx.newaxis])
            vs.dmax_nit_rz = update(vs.dmax_nit_rz, at[2:-2, 2:-2], self._params[:, -4, npx.newaxis])
            vs.dmax_nit_ss = update(vs.dmax_nit_ss, at[2:-2, 2:-2], self._params[:, -3, npx.newaxis])
            vs.kmin_rz = update(vs.kmin_rz, at[2:-2, 2:-2], self._params[:, -2, npx.newaxis])
            vs.kmin_ss = update(vs.kmin_ss, at[2:-2, 2:-2], self._params[:, -1, npx.newaxis])

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
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 1)
            elif settings.tm_structure == "advection-dispersion-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 1], 100)
            elif settings.tm_structure == "time-variant advection-dispersion-power":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_transp", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_rz", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, :], self._read_var_from_nc("sas_params_q_ss", self._base_path, 'tm_bromide_params.nc', group=_lys, subgroup=settings.tm_structure))
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 1], 0.1)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 6)
                vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 1], 100)

        @roger_routine
        def set_parameters(self, state):
            vs = state.variables
            settings = state.settings

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
                "NMIN_IN",
                "NORG_IN",
                "M_IN",
                "C_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables
            settings = state.settings

            vs.PREC_DIST_DAILY = update(vs.PREC_DIST_DAILY, at[2:-2, 2:-2, :], self._read_var_from_nc("prec", self._base_path, 'states_hm.mc'))
            vs.INF_MAT_RZ = update(vs.INF_MAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mat_rz", self._base_path, 'states_hm.mc'))
            vs.INF_PF_RZ = update(vs.INF_PF_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_mp_rz", self._base_path, 'states_hm.mc') + self._read_var_from_nc("inf_sc_rz", self._base_path, 'states_hm.mc'))
            vs.INF_PF_SS = update(vs.INF_PF_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("inf_ss", self._base_path, 'states_hm.mc'))
            vs.TRANSP = update(vs.TRANSP, at[2:-2, 2:-2, :], self._read_var_from_nc("transp", self._base_path, 'states_hm.mc'))
            vs.EVAP_SOIL = update(vs.EVAP_SOIL, at[2:-2, 2:-2, :], self._read_var_from_nc("evap_soil", self._base_path, 'states_hm.mc'))
            vs.CPR_RZ = update(vs.CPR_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("cpr_rz", self._base_path, 'states_hm.mc'))
            vs.Q_RZ = update(vs.Q_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("q_rz", self._base_path, 'states_hm.mc'))
            vs.Q_SS = update(vs.Q_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("q_ss", self._base_path, 'states_hm.mc'))
            vs.RE_RG = update(vs.RE_RG, at[2:-2, 2:-2, :], self._read_var_from_nc("re_rg", self._base_path, 'states_hm.mc'))
            vs.RE_RL = update(vs.RE_RL, at[2:-2, 2:-2, :], self._read_var_from_nc("re_rl", self._base_path, 'states_hm.mc'))
            vs.S_RZ = update(vs.S_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.mc'))
            vs.S_PWP_RZ = update(vs.S_PWP_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.mc'))
            vs.S_SAT_RZ = update(vs.S_SAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.mc'))
            vs.S_SS = update(vs.S_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.mc'))
            vs.S_PWP_SS = update(vs.S_PWP_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.mc'))
            vs.S_SAT_SS = update(vs.S_SAT_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.mc'))
            vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
            TA = allocate(state.dimensions, ("x", "y", "t"))
            TA = update(TA, at[2:-2, 2:-2, :], self._read_var_from_nc("ta", self._input_dir, 'states_hm.mc')[npx.newaxis, :, :])

            # convert kg N/ha to mg/square meter
            vs.NMIN_IN = update(vs.NMIN_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("Nmin", self._input_dir, 'forcing_tracer.nc') * 100 * settings.dx * settings.dy)
            vs.NORG_IN = update(vs.NORG_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("Norg", self._input_dir, 'forcing_tracer.nc') * 100 * settings.dx * settings.dy)

            INF = vs.INF_MAT_RZ + vs.INF_PF_RZ + vs.INF_PF_SS
            mask_rain = (INF > 0)
            mask_sol = (vs.NMIN_IN > 0)
            nn_rain = npx.int64(npx.sum(npx.any(mask_rain, axis=(0, 1))))
            nn_sol = npx.int64(npx.sum(npx.any(mask_sol, axis=(0, 1))))
            M_IN, C_IN, NMIN_IN = self._set_nitrate_input(state, nn_rain, nn_sol, INF)
            vs.M_IN = update(vs.M_IN, at[:, :, :], M_IN)
            vs.C_IN = update(vs.C_IN, at[:, :, :], C_IN)
            vs.NMIN_IN = update(vs.NMIN_IN, at[:, :, :], NMIN_IN)

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
            vs.Nmin_in = update(vs.Nmin_in, at[2:-2, 2:-2], vs.NMIN_IN[2:-2, 2:-2, vs.itt])
            vs.Norg_in = update(vs.Norg_in, at[2:-2, 2:-2], vs.NORG_IN[2:-2, 2:-2, vs.itt])

        @roger_routine
        def set_diagnostics(self, state):
            diagnostics = state.diagnostics

            diagnostics["rate"].output_variables = ["q_ss", "M_q_ss", "transp", "M_transp"]
            diagnostics["rate"].output_frequency = 24 * 60 * 60
            diagnostics["rate"].sampling_frequency = 1

            diagnostics["average"].output_variables = ["C_q_ss"]
            diagnostics["average"].output_frequency = 24 * 60 * 60
            diagnostics["average"].sampling_frequency = 1

            diagnostics["constant"].output_variables = ["sas_params_transp", "sas_params_q_rz",
                                                        "sas_params_q_ss", "alpha_transp",
                                                        "alpha_q", "km_denit_rz",
                                                        "km_denit_ss", "dmax_denit_rz",
                                                        "dmax_denit_ss", "km_nit_rz",
                                                        "km_nit_ss", "dmax_nit_rz",
                                                        "dmax_nit_ss", "kmin_rz",
                                                        "kmin_ss", "lut_crop_scale"]
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
    if sas_solver in ['RK4', 'Euler']:
        identifier = f'SVATCROPTRANSPORT_{lys_experiment}_{tms}_{sas_solver}_nitrate'
    else:
        identifier = f'SVATCROPTRANSPORT_{lys_experiment}_{tms}_nitrate'
    model._set_identifier(identifier)
    model._sample_params(nsamples)
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
