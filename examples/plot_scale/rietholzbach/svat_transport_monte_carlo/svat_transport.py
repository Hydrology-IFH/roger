from pathlib import Path
import os
import h5netcdf
import numpy as onp
import click
from roger.cli.roger_run_base import roger_base_cli


@click.option("-ns", "--nsamples", type=int, default=10000)
@click.option("-tms", "--transport-model-structure", type=click.Choice(['complete-mixing', 'piston', 'preferential', 'advection-dispersion', 'time-variant_preferential', 'time-variant_advection-dispersion', 'time-variant']), default='complete-mixing')
@roger_base_cli
def main(nsamples, transport_model_structure):
    from roger import runtime_settings as rs, runtime_state as rst
    from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, random_uniform, for_loop
    from roger.tools.setup import write_forcing_tracer

    class SVATTRANSPORTSetup(RogerSetup):
        """A SVAT transport model.
        """
        _base_path = Path(__file__).parent
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

        def _set_tm_structure(self, tm_structure):
            self._tm_structure = tm_structure

        def _set_identifier(self, identifier):
            self._identifier = identifier

        def _set_nsamples(self, nsamples):
            self._nsamples = nsamples

        def _ffill_3d(self, state, arr):
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
                at[2:-2, 2:-2, :], _ffill(idx)[2:-2, 2:-2, :],
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

            settings.nx, settings.ny, settings.nz = self._nsamples, 1, 1
            settings.nitt = self._get_nitt(self._input_dir, 'forcing_tracer.nc')
            settings.ages = settings.nitt
            settings.nages = settings.nitt + 1
            settings.runlen = self._get_runlen(self._input_dir, 'forcing_tracer.nc')

            # lysimeter surface 3.14 square meter (2m diameter)
            settings.dx = 2
            settings.dy = 2
            settings.dz = 1

            settings.x_origin = 0.0
            settings.y_origin = 0.0
            settings.time_origin = "1996-12-31 00:00:00"

            settings.enable_offline_transport = True
            settings.enable_oxygen18 = True
            settings.tm_structure = self._tm_structure

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "DT_SECS",
                "DT",
                "dt_secs",
                "dt",
                "t",
                "ages",
                "nages",
                "itt",
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
            vs.DT_SECS = update(vs.DT_SECS, at[:], vs.dt_secs)
            vs.DT = update(vs.DT, at[:], vs.dt)
            vs.t = update(vs.t, at[:], npx.cumsum(vs.DT))
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

            vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt])

            if settings.tm_structure == "complete-mixing":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 1)
            elif settings.tm_structure == "piston":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 22)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 22)
            elif settings.tm_structure == "preferential":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], random_uniform(1, 90, tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], random_uniform(1, 90, tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], random_uniform(1, 90, tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])))[2:-2, 2:-2])
            elif settings.tm_structure == "advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], random_uniform(1, 90, tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], random_uniform(1, 90, tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], random_uniform(1, 90, tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
            elif settings.tm_structure == "complete-mixing + advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], random_uniform(1, 90, tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], random_uniform(1, 90, tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
            elif settings.tm_structure == "time-variant advection-dispersion":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 31)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], random_uniform(1, 90, tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])))[2:-2, 2:-2])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 32)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], random_uniform(1, 90, tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 32)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], random_uniform(1, 90, tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
            elif settings.tm_structure == "time-variant preferential":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 31)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], random_uniform(1, 90, tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])))[2:-2, 2:-2])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 31)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], random_uniform(1, 90, tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 31)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], random_uniform(1, 90, tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
            elif settings.tm_structure == "time-variant":
                vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 35)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], random_uniform(1, 90, tuple((vs.sas_params_transp.shape[0], vs.sas_params_transp.shape[1])))[2:-2, 2:-2])
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 35)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], random_uniform(1, 90, tuple((vs.sas_params_q_rz.shape[0], vs.sas_params_q_rz.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 35)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], random_uniform(1, 90, tuple((vs.sas_params_q_ss.shape[0], vs.sas_params_q_ss.shape[1])))[2:-2, 2:-2])
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
                vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])

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

            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_snow", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, :vs.taup1], vs.S_rz[2:-2, 2:-2, :vs.taup1] + vs.S_ss[2:-2, 2:-2, :vs.taup1])
            vs.S_rz_init = update(vs.S_rz_init, at[2:-2, 2:-2], vs.S_rz[2:-2, 2:-2, 0])
            vs.S_ss_init = update(vs.S_ss_init, at[2:-2, 2:-2], vs.S_ss[2:-2, 2:-2, 0])

        @roger_routine
        def set_initial_conditions(self, state):
            vs = state.variables
            settings = state.settings

            # uniform StorAge
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

            if (settings.enable_oxygen18 | settings.enable_deuterium):
                vs.C_snow = update(vs.C_snow, at[2:-2, 2:-2, :vs.taup1], npx.nan)
                vs.C_rz = update(vs.C_rz, at[2:-2, 2:-2, :vs.taup1], -13)
                vs.C_ss = update(vs.C_ss, at[2:-2, 2:-2, :vs.taup1], -7)
                vs.msa_rz = update(
                    vs.msa_rz,
                    at[2:-2, 2:-2, :vs.taup1, :], vs.C_rz[2:-2, 2:-2, :vs.taup1, npx.newaxis],
                )
                vs.msa_rz = update(
                    vs.msa_rz,
                    at[2:-2, 2:-2, :vs.taup1, 0], npx.nan,
                )
                vs.msa_ss = update(
                    vs.msa_ss,
                    at[2:-2, 2:-2, :vs.taup1, :], vs.C_ss[2:-2, 2:-2, :vs.taup1, npx.newaxis],
                )
                vs.msa_ss = update(
                    vs.msa_ss,
                    at[2:-2, 2:-2, :vs.taup1, 0], npx.nan,
                )
                iso_rz = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
                iso_ss = allocate(state.dimensions, ("x", "y", "timesteps", "ages"))
                iso_rz = update(
                    iso_rz,
                    at[2:-2, 2:-2, :, :], npx.where(npx.isnan(vs.msa_rz), 0, vs.msa_rz)[2:-2, 2:-2, :, :],
                )
                iso_ss = update(
                    iso_ss,
                    at[2:-2, 2:-2, :, :], npx.where(npx.isnan(vs.msa_ss), 0, vs.msa_ss)[2:-2, 2:-2, :, :],
                )
                vs.msa_s = update(
                    vs.msa_s,
                    at[2:-2, 2:-2, :, :], (vs.sa_rz[2:-2, 2:-2, :, :] / vs.sa_s[2:-2, 2:-2, :, :]) * iso_rz[2:-2, 2:-2, :, :] + (vs.sa_ss[2:-2, 2:-2, :, :] / vs.sa_s[2:-2, 2:-2, :, :]) * iso_ss[2:-2, 2:-2, :, :],
                )

                vs.C_s = update(
                    vs.C_s,
                    at[2:-2, 2:-2, vs.tau], calc_conc_iso_storage(state, vs.sa_s, vs.msa_s)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
                )

                vs.C_s = update(
                    vs.C_s,
                    at[2:-2, 2:-2, vs.taum1], vs.C_s[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
                )

        @roger_routine(
            dist_safe=False,
            local_variables=[
                "C_IN",
            ],
        )
        def set_forcing_setup(self, state):
            vs = state.variables
            settings = state.settings

            if settings.enable_deuterium:
                vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, 0], npx.nan)
                vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("d2H", self._input_dir, 'forcing_tracer.nc'))

            if settings.enable_oxygen18:
                vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, 0], npx.nan)
                vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, 1:], self._read_var_from_nc("d18O", self._input_dir, 'forcing_tracer.nc'))

            if settings.enable_deuterium or settings.enable_oxygen18:
                vs.C_IN = update(vs.C_IN, at[2:-2, 2:-2, :], self._ffill_3d(state, vs.C_IN)[2:-2, 2:-2, :])

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
                "C_IN",
                "C_snow",
                "M_in"

            ],
        )
        def set_forcing(self, state):
            vs = state.variables

            vs.ta = update(vs.ta, at[2:-2, 2:-2], self._read_var_from_nc("ta", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("prec", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mat_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], self._read_var_from_nc("inf_mp_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt] + self._read_var_from_nc("inf_sc_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], self._read_var_from_nc("inf_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.transp = update(vs.transp, at[2:-2, 2:-2], self._read_var_from_nc("transp", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], self._read_var_from_nc("evap_soil", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.cpr_rz = update(vs.cpr_rz, at[2:-2, 2:-2], self._read_var_from_nc("cpr_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], self._read_var_from_nc("q_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt])
            vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], self._read_var_from_nc("q_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt])

            vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc')[:, :, vs.itt] - vs.S_pwp_ss[2:-2, 2:-2])
            vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_rz[2:-2, 2:-2, vs.tau] + vs.S_ss[2:-2, 2:-2, vs.tau])
            vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("S_snow", self._base_path, 'states_hm.nc')[:, :, vs.itt])

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

            # mix isotopes from snow melt and rainfall
            vs.C_in = update(
                vs.C_in,
                at[2:-2, 2:-2], npx.where(npx.isfinite(vs.C_snow[2:-2, 2:-2, vs.taum1]), vs.C_snow[2:-2, 2:-2, vs.taum1], npx.where(vs.prec[2:-2, 2:-2, vs.tau] > 0, vs.C_IN[2:-2, 2:-2, vs.itt], npx.nan)),
            )
            vs.M_in = update(
                vs.M_in,
                at[2:-2, 2:-2], vs.C_in[2:-2, 2:-2] * vs.prec[2:-2, 2:-2, vs.tau],
            )

        @roger_routine
        def set_diagnostics(self, state):
            diagnostics = state.diagnostics

            diagnostics["rates"].output_variables = ["q_ss"]
            diagnostics["rates"].output_frequency = 24 * 60 * 60
            diagnostics["rates"].sampling_frequency = 1

            diagnostics["averages"].output_variables = ["C_q_ss"]
            diagnostics["averages"].output_frequency = 24 * 60 * 60
            diagnostics["averages"].sampling_frequency = 1

            diagnostics["constant"].output_variables = ["sas_params_transp", "sas_params_q_rz", "sas_params_q_ss"]
            diagnostics["constant"].output_frequency = 0
            diagnostics["constant"].sampling_frequency = 1

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
        vs.C_snow = update(
            vs.C_snow,
            at[2:-2, 2:-2, vs.taum1], vs.C_snow[2:-2, 2:-2, vs.tau],
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
            C_snow=vs.C_snow,
            )

    @roger_kernel
    def calc_conc_iso_storage(state, sa, msa):
        """Calculates isotope signal of storage.
        """
        vs = state.variables

        mask = npx.isfinite(msa[:, :, vs.tau, :])
        vals = allocate(state.dimensions, ("x", "y", "ages"))
        weights = allocate(state.dimensions, ("x", "y", "ages"))
        vals = update(
            vals,
            at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2, :], msa[2:-2, 2:-2, vs.tau, :], 0),
        )
        weights = update(
            weights,
            at[2:-2, 2:-2, :], npx.where(sa[2:-2, 2:-2, vs.tau, :] * mask[2:-2, 2:-2, :] > 0, sa[2:-2, 2:-2, vs.tau, :] / npx.sum(sa[2:-2, 2:-2, vs.tau, :] * mask[2:-2, 2:-2, :], axis=-1)[:, :, npx.newaxis], 0),
        )
        conc = allocate(state.dimensions, ("x", "y"))
        # calculate weighted average
        conc = update(
            conc,
            at[2:-2, 2:-2], npx.sum(vals[2:-2, 2:-2, :] * weights[2:-2, 2:-2, :], axis=-1),
        )
        conc = update(
            conc,
            at[2:-2, 2:-2], npx.where(conc[2:-2, 2:-2] != 0, conc[2:-2, 2:-2], npx.nan),
        )

        return conc

    @roger_kernel
    def _ffill(loop_arr):
        def loop_body(i, loop_arr):
            loop_arr = update(
                loop_arr,
                at[:, :, i], npx.where(loop_arr[:, :, i] == 0, loop_arr[:, :, i - 1], loop_arr[:, :, i]),
            )

            return loop_arr

        loop_arr = for_loop(1, loop_arr.shape[2], loop_body, loop_arr)

        return loop_arr

    tms = transport_model_structure.replace("_", " ")
    model = SVATTRANSPORTSetup()
    if tms not in ['complete-mixing', 'piston']:
        model._set_nsamples(nsamples)
    else:
        if rs.mpi_comm:
            model._set_nsamples(rst.proc_num)
    model._set_tm_structure(tms)
    identifier = f'SVATTRANSPORT_{transport_model_structure}'
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
