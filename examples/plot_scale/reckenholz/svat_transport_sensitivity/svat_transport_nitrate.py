from pathlib import Path
import os
import h5netcdf
from SALib.sample import saltelli
import numpy as onp

from roger import runtime_settings as rs, runtime_state as rst
rs.backend = "numpy"
rs.force_overwrite = True
if rs.mpi_comm:
    rs.num_proc = (rst.proc_num, 1)
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at, where, random_uniform, scipy_stats as sstx
from roger.tools.setup import write_forcing_tracer


class SVATCROPTRANSPORTSetup(RogerSetup):
    """A SVAT transport model for nitrate including
    crop phenology/crop rotation.
    """
    _base_path = Path(__file__).parent
    _bounds = None
    _params = None
    _nrows = None
    _tm_structure = None
    _identifier = None
    _input_dir = None

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

    def _set_tm_structure(self, tm_structure):
        self._tm_structure = tm_structure

    def _set_identifier(self, identifier):
        self._identifier = identifier

    def _set_nsamples(self, nsamples):
        self._nsamples = nsamples

    def _sample_params(self, nsamples):
        if self._tm_structure == "complete-mixing":
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

        elif self._tm_structure == "piston":
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

        elif self._tm_structure == "preferential":
            self._bounds = {
                'num_vars': 15,
                'names': ['b_transp', 'b_q_rz', 'b_q_ss', 'alpha_transp', 'alpha_q',
                          'km_denit_rz', 'km_denit_ss', 'dmax_denit_rz',
                          'dmax_denit_ss', 'km_nit_rz', 'km_nit_ss',
                          'dmax_nit_rz', 'dmax_nit_ss', 'kmin_rz', 'kmin_ss'],
                'bounds': [[1, 90],
                           [1, 90],
                           [1, 90],
                           [0.01, 1.5],
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

        elif self._tm_structure == "advection-dispersion":
            self._bounds = {
                'num_vars': 15,
                'names': ['b_transp', 'a_q_rz', 'a_q_ss', 'alpha_transp', 'alpha_q',
                          'km_denit_rz', 'km_denit_ss', 'dmax_denit_rz',
                          'dmax_denit_ss', 'km_nit_rz', 'km_nit_ss',
                          'dmax_nit_rz', 'dmax_nit_ss', 'kmin_rz', 'kmin_ss'],
                'bounds': [[1, 90],
                           [1, 90],
                           [1, 90],
                           [0.01, 1.5],
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

        elif self._tm_structure == "complete-mixing + advection-dispersion":
            self._bounds = {
                'num_vars': 14,
                'names': ['a_q_rz', 'a_q_ss', 'alpha_transp', 'alpha_q',
                          'km_denit_rz', 'km_denit_ss', 'dmax_denit_rz',
                          'dmax_denit_ss', 'km_nit_rz', 'km_nit_ss',
                          'dmax_nit_rz', 'dmax_nit_ss', 'kmin_rz', 'kmin_ss'],
                'bounds': [[1, 90],
                           [1, 90],
                           [0.01, 1.5],
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

        elif self._tm_structure == "time-variant complete-mixing + advection-dispersion":
            self._bounds = {
                'num_vars': 14,
                'names': ['a_q_rz', 'a_q_ss', 'alpha_transp', 'alpha_q',
                          'km_denit_rz', 'km_denit_ss', 'dmax_denit_rz',
                          'dmax_denit_ss', 'km_nit_rz', 'km_nit_ss',
                          'dmax_nit_rz', 'dmax_nit_ss', 'kmin_rz', 'kmin_ss'],
                'bounds': [[1, 90],
                           [1, 90],
                           [0.01, 1.5],
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

        elif self._tm_structure == "time-variant preferential":
            self._bounds = {
                'num_vars': 15,
                'names': ['b_transp', 'b_q_rz', 'b_q_ss', 'alpha_transp', 'alpha_q',
                          'km_denit_rz', 'km_denit_ss', 'dmax_denit_rz',
                          'dmax_denit_ss', 'km_nit_rz', 'km_nit_ss',
                          'dmax_nit_rz', 'dmax_nit_ss', 'kmin_rz', 'kmin_ss'],
                'bounds': [[1, 90],
                           [1, 90],
                           [1, 90],
                           [0.01, 1.5],
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

        elif self._tm_structure == "time-variant advection-dispersion":
            self._bounds = {
                'num_vars': 15,
                'names': ['b_transp', 'a_q_rz', 'a_q_ss', 'alpha_transp', 'alpha_q',
                          'km_denit_rz', 'km_denit_ss', 'dmax_denit_rz',
                          'dmax_denit_ss', 'km_nit_rz', 'km_nit_ss',
                          'dmax_nit_rz', 'dmax_nit_ss', 'kmin_rz', 'kmin_ss'],
                'bounds': [[1, 90],
                           [1, 90],
                           [1, 90],
                           [0.01, 1.5],
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

        elif self._tm_structure == "time-variant":
            self._bounds = {
                'num_vars': 15,
                'names': ['ab_transp', 'ab_q_rz', 'ab_q_ss', 'alpha_transp', 'alpha_q',
                          'km_denit_rz', 'km_denit_ss', 'dmax_denit_rz',
                          'dmax_denit_ss', 'km_nit_rz', 'km_nit_ss',
                          'dmax_nit_rz', 'dmax_nit_ss', 'kmin_rz', 'kmin_ss'],
                'bounds': [[1, 90],
                           [1, 90],
                           [1, 90],
                           [0.01, 1.5],
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

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = self._identifier

        settings.nx, settings.ny, settings.nz = 1, 1, 1
        settings.nitt = self._get_nitt(self._base_path, 'forcing_tracer.nc')
        settings.ages = settings.nitt
        settings.nages = settings.nitt + 1
        settings.runlen = self._get_runlen(self._base_path, 'forcing_tracer.nc')

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
            "S_PWP_RZ",
            "S_PWP_SS",
            "S_SAT_RZ",
            "S_SAT_SS",
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
        ],
    )
    def set_parameters_setup(self, state):
        vs = state.variables
        settings = state.settings

        vs.S_PWP_RZ = update(vs.S_PWP_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_rz", self._base_path, 'states_hm.nc'))
        vs.S_PWP_SS = update(vs.S_PWP_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_pwp_ss", self._base_path, 'states_hm.nc'))
        vs.S_SAT_RZ = update(vs.S_SAT_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_rz", self._base_path, 'states_hm.nc'))
        vs.S_SAT_SS = update(vs.S_SAT_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_sat_ss", self._base_path, 'states_hm.nc'))

        vs.S_pwp_rz = update(vs.S_pwp_rz, at[2:-2, 2:-2], vs.S_PWP_RZ[2:-2, 2:-2, 0])
        vs.S_pwp_ss = update(vs.S_pwp_ss, at[2:-2, 2:-2], vs.S_PWP_SS[2:-2, 2:-2, 0])
        vs.S_sat_rz = update(vs.S_sat_rz, at[2:-2, 2:-2], vs.S_SAT_RZ[2:-2, 2:-2, 0])
        vs.S_sat_ss = update(vs.S_sat_ss, at[2:-2, 2:-2], vs.S_SAT_SS[2:-2, 2:-2, 0])

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
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], self._params[:, 0, npx.newaxis])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], self._params[:, 1, npx.newaxis])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], self._params[:, 2, npx.newaxis])
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
        elif settings.tm_structure == "advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 1], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 2], self._params[:, 0, npx.newaxis])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], self._params[:, 1, npx.newaxis])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], self._params[:, 2, npx.newaxis])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
        elif settings.tm_structure == "complete-mixing + advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 1], self._params[:, 0, npx.newaxis])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 2], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 3)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 1], self._params[:, 1, npx.newaxis])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 2], 1)
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
        elif settings.tm_structure == "time-variant complete-mixing + advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 32)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], self._params[:, 0, npx.newaxis])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 32)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], self._params[:, 1, npx.newaxis])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
        elif settings.tm_structure == "time-variant advection-dispersion":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], self._params[:, 0, npx.newaxis])
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 32)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], self._params[:, 1, npx.newaxis])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 32)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], self._params[:, 2, npx.newaxis])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
        elif settings.tm_structure == "time-variant preferential":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], self._params[:, 0, npx.newaxis])
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], self._params[:, 1, npx.newaxis])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 31)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], self._params[:, 2, npx.newaxis])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)
        elif settings.tm_structure == "time-variant":
            vs.sas_params_evap_soil = update(vs.sas_params_evap_soil, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_cpr_rz = update(vs.sas_params_cpr_rz, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 0], 35)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 4], self._params[:, 0, npx.newaxis])
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_transp = update(vs.sas_params_transp, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 0], 35)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 4], self._params[:, 1, npx.newaxis])
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_rz = update(vs.sas_params_q_rz, at[2:-2, 2:-2, 6], vs.S_sat_rz[2:-2, 2:-2] - vs.S_pwp_rz[2:-2, 2:-2])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 0], 35)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 3], 1)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 4], self._params[:, 2, npx.newaxis])
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 5], 0)
            vs.sas_params_q_ss = update(vs.sas_params_q_ss, at[2:-2, 2:-2, 6], vs.S_sat_ss[2:-2, 2:-2] - vs.S_pwp_ss[2:-2, 2:-2])
            vs.sas_params_re_rg = update(vs.sas_params_re_rg, at[2:-2, 2:-2, 0], 21)
            vs.sas_params_re_rl = update(vs.sas_params_re_rl, at[2:-2, 2:-2, 0], 22)

    @roger_routine
    def set_parameters(self, state):
        pass

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "S_RZ",
            "S_SS",
        ],
    )
    def set_initial_conditions_setup(self, state):
        vs = state.variables

        vs.S_RZ = update(vs.S_RZ, at[2:-2, 2:-2, :], self._read_var_from_nc("S_rz", self._base_path, 'states_hm.nc'))
        vs.S_SS = update(vs.S_SS, at[2:-2, 2:-2, :], self._read_var_from_nc("S_ss", self._base_path, 'states_hm.nc'))

    @roger_routine
    def set_initial_conditions(self, state):
        vs = state.variables
        settings = state.settings

        vs.S_S = update(vs.S_S, at[2:-2, 2:-2, :], vs.S_RZ[2:-2, 2:-2, :] + vs.S_SS[2:-2, 2:-2, :])
        vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], vs.S_RZ[2:-2, 2:-2, 0, npx.newaxis] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
        vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], vs.S_SS[2:-2, 2:-2, 0, npx.newaxis] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
        vs.S_s = update(vs.S_s, at[2:-2, 2:-2, :vs.taup1], vs.S_S[2:-2, 2:-2, 0, npx.newaxis] - (vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis] + vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis]))

        vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, :vs.taup1], vs.S_RZ[2:-2, 2:-2, 0, npx.newaxis] - vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis])
        vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, :vs.taup1], vs.S_SS[2:-2, 2:-2, 0, npx.newaxis] - vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis])
        vs.S_s = update(vs.S_s, at[2:-2, 2:-2, :vs.taup1], vs.S_S[2:-2, 2:-2, 0, npx.newaxis] - (vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis] + vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis]))

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

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "PREC",
            "TA",
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
            "NMIN_IN",
            "NORG_IN",
            "M_IN",
            "C_IN",
        ],
    )
    def set_forcing_setup(self, state):
        vs = state.variables
        settings = state.settings

        vs.TA = update(vs.TA, at[2:-2, 2:-2, :], self._read_var_from_nc("ta", self._base_path, 'states_hm.nc'))
        vs.PREC = update(vs.PREC, at[2:-2, 2:-2, :], self._read_var_from_nc("prec", self._base_path, 'states_hm.nc'))
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

        # convert kg N/ha to mg/square meter
        vs.NMIN_IN = update(vs.NMIN_IN, at[2:-2, 2:-2, :], self._read_var_from_nc("Nmin", 'tracer_input.nc') * 100 * settings.dx * settings.dy)
        vs.NORG_IN = update(vs.NORG_IN, at[2:-2, 2:-2, :], self._read_var_from_nc("Norg", 'tracer_input.nc') * 100 * settings.dx * settings.dy)

        mask_rain = (vs.PREC > 0) & (vs.TA > 0)
        mask_sol = (vs.NMIN_IN > 0)
        nn_rain = npx.int64(npx.sum(npx.any(mask_rain, axis=(0, 1))))
        nn_sol = npx.int64(npx.sum(npx.any(mask_sol, axis=(0, 1))))
        vs.update(set_nitrate_input_kernel(state, nn_rain, nn_sol))

    @roger_routine
    def set_forcing(self, state):
        vs = state.variables

        vs.update(set_states_kernel(state))
        vs.update(set_forcing_nitrate_kernel(state))

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


@roger_kernel(static_args=("nn_rain", "nn_sol"))
def set_nitrate_input_kernel(state, nn_rain, nn_sol):
    vs = state.variables

    NMIN_IN = allocate(state.dimensions, ("x", "y", "t"))

    mask_rain = (vs.PREC > 0) & (vs.TA > 0)
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
        input_itt = npx.nanargmin(npx.where(rain_idx - sol_idx[i] < 0, npx.NaN, rain_idx - sol_idx[i]))
        start_rain = rain_idx[input_itt]
        rain_sum = update(
            rain_sum,
            at[:, :], npx.max(npx.where(npx.cumsum(vs.PREC[:, :, start_rain:], axis=-1) <= 20, npx.max(npx.cumsum(vs.PREC[:, :, start_rain:], axis=-1), axis=-1), 0), axis=-1),
        )
        nn_end = npx.max(npx.where(npx.cumsum(vs.PREC[:, :, start_rain:]) <= 20, npx.max(npx.arange(npx.shape(vs.PREC)[2])[npx.newaxis, npx.newaxis, npx.shape(vs.PREC)[2]-start_rain], axis=-1), 0))
        end_rain = update(end_rain, at[:], start_rain + nn_end)
        end_rain = update(end_rain, at[:], npx.where(end_rain > npx.shape(vs.PREC)[2], npx.shape(vs.PREC)[2], end_rain))

        # proportions for redistribution
        NMIN_IN = update(
            NMIN_IN,
            at[:, :, start_rain:end_rain[0]], vs.M_IN[:, :, sol_idx[i], npx.newaxis] * (vs.PREC[:, :, start_rain:end_rain[0]] / rain_sum[:, :, npx.newaxis]),
        )

    # solute input concentration
    vs.M_IN = update(
        vs.M_IN,
        at[2:-2, 2:-2, :], NMIN_IN[2:-2, 2:-2, :] * 0.3,
    )
    vs.C_IN = update(
        vs.C_IN,
        at[2:-2, 2:-2, :], npx.where(vs.PREC > 0, vs.M_IN / vs.PREC, 0)[2:-2, 2:-2, :],
    )
    vs.NMIN_IN = update(
        vs.NMIN_IN,
        at[2:-2, 2:-2, :], NMIN_IN[2:-2, 2:-2, :] * 0.7,
    )

    return KernelOutput(
        M_IN=vs.M_IN,
        C_IN=vs.C_IN,
        NMIN_IN=vs.NMIN_IN,
    )


@roger_kernel
def set_forcing_nitrate_kernel(state):
    vs = state.variables

    vs.C_in = update(
        vs.C_in,
        at[2:-2, 2:-2], vs.C_IN[2:-2, 2:-2, vs.itt] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.M_in = update(
        vs.M_in,
        at[2:-2, 2:-2], vs.M_IN[2:-2, 2:-2, vs.itt] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.Nmin_in = update(
        vs.Nmin_in,
        at[2:-2, 2:-2], vs.NMIN_IN[2:-2, 2:-2, vs.itt] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.Norg_in = update(
        vs.Norg_in,
        at[2:-2, 2:-2], vs.NORG_IN[2:-2, 2:-2, vs.itt] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        M_in=vs.M_in,
        C_in=vs.C_in,
        Nmin_in=vs.Nmin_in,
        Norg_in=vs.Norg_in,
    )


@roger_kernel
def set_states_kernel(state):
    vs = state.variables

    vs.inf_mat_rz = update(vs.inf_mat_rz, at[2:-2, 2:-2], vs.INF_MAT_RZ[2:-2, 2:-2, vs.itt])
    vs.inf_pf_rz = update(vs.inf_pf_rz, at[2:-2, 2:-2], vs.INF_PF_RZ[2:-2, 2:-2, vs.itt])
    vs.inf_pf_ss = update(vs.inf_pf_ss, at[2:-2, 2:-2], vs.INF_PF_SS[2:-2, 2:-2, vs.itt])
    vs.transp = update(vs.transp, at[2:-2, 2:-2], vs.TRANSP[2:-2, 2:-2, vs.itt])
    vs.evap_soil = update(vs.evap_soil, at[2:-2, 2:-2], vs.EVAP_SOIL[2:-2, 2:-2, vs.itt])
    vs.q_rz = update(vs.q_rz, at[2:-2, 2:-2], vs.Q_RZ[2:-2, 2:-2, vs.itt])
    vs.q_ss = update(vs.q_ss, at[2:-2, 2:-2], vs.Q_SS[2:-2, 2:-2, vs.itt])
    vs.cpr_rz = update(vs.cpr_ss, at[2:-2, 2:-2], vs.CPR_RZ[2:-2, 2:-2, vs.itt])
    vs.re_rg = update(vs.re_rg, at[2:-2, 2:-2], vs.RE_RG[2:-2, 2:-2, vs.itt])
    vs.re_rl = update(vs.re_rl, at[2:-2, 2:-2], vs.RE_RL[2:-2, 2:-2, vs.itt])

    vs.S_rz = update(vs.S_rz, at[2:-2, 2:-2, vs.tau], vs.S_RZ[2:-2, 2:-2, vs.itt])
    vs.S_ss = update(vs.S_ss, at[2:-2, 2:-2, vs.tau], vs.S_SS[2:-2, 2:-2, vs.itt])
    vs.S_s = update(vs.S_s, at[2:-2, 2:-2, vs.tau], vs.S_S[2:-2, 2:-2, vs.itt])

    return KernelOutput(
        inf_mat_rz=vs.inf_mat_rz,
        inf_pf_rz=vs.inf_pf_rz,
        inf_pf_ss=vs.inf_pf_ss,
        transp=vs.transp,
        evap_soil=vs.evap_soil,
        q_rz=vs.q_rz,
        q_ss=vs.q_ss,
        cpr_rz=vs.cpr_rz,
        re_rg=vs.re_rg,
        re_rl=vs.re_rl,
        S_rz=vs.S_rz,
        S_ss=vs.S_ss,
    )


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


nsamples = 2**10  # number of samples
lys_experiments = ["lys2", "lys3", "lys4", "lys8", "lys9"]
tm_structures = ['complete-mixing', 'piston',
                 'preferential', 'complete-mixing + advection-dispersion',
                 'time-variant preferential',
                 'time-variant complete-mixing + advection-dispersion']
for lys_experiment in lys_experiments:
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        model = SVATCROPTRANSPORTSetup()
        model._set_nsamples(nsamples)
        model._set_tm_structure(tm_structure)
        identifier = f'SVATCROPTRANSPORT_{tms}_{lys_experiment}_nitrate'
        model._set_identifier(identifier)
        model._sample_params(nsamples)
        input_path = model._base_path / "input" / lys_experiment
        model._set_input_dir(input_path)
        forcing_path = model._input_dir / "forcing_tracer.nc"
        if not os.path.exists(forcing_path):
            write_forcing_tracer(input_path, 'NO3')
        model.setup()
        model.warmup()
        model.run()
