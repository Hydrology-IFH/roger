from pathlib import Path
import h5netcdf
import pandas as pd
import numpy as onp

from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at
import roger.lookuptables as lut


class ONEDEVENTSetup(RogerSetup):
    """A 1D model for a single event.
    """
    _base_path = Path(__file__).parent
    _input_dir = _base_path / "input"

    # custom helper functions
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

    def _get_nitt(self, path_dir, file):
        nc_file = path_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['Time']
            return len(onp.array(var_obj))

    def _get_runlen(self, path_dir, file):
        nc_file = path_dir / file
        with h5netcdf.File(nc_file, "r", decode_vlen_strings=False) as infile:
            var_obj = infile.variables['dt']
            return onp.sum(var_obj)

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "ONEDEVENT"

        # total grid numbers in x-- and y-direction
        settings.nx, settings.ny = 1, 1
        # derive total number of time steps from forcing
        settings.nitt = self._get_nitt(self._input_dir, 'forcing.nc')
        # length of simulation (in seconds)
        settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')

        # spatial discretization (in meters)
        settings.dx = 1
        settings.dy = 1

        # origin of spatial grid
        settings.x_origin = 0.0
        settings.y_origin = 0.0

        # enable specific processes
        settings.enable_groundwater_boundary = False
        settings.enable_lateral_flow = True

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
        settings = state.settings

        # temporal grid
        vs.dt_secs = 10 * 60
        vs.dt = 1 / 6
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
        pass

    @roger_routine
    def set_parameters_setup(self, state):
        vs = state.variables

        # land use ID (see README for description)
        vs.lu_id = update(vs.lu_id, at[2:-2, 2:-2], 8)
        # degree of sealing (-)
        vs.sealing = update(vs.sealing, at[2:-2, 2:-2], 0)
        # surface slope (-)
        vs.slope = update(vs.slope, at[2:-2, 2:-2], 0.05)
        # convert slope to percentage
        vs.slope_per = update(vs.slope_per, at[2:-2, 2:-2], vs.slope[2:-2, 2:-2] * 100)
        # total surface depression storage (mm)
        vs.S_dep_tot = update(vs.S_dep_tot, at[2:-2, 2:-2], 0)
        # soil depth (mm)
        vs.z_soil = update(vs.z_soil, at[2:-2, 2:-2], 2000)
        # density of vertical macropores (1/m2)
        vs.dmpv = update(vs.dmpv, at[2:-2, 2:-2], 50)
        # density of horizontal macropores (1/m2)
        vs.dmph = update(vs.dmph, at[2:-2, 2:-2], 50)
        # total length of vertical macropores (mm)
        vs.lmpv = update(vs.lmpv, at[2:-2, 2:-2], 1000)
        # air capacity (-)
        vs.theta_ac = update(vs.theta_ac, at[2:-2, 2:-2], 0.1)
        # usable field capacity (-)
        vs.theta_ufc = update(vs.theta_ufc, at[2:-2, 2:-2], 0.1)
        # permanent wilting point (-)
        vs.theta_pwp = update(vs.theta_pwp, at[2:-2, 2:-2], 0.2)
        # saturated hydraulic conductivity (mm/h)
        vs.ks = update(vs.ks, at[2:-2, 2:-2], 5)
        # hydraulic conductivity of bedrock/saturated zone (mm/h)
        vs.kf = update(vs.kf, at[2:-2, 2:-2], 2500)

    @roger_routine
    def set_parameters(self, state):
        pass

    @roger_routine
    def set_initial_conditions_setup(self, state):
        pass

    @roger_routine
    def set_initial_conditions(self, state):
        vs = state.variables

        # interception storage of upper surface layer (mm)
        vs.S_int_top = update(vs.S_int_top, at[2:-2, 2:-2, :vs.taup1], 0)
        # snow water equivalent stored in upper surface layer (mm)
        vs.swe_top = update(vs.swe_top, at[2:-2, 2:-2, :vs.taup1], 0)
        # interception storage of lower surface layer (mm)
        vs.S_int_ground = update(vs.S_int_ground, at[2:-2, 2:-2, :vs.taup1], 0)
        # snow water equivalent stored in lower surface layer (mm)
        vs.swe_ground = update(vs.swe_ground, at[2:-2, 2:-2, :vs.taup1], 0)
        # surface depression storage (mm)
        vs.S_dep = update(vs.S_dep, at[2:-2, 2:-2, :vs.taup1], 0)
        # snow cover storage (mm)
        vs.S_snow = update(vs.S_snow, at[2:-2, 2:-2, :vs.taup1], 0)
        # snow water equivalent of snow cover (mm)
        vs.swe = update(vs.swe, at[2:-2, 2:-2, :vs.taup1], 0)
        # soil water content of root zone/upper soil layer (-)
        vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], 0.3)
        # soil water content of subsoil/lower soil layer (-)
        vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], 0.3)

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

        vs.prec = update(vs.prec, at[2:-2, 2:-2, vs.tau], self._read_var_from_nc("PREC", self._input_dir, 'forcing.nc')[:, :, vs.itt])
        vs.ta = update(vs.ta, at[2:-2, 2:-2], self._read_var_from_nc("TA", self._input_dir, 'forcing.nc')[:, :, vs.itt])
        vs.event_id = update(vs.event_id, at[vs.tau], 1)

    @roger_routine
    def set_diagnostics(self, state):
        pass

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
