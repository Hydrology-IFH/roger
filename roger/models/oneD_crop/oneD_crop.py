from pathlib import Path
import h5netcdf
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, at
from roger.core.surface import calc_parameters_surface_kernel
import roger.lookuptables as lut
import numpy as onp


class ONEDCROPSetup(RogerSetup):
    """A 1D-model including crop phenology/crop rotation.
    """
    _base_path = Path(__file__).parent
    _input_dir = _base_path / "input"

    # custom helper functions
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
        settings.identifier = "SVATCROP"

        # output frequency (in seconds)
        settings.output_frequency = 86400
        # total grid numbers in x- and y-direction
        settings.nx, settings.ny = 1, 1
        # length of simulation (in seconds)
        settings.runlen = self._get_runlen(self._input_dir, 'forcing.nc')
        # number of time steps in meteorological data
        settings.nitt_forc = len(self._read_var_from_nc("Time", self._input_dir, "forcing.nc"))

        # spatial discretization (in meters)
        settings.dx = 1
        settings.dy = 1

        # origin of spatial grid
        settings.x_origin = 0.0
        settings.y_origin = 0.0
        # origin of time steps (e.g. 01-01-2023)
        settings.time_origin = self._get_time_origin(self._input_dir, 'forcing.nc')

        # enable specific processes
        settings.enable_lateral_flow = True
        settings.enable_groundwater_boundary = False
        settings.enable_crop_water_stress = True
        settings.enable_crop_phenology = True
        settings.enable_crop_rotation = True
        settings.enable_macropore_lower_boundary_condition = False
        settings.enable_adaptive_time_stepping = True

        if settings.enable_crop_rotation:
            settings.ncrops = 3
            settings.ncr = 3

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

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "lut_ilu",
            "lut_gc",
            "lut_gcm",
            "lut_is",
            "lut_rdlu",
            "lut_mlms",
            "lut_crops",
        ],
    )
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
        # crop parameters
        vs.lut_crops = update(vs.lut_crops, at[:, :], lut.ARR_CP)

    @roger_routine
    def set_topography(self, state):
        pass

    @roger_routine(
        dist_safe=False,
        local_variables=[
            "lu_id",
            "sealing",
            "slope",
            "slope_per",
            "S_dep_tot",
            "z_soil",
            "dmpv",
            "dmph",
            "lmpv",
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

        vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 0], 599)
        vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 1], 539)
        vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 2], 564)

        vs.z_root = update(vs.z_root, at[2:-2, 2:-2, :2], 200)
        vs.z_root_crop = update(
            vs.z_root_crop,
            at[2:-2, 2:-2, :2, 0], 200
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

        vs.theta_rz = update(vs.theta_rz, at[2:-2, 2:-2, :vs.taup1], 0.3)
        vs.theta_ss = update(vs.theta_ss, at[2:-2, 2:-2, :vs.taup1], 0.3)

        vs.update(set_initial_conditions_crops_kernel(state))

    @roger_routine
    def set_boundary_conditions_setup(self, state):
        pass

    @roger_routine
    def set_boundary_conditions(self, state):
        pass

    @roger_routine(
        dist_safe=False,
        local_variables=["PREC", "TA", "TA_MIN", "TA_MAX", "PET"],
    )
    def set_forcing_setup(self, state):
        vs = state.variables

        vs.PREC = update(vs.PREC, at[:], self._read_var_from_nc("PREC", self._input_dir, "forcing.nc")[0, 0, :])
        vs.TA = update(vs.TA, at[:], self._read_var_from_nc("TA", self._input_dir, "forcing.nc")[0, 0, :])
        vs.TA_MIN = update(
            vs.TA_MIN, at[:], self._read_var_from_nc("TA_min", self._input_dir, "forcing.nc")[0, 0, :]
        )
        vs.TA_MAX = update(
            vs.TA_MAX, at[:], self._read_var_from_nc("TA_max", self._input_dir, "forcing.nc")[0, 0, :]
        )
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
                vs.prec_day, at[:, :, :], vs.PREC[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24]
            )
            vs.ta_day = update(
                vs.ta_day, at[:, :, :], vs.TA[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24]
            )
            vs.ta_min = update(
                vs.ta_min,
                at[:, :],
                npx.min(vs.TA_MIN[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24], axis=-1),
            )
            vs.ta_max = update(
                vs.ta_max,
                at[:, :],
                npx.max(vs.TA_MAX[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24], axis=-1),
            )
            vs.pet_day = update(
                vs.pet_day, at[:, :, :], vs.PET[npx.newaxis, npx.newaxis, vs.itt_forc : vs.itt_forc + 6 * 24]
            )
            vs.itt_forc = vs.itt_forc + 6 * 24

        if (vs.year[1] != vs.year[0]) & (vs.itt > 1):
            vs.itt_cr = vs.itt_cr + 2
            vs.crop_type = update(vs.crop_type, at[2:-2, 2:-2, 0], 599)
            vs.crop_type = update(
                vs.crop_type,
                at[2:-2, 2:-2, 1],
                539,
            )
            vs.crop_type = update(
                vs.crop_type,
                at[2:-2, 2:-2, 2],
                599,
            )


    @roger_routine
    def set_diagnostics(self, state):
        pass

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
    vs.re_rg_pwp = update(vs.re_rg_pwp, at[2:-2, 2:-2], 0)
    vs.re_rg = update(vs.re_rg, at[2:-2, 2:-2], 0)
    vs.re_rl_pwp = update(vs.re_rl_pwp, at[2:-2, 2:-2], 0)
    vs.re_rl = update(vs.re_rl, at[2:-2, 2:-2], 0)

    return KernelOutput(
        ta_min=vs.ta_min,
        ta_max=vs.ta_max,
        gdd_sum=vs.gdd_sum,
        t_grow_cc=vs.t_grow_cc,
        t_grow_root=vs.t_grow_root,
        ccc=vs.ccc,
        z_root_crop=vs.z_root_crop,
        re_rg_pwp=vs.re_rg_pwp,
        re_rg=vs.re_rg,
        re_rl_pwp=vs.re_rl_pwp,
        re_rl=vs.re_rl,
    )
