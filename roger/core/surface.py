from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core import numerics
from roger.core.utilities import _get_row_no
from roger.core.operators import numpy as npx, update, at, for_loop


@roger_kernel
def calc_S(state):
    """
    Calculates soil water content
    """
    vs = state.variables

    vs.S_sur = update(
        vs.S_sur,
        at[2:-2, 2:-2, vs.tau],
        (
            vs.S_int_top[2:-2, 2:-2, vs.tau]
            + vs.S_int_ground[2:-2, 2:-2, vs.tau]
            + vs.S_dep[2:-2, 2:-2, vs.tau]
            + vs.S_snow[2:-2, 2:-2, vs.tau]
            + vs.z0[2:-2, 2:-2, vs.tau]
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_sur=vs.S_sur)


@roger_routine
def calculate_surface(state):
    """
    Calculates soil storage and storage dependent variables
    """
    vs = state.variables
    vs.update(calc_S(state))


@roger_kernel
def calc_topo_kernel(state):
    vs = state.variables
    """
    River mask
    """
    river_mask = vs.lu_id == 20
    vs.maskRiver = update(vs.maskRiver, at[...], river_mask)

    """
    lake mask
    """
    lake_mask = vs.lu_id == 14
    vs.maskLake = update(vs.maskLake, at[...], lake_mask)

    """
    Catchment mask
    """
    catch_mask = (vs.lu_id != 14) & (vs.lu_id != 20) & (vs.lu_id != 999)
    vs.maskCatch = update(vs.maskCatch, at[...], catch_mask)
    """
    Urban mask
    """
    urban_mask = (vs.lu_id == 0) & (vs.lu_id == 31) & (vs.lu_id == 32) & (vs.lu_id == 33)
    vs.maskUrban = update(vs.maskUrban, at[...], urban_mask)

    return KernelOutput(
        maskLake=vs.maskLake,
        maskRiver=vs.maskRiver,
        maskCatch=vs.maskCatch,
        maskUrban=vs.maskUrban,
    )


@roger_kernel
def calc_parameters_surface_kernel(state):
    vs = state.variables

    # land use dependent upper interception storage
    S_int_top_tot = allocate(state.dimensions, ("x", "y"))
    trees_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    trees_cond = update(
        trees_cond,
        at[:, :],
        npx.isin(vs.lu_id, npx.array([10, 11, 12, 15, 17])),
    )

    def loop_body_S_int_top_tot(i, S_int_top_tot):
        mask = (vs.lu_id == i) & trees_cond
        row_no = _get_row_no(vs.lut_ilu[:, 0], i)
        S_int_top_tot = update(
            S_int_top_tot,
            at[2:-2, 2:-2],
            npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month[vs.tau]], S_int_top_tot[2:-2, 2:-2])
            * vs.maskCatch[2:-2, 2:-2],
        )

        return S_int_top_tot

    S_int_top_tot = for_loop(10, 16, loop_body_S_int_top_tot, S_int_top_tot)

    vs.S_int_top_tot = update(
        vs.S_int_top_tot, at[2:-2, 2:-2], S_int_top_tot[2:-2, 2:-2] * vs.c_int[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    # land use dependent lower interception storage
    S_int_ground_tot = allocate(state.dimensions, ("x", "y"))

    ground_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    ground_cond = update(
        ground_cond, at[:, :], npx.isin(vs.lu_id, npx.array([0, 5, 6, 7, 8, 9, 13, 98, 31, 32, 33, 40, 41, 50, 98]))
    )

    def loop_body_S_int_ground_tot(i, S_int_ground_tot):
        mask = (vs.lu_id == i) & ground_cond
        row_no = _get_row_no(vs.lut_ilu[:, 0], i)
        S_int_ground_tot = update(
            S_int_ground_tot,
            at[2:-2, 2:-2],
            npx.where(mask[2:-2, 2:-2], vs.lut_ilu[row_no, vs.month[vs.tau]], S_int_ground_tot[2:-2, 2:-2])
            * vs.maskCatch[2:-2, 2:-2],
        )

        return S_int_ground_tot

    trees_ground_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    trees_ground_cond = update(trees_ground_cond, at[:, :], npx.isin(vs.lu_id, npx.array([10, 11, 12, 15, 16])))

    def loop_body_S_int_ground_tot_trees(i, S_int_ground_tot):
        mask = (vs.lu_id == i) & trees_ground_cond
        S_int_ground_tot = update(
            S_int_ground_tot,
            at[2:-2, 2:-2],
            npx.where(mask[2:-2, 2:-2], 1, S_int_ground_tot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
        )

        return S_int_ground_tot

    S_int_ground_tot = for_loop(0, 51, loop_body_S_int_ground_tot, S_int_ground_tot)
    S_int_ground_tot = for_loop(10, 16, loop_body_S_int_ground_tot_trees, S_int_ground_tot)

    vs.S_int_ground_tot = update(
        vs.S_int_ground_tot,
        at[2:-2, 2:-2],
        S_int_ground_tot[2:-2, 2:-2] * vs.c_int[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # land use dependent ground cover (canopy cover)
    ground_cover = allocate(state.dimensions, ("x", "y"))

    cc_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    cc_cond = update(
        cc_cond,
        at[:, :],
        npx.isin(vs.lu_id, npx.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 98, 31, 32, 33, 40, 41, 50, 90, 98])),
    )

    def loop_body_ground_cover(i, ground_cover):
        mask = (vs.lu_id == i) & cc_cond
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        ground_cover = update(
            ground_cover,
            at[2:-2, 2:-2],
            npx.where(mask[2:-2, 2:-2], vs.lut_gc[row_no, vs.month[vs.tau]], ground_cover[2:-2, 2:-2])
            * vs.maskCatch[2:-2, 2:-2],
        )

        return ground_cover

    ground_cover = for_loop(0, 51, loop_body_ground_cover, ground_cover)

    vs.ground_cover = update(
        vs.ground_cover, at[2:-2, 2:-2, vs.tau], ground_cover[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    # land use dependent transpiration coeffcient
    basal_transp_coeff = allocate(state.dimensions, ("x", "y"))

    def loop_body_basal_transp_coeff(i, basal_transp_coeff):
        mask = (vs.lu_id == i) & cc_cond
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        basal_transp_coeff = update(
            basal_transp_coeff,
            at[2:-2, 2:-2],
            npx.where(
                mask[2:-2, 2:-2],
                vs.lut_gc[row_no, vs.month[vs.tau]] / vs.lut_gcm[row_no, 1],
                basal_transp_coeff[2:-2, 2:-2],
            )
            * vs.maskCatch[2:-2, 2:-2],
        )

        return basal_transp_coeff

    basal_transp_coeff = for_loop(0, 51, loop_body_basal_transp_coeff, basal_transp_coeff)

    basal_transp_coeff = update(
        basal_transp_coeff,
        at[2:-2, 2:-2],
        npx.where(vs.maskRiver[2:-2, 2:-2] | vs.maskLake[2:-2, 2:-2], 0, basal_transp_coeff[2:-2, 2:-2]),
    )

    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff, at[2:-2, 2:-2], basal_transp_coeff[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    # land use dependent evaporation coeffcient
    basal_evap_coeff = allocate(state.dimensions, ("x", "y"))

    def loop_body_basal_evap_coeff(i, basal_evap_coeff):
        mask = (vs.lu_id == i) & cc_cond
        row_no = _get_row_no(vs.lut_gc[:, 0], i)
        basal_evap_coeff = update(
            basal_evap_coeff,
            at[2:-2, 2:-2],
            npx.where(
                mask[2:-2, 2:-2],
                1 - ((vs.lut_gc[row_no, vs.month[vs.tau]] / vs.lut_gcm[row_no, 1]) * vs.lut_gcm[row_no, 1]),
                basal_evap_coeff[2:-2, 2:-2],
            )
            * vs.maskCatch[2:-2, 2:-2],
        )

        return basal_evap_coeff

    basal_evap_coeff = for_loop(0, 51, loop_body_basal_evap_coeff, basal_evap_coeff)

    basal_evap_coeff = update(
        basal_evap_coeff,
        at[2:-2, 2:-2],
        npx.where(vs.maskRiver[2:-2, 2:-2] | vs.maskLake[2:-2, 2:-2], 1, basal_evap_coeff[2:-2, 2:-2]),
    )

    vs.basal_evap_coeff = update(
        vs.basal_evap_coeff, at[2:-2, 2:-2], basal_evap_coeff[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    # maximum snow interception storage
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2],
        npx.where((vs.ta[2:-2, 2:-2, vs.tau] > -1) & (vs.lu_id[2:-2, 2:-2] == 10), 9, vs.swe_top_tot[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2],
        npx.where((vs.ta[2:-2, 2:-2, vs.tau] > -1) & (vs.lu_id[2:-2, 2:-2] == 11), 15, vs.swe_top_tot[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2],
        npx.where((vs.ta[2:-2, 2:-2, vs.tau] > -1) & (vs.lu_id[2:-2, 2:-2] == 12), 25, vs.swe_top_tot[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2],
        npx.where(
            (vs.ta[2:-2, 2:-2, vs.tau] >= -3) & (vs.ta[2:-2, 2:-2, vs.tau] <= -1) & (vs.lu_id[2:-2, 2:-2] == 10),
            2.5 + 0.5 * vs.ta[2:-2, 2:-2, vs.tau] * 9,
            vs.swe_top_tot[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2],
        npx.where(
            (vs.ta[2:-2, 2:-2, vs.tau] >= -3) & (vs.ta[2:-2, 2:-2, vs.tau] <= -1) & (vs.lu_id[2:-2, 2:-2] == 11),
            2.5 + 0.5 * vs.ta[2:-2, 2:-2, vs.tau] * 15,
            vs.swe_top_tot[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2],
        npx.where(
            (vs.ta[2:-2, 2:-2, vs.tau] >= -3) & (vs.ta[2:-2, 2:-2, vs.tau] <= -1) & (vs.lu_id[2:-2, 2:-2] == 12),
            2.5 + 0.5 * vs.ta[2:-2, 2:-2, vs.tau] * 25,
            vs.swe_top_tot[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2],
        npx.where((vs.ta[2:-2, 2:-2, vs.tau] < -3) & (vs.lu_id[2:-2, 2:-2] == 10), 18, vs.swe_top_tot[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2],
        npx.where((vs.ta[2:-2, 2:-2, vs.tau] < -3) & (vs.lu_id[2:-2, 2:-2] == 11), 30, vs.swe_top_tot[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.swe_top_tot = update(
        vs.swe_top_tot,
        at[2:-2, 2:-2],
        npx.where((vs.ta[2:-2, 2:-2, vs.tau] < -3) & (vs.lu_id[2:-2, 2:-2] == 12), 50, vs.swe_top_tot[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.lai = update(
        vs.lai,
        at[2:-2, 2:-2],
        npx.log(1 / (1 - vs.ground_cover[2:-2, 2:-2, vs.tau])) / npx.log(1 / 0.7) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.throughfall_coeff_top = update(
        vs.throughfall_coeff_top,
        at[2:-2, 2:-2],
        npx.where(
            npx.isin(vs.lu_id[2:-2, 2:-2], npx.array([10, 11, 12])),
            npx.where(vs.lai[2:-2, 2:-2] > 1, 0.1, 1.1 - vs.lai[2:-2, 2:-2]),
            0,
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.throughfall_coeff_ground = update(
        vs.throughfall_coeff_ground,
        at[2:-2, 2:-2],
        npx.where(
            npx.isin(vs.lu_id[2:-2, 2:-2], npx.arange(500, 598)),
            npx.where(vs.lai[2:-2, 2:-2] > 1, 0.1, 1.1 - vs.lai[2:-2, 2:-2]),
            0,
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        S_int_top_tot=vs.S_int_top_tot,
        S_int_ground_tot=vs.S_int_ground_tot,
        ground_cover=vs.ground_cover,
        basal_transp_coeff=vs.basal_transp_coeff,
        basal_evap_coeff=vs.basal_evap_coeff,
        swe_top_tot=vs.swe_top_tot,
        lai=vs.lai,
        throughfall_coeff_top=vs.throughfall_coeff_top,
        throughfall_coeff_ground=vs.throughfall_coeff_ground,
    )


@roger_kernel
def calc_parameters_crops_kernel(state):
    vs = state.variables

    for i in range(500, 600):
        mask = vs.crop_type == i
        row_no = _get_row_no(vs.lut_crops[:, 0], i)
        vs.z_root_crop_max = update(
            vs.z_root_crop_max,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 15] * 1000, vs.z_root_crop_max[2:-2, 2:-2, :]),
        )
        vs.root_growth_rate = update(
            vs.root_growth_rate,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 19], vs.root_growth_rate[2:-2, 2:-2, :]),
        )
        vs.water_stress_coeff_crop = update(
            vs.water_stress_coeff_crop,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 20], vs.water_stress_coeff_crop[2:-2, 2:-2, :]),
        )

    vs.theta_water_stress_crop = update(
        vs.theta_water_stress_crop,
        at[2:-2, 2:-2, :],
        (1 - vs.water_stress_coeff_crop[2:-2, 2:-2])
        * (vs.theta_fc[2:-2, 2:-2, npx.newaxis] - vs.theta_pwp[2:-2, 2:-2, npx.newaxis])
        + vs.theta_pwp[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        z_root_crop_max=vs.z_root_crop_max,
        root_growth_rate=vs.root_growth_rate,
        water_stress_coeff_crop=vs.water_stress_coeff_crop,
        theta_water_stress_crop=vs.theta_water_stress_crop,
    )


@roger_routine
def calculate_parameters(state):
    vs = state.variables
    settings = state.settings

    vs.update(calc_topo_kernel(state))
    if not settings.enable_offline_transport:
        numerics.validate_parameters_surface(state)
        vs.update(calc_parameters_surface_kernel(state))
        if settings.enable_crop_phenology:
            vs.update(calc_parameters_crops_kernel(state))


@roger_kernel
def calc_initial_conditions_surface_kernel(state):
    vs = state.variables

    vs.S_sur = update(
        vs.S_sur,
        at[2:-2, 2:-2, :2],
        (
            vs.S_int_top[2:-2, 2:-2, :2]
            + vs.S_int_ground[2:-2, 2:-2, :2]
            + vs.S_dep[2:-2, 2:-2, :2]
            + vs.S_snow[2:-2, 2:-2, :2]
        )
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(S_sur=vs.S_sur)


@roger_routine
def calculate_initial_conditions(state):
    """
    calculate storage, concentrations, etc
    """
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        numerics.validate_initial_conditions_surface(state)
        vs.update(calc_initial_conditions_surface_kernel(state))
