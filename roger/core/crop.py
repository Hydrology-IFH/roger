from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at, for_loop
import roger.lookuptables as lut
from roger.core.utilities import _get_row_no
from roger.core import transport


@roger_kernel
def calc_gdd(state):
    """Calculates growing degree days"""
    vs = state.variables

    ta = allocate(state.dimensions, ("x", "y"))
    ta = update(
        ta,
        at[2:-2, 2:-2],
        ((vs.ta_max[2:-2, 2:-2, vs.tau] + vs.ta_min[2:-2, 2:-2, vs.tau]) / 2),
    )
    mask = (ta[:, :, npx.newaxis] > vs.ta_base) & (ta[:, :, npx.newaxis] < vs.ta_ceil)
    vs.gdd = update(
        vs.gdd,
        at[2:-2, 2:-2, :],
        npx.where(mask[2:-2, 2:-2], ta[2:-2, 2:-2, npx.newaxis] - vs.ta_base[2:-2, 2:-2], 0),
    )
    vs.gdd_sum = update_add(
        vs.gdd_sum,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :],
    )

    return KernelOutput(gdd=vs.gdd, gdd_sum=vs.gdd_sum)


@roger_kernel
def calc_k_stress_transp_crop(state):
    """Calculates water stress coeffcient of transpiration"""
    vs = state.variables

    mask = (vs.theta_rz[:, :, vs.tau, npx.newaxis] > vs.theta_water_stress_crop) & npx.isin(
        vs.crop_type, npx.arange(500, 600, 1, dtype=int)
    )
    vs.k_stress_transp_crop = update(
        vs.k_stress_transp_crop,
        at[2:-2, 2:-2, :],
        npx.where(
            mask[2:-2, 2:-2, :],
            1,
            (vs.theta_rz[2:-2, 2:-2, vs.tau, npx.newaxis] - vs.theta_pwp[2:-2, 2:-2, npx.newaxis])
            / (vs.theta_water_stress_crop[2:-2, 2:-2, :] - vs.theta_pwp[2:-2, 2:-2, npx.newaxis]),
        ),
    )

    return KernelOutput(k_stress_transp_crop=vs.k_stress_transp_crop)


@roger_kernel
def calc_basal_evap_coeff_crop(state):
    """Calculates crop evaporation coeffcient"""
    vs = state.variables

    vs.basal_evap_coeff_crop = update(
        vs.basal_evap_coeff_crop,
        at[2:-2, 2:-2, :],
        1 - vs.ccc[2:-2, 2:-2, vs.tau, :],
    )

    return KernelOutput(basal_evap_coeff_crop=vs.basal_evap_coeff_crop)


@roger_kernel
def calc_t_grow(state):
    """
    Calculates time since growing
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y", "crops"))
    mask_summer = npx.isin(vs.crop_type, lut.SUMMER_CROPS)
    mask_winter = npx.isin(vs.crop_type, lut.WINTER_CROPS)
    mask_winter_catch = npx.isin(vs.crop_type, lut.WINTER_CATCH_CROPS)
    mask_my_init_winter = npx.isin(vs.crop_type, lut.WINTER_MULTI_YEAR_CROPS_INIT)
    mask_my_cont_summer = npx.isin(vs.crop_type, lut.SUMMER_MULTI_YEAR_CROPS_CONT)
    mask_my_init_summer = npx.isin(vs.crop_type, lut.SUMMER_MULTI_YEAR_CROPS_INIT)
    mask_my_cont_winter = npx.isin(vs.crop_type, lut.WINTER_MULTI_YEAR_CROPS_CONT)

    mask1 = mask_summer & (vs.doy[vs.tau] < vs.doy_start)
    mask2 = mask_summer & (vs.doy[vs.tau] >= vs.doy_start) & (vs.doy[vs.tau] <= vs.doy_end)
    mask3 = mask_summer & (vs.doy[vs.tau] > vs.doy_end)
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask1[2:-2, 2:-2, :], 0, vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
    )
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_transp_crop[2:-2, 2:-2, :] * mask2[2:-2, 2:-2, :],
    )
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask3[2:-2, 2:-2, :], 0, vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
    )

    mask7 = mask_winter & (
        (vs.doy[vs.tau] >= vs.doy_start)
        | ((vs.doy[vs.tau] <= vs.doy_end) & (vs.doy[vs.tau] > arr0) & (vs.ccc[:, :, vs.tau, :] > 0))
    )
    mask8 = mask_winter & (vs.doy[vs.tau] > vs.doy_end) & (vs.doy[vs.tau] < vs.doy_start)
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_transp_crop[2:-2, 2:-2, :] * mask7[2:-2, 2:-2, :],
    )
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask8[2:-2, 2:-2, :], 0, vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
    )
    mask9 = mask_winter_catch & (
        (vs.doy[vs.tau] >= vs.doy_start)
        | ((vs.doy[vs.tau] <= vs.doy_end) & (vs.doy[vs.tau] > arr0) & (vs.ccc[:, :, vs.tau, :] > 0))
    )
    mask10 = mask_winter_catch & (vs.doy[vs.tau] > vs.doy_end) & (vs.doy[vs.tau] < vs.doy_start)
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_transp_crop[2:-2, 2:-2, :] * mask9[2:-2, 2:-2, :],
    )
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask10[2:-2, 2:-2, :], 0, vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
    )
    mask11 = mask_my_init_winter & (
        (vs.doy[vs.tau] >= vs.doy_start) | ((vs.doy[vs.tau] <= vs.doy_end) & (vs.doy[vs.tau] > arr0) & (vs.ccc[:, :, vs.tau, :] > 0))
    )
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_transp_crop[2:-2, 2:-2, :] * mask11[2:-2, 2:-2, :],
    )

    mask121 = mask_my_init_winter[:, :, 0] & mask_my_cont_summer[:, :, 1] & (vs.doy[vs.tau] == vs.doy_end[:, :, 0])
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(mask121[2:-2, 2:-2], vs.t_grow_cc[2:-2, 2:-2, vs.tau, 0], vs.t_grow_cc[2:-2, 2:-2, vs.tau, 1]),
    )

    mask12 = mask_my_init_winter[:, :, 0] & mask_my_cont_summer[:, :, 1] & (vs.doy[vs.tau] >= vs.doy_start[:, :, 1]) & (vs.doy[vs.tau] <= vs.doy_end[:, :, 1]) & (vs.ccc[:, :, vs.tau, 1] > 0)
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, 1],
        vs.gdd[2:-2, 2:-2, 1] * vs.k_stress_transp_crop[2:-2, 2:-2, 1] * mask12[2:-2, 2:-2],
    )

    mask13 = mask_my_init_summer & (vs.doy[vs.tau] >= vs.doy_start) & (vs.doy[vs.tau] <= vs.doy_end)
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_transp_crop[2:-2, 2:-2, :] * mask13[2:-2, 2:-2, :],
    )

    mask131 = mask_my_init_summer[:, :, 1] & mask_my_cont_winter[:, :, 2] & (vs.doy[vs.tau] == vs.doy_end[:, :, 1])
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, 2],
        npx.where(mask131[2:-2, 2:-2], vs.t_grow_cc[2:-2, 2:-2, vs.tau, 1], vs.t_grow_cc[2:-2, 2:-2, vs.tau, 2]),
    )

    # cutting of grass
    mask_grass = vs.crop_type == 573
    mask21 = mask_grass[:, :, 1] & ((vs.doy[vs.tau] == 167) | (vs.doy[vs.tau] == 223))
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(mask21[2:-2, 2:-2], 0, vs.t_grow_cc[2:-2, 2:-2, vs.tau, 1]),
    )

    mask22 = mask_grass[:, :, 1] & (vs.doy[vs.tau] >= vs.doy_start[:, :, 1]) & (vs.doy[vs.tau] <= vs.doy_end[:, :, 1])
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, 1],
        vs.gdd[2:-2, 2:-2, 1] * vs.k_stress_transp_crop[2:-2, 2:-2, 1] * mask22[2:-2, 2:-2],
    )

    mask_miscanthus = vs.crop_type == 591
    mask23 = mask_miscanthus[:, :, 1] & (vs.doy[vs.tau] >= vs.doy_start[:, :, 1]) & (vs.doy[vs.tau] <= vs.doy_end[:, :, 1])
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, 1],
        vs.gdd[2:-2, 2:-2, 1] * vs.k_stress_transp_crop[2:-2, 2:-2, 1] * mask23[2:-2, 2:-2],
    )

    mask1 = mask_summer & (vs.doy[vs.tau] < vs.doy_start)
    mask2 = mask_summer & (vs.doy[vs.tau] >= vs.doy_start) & (vs.doy[vs.tau] <= vs.doy_end)
    mask3 = mask_summer & (vs.doy[vs.tau] > vs.doy_end)
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask1[2:-2, 2:-2, :], 0, vs.t_grow_root[2:-2, 2:-2, vs.tau, :]),
    )
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_root_growth[2:-2, 2:-2, :] * mask2[2:-2, 2:-2, :],
    )
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask3[2:-2, 2:-2, :], 0, vs.t_grow_root[2:-2, 2:-2, vs.tau, :]),
    )

    mask7 = mask_winter & (
        (vs.doy[vs.tau] >= vs.doy_start)
        | ((vs.doy[vs.tau] <= vs.doy_end) & (vs.doy[vs.tau] > arr0) & (vs.ccc[:, :, vs.tau, :] > 0))
    )
    mask8 = mask_winter & (vs.doy[vs.tau] > vs.doy_end) & (vs.doy[vs.tau] < vs.doy_start)
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_root_growth[2:-2, 2:-2, :] * mask7[2:-2, 2:-2, :],
    )
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask8[2:-2, 2:-2, :], 0, vs.t_grow_root[2:-2, 2:-2, vs.tau, :]),
    )
    mask9 = mask_winter_catch & (
        (vs.doy[vs.tau] >= vs.doy_start)
        | ((vs.doy[vs.tau] <= vs.doy_end) & (vs.doy[vs.tau] > arr0) & (vs.ccc[:, :, vs.tau, :] > 0))
    )
    mask10 = mask_winter_catch & (vs.doy[vs.tau] > vs.doy_end) & (vs.doy[vs.tau] < vs.doy_start)
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_root_growth[2:-2, 2:-2, :] * mask9[2:-2, 2:-2, :],
    )
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask10[2:-2, 2:-2, :], 0, vs.t_grow_root[2:-2, 2:-2, vs.tau, :]),
    )

    mask11 = mask_my_init_winter & (
        (vs.doy[vs.tau] >= vs.doy_start) | ((vs.doy[vs.tau] <= vs.doy_end) & (vs.doy[vs.tau] > arr0) & (vs.ccc[:, :, vs.tau, :] > 0))
    )
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_root_growth[2:-2, 2:-2, :] * mask11[2:-2, 2:-2, :],
    )

    mask121 = mask_my_init_winter[:, :, 0] & mask_my_cont_summer[:, :, 1] & (vs.doy[vs.tau] == vs.doy_end[:, :, 0])
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(mask121[2:-2, 2:-2], vs.t_grow_root[2:-2, 2:-2, vs.tau, 0], vs.t_grow_root[2:-2, 2:-2, vs.tau, 1]),
    )

    mask12 = mask_my_init_winter[:, :, 0] & mask_my_cont_summer[:, :, 1] & (vs.doy[vs.tau] >= vs.doy_start[:, :, 1]) & (vs.doy[vs.tau] <= vs.doy_end[:, :, 1]) & (vs.ccc[:, :, vs.tau, 1] > 0)
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, 1],
        vs.gdd[2:-2, 2:-2, 1] * vs.k_stress_root_growth[2:-2, 2:-2, 1] * mask12[2:-2, 2:-2],
    )

    mask13 = mask_my_init_summer & (vs.doy[vs.tau] >= vs.doy_start) & (vs.doy[vs.tau] <= vs.doy_end)
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        vs.gdd[2:-2, 2:-2, :] * vs.k_stress_root_growth[2:-2, 2:-2, :] * mask13[2:-2, 2:-2, :],
    )

    # no crop growth if soil water content is greater than 80% of usable field capacity
    mask14 = (vs.theta_rz[:, :, vs.tau] > (vs.theta_ufc * 1.0) + vs.theta_pwp)
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask14[2:-2, 2:-2, npx.newaxis], 0, vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
    )
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask14[2:-2, 2:-2, npx.newaxis], 0, vs.t_grow_root[2:-2, 2:-2, vs.tau, :]),
    )

    return KernelOutput(t_grow_cc=vs.t_grow_cc, t_grow_root=vs.t_grow_root)


@roger_kernel
def calc_t_decay(state):
    """
    Calculates time since decay of crop canopy
    """
    vs = state.variables

    mask = vs.doy[vs.tau] == vs.doy_dec
    vs.t_decay = update(
        vs.t_decay,
        at[2:-2, 2:-2, :],
        npx.where(mask[2:-2, 2:-2, :], vs.t_grow_cc[2:-2, 2:-2, vs.tau, :], vs.t_decay[2:-2, 2:-2, :]),
    )

    return KernelOutput(t_decay=vs.t_decay)


@roger_kernel
def calc_t_half_mid(state):
    """
    Calculates half time of crop growing
    """
    vs = state.variables

    mask = vs.ccc[:, :, vs.taum1, :] <= (vs.ccc_max / 2)
    vs.t_half_mid = update(
        vs.t_half_mid,
        at[2:-2, 2:-2, :],
        npx.where(mask[2:-2, 2:-2, :], vs.t_grow_cc[2:-2, 2:-2, vs.tau, :], vs.t_half_mid[2:-2, 2:-2, :]),
    )

    return KernelOutput(t_half_mid=vs.t_half_mid)


@roger_kernel
def calc_canopy_cover(state):
    """
    Calculates development of crop canopy cover
    """
    vs = state.variables
    settings = state.settings

    arr0 = allocate(state.dimensions, ("x", "y", "crops"))
    mask_summer = npx.isin(vs.crop_type, lut.SUMMER_CROPS)
    mask_winter = npx.isin(vs.crop_type, lut.WINTER_CROPS)
    mask_winter_catch = npx.isin(vs.crop_type, lut.WINTER_CATCH_CROPS)
    mask_growing_summer = npx.isin(vs.crop_type, npx.array([571, 580, 589], dtype=int))
    mask_growing_winter = npx.isin(vs.crop_type, npx.array([572, 583], dtype=int))
    mask_cont_summer = npx.isin(vs.crop_type, lut.SUMMER_MULTI_YEAR_CROPS_CONT)
    mask_cont_summer_grow = npx.isin(vs.crop_type, lut.SUMMER_MULTI_YEAR_CROPS_CONT_GROW)
    mask_cont_winter = npx.isin(vs.crop_type, lut.WINTER_MULTI_YEAR_CROPS_CONT)
    mask_bare = vs.crop_type == 599

    mask1 = mask_summer & (vs.doy[vs.tau] > vs.doy_mid) & (vs.doy[vs.tau] < vs.doy_dec)
    mask2 = mask_summer & (vs.doy[vs.tau] < vs.doy_start)
    mask3 = (
        mask_summer
        & (vs.doy[vs.tau] >= vs.doy_start)
        & (vs.ccc[:, :, vs.tau, :] < vs.ccc_max)
        & (vs.doy[vs.tau] <= vs.doy_dec)
    )
    mask4 = mask_summer & (vs.doy[vs.tau] > vs.doy_dec) & (vs.doy[vs.tau] <= vs.doy_end)
    mask5 = mask_summer & (vs.doy[vs.tau] > vs.doy_end)
    # mature crop
    vs.ccc_mid = update(
        vs.ccc_mid,
        at[2:-2, 2:-2, :],
        npx.where(mask1[2:-2, 2:-2, :], vs.ccc[2:-2, 2:-2, vs.tau, :], vs.ccc_mid[2:-2, 2:-2, :]),
    )
    # before growing period
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask2[2:-2, 2:-2, :], 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )
    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask3[2:-2, 2:-2, :],
            npx.where(
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :])
                <= vs.ccc_max[2:-2, 2:-2, :] / 2,
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
                vs.ccc_max[2:-2, 2:-2, :]
                - (vs.ccc_max[2:-2, 2:-2, :] / 2 - vs.ccc_min[2:-2, 2:-2, :])
                * npx.exp(
                    -vs.ccc_growth_rate[2:-2, 2:-2, :]
                    * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, :] - vs.t_half_mid[2:-2, 2:-2, :])
                ),
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, :],
        ),
    )

    # decay of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask4[2:-2, 2:-2, :],
            vs.ccc_mid[2:-2, 2:-2, :]
            * (
                1
                - 0.05
                * (
                    npx.exp(
                        (settings.ccc_decay_rate / vs.ccc_mid[2:-2, 2:-2, :])
                        * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, :] - vs.t_decay[2:-2, 2:-2, :])
                        - 1
                    )
                )
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, :],
        ),
    )

    # harvesting
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask5[2:-2, 2:-2, :], 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )

    mask6 = mask_winter & (vs.doy[vs.tau] > vs.doy_mid) & (vs.doy[vs.tau] < vs.doy_dec)
    mask7 = mask_winter & (vs.t_grow_cc[:, :, vs.tau, :] <= 0)
    mask8 = (
        mask_winter
        & (vs.ccc[:, :, vs.tau, :] < vs.ccc_max)
        & (
            (vs.doy[vs.tau] >= vs.doy_start)
            | (vs.doy[vs.tau] <= vs.doy_dec) & (vs.doy[vs.tau] > arr0) & (vs.t_grow_cc[:, :, vs.tau, :] > 0)
        )
    )
    mask9 = (
        mask_winter
        & (vs.doy[vs.tau] > vs.doy_dec)
        & (vs.doy[vs.tau] <= vs.doy_end)
        & (vs.t_grow_cc[:, :, vs.tau, :] > 0)
    )
    mask10 = mask_winter & (vs.doy[vs.tau] > vs.doy_end) & (vs.doy[vs.tau] < vs.doy_start)
    # mature crop
    vs.ccc_mid = update(
        vs.ccc_mid,
        at[2:-2, 2:-2, :],
        npx.where(mask6[2:-2, 2:-2, :], vs.ccc[2:-2, 2:-2, vs.tau, :], vs.ccc_mid[2:-2, 2:-2, :]),
    )
    # before growing period
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask7[2:-2, 2:-2, :], 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )
    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask8[2:-2, 2:-2, :],
            npx.where(
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :])
                <= vs.ccc_max[2:-2, 2:-2, :] / 2,
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
                vs.ccc_max[2:-2, 2:-2, :]
                - (vs.ccc_max[2:-2, 2:-2, :] / 2 - vs.ccc_min[2:-2, 2:-2, :])
                * npx.exp(
                    -vs.ccc_growth_rate[2:-2, 2:-2, :]
                    * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, :] - vs.t_half_mid[2:-2, 2:-2, :])
                ),
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, :],
        ),
    )
    # decay of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask9[2:-2, 2:-2, :],
            vs.ccc_mid[2:-2, 2:-2, :]
            * (
                1
                - 0.05
                * (
                    npx.exp(
                        (settings.ccc_decay_rate / vs.ccc_mid[2:-2, 2:-2, :])
                        * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, :] - vs.t_decay[2:-2, 2:-2, :])
                        - 1
                    )
                )
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, :],
        ),
    )
    # harvesting
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask10[2:-2, 2:-2, :], 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )

    # winter catch crops
    mask11 = mask_winter_catch & (
        (vs.doy[vs.tau] > vs.doy_mid) | ((vs.doy[vs.tau] < vs.doy_dec) & (vs.doy[vs.tau] > arr0))
    )
    mask12 = mask_winter_catch & (vs.t_grow_cc[:, :, vs.tau, :] <= 0)
    mask13 = (
        mask_winter_catch
        & (vs.ccc[:, :, vs.tau, :] < vs.ccc_max)
        & (
            (vs.doy[vs.tau] >= vs.doy_start)
            | (vs.doy[vs.tau] <= vs.doy_dec) & (vs.doy[vs.tau] > arr0) & (vs.t_grow_cc[:, :, vs.tau, :] > 0)
        )
    )
    mask14 = mask_winter_catch & (
        ((vs.doy[vs.tau] > vs.doy_dec) & (vs.doy[vs.tau] < vs.doy_start))
        | (
            (vs.doy[vs.tau] <= vs.doy_end)
            & (vs.doy[vs.tau] > vs.doy_dec)
            & (vs.doy[vs.tau] > arr0)
            & (vs.t_grow_cc[:, :, vs.tau, :] > 0)
        )
    )
    mask15 = mask_winter_catch & (vs.doy[vs.tau] > vs.doy_end) & (vs.doy[vs.tau] < vs.doy_start)
    # mature crop
    vs.ccc_mid = update(
        vs.ccc_mid,
        at[2:-2, 2:-2, :],
        npx.where(mask11[2:-2, 2:-2, :], vs.ccc[2:-2, 2:-2, vs.tau, :], vs.ccc_mid[2:-2, 2:-2, :]),
    )
    # before growing period
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask12[2:-2, 2:-2, :], 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )

    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask13[2:-2, 2:-2, :],
            npx.where(
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :])
                <= vs.ccc_max[2:-2, 2:-2, :] / 2,
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
                vs.ccc_max[2:-2, 2:-2, :]
                - (vs.ccc_max[2:-2, 2:-2, :] / 2 - vs.ccc_min[2:-2, 2:-2, :])
                * npx.exp(
                    -vs.ccc_growth_rate[2:-2, 2:-2, :]
                    * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, :] - vs.t_half_mid[2:-2, 2:-2, :])
                ),
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, :],
        ),
    )
    # decay of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask14[2:-2, 2:-2, :],
            vs.ccc_mid[2:-2, 2:-2, :]
            * (
                1
                - 0.05
                * (
                    npx.exp(
                        (settings.ccc_decay_rate / vs.ccc_mid[2:-2, 2:-2, :])
                        * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, :] - vs.t_decay[2:-2, 2:-2, :])
                        - 1
                    )
                )
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, :],
        ),
    )
    # harvesting
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask15[2:-2, 2:-2, :], 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )

    # multi-year crop starting in summer
    mask16 = mask_growing_summer & (vs.doy[vs.tau] < vs.doy_start)
    mask17 = (
        mask_growing_summer
        & (vs.doy[vs.tau] >= vs.doy_start)
        & (vs.ccc[:, :, vs.tau, :] < vs.ccc_max)
        & (vs.doy[vs.tau] <= vs.doy_end)
    )
    # before growing period
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask16[2:-2, 2:-2, :], 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )
    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask17[2:-2, 2:-2, :],
            npx.where(
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :])
                <= vs.ccc_max[2:-2, 2:-2, :] / 2,
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
                vs.ccc_max[2:-2, 2:-2, :]
                - (vs.ccc_max[2:-2, 2:-2, :] / 2 - vs.ccc_min[2:-2, 2:-2, :])
                * npx.exp(
                    -vs.ccc_growth_rate[2:-2, 2:-2, :]
                    * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, :] - vs.t_half_mid[2:-2, 2:-2, :])
                ),
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, :],
        ),
    )

    # multi-year crop starting in winter
    mask18 = mask_growing_winter & (
        (vs.doy[vs.tau] > vs.doy_mid) | ((vs.doy[vs.tau] < vs.doy_dec) & (vs.doy[vs.tau] > arr0))
    )
    mask19 = (
        mask_growing_winter
        & (
            (vs.doy[vs.tau] >= vs.doy_start)
            | (vs.doy[vs.tau] <= vs.doy_end) & (vs.doy[vs.tau] > arr0) & (vs.t_grow_cc[:, :, vs.tau, :] > 0)
        )
    )
    # mature crop
    vs.ccc_mid = update(
        vs.ccc_mid,
        at[2:-2, 2:-2, :],
        npx.where(mask18[2:-2, 2:-2, :], vs.ccc[2:-2, 2:-2, vs.tau, :], vs.ccc_mid[2:-2, 2:-2, :]),
    )
    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask19[2:-2, 2:-2, :],
            npx.where(
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :])
                <= vs.ccc_max[2:-2, 2:-2, :] / 2,
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
                vs.ccc_max[2:-2, 2:-2, :]
                - (vs.ccc_max[2:-2, 2:-2, :] / 2 - vs.ccc_min[2:-2, 2:-2, :])
                * npx.exp(
                    -vs.ccc_growth_rate[2:-2, 2:-2, :]
                    * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, :] - vs.t_half_mid[2:-2, 2:-2, :])
                ),
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, :],
        ),
    )

    # multi-year crop continued in summer
    mask20 = mask_cont_summer & (
        (vs.doy[vs.tau] > vs.doy_mid) & (vs.doy[vs.tau] < vs.doy_end)
    )
    mask21 = (
        mask_cont_summer[:, :, 1] & mask_growing_winter[:, :, 0]
        & (vs.doy[vs.tau] >= vs.doy_start[:, :, 1])
        & (vs.doy[vs.tau] <= vs.doy_end[:, :, 1])
    )
    # mature crop
    vs.ccc_mid = update(
        vs.ccc_mid,
        at[2:-2, 2:-2, :],
        npx.where(mask20[2:-2, 2:-2, :], vs.ccc[2:-2, 2:-2, vs.tau, :], vs.ccc_mid[2:-2, 2:-2, :]),
    )
    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(
            mask21[2:-2, 2:-2],
            npx.where(
                vs.ccc_min[2:-2, 2:-2, 1]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, 1] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, 1])
                <= vs.ccc_max[2:-2, 2:-2, 1] / 2,
                vs.ccc_min[2:-2, 2:-2, 1]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, 1] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, 1]),
                vs.ccc_max[2:-2, 2:-2, 1]
                - (vs.ccc_max[2:-2, 2:-2, 1] / 2 - vs.ccc_min[2:-2, 2:-2, 1])
                * npx.exp(
                    -vs.ccc_growth_rate[2:-2, 2:-2, 1]
                    * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, 1] - vs.t_half_mid[2:-2, 2:-2, 1])
                ),
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, 1],
        ),
    )

    # multi-year crop continued in winter
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, -1],
        npx.where(
            mask_cont_winter[2:-2, 2:-2, -1] & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 1]),
            vs.ccc[2:-2, 2:-2, vs.tau, 1],
            vs.ccc[2:-2, 2:-2, vs.tau, -1]
        ),
    )

    # multi-year crop continued in summer
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(
            mask_cont_winter[2:-2, 2:-2, 0] & mask_cont_summer[2:-2, 2:-2, 1] & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 0]),
            vs.ccc[2:-2, 2:-2, vs.tau, 0],
            vs.ccc[2:-2, 2:-2, vs.tau, 1]
        ),
    )

    # growth after cutting
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(
            mask_cont_winter[2:-2, 2:-2, 0] & mask_cont_summer_grow[2:-2, 2:-2, 1] & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 0]),
            0.3,
            vs.ccc[2:-2, 2:-2, vs.tau, 1]
        ),
    )
    mask22 = mask_cont_summer_grow & (vs.doy[vs.tau] > vs.doy_start) & (vs.doy[vs.tau] <= vs.doy_end)
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask22[2:-2, 2:-2, :],
            npx.where(
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :])
                <= vs.ccc_max[2:-2, 2:-2, :] / 2,
                vs.ccc_min[2:-2, 2:-2, :]
                * npx.exp(vs.ccc_growth_rate[2:-2, 2:-2, :] * vs.t_grow_cc[2:-2, 2:-2, vs.tau, :]),
                vs.ccc_max[2:-2, 2:-2, :]
                - (vs.ccc_max[2:-2, 2:-2, :] / 2 - vs.ccc_min[2:-2, 2:-2, :])
                * npx.exp(
                    -vs.ccc_growth_rate[2:-2, 2:-2, :]
                    * (vs.t_grow_cc[2:-2, 2:-2, vs.tau, :] - vs.t_half_mid[2:-2, 2:-2, :])
                ),
            ),
            vs.ccc[2:-2, 2:-2, vs.tau, :],
        ),
    )

    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, 1], npx.where(mask22[2:-2, 2:-2, 1] & (vs.ccc[2:-2, 2:-2, vs.tau, 1] <= 0.3), 0.3, vs.ccc[2:-2, 2:-2, vs.tau, 1]))

    # multi-year crop stops in winter
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, 0],
        npx.where(
            (mask_cont_winter[2:-2, 2:-2, 0]) & (mask_cont_summer[2:-2, 2:-2, 1] == False) & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 0]),
            0,
            vs.ccc[2:-2, 2:-2, vs.tau, 0]
        ),
    )

    # multi-year crop stops in summer
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(
            (mask_cont_winter[2:-2, 2:-2, -1] == False) & mask_cont_summer[2:-2, 2:-2, 0] & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 1]),
            0,
            vs.ccc[2:-2, 2:-2, vs.tau, 1]
        ),
    )

    # harvesting of miscanthus
    mask_miscanthus = vs.crop_type == 590
    mask23 = mask_miscanthus & (vs.doy[vs.tau] == 90)
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, 0],
        npx.where(mask23[2:-2, 2:-2, 0], 0.3, vs.ccc[2:-2, 2:-2, vs.tau, 0]),
    )

    mask24 = mask_summer & (vs.doy[vs.tau] > vs.doy_end)
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask24[2:-2, 2:-2, :], 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )

    # bare
    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask_bare[2:-2, 2:-2, :], 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )

    vs.ccc = update(
        vs.ccc,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(vs.ccc[2:-2, 2:-2, vs.tau, :] <= 0, 0, vs.ccc[2:-2, 2:-2, vs.tau, :]),
    )

    return KernelOutput(ccc=vs.ccc)


@roger_kernel
def calc_crop_height(state):
    """
    Calculates crop height
    """
    vs = state.variables

    vs.crop_height = update(
        vs.crop_height,
        at[2:-2, 2:-2, :],
        npx.where(
            vs.ccc_max[2:-2, 2:-2, :] > 0,
            (vs.ccc[2:-2, 2:-2, vs.tau, :] / vs.ccc_max[2:-2, 2:-2, :]) * vs.crop_height_max[2:-2, 2:-2, :],
            0,
        ),
    )

    return KernelOutput(crop_height=vs.crop_height)


@roger_kernel
def calc_crop_dev_coeff(state):
    """
    Calculates crop development coefficient
    """
    vs = state.variables

    crop_dev_coeff = allocate(state.dimensions, ("x", "y", "crops", 3))

    crop_dev_coeff = update(
        crop_dev_coeff,
        at[2:-2, 2:-2, :, 0],
        1,
    )
    crop_dev_coeff = update(
        crop_dev_coeff,
        at[2:-2, 2:-2, :, 1],
        npx.where(vs.crop_height[2:-2, 2:-2, :] <= 0, 0, npx.where(vs.crop_height[2:-2, 2:-2, :] > 1, 2, 1.5))
        * vs.ccc[2:-2, 2:-2, vs.tau, :],
    )
    crop_dev_coeff = update(
        crop_dev_coeff,
        at[2:-2, 2:-2, :, 2],
        vs.ccc[2:-2, 2:-2, vs.tau, :] ** (1 / (1 + vs.crop_height[2:-2, 2:-2, :])),
    )
    vs.crop_dev_coeff = update(
        vs.crop_dev_coeff,
        at[2:-2, 2:-2, :],
        npx.nanmin(crop_dev_coeff[2:-2, 2:-2, :], axis=-1),
    )

    return KernelOutput(crop_dev_coeff=vs.crop_dev_coeff)


@roger_kernel
def calc_basal_crop_coeff(state):
    """
    Calculates basal crop coefficient
    """
    vs = state.variables
    settings = state.settings

    vs.basal_crop_coeff = update(
        vs.basal_crop_coeff,
        at[2:-2, 2:-2, :],
        settings.basal_crop_coeff_min
        + vs.ccc[2:-2, 2:-2, vs.tau, :] * (vs.basal_crop_coeff_mid[2:-2, 2:-2, :] - settings.basal_crop_coeff_min),
    )

    # bare
    mask_bare = vs.crop_type == 599
    vs.basal_crop_coeff = update(
        vs.basal_crop_coeff,
        at[2:-2, 2:-2, :],
        npx.where(mask_bare[2:-2, 2:-2, :], 0, vs.basal_crop_coeff[2:-2, 2:-2, :]),
    )

    return KernelOutput(basal_crop_coeff=vs.basal_crop_coeff)


@roger_kernel
def calc_S_int_tot(state):
    """
    Calculates potential crop interception storage
    """
    vs = state.variables

    vs.lai_crop = update(
        vs.lai_crop,
        at[2:-2, 2:-2, :],
        npx.log(1 / (1 - vs.ccc[2:-2, 2:-2, vs.tau, :])) / npx.log(1 / 0.7),
    )

    vs.S_int_tot_crop = update(
        vs.S_int_tot_crop,
        at[2:-2, 2:-2, :],
        0.2 * vs.lai_crop[2:-2, 2:-2, :],
    )

    return KernelOutput(lai_crop=vs.lai_crop, S_int_tot_crop=vs.S_int_tot_crop)


@roger_kernel
def calc_k_stress_root_growth(state):
    """
    Calculates water stress coeffcient of crop root growth
    """
    vs = state.variables

    mask = vs.lu_id[:, :, npx.newaxis] == vs.crop_type
    vs.k_stress_root_growth = update(
        vs.k_stress_root_growth,
        at[2:-2, 2:-2, :],
        npx.where(
            mask[2:-2, 2:-2, :],
            1,
            (vs.theta_rz[2:-2, 2:-2, vs.tau, npx.newaxis] - vs.theta_pwp[2:-2, 2:-2, npx.newaxis])
            / (vs.theta_water_stress_crop[2:-2, 2:-2, :] - vs.theta_pwp[2:-2, 2:-2, npx.newaxis]),
        ),
    )
    vs.k_stress_root_growth = update(
        vs.k_stress_root_growth,
        at[2:-2, 2:-2, :],
        npx.where(vs.k_stress_root_growth[2:-2, 2:-2, :] > 1, 1, vs.k_stress_root_growth[2:-2, 2:-2, :]),
    )

    return KernelOutput(k_stress_root_growth=vs.k_stress_root_growth)


@roger_kernel
def calc_root_growth(state):
    """
    Calculates root growth of crops
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y", "crops"))
    mask_summer = npx.isin(vs.crop_type, lut.SUMMER_CROPS)
    mask_winter = npx.isin(vs.crop_type, lut.WINTER_CROPS)
    mask_winter_catch = npx.isin(vs.crop_type, lut.WINTER_CATCH_CROPS)
    mask_growing_summer = npx.isin(vs.crop_type, npx.array([571, 580, 589], dtype=int))
    mask_growing_winter = npx.isin(vs.crop_type, npx.array([572, 583], dtype=int))
    mask_cont_summer = npx.isin(vs.crop_type, lut.SUMMER_MULTI_YEAR_CROPS_CONT)
    mask_cont_summer_grow = npx.isin(vs.crop_type, lut.SUMMER_MULTI_YEAR_CROPS_CONT_GROW)
    mask_cont_winter = npx.isin(vs.crop_type, lut.WINTER_MULTI_YEAR_CROPS_CONT)
    mask_bare = vs.crop_type == 599

    # summer crops
    mask1 = mask_summer & (vs.doy[vs.tau] < vs.doy_start)
    mask2 = mask_summer & (vs.doy[vs.tau] >= vs.doy_start) & (vs.doy[vs.tau] <= vs.doy_mid)
    mask3 = mask_summer & (vs.doy[vs.tau] > vs.doy_end)

    # before growing period
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask1[2:-2, 2:-2, :], vs.z_evap[2:-2, 2:-2, npx.newaxis], vs.z_root_crop[2:-2, 2:-2, vs.tau, :]),
    )
    # root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask2[2:-2, 2:-2, :],
            (
                (vs.z_root_crop_max[2:-2, 2:-2, :] / 1000)
                - ((vs.z_root_crop_max[2:-2, 2:-2, :] - vs.z_evap[2:-2, 2:-2, npx.newaxis]) / 1000)
                * npx.exp(vs.root_growth_rate[2:-2, 2:-2, :] * vs.t_grow_root[2:-2, 2:-2, vs.tau, :])
            )
            * 1000,
            vs.z_root_crop[2:-2, 2:-2, vs.tau, :],
        ),
    )
    # harvesting
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask3[2:-2, 2:-2, :], vs.z_evap[2:-2, 2:-2, npx.newaxis], vs.z_root_crop[2:-2, 2:-2, vs.tau, :]),
    )

    # winter crops
    mask4 = mask_winter & (vs.t_grow_root[:, :, vs.tau, :] <= 0)
    mask5 = mask_winter & (
        (vs.doy[vs.tau] >= vs.doy_start)
        | (vs.doy[vs.tau] <= vs.doy_mid) & (vs.doy[vs.tau] > arr0) & (vs.t_grow_root[:, :, vs.tau, :] > 0)
    )
    mask6 = mask_winter & (vs.doy[vs.tau] > vs.doy_end) & (vs.doy[vs.tau] < vs.doy_start)
    # before growing period
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask4[2:-2, 2:-2, :], vs.z_evap[2:-2, 2:-2, npx.newaxis], vs.z_root_crop[2:-2, 2:-2, vs.tau, :]),
    )
    # root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask5[2:-2, 2:-2, :],
            (
                (vs.z_root_crop_max[2:-2, 2:-2, :] / 1000)
                - ((vs.z_root_crop_max[2:-2, 2:-2, :] - vs.z_evap[2:-2, 2:-2, npx.newaxis]) / 1000)
                * npx.exp(vs.root_growth_rate[2:-2, 2:-2, :] * vs.t_grow_root[2:-2, 2:-2, vs.tau, :])
            )
            * 1000,
            vs.z_root_crop[2:-2, 2:-2, vs.tau, :],
        ),
    )
    # harvesting
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask6[2:-2, 2:-2, :], vs.z_evap[2:-2, 2:-2, npx.newaxis], vs.z_root_crop[2:-2, 2:-2, vs.tau, :]),
    )

    mask7 = mask_winter_catch & (vs.t_grow_root[:, :, vs.tau, :] <= 0)
    mask8 = mask_winter_catch & (vs.doy[vs.tau] >= vs.doy_start) & (vs.doy[vs.tau] <= vs.doy_mid)
    mask9 = mask_winter_catch & (vs.doy[vs.tau] > vs.doy_end) & (vs.doy[vs.tau] < vs.doy_start)
    # before growing period
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask7[2:-2, 2:-2, :], vs.z_evap[2:-2, 2:-2, npx.newaxis], vs.z_root_crop[2:-2, 2:-2, vs.tau, :]),
    )
    # root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask8[2:-2, 2:-2, :],
            (
                (vs.z_root_crop_max[2:-2, 2:-2, :] / 1000)
                - ((vs.z_root_crop_max[2:-2, 2:-2, :] - vs.z_evap[2:-2, 2:-2, npx.newaxis]) / 1000)
                * npx.exp(vs.root_growth_rate[2:-2, 2:-2, :] * vs.t_grow_root[2:-2, 2:-2, vs.tau, :])
            )
            * 1000,
            vs.z_root_crop[2:-2, 2:-2, vs.tau, :],
        ),
    )
    # harvesting
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask9[2:-2, 2:-2, :], vs.z_evap[2:-2, 2:-2, npx.newaxis], vs.z_root_crop[2:-2, 2:-2, vs.tau, :]),
    )

    # multi-year crop starting in summer
    mask10 = mask_growing_summer & (vs.doy[vs.tau] < vs.doy_start)
    mask11 = mask_growing_summer & (vs.doy[vs.tau] >= vs.doy_start) & (vs.doy[vs.tau] <= vs.doy_mid)
    # before growing period
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask10[2:-2, 2:-2, :], vs.z_evap[2:-2, 2:-2, npx.newaxis], vs.z_root_crop[2:-2, 2:-2, vs.tau, :]),
    )
    # root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask11[2:-2, 2:-2, :],
            (
                (vs.z_root_crop_max[2:-2, 2:-2, :] / 1000)
                - ((vs.z_root_crop_max[2:-2, 2:-2, :] - vs.z_evap[2:-2, 2:-2, npx.newaxis]) / 1000)
                * npx.exp(vs.root_growth_rate[2:-2, 2:-2, :] * vs.t_grow_root[2:-2, 2:-2, vs.tau, :])
            )
            * 1000,
            vs.z_root_crop[2:-2, 2:-2, vs.tau, :],
        ),
    )

    mask13 = (
        mask_growing_winter
        & (
            (vs.doy[vs.tau] >= vs.doy_start)
            | (vs.doy[vs.tau] <= vs.doy_end) & (vs.doy[vs.tau] > arr0) & (vs.t_grow_cc[:, :, vs.tau, :] > 0)
        )
    )

    # crop root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask13[2:-2, 2:-2, :],
            (
                (vs.z_root_crop_max[2:-2, 2:-2, :] / 1000)
                - ((vs.z_root_crop_max[2:-2, 2:-2, :] - vs.z_evap[2:-2, 2:-2, npx.newaxis]) / 1000)
                * npx.exp(vs.root_growth_rate[2:-2, 2:-2, :] * vs.t_grow_root[2:-2, 2:-2, vs.tau, :])
            )
            * 1000,
            vs.z_root_crop[2:-2, 2:-2, vs.tau, :],
        ),
    )

    # multi-year crop continued in summer
    mask14 = (
        mask_cont_summer[:, :, 1] & mask_growing_winter[:, :, 0]
        & (vs.doy[vs.tau] >= vs.doy_start[:, :, 1])
        & (vs.doy[vs.tau] <= vs.doy_end[:, :, 1])
    )

    # crop root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(
            mask14[2:-2, 2:-2],
            (
                (vs.z_root_crop_max[2:-2, 2:-2, 1] / 1000)
                - ((vs.z_root_crop_max[2:-2, 2:-2, 1] - vs.z_evap[2:-2, 2:-2]) / 1000)
                * npx.exp(vs.root_growth_rate[2:-2, 2:-2, 1] * vs.t_grow_root[2:-2, 2:-2, vs.tau, 1])
            )
            * 1000,
            vs.z_root_crop[2:-2, 2:-2, vs.tau, 1],
        ),
    )

    # multi-year crop continued in winter
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, -1],
        npx.where(
            mask_cont_winter[2:-2, 2:-2, -1] & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 1]),
            vs.z_root_crop[2:-2, 2:-2, vs.tau, 1],
            vs.z_root_crop[2:-2, 2:-2, vs.tau, -1]
        ),
    )

    # multi-year crop continued in summer
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(
            mask_cont_winter[2:-2, 2:-2, 0] & mask_cont_summer[2:-2, 2:-2, 1] & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 0]),
            vs.z_root_crop[2:-2, 2:-2, vs.tau, 0],
            vs.z_root_crop[2:-2, 2:-2, vs.tau, 1]
        ),
    )

    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(
            mask_cont_winter[2:-2, 2:-2, 0] & mask_cont_summer_grow[2:-2, 2:-2, 1] & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 0]),
            vs.z_root_crop[2:-2, 2:-2, vs.tau, 0],
            vs.z_root_crop[2:-2, 2:-2, vs.tau, 1]
        ),
    )

    # multi-year crop stops in winter
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, 0],
        npx.where(
            (mask_cont_winter[2:-2, 2:-2, 0]) & (mask_cont_summer[2:-2, 2:-2, 1] == False) & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 0]),
            vs.z_evap[2:-2, 2:-2],
            vs.z_root_crop[2:-2, 2:-2, vs.tau, 0]
        ),
    )

    # multi-year crop stops in summer
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, 1],
        npx.where(
            (mask_cont_winter[2:-2, 2:-2, -1] == False) & mask_cont_summer[2:-2, 2:-2, 0] & (vs.doy[vs.tau] == vs.doy_end[2:-2, 2:-2, 1]),
            vs.z_evap[2:-2, 2:-2],
            vs.z_root_crop[2:-2, 2:-2, vs.tau, 1]
        ),
    )

    # crop root growth stops if 70 % of total soil depth is reached
    mask_stop_growth = vs.z_root_crop[:, :, vs.tau, :] >= vs.zroot_to_zsoil_max[:, :, npx.newaxis] * vs.z_soil[:, :, npx.newaxis]
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            mask_stop_growth[2:-2, 2:-2, :],
            vs.zroot_to_zsoil_max[2:-2, 2:-2, npx.newaxis] * vs.z_soil[2:-2, 2:-2, npx.newaxis],
            vs.z_root_crop[2:-2, 2:-2, vs.tau, :],
        ),
    )

    # bare
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(mask_bare[2:-2, 2:-2, :], vs.z_evap[2:-2, 2:-2, npx.newaxis], vs.z_root_crop[2:-2, 2:-2, vs.tau, :]),
    )

    return KernelOutput(z_root_crop=vs.z_root_crop)


@roger_kernel
def update_lu_id(state):
    """
    Updates land use while crop rotation
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y"))
    mask1 = (
        (vs.doy[vs.tau] >= arr0)
        & (vs.doy[vs.tau] <= vs.doy_end[:, :, 0])
        & (vs.doy_start[:, :, 0] != 0)
        & (vs.doy_end[:, :, 0] != 0)
    )
    mask2 = (
        (vs.doy[vs.tau] >= vs.doy_start[:, :, 1])
        & (vs.doy[vs.tau] <= vs.doy_end[:, :, 1])
        & (vs.doy_start[:, :, 1] != 0)
        & (vs.doy_end[:, :, 1] != 0)
    )
    mask3 = (
        (vs.doy[vs.tau] >= vs.doy_start[:, :, 2])
        & (vs.doy[vs.tau] > vs.doy_end[:, :, 0])
        & (vs.doy_start[:, :, 2] != 0)
        & (vs.doy_end[:, :, 2] != 0)
    )

    vs.lu_id = update(
        vs.lu_id,
        at[2:-2, 2:-2],
        npx.where(npx.any(vs.crop_type[2:-2, 2:-2, :] == 598, axis=-1), vs.lu_id[2:-2, 2:-2], 599),
    )
    vs.lu_id = update(
        vs.lu_id, at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], vs.crop_type[2:-2, 2:-2, 0], vs.lu_id[2:-2, 2:-2])
    )
    vs.lu_id = update(
        vs.lu_id, at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.crop_type[2:-2, 2:-2, 1], vs.lu_id[2:-2, 2:-2])
    )
    vs.lu_id = update(
        vs.lu_id, at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], vs.crop_type[2:-2, 2:-2, 2], vs.lu_id[2:-2, 2:-2])
    )

    return KernelOutput(lu_id=vs.lu_id)


@roger_kernel
def calc_irrig(state):
    """
    Calculates irrigation
    """
    vs = state.variables

    vs.irrig = update(
        vs.irrig, at[2:-2, 2:-2], npx.where(vs.irr_demand[2:-2, 2:-2] > 0, 30, 0)
    )

    return KernelOutput(irrig=vs.irrig)



@roger_kernel
def update_theta_irr(state):
    """
    Updates irrigation threshold while crop rotation
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y"))
    mask1 = (
        (vs.doy[vs.tau] >= arr0)
        & (vs.doy[vs.tau] <= vs.doy_end[:, :, 0])
        & (vs.doy_start[:, :, 0] != 0)
        & (vs.doy_end[:, :, 0] != 0)
    )
    mask2 = (
        (vs.doy[vs.tau] >= vs.doy_start[:, :, 1])
        & (vs.doy[vs.tau] <= vs.doy_end[:, :, 1])
        & (vs.doy_start[:, :, 1] != 0)
        & (vs.doy_end[:, :, 1] != 0)
    )
    mask3 = (
        (vs.doy[vs.tau] >= vs.doy_start[:, :, 2])
        & (vs.doy[vs.tau] > vs.doy_end[:, :, 0])
        & (vs.doy_start[:, :, 2] != 0)
        & (vs.doy_end[:, :, 2] != 0)
    )

    vs.theta_irr = update(
        vs.theta_irr, at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], vs.theta_irr_crop[2:-2, 2:-2, 0], vs.theta_irr[2:-2, 2:-2])
    )
    vs.theta_irr = update(
        vs.theta_irr, at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.theta_irr_crop[2:-2, 2:-2, 1], vs.theta_irr[2:-2, 2:-2])
    )
    vs.theta_irr = update(
        vs.theta_irr, at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], vs.theta_irr_crop[2:-2, 2:-2, 2], vs.theta_irr[2:-2, 2:-2])
    )

    return KernelOutput(theta_irr=vs.theta_irr)


@roger_kernel
def update_ground_cover(state):
    """
    Updates ground cover
    """
    vs = state.variables

    mask = vs.lu_id[:, :, npx.newaxis] == vs.crop_type

    ccc = allocate(state.dimensions, ("x", "y", "crops"))
    ccc = update(ccc, at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2, :], vs.ccc[2:-2, 2:-2, vs.tau, :], 0))
    vs.ground_cover = update(
        vs.ground_cover,
        at[2:-2, 2:-2, vs.tau],
        npx.where(
            npx.any(vs.crop_type[2:-2, 2:-2, :] == 598, axis=-1),
            vs.ground_cover[2:-2, 2:-2, vs.tau],
            npx.nanmax(ccc[2:-2, 2:-2, :], axis=-1),
        ),
    )

    return KernelOutput(ground_cover=vs.ground_cover)


@roger_kernel
def update_k_stress_transp(state):
    """
    Updates water stress coeffcient of transpiration
    """
    vs = state.variables

    mask = vs.lu_id[:, :, npx.newaxis] == vs.crop_type

    k_stress_transp_crop = allocate(state.dimensions, ("x", "y", "crops"))
    k_stress_transp_crop = update(
        k_stress_transp_crop,
        at[2:-2, 2:-2, :],
        npx.where(mask[2:-2, 2:-2, :], vs.k_stress_transp_crop[2:-2, 2:-2, :], 1),
    )
    vs.k_stress_transp = update(
        vs.k_stress_transp,
        at[2:-2, 2:-2],
        npx.where(
            npx.any(vs.crop_type[2:-2, 2:-2, :] == 598, axis=-1),
            vs.k_stress_transp[2:-2, 2:-2],
            npx.nanmin(k_stress_transp_crop[2:-2, 2:-2, :], axis=-1),
        ),
    )

    return KernelOutput(k_stress_transp=vs.k_stress_transp)


@roger_kernel
def update_basal_transp_coeff(state):
    """
    Updates transpiration coeffcient
    """
    vs = state.variables

    mask = vs.lu_id[:, :, npx.newaxis] == vs.crop_type

    basal_crop_coeff = allocate(state.dimensions, ("x", "y", "crops"))
    basal_crop_coeff = update(
        basal_crop_coeff, at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2, :], vs.basal_crop_coeff[2:-2, 2:-2, :], 0)
    )
    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff,
        at[2:-2, 2:-2],
        npx.where(
            npx.any(vs.crop_type[2:-2, 2:-2, :] == 598, axis=-1),
            vs.basal_transp_coeff[2:-2, 2:-2],
            npx.nanmax(basal_crop_coeff[2:-2, 2:-2, :], axis=-1),
        ),
    )

    return KernelOutput(basal_transp_coeff=vs.basal_transp_coeff)


@roger_kernel
def update_basal_evap_coeff(state):
    """
    Updates evaporation coeffcient
    """
    vs = state.variables

    mask = vs.lu_id[:, :, npx.newaxis] == vs.crop_type

    basal_evap_coeff_crop = allocate(state.dimensions, ("x", "y", "crops"))
    basal_evap_coeff_crop = update(
        basal_evap_coeff_crop,
        at[2:-2, 2:-2, :],
        npx.where(mask[2:-2, 2:-2, :], vs.basal_evap_coeff_crop[2:-2, 2:-2, :], 0),
    )
    vs.basal_evap_coeff = update(
        vs.basal_evap_coeff,
        at[2:-2, 2:-2],
        npx.where(
            npx.any(vs.crop_type[2:-2, 2:-2, :] == 598, axis=-1),
            vs.basal_evap_coeff[2:-2, 2:-2],
            npx.nanmax(basal_evap_coeff_crop[2:-2, 2:-2, :], axis=-1),
        ),
    )

    return KernelOutput(basal_evap_coeff=vs.basal_evap_coeff)


@roger_kernel
def update_S_int_ground_tot(state):
    """
    Updates total lower interception storage
    """
    vs = state.variables

    mask = vs.lu_id[:, :, npx.newaxis] == vs.crop_type

    S_int_tot_crop = allocate(state.dimensions, ("x", "y", "crops"))
    S_int_tot_crop = update(
        S_int_tot_crop, at[2:-2, 2:-2, :], npx.where(mask[2:-2, 2:-2, :], vs.S_int_tot_crop[2:-2, 2:-2, :], 0)
    )
    vs.S_int_ground_tot = update(
        vs.S_int_ground_tot,
        at[2:-2, 2:-2],
        npx.where(
            npx.any(vs.crop_type[2:-2, 2:-2, :] == 598, axis=-1),
            vs.S_int_ground_tot[2:-2, 2:-2],
            npx.nanmax(S_int_tot_crop[2:-2, 2:-2, :], axis=-1),
        ),
    )

    return KernelOutput(S_int_ground_tot=vs.S_int_ground_tot)


@roger_kernel
def update_z_root(state):
    """
    Updates root depth
    """
    vs = state.variables

    mask = vs.lu_id[:, :, npx.newaxis] == vs.crop_type

    z_root_crop = allocate(state.dimensions, ("x", "y", "crops"))
    z_root_crop = update(
        z_root_crop,
        at[2:-2, 2:-2, :],
        npx.where(mask[2:-2, 2:-2, :], vs.z_root_crop[2:-2, 2:-2, vs.tau, :], vs.z_evap[2:-2, 2:-2, npx.newaxis]),
    )
    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, vs.tau],
        npx.where(
            npx.any(vs.crop_type[2:-2, 2:-2, :] == 598, axis=-1),
            vs.z_root[2:-2, 2:-2, vs.tau],
            npx.nanmax(z_root_crop[2:-2, 2:-2, :], axis=-1),
        ),
    )

    # set thickness of upper soil water storage to 20 cm for bare soils
    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, vs.tau],
        npx.where(vs.z_root[2:-2, 2:-2, vs.tau] < 200, 200, vs.z_root[2:-2, 2:-2, vs.tau]),
    )
    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, vs.tau],
        npx.where(
            vs.z_root[2:-2, 2:-2, vs.tau] < vs.z_soil[2:-2, 2:-2],
            vs.z_root[2:-2, 2:-2, vs.tau],
            vs.z_soil[2:-2, 2:-2] * vs.zroot_to_zsoil_max[2:-2, 2:-2],
        ),
    )

    return KernelOutput(z_root=vs.z_root)


@roger_kernel
def redistribution_pwp(state):
    """
    Calculates redistribution between root zone and subsoil after
    root growth/root loss for immobile soil water storage (i.e. soil water below permanent wilting point)
    """
    vs = state.variables

    uplift_root_growth_pwp = allocate(state.dimensions, ("x", "y"))
    drain_root_loss_pwp = allocate(state.dimensions, ("x", "y"))

    mask_root_growth = vs.z_root[:, :, vs.tau] > vs.z_root[:, :, vs.taum1]
    mask_root_loss = vs.z_root[:, :, vs.tau] < vs.z_root[:, :, vs.taum1]

    uplift_root_growth_pwp = update(
        uplift_root_growth_pwp,
        at[2:-2, 2:-2],
        (vs.z_root[2:-2, 2:-2, vs.tau] - vs.z_root[2:-2, 2:-2, vs.taum1])
        * vs.theta_pwp[2:-2, 2:-2]
        * mask_root_growth[2:-2, 2:-2],
    )
    uplift_root_growth_pwp = update(
        uplift_root_growth_pwp,
        at[2:-2, 2:-2],
        npx.where(uplift_root_growth_pwp[2:-2, 2:-2] <= 0, 0, uplift_root_growth_pwp[2:-2, 2:-2]),
    )
    drain_root_loss_pwp = update(
        drain_root_loss_pwp,
        at[2:-2, 2:-2],
        npx.abs(vs.z_root[2:-2, 2:-2, vs.taum1] - vs.z_root[2:-2, 2:-2, vs.tau])
        * vs.theta_pwp[2:-2, 2:-2]
        * mask_root_loss[2:-2, 2:-2],
    )
    drain_root_loss_pwp = update(
        drain_root_loss_pwp,
        at[2:-2, 2:-2],
        npx.where(drain_root_loss_pwp[2:-2, 2:-2] <= 0, 0, drain_root_loss_pwp[2:-2, 2:-2]),
    )

    # add redistribution of immobile soil water storage (i.e. soil water below permanent wilting point)
    vs.re_rg_pwp = update(
        vs.re_rg_pwp, at[2:-2, 2:-2], npx.where(mask_root_growth[2:-2, 2:-2], uplift_root_growth_pwp[2:-2, 2:-2], 0)
    )
    vs.re_rl_pwp = update(
        vs.re_rl_pwp, at[2:-2, 2:-2], npx.where(mask_root_loss[2:-2, 2:-2], drain_root_loss_pwp[2:-2, 2:-2], 0)
    )

    return KernelOutput(
        re_rg_pwp=vs.re_rg_pwp,
        re_rl_pwp=vs.re_rl_pwp,
    )


@roger_kernel
def redistribution(state):
    """
    Calculates redistribution between root zone and subsoil after
    root growth/root loss.
    """
    vs = state.variables

    uplift_root_growth_lp = allocate(state.dimensions, ("x", "y"))
    uplift_root_growth_fp = allocate(state.dimensions, ("x", "y"))
    drain_root_loss_lp = allocate(state.dimensions, ("x", "y"))
    drain_root_loss_fp = allocate(state.dimensions, ("x", "y"))

    mask_root_growth = vs.z_root[:, :, vs.tau] > vs.z_root[:, :, vs.taum1]
    mask_root_loss = vs.z_root[:, :, vs.tau] < vs.z_root[:, :, vs.taum1]

    uplift_root_growth_lp = update(
        uplift_root_growth_lp,
        at[2:-2, 2:-2],
        npx.where(
            mask_root_growth[2:-2, 2:-2],
            (
                (vs.z_root[2:-2, 2:-2, vs.tau] - vs.z_root[2:-2, 2:-2, vs.taum1])
                / (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.taum1])
            )
            * vs.S_lp_ss[2:-2, 2:-2],
            0,
        ),
    )
    uplift_root_growth_fp = update(
        uplift_root_growth_fp,
        at[2:-2, 2:-2],
        npx.where(
            mask_root_growth[2:-2, 2:-2],
            (
                (vs.z_root[2:-2, 2:-2, vs.tau] - vs.z_root[2:-2, 2:-2, vs.taum1])
                / (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.taum1])
            )
            * vs.S_fp_ss[2:-2, 2:-2],
            0,
        ),
    )
    uplift_root_growth_lp = update(
        uplift_root_growth_lp,
        at[2:-2, 2:-2],
        npx.where(uplift_root_growth_lp[2:-2, 2:-2] <= 0, 0, uplift_root_growth_lp[2:-2, 2:-2]),
    )
    uplift_root_growth_fp = update(
        uplift_root_growth_fp,
        at[2:-2, 2:-2],
        npx.where(uplift_root_growth_fp[2:-2, 2:-2] <= 0, 0, uplift_root_growth_fp[2:-2, 2:-2]),
    )

    drain_root_loss_lp = update(
        drain_root_loss_lp,
        at[2:-2, 2:-2],
        npx.where(
            mask_root_loss[2:-2, 2:-2],
            ((vs.z_root[2:-2, 2:-2, vs.taum1] - vs.z_root[2:-2, 2:-2, vs.tau]) / vs.z_root[2:-2, 2:-2, vs.taum1])
            * vs.S_lp_rz[2:-2, 2:-2],
            0,
        ),
    )
    drain_root_loss_fp = update(
        drain_root_loss_fp,
        at[2:-2, 2:-2],
        npx.where(
            mask_root_loss[2:-2, 2:-2],
            ((vs.z_root[2:-2, 2:-2, vs.taum1] - vs.z_root[2:-2, 2:-2, vs.tau]) / vs.z_root[2:-2, 2:-2, vs.taum1])
            * vs.S_fp_rz[2:-2, 2:-2],
            0,
        ),
    )
    drain_root_loss_lp = update(
        drain_root_loss_lp,
        at[2:-2, 2:-2],
        npx.where(drain_root_loss_lp[2:-2, 2:-2] <= 0, 0, drain_root_loss_lp[2:-2, 2:-2]),
    )
    drain_root_loss_fp = update(
        drain_root_loss_fp,
        at[2:-2, 2:-2],
        npx.where(drain_root_loss_fp[2:-2, 2:-2] <= 0, 0, drain_root_loss_fp[2:-2, 2:-2]),
    )

    vs.re_rg = update(
        vs.re_rg,
        at[2:-2, 2:-2],
        npx.where(
            mask_root_growth[2:-2, 2:-2], uplift_root_growth_fp[2:-2, 2:-2] + uplift_root_growth_lp[2:-2, 2:-2], 0
        ),
    )
    vs.re_rl = update(
        vs.re_rl,
        at[2:-2, 2:-2],
        npx.where(mask_root_loss[2:-2, 2:-2], drain_root_loss_fp[2:-2, 2:-2] + drain_root_loss_lp[2:-2, 2:-2], 0),
    )

    # uplift from subsoil large pores
    vs.S_lp_ss = update_add(
        vs.S_lp_ss, at[2:-2, 2:-2], npx.where(mask_root_growth[2:-2, 2:-2], -uplift_root_growth_lp[2:-2, 2:-2], 0)
    )
    # uplift from subsoil fine pores
    vs.S_fp_ss = update_add(
        vs.S_fp_ss, at[2:-2, 2:-2], npx.where(mask_root_growth[2:-2, 2:-2], -uplift_root_growth_fp[2:-2, 2:-2], 0)
    )

    # update root zone storage
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[2:-2, 2:-2],
        vs.re_rg[2:-2, 2:-2],
    )

    # fine pore excess fills large pores
    mask1 = (vs.S_fp_rz > vs.S_ufc_rz) & (vs.re_rg > 0)
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[2:-2, 2:-2],
        npx.where(mask1[2:-2, 2:-2], (vs.S_fp_rz[2:-2, 2:-2] - vs.S_ufc_rz[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[2:-2, 2:-2],
        npx.where(mask1[2:-2, 2:-2], vs.S_ufc_rz[2:-2, 2:-2], vs.S_fp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # drainage from root zone large pores
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[2:-2, 2:-2],
        npx.where(mask_root_loss[2:-2, 2:-2], -drain_root_loss_lp[2:-2, 2:-2], 0),
    )
    # drainage from root zone fine pores
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[2:-2, 2:-2],
        npx.where(mask_root_loss[2:-2, 2:-2], -drain_root_loss_fp[2:-2, 2:-2], 0),
    )

    # update subsoil storage
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[2:-2, 2:-2],
        vs.re_rl[2:-2, 2:-2],
    )

    # fine pore excess storage fills large pores
    mask2 = (vs.S_fp_ss > vs.S_ufc_ss) & (vs.re_rl > 0)
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[2:-2, 2:-2],
        npx.where(mask2[2:-2, 2:-2], (vs.S_fp_ss[2:-2, 2:-2] - vs.S_ufc_ss[2:-2, 2:-2]), 0),
    )
    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[2:-2, 2:-2],
        npx.where(mask2[2:-2, 2:-2], vs.S_ufc_ss[2:-2, 2:-2], vs.S_fp_ss[2:-2, 2:-2]),
    )

    # add redistribution of immobile soil water storage (i.e. soil water below permanent wilting point)
    vs.re_rg = update_add(
        vs.re_rg, at[2:-2, 2:-2], npx.where(mask_root_growth[2:-2, 2:-2], vs.re_rg_pwp[2:-2, 2:-2], 0)
    )
    vs.re_rl = update_add(vs.re_rl, at[2:-2, 2:-2], npx.where(mask_root_loss[2:-2, 2:-2], vs.re_rl_pwp[2:-2, 2:-2], 0))

    return KernelOutput(
        re_rg=vs.re_rg,
        re_rl=vs.re_rl,
        S_fp_rz=vs.S_fp_rz,
        S_lp_rz=vs.S_lp_rz,
        S_fp_ss=vs.S_fp_ss,
        S_lp_ss=vs.S_lp_ss,
    )


@roger_kernel
def recalc_soil_params(state):
    """
    Recalculates parameters of root zone and subsoil after root growth/root loss.
    """
    vs = state.variables

    vs.S_ac_rz = update(
        vs.S_ac_rz,
        at[2:-2, 2:-2],
        (vs.theta_ac[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_ufc_rz = update(
        vs.S_ufc_rz,
        at[2:-2, 2:-2],
        (vs.theta_ufc[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_pwp_rz = update(
        vs.S_pwp_rz,
        at[2:-2, 2:-2],
        (vs.theta_pwp[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_sat_rz = update(
        vs.S_sat_rz,
        at[2:-2, 2:-2],
        (
            (vs.theta_ac[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2])
            * vs.z_root[2:-2, 2:-2, vs.tau]
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_fc_rz = update(
        vs.S_fc_rz,
        at[2:-2, 2:-2],
        ((vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * vs.z_root[2:-2, 2:-2, vs.tau])
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_ac_ss = update(
        vs.S_ac_ss,
        at[2:-2, 2:-2],
        (vs.theta_ac[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_ufc_ss = update(
        vs.S_ufc_ss,
        at[2:-2, 2:-2],
        (vs.theta_ufc[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_pwp_ss = update(
        vs.S_pwp_ss,
        at[2:-2, 2:-2],
        (vs.theta_pwp[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_sat_ss = update(
        vs.S_sat_ss,
        at[2:-2, 2:-2],
        (
            (vs.theta_ac[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2])
            * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_fc_ss = update(
        vs.S_fc_ss,
        at[2:-2, 2:-2],
        (
            (vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2])
            * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        S_ac_rz=vs.S_ac_rz,
        S_ufc_rz=vs.S_ufc_rz,
        S_pwp_rz=vs.S_pwp_rz,
        S_fc_rz=vs.S_fc_rz,
        S_sat_rz=vs.S_sat_rz,
        S_ac_ss=vs.S_ac_ss,
        S_ufc_ss=vs.S_ufc_ss,
        S_pwp_ss=vs.S_pwp_ss,
        S_fc_ss=vs.S_fc_ss,
        S_sat_ss=vs.S_sat_ss,
    )


@roger_kernel
def set_crop_params(state):
    """
    Recalculates parameters of root zone and subsoil after root growth/root loss.
    """
    vs = state.variables
    settings = state.settings

    for i in range(500, 600):
        mask = vs.crop_type == i
        row_no = _get_row_no(vs.lut_crops[:, 0], i)
        vs.doy_start = update(
            vs.doy_start,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 1], vs.doy_start[2:-2, 2:-2, :]),
        )
        vs.doy_mid = update(
            vs.doy_mid,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 2], vs.doy_mid[2:-2, 2:-2, :]),
        )
        vs.doy_dec = update(
            vs.doy_dec,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 3], vs.doy_dec[2:-2, 2:-2, :]),
        )
        vs.doy_end = update(
            vs.doy_end,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 4], vs.doy_end[2:-2, 2:-2, :]),
        )
        vs.ta_base = update(
            vs.ta_base,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 11], vs.ta_base[2:-2, 2:-2, :]),
        )
        vs.ta_ceil = update(
            vs.ta_ceil,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 12], vs.ta_ceil[2:-2, 2:-2, :]),
        )
        vs.ccc_min = update(
            vs.ccc_min,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 13], vs.ccc_min[2:-2, 2:-2, :]),
        )
        vs.ccc_max = update(
            vs.ccc_max,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 14], vs.ccc_max[2:-2, 2:-2, :]),
        )
        vs.crop_height_max = update(
            vs.crop_height_max,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 16] * 1000, vs.crop_height_max[2:-2, 2:-2, :]),
        )
        vs.ccc_growth_rate = update(
            vs.ccc_growth_rate,
            at[2:-2, 2:-2, :],
            npx.where(
                mask[2:-2, 2:-2, :],
                vs.lut_crops[row_no, 18] * vs.canopy_growth_scale[2:-2, 2:-2, npx.newaxis],
                vs.ccc_growth_rate[2:-2, 2:-2, :],
            ),
        )
        vs.basal_crop_coeff_mid = update(
            vs.basal_crop_coeff_mid,
            at[2:-2, 2:-2, :],
            npx.where(
                mask[2:-2, 2:-2, :],
                vs.lut_crops[row_no, 21] * vs.basal_crop_coeff_scale[2:-2, 2:-2, npx.newaxis],
                vs.basal_crop_coeff_mid[2:-2, 2:-2, :],
            ),
        )
        vs.z_root_crop_max = update(
            vs.z_root_crop_max,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 15] * 1000, vs.z_root_crop_max[2:-2, 2:-2, :]),
        )
        vs.root_growth_rate = update(
            vs.root_growth_rate,
            at[2:-2, 2:-2, :],
            npx.where(
                mask[2:-2, 2:-2, :],
                vs.lut_crops[row_no, 19] * vs.root_growth_scale[2:-2, 2:-2, npx.newaxis],
                vs.root_growth_rate[2:-2, 2:-2, :],
            ),
        )
        vs.water_stress_coeff_crop = update(
            vs.water_stress_coeff_crop,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2, :], vs.lut_crops[row_no, 20], vs.water_stress_coeff_crop[2:-2, 2:-2, :]),
        )

    vs.theta_water_stress_crop = update(
        vs.theta_water_stress_crop,
        at[2:-2, 2:-2, :],
        (vs.water_stress_coeff_crop[2:-2, 2:-2, :] * vs.theta_ufc[2:-2, 2:-2, npx.newaxis])
        + vs.theta_pwp[2:-2, 2:-2, npx.newaxis],
    )

    if settings.enable_crop_specific_irrigation_demand:
        vs.theta_irr_crop = update(
            vs.theta_irr_crop,
            at[2:-2, 2:-2, :],
            ((vs.water_stress_coeff_crop[2:-2, 2:-2, :] + 0.2) * vs.theta_ufc[2:-2, 2:-2, npx.newaxis])
            + vs.theta_pwp[2:-2, 2:-2, npx.newaxis],
        )

    return KernelOutput(
        doy_start=vs.doy_start,
        doy_mid=vs.doy_mid,
        doy_dec=vs.doy_dec,
        doy_end=vs.doy_end,
        ta_base=vs.ta_base,
        ta_ceil=vs.ta_ceil,
        ccc_min=vs.ccc_min,
        ccc_max=vs.ccc_max,
        crop_height_max=vs.crop_height_max,
        ccc_growth_rate=vs.ccc_growth_rate,
        basal_crop_coeff_mid=vs.basal_crop_coeff_mid,
        z_root_crop_max=vs.z_root_crop_max,
        root_growth_rate=vs.root_growth_rate,
        water_stress_coeff_crop=vs.water_stress_coeff_crop,
        theta_water_stress_crop=vs.theta_water_stress_crop,
        theta_irr_crop=vs.theta_irr_crop,
    )


@roger_routine
def calculate_crop_phenology(state):
    """
    Calculates crop phenology
    """
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        if (vs.year[vs.tau] > vs.year[vs.taum1]) & (vs.itt > 1):
            if settings.enable_crop_rotation:
                vs.ccc = update(
                    vs.ccc,
                    at[2:-2, 2:-2, :2, 0],
                    vs.ccc[2:-2, 2:-2, :2, 2],
                )
                vs.ccc = update(
                    vs.ccc,
                    at[2:-2, 2:-2, :2, 1:],
                    0,
                )
                vs.z_root_crop = update(
                    vs.z_root_crop,
                    at[2:-2, 2:-2, :2, 0],
                    vs.z_root_crop[2:-2, 2:-2, :2, 2],
                )
                vs.z_root_crop = update(
                    vs.z_root_crop,
                    at[2:-2, 2:-2, :2, 1:],
                    vs.z_evap[2:-2, 2:-2, npx.newaxis, npx.newaxis],
                )
                vs.t_grow_cc = update(
                    vs.t_grow_cc,
                    at[2:-2, 2:-2, :, 0],
                    vs.t_grow_cc[2:-2, 2:-2, :, 2],
                )
                vs.t_grow_cc = update(
                    vs.t_grow_cc,
                    at[2:-2, 2:-2, :, 1:],
                    0,
                )
                vs.t_grow_root = update(
                    vs.t_grow_root,
                    at[2:-2, 2:-2, :, 0],
                    vs.t_grow_root[2:-2, 2:-2, :, 2],
                )
                vs.t_grow_root = update(
                    vs.t_grow_root,
                    at[2:-2, 2:-2, :, 1:],
                    0,
                )
                vs.gdd_sum = update(
                    vs.gdd_sum,
                    at[2:-2, 2:-2, :, :],
                    0,
                )
                vs.ccc_mid = update(
                    vs.ccc_mid,
                    at[2:-2, 2:-2, 0],
                    vs.ccc_mid[2:-2, 2:-2, 2],
                )
                vs.t_half_mid = update(
                    vs.t_half_mid,
                    at[2:-2, 2:-2, 0],
                    vs.t_half_mid[2:-2, 2:-2, 2],
                )
                vs.t_half_mid = update(
                    vs.t_half_mid,
                    at[2:-2, 2:-2, 2],
                    0,
                )
                vs.t_half_mid = update(
                    vs.t_half_mid,
                    at[2:-2, 2:-2, 1],
                    0,
                )
                vs.t_decay = update(
                    vs.t_decay,
                    at[2:-2, 2:-2, 0],
                    vs.t_decay[2:-2, 2:-2, 2],
                )
                vs.t_decay = update(
                    vs.t_decay,
                    at[2:-2, 2:-2, 2],
                    0,
                )
                vs.t_decay = update(
                    vs.t_decay,
                    at[2:-2, 2:-2, 1],
                    0,
                )

            else:
                vs.gdd_sum = update(
                    vs.gdd_sum,
                    at[2:-2, 2:-2, :, 0],
                    0,
                )
                vs.t_half_mid = update(
                    vs.t_half_mid,
                    at[2:-2, 2:-2, 0],
                    0,
                )
                vs.t_decay = update(
                    vs.t_decay,
                    at[2:-2, 2:-2, 0],
                    0,
                )

            vs.update(set_crop_params(state))

        if vs.itt <= 0:
            vs.update(set_crop_params(state))

        if vs.time % (24 * 60 * 60) == 0:
            if settings.enable_crop_water_stress:
                vs.update(calc_k_stress_transp_crop(state))
                vs.update(calc_k_stress_root_growth(state))
            if settings.enable_crop_specific_irrigation_demand:
                vs.update(update_theta_irr(state))
            if settings.enable_irrigation:
                vs.update(calc_irrig(state))
            vs.update(calc_gdd(state))
            vs.update(calc_t_grow(state))
            vs.update(calc_t_half_mid(state))
            vs.update(calc_t_decay(state))
            vs.update(calc_canopy_cover(state))
            vs.update(calc_crop_height(state))
            vs.update(calc_crop_dev_coeff(state))
            vs.update(calc_basal_crop_coeff(state))
            vs.update(calc_basal_evap_coeff_crop(state))
            vs.update(calc_S_int_tot(state))
            vs.update(calc_root_growth(state))
            vs.update(update_lu_id(state))
            vs.update(update_ground_cover(state))
            vs.update(update_k_stress_transp(state))
            vs.update(update_basal_transp_coeff(state))
            vs.update(update_basal_evap_coeff(state))

        if vs.event_id[vs.tau] == 0:
            vs.update(update_lu_id(state))
            vs.update(update_ground_cover(state))
            vs.update(update_k_stress_transp(state))
            vs.update(update_basal_transp_coeff(state))
            vs.update(update_basal_evap_coeff(state))
            vs.update(update_S_int_ground_tot(state))
            vs.update(update_z_root(state))
            vs.update(recalc_soil_params(state))
            vs.update(redistribution_pwp(state))
            vs.update(redistribution(state))


@roger_kernel
def update_alpha_transp(state):
    """
    Updates crop specific partition coefficient of transpiration
    """
    vs = state.variables

    # land use dependent upper interception storage
    alpha_transp = allocate(state.dimensions, ("x", "y"))

    def loop_body_alpha_transp(i, alpha_transp):
        mask = (vs.lu_id == i)
        row_no = _get_row_no(vs.lut_crops[:, 0], i)
        alpha_transp = update(
            alpha_transp,
            at[2:-2, 2:-2],
            npx.where(mask[2:-2, 2:-2], vs.lut_crop_scale[2:-2, 2:-2, row_no], alpha_transp[2:-2, 2:-2]),
        )

        return alpha_transp
    
    alpha_transp = for_loop(500, 600, loop_body_alpha_transp, alpha_transp)

    vs.alpha_transp = update(
        vs.alpha_transp, at[2:-2, 2:-2], alpha_transp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2]
    )

    return KernelOutput(alpha_transp=vs.alpha_transp)


@roger_kernel
def calc_redistribution_root_growth_transport_kernel(state):
    """
    Calculates transport of redistribution after root growth
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :],
        transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.re_rg = update(
        vs.re_rg,
        at[2:-2, 2:-2],
        npx.where(
            vs.re_rg[2:-2, 2:-2] > npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1),
            npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1),
            vs.re_rg[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.tt_re_rg = update(
        vs.tt_re_rg,
        at[2:-2, 2:-2, :],
        transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.re_rg, vs.sas_params_re_rg)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_re_rg = update(
        vs.TT_re_rg,
        at[2:-2, 2:-2, 1:],
        npx.cumsum(vs.tt_re_rg[2:-2, 2:-2, :], axis=-1),
    )

    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :],
        transport.update_sa(state, vs.sa_ss, vs.tt_re_rg, vs.re_rg)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, :],
        vs.tt_re_rg[2:-2, 2:-2, :] * vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, tt_re_rg=vs.tt_re_rg, TT_re_rg=vs.TT_re_rg, re_rg=vs.re_rg)


@roger_kernel
def calc_redistribution_root_growth_transport_iso_kernel(state):
    """
    Calculates isotope transport of redistribution after root growth
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :],
        transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.re_rg = update(
        vs.re_rg,
        at[2:-2, 2:-2],
        npx.where(
            vs.re_rg[2:-2, 2:-2] > npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1),
            npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1),
            vs.re_rg[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.tt_re_rg = update(
        vs.tt_re_rg,
        at[2:-2, 2:-2, :],
        transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.re_rg, vs.sas_params_re_rg)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_re_rg = update(
        vs.TT_re_rg,
        at[2:-2, 2:-2, 1:],
        npx.cumsum(vs.tt_re_rg[2:-2, 2:-2, :], axis=-1),
    )

    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_re_rg = update(
        vs.mtt_re_rg,
        at[2:-2, 2:-2, :],
        transport.calc_mtt(state, vs.sa_ss, vs.tt_re_rg, vs.re_rg, vs.msa_ss, alpha)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_re_rg = update(
        vs.C_re_rg,
        at[2:-2, 2:-2],
        transport.calc_conc_iso_flux(state, vs.mtt_re_rg, vs.tt_re_rg, vs.re_rg)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_re_rg = update(
        vs.C_iso_re_rg,
        at[2:-2, 2:-2],
        transport.conc_to_delta(state, vs.C_re_rg)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update isotope StorAge
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            vs.tt_re_rg[2:-2, 2:-2, :] * vs.re_rg[2:-2, 2:-2, npx.newaxis] + vs.sa_rz[2:-2, 2:-2, vs.tau, :] > 0,
            vs.msa_rz[2:-2, 2:-2, vs.tau, :]
            * (
                vs.sa_rz[2:-2, 2:-2, vs.tau, :]
                / (vs.tt_re_rg[2:-2, 2:-2, :] * vs.re_rg[2:-2, 2:-2, npx.newaxis] + vs.sa_rz[2:-2, 2:-2, vs.tau, :])
            )
            + vs.mtt_re_rg[2:-2, 2:-2, :]
            * (
                (vs.tt_re_rg[2:-2, 2:-2, :] * vs.re_rg[2:-2, 2:-2, npx.newaxis])
                / ((vs.tt_re_rg[2:-2, 2:-2, :] * vs.re_rg[2:-2, 2:-2, npx.newaxis]) + vs.sa_rz[2:-2, 2:-2, vs.tau, :])
            ),
            vs.msa_rz[2:-2, 2:-2, vs.tau, :],
        )
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :],
        transport.update_sa(state, vs.sa_ss, vs.tt_re_rg, vs.re_rg)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, :],
        vs.tt_re_rg[2:-2, 2:-2, :] * vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update isotope StorAge
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(vs.sa_ss[2:-2, 2:-2, vs.tau, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, vs.tau, :])
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        sa_rz=vs.sa_rz,
        sa_ss=vs.sa_ss,
        tt_re_rg=vs.tt_re_rg,
        TT_re_rg=vs.TT_re_rg,
        msa_rz=vs.msa_rz,
        msa_ss=vs.msa_ss,
        mtt_re_rg=vs.mtt_re_rg,
        C_re_rg=vs.C_re_rg,
        C_iso_re_rg=vs.C_iso_re_rg,
        re_rg=vs.re_rg,
    )


@roger_kernel
def calc_redistribution_root_growth_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of redistribution after root growth
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :],
        transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.re_rg = update(
        vs.re_rg,
        at[2:-2, 2:-2],
        npx.where(
            vs.re_rg[2:-2, 2:-2] > npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1),
            npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1),
            vs.re_rg[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.tt_re_rg = update(
        vs.tt_re_rg,
        at[2:-2, 2:-2, :],
        transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.re_rg, vs.sas_params_re_rg)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_re_rg = update(
        vs.TT_re_rg,
        at[2:-2, 2:-2, 1:],
        npx.cumsum(vs.tt_re_rg[2:-2, 2:-2, :], axis=2),
    )

    # calculate anion travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_re_rg = update(
        vs.mtt_re_rg,
        at[2:-2, 2:-2, :],
        transport.calc_mtt(state, vs.sa_ss, vs.tt_re_rg, vs.re_rg, vs.msa_ss, alpha)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_re_rg = update(
        vs.C_re_rg,
        at[2:-2, 2:-2],
        npx.where(vs.re_rg[2:-2, 2:-2] > 0, npx.sum(vs.mtt_re_rg[2:-2, 2:-2, :], axis=-1) / vs.re_rg[2:-2, 2:-2], 0)
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.M_re_rg = update(
        vs.M_re_rg,
        at[2:-2, 2:-2],
        npx.sum(vs.mtt_re_rg[2:-2, 2:-2, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :],
        transport.update_sa(state, vs.sa_ss, vs.tt_re_rg, vs.re_rg)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.sa_rz = update_add(
        vs.sa_rz,
        at[2:-2, 2:-2, vs.tau, :],
        vs.tt_re_rg[2:-2, 2:-2, :] * vs.re_rg[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update solute StorAge of root zone
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :],
        vs.mtt_re_rg[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # update solute StorAge of subsoil
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :],
        -vs.mtt_re_rg[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        sa_rz=vs.sa_rz,
        sa_ss=vs.sa_ss,
        tt_re_rg=vs.tt_re_rg,
        TT_re_rg=vs.TT_re_rg,
        msa_rz=vs.msa_rz,
        msa_ss=vs.msa_ss,
        mtt_re_rg=vs.mtt_re_rg,
        M_re_rg=vs.M_re_rg,
        re_rg=vs.re_rg,
    )


@roger_kernel
def calc_redistribution_root_loss_transport_kernel(state):
    """
    Calculates transport of redistribution after root loss
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :],
        transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.re_rl = update(
        vs.re_rl,
        at[2:-2, 2:-2],
        npx.where(
            vs.re_rl[2:-2, 2:-2] > npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1),
            npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1),
            vs.re_rl[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.tt_re_rl = update(
        vs.tt_re_rl,
        at[2:-2, 2:-2, :],
        transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.re_rl, vs.sas_params_re_rl)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_re_rl = update(
        vs.TT_re_rl,
        at[2:-2, 2:-2, 1:],
        npx.cumsum(vs.tt_re_rl[2:-2, 2:-2, :], axis=2),
    )

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :],
        transport.update_sa(state, vs.sa_rz, vs.tt_re_rl, vs.re_rl)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, :],
        vs.tt_re_rl[2:-2, 2:-2, :] * vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, sa_ss=vs.sa_ss, tt_re_rl=vs.tt_re_rl, TT_re_rl=vs.TT_re_rl, re_rl=vs.re_rl)


@roger_kernel
def calc_redistribution_root_loss_transport_iso_kernel(state):
    """
    Calculates isotope transport of redistribution after root loss
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :],
        transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.re_rl = update(
        vs.re_rl,
        at[2:-2, 2:-2],
        npx.where(
            vs.re_rl[2:-2, 2:-2] > npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1),
            npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1),
            vs.re_rl[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.tt_re_rl = update(
        vs.tt_re_rl,
        at[2:-2, 2:-2, :],
        transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.re_rl, vs.sas_params_re_rl)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_re_rl = update(
        vs.TT_re_rl,
        at[2:-2, 2:-2, 1:],
        npx.cumsum(vs.tt_re_rl[2:-2, 2:-2, :], axis=-1),
    )

    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_re_rl = update(
        vs.mtt_re_rl,
        at[2:-2, 2:-2, :],
        transport.calc_mtt(state, vs.sa_rz, vs.tt_re_rl, vs.re_rl, vs.msa_rz, alpha)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_re_rl = update(
        vs.C_re_rl,
        at[2:-2, 2:-2],
        transport.calc_conc_iso_flux(state, vs.mtt_re_rl, vs.tt_re_rl, vs.re_rl)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_re_rl = update(
        vs.C_iso_re_rl,
        at[2:-2, 2:-2],
        transport.conc_to_delta(state, vs.C_re_rl)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update isotope StorAge
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(
            vs.tt_re_rl[2:-2, 2:-2, :] * vs.re_rl[2:-2, 2:-2, npx.newaxis] + vs.sa_ss[2:-2, 2:-2, vs.tau, :] > 0,
            vs.msa_ss[2:-2, 2:-2, vs.tau, :]
            * (
                vs.sa_ss[2:-2, 2:-2, vs.tau, :]
                / (vs.tt_re_rl[2:-2, 2:-2, :] * vs.re_rl[2:-2, 2:-2, npx.newaxis] + vs.sa_ss[2:-2, 2:-2, vs.tau, :])
            )
            + vs.mtt_re_rl[2:-2, 2:-2, :]
            * (
                (vs.tt_re_rl[2:-2, 2:-2, :] * vs.re_rl[2:-2, 2:-2, npx.newaxis])
                / ((vs.tt_re_rl[2:-2, 2:-2, :] * vs.re_rl[2:-2, 2:-2, npx.newaxis]) + vs.sa_ss[2:-2, 2:-2, vs.tau, :])
            ),
            vs.msa_ss[2:-2, 2:-2, vs.tau, :],
        )
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :],
        transport.update_sa(state, vs.sa_rz, vs.tt_re_rl, vs.re_rl)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, :],
        vs.tt_re_rl[2:-2, 2:-2, :] * vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update isotope StorAge
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :],
        npx.where(vs.sa_rz[2:-2, 2:-2, vs.tau, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, vs.tau, :])
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        sa_rz=vs.sa_rz,
        sa_ss=vs.sa_ss,
        tt_re_rl=vs.tt_re_rl,
        TT_re_rl=vs.TT_re_rl,
        msa_rz=vs.msa_rz,
        msa_ss=vs.msa_ss,
        mtt_re_rl=vs.mtt_re_rl,
        C_re_rl=vs.C_re_rl,
        C_iso_re_rl=vs.C_iso_re_rl,
        re_rl=vs.re_rl,
    )


@roger_kernel
def calc_redistribution_root_loss_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of redistribution after root loss
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :],
        transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.re_rl = update(
        vs.re_rl,
        at[2:-2, 2:-2],
        npx.where(
            vs.re_rl[2:-2, 2:-2] > npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1),
            npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1),
            vs.re_rl[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.tt_re_rl = update(
        vs.tt_re_rl,
        at[2:-2, 2:-2, :],
        transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.re_rl, vs.sas_params_re_rl)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_re_rl = update(
        vs.TT_re_rl,
        at[2:-2, 2:-2, 1:],
        npx.cumsum(vs.tt_re_rl[2:-2, 2:-2, :], axis=2),
    )

    # calculate anion travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_re_rl = update(
        vs.mtt_re_rl,
        at[2:-2, 2:-2, :],
        transport.calc_mtt(state, vs.sa_rz, vs.tt_re_rl, vs.re_rl, vs.msa_rz, alpha)[2:-2, 2:-2, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_re_rl = update(
        vs.C_re_rl,
        at[2:-2, 2:-2],
        npx.where(vs.re_rl[2:-2, 2:-2] > 0, npx.sum(vs.mtt_re_rl[2:-2, 2:-2, :], axis=-1) / vs.re_rl[2:-2, 2:-2], 0)
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.M_re_rl = update(
        vs.M_re_rl,
        at[2:-2, 2:-2],
        npx.sum(vs.mtt_re_rl[2:-2, 2:-2, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :],
        transport.update_sa(state, vs.sa_rz, vs.tt_re_rl, vs.re_rl)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, :],
        vs.tt_re_rl[2:-2, 2:-2, :] * vs.re_rl[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update solute StorAge of root zone
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :],
        -vs.mtt_re_rl[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # update solute StorAge of subsoil
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :],
        vs.mtt_re_rl[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        sa_rz=vs.sa_rz,
        sa_ss=vs.sa_ss,
        tt_re_rl=vs.tt_re_rl,
        TT_re_rl=vs.TT_re_rl,
        msa_rz=vs.msa_rz,
        msa_ss=vs.msa_ss,
        mtt_re_rl=vs.mtt_re_rl,
        C_re_rl=vs.C_re_rl,
        M_re_rl=vs.M_re_rl,
        re_rl=vs.re_rl,
    )


@roger_routine
def calculate_redistribution_transport(state):
    """
    Calculates trasnport of redistribution after root growth/root loss
    """
    vs = state.variables
    settings = state.settings

    if (
        settings.enable_offline_transport
        and settings.enable_crop_phenology
        and not (
            settings.enable_chloride
            | settings.enable_bromide
            | settings.enable_oxygen18
            | settings.enable_deuterium
            | settings.enable_nitrate
            | settings.enable_virtualtracer
        )
    ):
        vs.update(calc_redistribution_root_growth_transport_kernel(state))
        vs.update(calc_redistribution_root_loss_transport_kernel(state))

    if (
        settings.enable_offline_transport
        and settings.enable_crop_phenology
        and (settings.enable_oxygen18 | settings.enable_deuterium)
    ):
        vs.update(calc_redistribution_root_growth_transport_iso_kernel(state))
        vs.update(calc_redistribution_root_loss_transport_iso_kernel(state))

    if (
        settings.enable_offline_transport
        and settings.enable_crop_phenology
        and (
            settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate | settings.enable_virtualtracer
        )
    ):
        vs.update(calc_redistribution_root_growth_transport_anion_kernel(state))
        vs.update(calc_redistribution_root_loss_transport_anion_kernel(state))
