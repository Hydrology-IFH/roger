from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at
import roger.lookuptables as lut
from roger.core.utilities import _get_row_no


@roger_kernel
def calc_gdd(state):
    """Calculates growing degree days
    """
    vs = state.variables

    ta = allocate(state.dimensions, ("x", "y"))
    ta = update(
        ta,
        at[:, :], ((vs.ta_max[:, :, vs.tau] + vs.ta_min[:, :, vs.tau]) / 2),
    )
    mask = (ta[:, :, npx.newaxis] > vs.ta_base) & (ta[:, :, npx.newaxis] < vs.ta_ceil)
    vs.gdd = update(
        vs.gdd,
        at[:, :, :], npx.where(mask, ta[:, :, npx.newaxis] - vs.ta_base, 0),
    )
    vs.gdd_sum = update_add(
        vs.gdd_sum,
        at[:, :, vs.tau, :], vs.gdd,
    )

    return KernelOutput(gdd=vs.gdd, gdd_sum=vs.gdd_sum)


@roger_kernel
def calc_k_stress_transp_crop(state):
    """Calculates water stress coeffcient of transpiration
    """
    vs = state.variables

    mask = (vs.theta_rz[:, :, vs.tau, npx.newaxis] > vs.theta_water_stress_crop) & npx.isin(vs.crop_type, npx.arange(500, 600, 1, dtype=int))
    vs.k_stress_transp_crop = update(
        vs.k_stress_transp_crop,
        at[:, :, :], npx.where(mask, 1, (vs.theta_rz[:, :, vs.tau, npx.newaxis] - vs.theta_pwp[:, :, npx.newaxis]) / (vs.theta_water_stress_crop - vs.theta_pwp[:, :, npx.newaxis])),
    )

    return KernelOutput(k_stress_transp_crop=vs.k_stress_transp_crop)


@roger_kernel
def calc_basal_evap_coeff_crop(state):
    """Calculates crop evaporation coeffcient
    """
    vs = state.variables

    vs.basal_evap_coeff_crop = update(
        vs.basal_evap_coeff_crop,
        at[:, :, :], 1 - vs.ccc[:, :, vs.tau, :],
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
    mask_my_init = npx.isin(vs.crop_type, lut.WINTER_MULTI_YEAR_CROPS_INIT)
    mask_my_cont = npx.isin(vs.crop_type, lut.SUMMER_MULTI_YEAR_CROPS_CONT)

    mask1 = mask_summer & (vs.doy < vs.doy_start)
    mask2 = mask_summer & (vs.doy >= vs.doy_start) & (vs.doy <= vs.doy_end)
    mask3 = mask_summer & (vs.doy > vs.doy_end)
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], npx.where(mask1, 0, vs.t_grow_cc[:, :, vs.tau, :]),
    )
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_transp_crop * mask2,
    )
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], npx.where(mask3, 0, vs.t_grow_cc[:, :, vs.tau, :]),
    )
    mask4 = mask_my_cont & (vs.doy < vs.doy_start)
    mask5 = mask_my_cont & (vs.doy >= vs.doy_start) & (vs.doy <= vs.doy_end)
    mask6 = mask_my_cont & (vs.doy > vs.doy_end)
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], npx.where(mask4, 0, vs.t_grow_cc[:, :, vs.tau, :]),
    )
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_transp_crop * mask5,
    )
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], npx.where(mask6, 0, vs.t_grow_cc[:, :, vs.tau, :]),
    )
    mask7 = mask_winter & ((vs.doy >= vs.doy_start) | ((vs.doy <= vs.doy_end) & (vs.doy > arr0) & (vs.ccc[:, :, vs.tau, :] > 0)))
    mask8 = mask_winter & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_transp_crop * mask7,
    )
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], npx.where(mask8, 0, vs.t_grow_cc[:, :, vs.tau, :]),
    )
    mask9 = mask_winter_catch & ((vs.doy >= vs.doy_start) | ((vs.doy <= vs.doy_end) & (vs.doy > arr0) & (vs.ccc[:, :, vs.tau, :] > 0)))
    mask10 = mask_winter_catch & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_transp_crop * mask9,
    )
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], npx.where(mask10, 0, vs.t_grow_cc[:, :, vs.tau, :]),
    )
    mask11 = mask_my_init & ((vs.doy >= vs.doy_start) | ((vs.doy <= vs.doy_end) & (vs.doy > arr0)))
    mask12 = mask_my_init & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    vs.t_grow_cc = update_add(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_transp_crop * mask11,
    )
    vs.t_grow_cc = update(
        vs.t_grow_cc,
        at[:, :, vs.tau, :], npx.where(mask12, 0, vs.t_grow_cc[:, :, vs.tau, :]),
    )

    mask1 = mask_summer & (vs.doy < vs.doy_start)
    mask2 = mask_summer & (vs.doy >= vs.doy_start) & (vs.doy <= vs.doy_end)
    mask3 = mask_summer & (vs.doy > vs.doy_end)
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[:, :, vs.tau, :], npx.where(mask1, 0, vs.t_grow_root[:, :, vs.tau, :]),
    )
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_root_growth * mask2,
    )
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[:, :, vs.tau, :], npx.where(mask3, 0, vs.t_grow_root[:, :, vs.tau, :]),
    )
    mask4 = mask_my_cont & (vs.doy < vs.doy_start)
    mask5 = mask_my_cont & (vs.doy >= vs.doy_start) & (vs.doy <= vs.doy_end)
    mask6 = mask_my_cont & (vs.doy > vs.doy_end)
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[:, :, vs.tau, :], npx.where(mask4, 0, vs.t_grow_root[:, :, vs.tau, :]),
    )
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_root_growth * mask5,
    )
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[:, :, vs.tau, :], npx.where(mask6, 0, vs.t_grow_root[:, :, vs.tau, :]),
    )
    mask7 = mask_winter & ((vs.doy >= vs.doy_start) | ((vs.doy <= vs.doy_end) & (vs.doy > arr0) & (vs.ccc[:, :, vs.tau, :] > 0)))
    mask8 = mask_winter & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_root_growth * mask7,
    )
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[:, :, vs.tau, :], npx.where(mask8, 0, vs.t_grow_root[:, :, vs.tau, :]),
    )
    mask9 = mask_winter_catch & ((vs.doy >= vs.doy_start) | ((vs.doy <= vs.doy_end) & (vs.doy > arr0) & (vs.ccc[:, :, vs.tau, :] > 0)))
    mask10 = mask_winter_catch & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_root_growth * mask9,
    )
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[:, :, vs.tau, :], npx.where(mask10, 0, vs.t_grow_root[:, :, vs.tau, :]),
    )
    mask11 = mask_my_init & ((vs.doy >= vs.doy_start) | ((vs.doy <= vs.doy_end) & (vs.doy > arr0)))
    mask12 = mask_my_init & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    vs.t_grow_root = update_add(
        vs.t_grow_root,
        at[:, :, vs.tau, :], vs.gdd * vs.k_stress_root_growth * mask11,
    )
    vs.t_grow_root = update(
        vs.t_grow_root,
        at[:, :, vs.tau, :], npx.where(mask12, 0, vs.t_grow_root[:, :, vs.tau, :]),
    )

    return KernelOutput(t_grow_cc=vs.t_grow_cc, t_grow_root=vs.t_grow_root)


@roger_kernel
def calc_t_decay(state):
    """
    Calculates time since decay of crop canopy
    """
    vs = state.variables

    mask = (vs.doy == vs.doy_dec)
    vs.t_decay = update(
        vs.t_decay,
        at[:, :, :], npx.where(mask, vs.t_grow_cc[:, :, vs.tau, :], vs.t_decay),
    )

    return KernelOutput(t_decay=vs.t_decay)


@roger_kernel
def calc_t_half_mid(state):
    """
    Calculates half time of crop growing
    """
    vs = state.variables

    mask = (vs.ccc[:, :, vs.taum1, :] <= (vs.ccc_max / 2))
    vs.t_half_mid = update(
        vs.t_half_mid,
        at[:, :, :], npx.where(mask, vs.t_grow_cc[:, :, vs.tau, :], vs.t_half_mid),
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
    mask_571 = npx.isin(vs.crop_type, npx.array([571], dtype=int))
    mask_572 = npx.isin(vs.crop_type, npx.array([572], dtype=int))
    mask_bare = (vs.crop_type == 599)

    mask1 = mask_summer & (vs.doy > vs.doy_mid) & (vs.doy < vs.doy_dec)
    mask2 = mask_summer & (vs.doy < vs.doy_start)
    mask3 = mask_summer & (vs.doy >= vs.doy_start) & (vs.ccc[:, :, vs.tau, :] < vs.ccc_max) & (vs.doy <= vs.doy_dec)
    mask4 = mask_summer & (vs.doy > vs.doy_dec) & (vs.doy <= vs.doy_end)
    mask5 = mask_summer & (vs.doy > vs.doy_end)
    # mature crop
    vs.ccc_mid = update(
        vs.ccc_mid,
        at[:, :, :], npx.where(mask1, vs.ccc[:, :, vs.tau, :], vs.ccc_mid),
    )
    # before growing period
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask2, 0, vs.ccc[:, :, vs.tau, :]),
    )
    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask3, npx.where(vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]) <= vs.ccc_max/2,
                                       vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]),
                                       vs.ccc_max - (vs.ccc_max/2 - vs.ccc_min) * npx.exp(-vs.ccc_growth_rate * (vs.t_grow_cc[:, :, vs.tau, :] - vs.t_half_mid))), vs.ccc[:, :, vs.tau, :]),
    )

    # decay of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask4, vs.ccc_mid * (1 - 0.05 * (npx.exp((settings.ccc_decay_rate / vs.ccc_mid) * (vs.t_grow_cc[:, :, vs.tau, :] - vs.t_decay) - 1))), vs.ccc[:, :, vs.tau, :]),
    )

    # harvesting
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask5, 0, vs.ccc[:, :, vs.tau, :]),
    )

    mask6 = mask_winter & (vs.doy > vs.doy_mid) & (vs.doy < vs.doy_dec)
    mask7 = mask_winter & (vs.t_grow_cc[:, :, vs.tau, :] <= 0)
    mask8 = mask_winter & (vs.ccc[:, :, vs.tau, :] < vs.ccc_max) & ((vs.doy >= vs.doy_start) | (vs.doy <= vs.doy_dec) & (vs.doy > arr0) & (vs.t_grow_cc[:, :, vs.tau, :] > 0))
    mask9 = mask_winter & (vs.doy > vs.doy_dec) & (vs.doy <= vs.doy_end) & (vs.t_grow_cc[:, :, vs.tau, :] > 0)
    mask10 = mask_winter & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    # mature crop
    vs.ccc_mid = update(
        vs.ccc_mid,
        at[:, :, :], npx.where(mask6, vs.ccc[:, :, vs.tau, :], vs.ccc_mid),
    )
    # before growing period
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask7, 0, vs.ccc[:, :, vs.tau, :]),
    )
    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask8, npx.where(vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]) <= vs.ccc_max/2,
                                       vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]),
                                       vs.ccc_max - (vs.ccc_max/2 - vs.ccc_min) * npx.exp(-vs.ccc_growth_rate * (vs.t_grow_cc[:, :, vs.tau, :] - vs.t_half_mid))), vs.ccc[:, :, vs.tau, :]),
    )
    # decay of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask9, vs.ccc_mid * (1 - 0.05 * (npx.exp((settings.ccc_decay_rate / vs.ccc_mid) * (vs.t_grow_cc[:, :, vs.tau, :] - vs.t_decay) - 1))), vs.ccc[:, :, vs.tau, :]),
    )
    # harvesting
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask10, 0, vs.ccc[:, :, vs.tau, :]),
    )

    mask11 = mask_winter_catch & ((vs.doy > vs.doy_mid) | ((vs.doy < vs.doy_dec) & (vs.doy > arr0)))
    mask12 = mask_winter_catch & (vs.t_grow_cc[:, :, vs.tau, :] <= 0)
    mask13 = mask_winter_catch & (vs.ccc[:, :, vs.tau, :] < vs.ccc_max) & ((vs.doy >= vs.doy_start) | (vs.doy <= vs.doy_dec) & (vs.doy > arr0) & (vs.t_grow_cc[:, :, vs.tau, :] > 0))
    mask14 = mask_winter_catch & ((vs.doy > vs.doy_dec) | ((vs.doy <= vs.doy_end) & (vs.doy > arr0) & (vs.t_grow_cc[:, :, vs.tau, :] > 0)))
    mask15 = mask_winter_catch & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    # mature crop
    vs.ccc_mid = update(
        vs.ccc_mid,
        at[:, :, :], npx.where(mask11, vs.ccc[:, :, vs.tau, :], vs.ccc_mid),
    )
    # before growing period
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask12, 0, vs.ccc[:, :, vs.tau, :]),
    )

    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask13, npx.where(vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]) <= vs.ccc_max/2,
                                       vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]),
                                       vs.ccc_max - (vs.ccc_max/2 - vs.ccc_min) * npx.exp(-vs.ccc_growth_rate * (vs.t_grow_cc[:, :, vs.tau, :] - vs.t_half_mid))), vs.ccc[:, :, vs.tau, :]),
    )
    # decay of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask14, vs.ccc_mid * (1 - 0.05 * (npx.exp((settings.ccc_decay_rate / vs.ccc_mid) * (vs.t_grow_cc[:, :, vs.tau, :] - vs.t_decay) - 1))), vs.ccc[:, :, vs.tau, :]),
    )
    # harvesting
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask15, 0, vs.ccc[:, :, vs.tau, :]),
    )

    mask16 = mask_571 & (vs.doy < vs.doy_start)
    mask17 = mask_571 & (vs.doy >= vs.doy_start) & (vs.ccc[:, :, vs.tau, :] < vs.ccc_max) & (vs.doy <= vs.doy_end)
    # before growing period
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask16, 0, vs.ccc[:, :, vs.tau, :]),
    )
    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask17, npx.where(vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]) <= vs.ccc_max/2,
                                       vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]),
                                       vs.ccc_max - (vs.ccc_max/2 - vs.ccc_min) * npx.exp(-vs.ccc_growth_rate * (vs.t_grow_cc[:, :, vs.tau, :] - vs.t_half_mid))), vs.ccc[:, :, vs.tau, :]),
    )

    mask18 = mask_572 & (vs.t_grow_cc[:, :, vs.tau, :] <= 0)
    mask19 = mask_572 & (vs.ccc[:, :, vs.tau, :] < vs.ccc_mid) & ((vs.doy >= vs.doy_start) | (vs.doy <= vs.doy_dec) & (vs.doy > arr0) & (vs.t_grow_cc[:, :, vs.tau, :] > 0))
    # before growing period
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask18, 0, vs.ccc[:, :, vs.tau, :]),
    )
    # growth of canopy cover
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask19, npx.where(vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]) <= vs.ccc_max/2,
                                       vs.ccc_min * npx.exp(vs.ccc_growth_rate * vs.t_grow_cc[:, :, vs.tau, :]),
                                       vs.ccc_max - (vs.ccc_max/2 - vs.ccc_min) * npx.exp(-vs.ccc_growth_rate * (vs.t_grow_cc[:, :, vs.tau, :] - vs.t_half_mid))), vs.ccc[:, :, vs.tau, :]),
    )

    # bare
    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(mask_bare, 0, vs.ccc[:, :, vs.tau, :]),
    )

    vs.ccc = update(
        vs.ccc,
        at[:, :, vs.tau, :], npx.where(vs.ccc[:, :, vs.tau, :] <= 0, 0, vs.ccc[:, :, vs.tau, :]),
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
        at[:, :, :], npx.where(vs.ccc_max > 0, (vs.ccc[:, :, vs.tau, :] / vs.ccc_max) * vs.crop_height_max, 0),
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
        at[:, :, :, 0], 1,
    )
    crop_dev_coeff = update(
        crop_dev_coeff,
        at[:, :, :, 1], npx.where(vs.crop_height <= 0, 0, npx.where(vs.crop_height > 1, 2, 1.5)) * vs.ccc[:, :, vs.tau, :],
    )
    crop_dev_coeff = update(
        crop_dev_coeff,
        at[:, :, :, 2], vs.ccc[:, :, vs.tau, :]**(1 / (1 + vs.crop_height)),
    )
    vs.crop_dev_coeff = update(
        vs.crop_dev_coeff,
        at[:, :, :], npx.nanmin(crop_dev_coeff, axis=-1),
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
        at[:, :, :], settings.basal_crop_coeff_min + vs.crop_dev_coeff * (vs.basal_crop_coeff_mid - settings.basal_crop_coeff_min),
    )

    # bare
    mask_bare = (vs.crop_type == 599)
    vs.basal_crop_coeff = update(
        vs.basal_crop_coeff,
        at[:, :, :], npx.where(mask_bare, 0, vs.basal_crop_coeff),
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
        at[:, :, :], npx.log(1 / (1 - vs.ccc[:, :, vs.tau, :])) / npx.log(1 / 0.7),
    )

    vs.S_int_tot_crop = update(
        vs.S_int_tot_crop,
        at[:, :, :], 0.2 * vs.lai_crop,
    )

    return KernelOutput(lai_crop=vs.lai_crop, S_int_tot_crop=vs.S_int_tot_crop)


@roger_kernel
def calc_k_stress_root_growth(state):
    """
    Calculates water stress coeffcient of crop root growth
    """
    vs = state.variables

    k_stress_root_growth = allocate(state.dimensions, ("x", "y", "crops", 2))

    k_stress_root_growth = update(
        k_stress_root_growth,
        at[:, :, :, 0], 1,
    )
    k_stress_root_growth = update(
        k_stress_root_growth,
        at[:, :, :, 1], 4 * ((vs.theta_rz[:, :, vs.tau, npx.newaxis] - vs.theta_pwp[:, :, npx.newaxis]) / (vs.theta_fc[:, :, npx.newaxis] - vs.theta_pwp[:, :, npx.newaxis])),
    )
    vs.k_stress_root_growth = update(
        vs.k_stress_root_growth,
        at[:, :, :], npx.nanmin(k_stress_root_growth, axis=-1)
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
    mask_571 = npx.isin(vs.crop_type, npx.array([571], dtype=int))
    mask_572 = npx.isin(vs.crop_type, npx.array([572], dtype=int))
    mask_bare = (vs.crop_type == 599)

    mask1 = mask_summer & (vs.doy < vs.doy_start)
    mask2 = mask_summer & (vs.doy >= vs.doy_start) & (vs.doy <= vs.doy_mid)
    mask3 = mask_summer & (vs.doy > vs.doy_end)

    # before growing period
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask1, vs.z_evap[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )
    # root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask2, ((vs.z_root_crop_max / 1000) - ((vs.z_root_crop_max - vs.z_evap[:, :, npx.newaxis]) / 1000) * npx.exp(vs.root_growth_rate * vs.t_grow_root[:, :, vs.tau, :])) * 1000, vs.z_root_crop[:, :, vs.tau, :]),
    )
    # harvesting
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask3, vs.z_evap[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )

    mask4 = mask_winter & (vs.t_grow_root[:, :, vs.tau, :] <= 0)
    mask5 = mask_winter & ((vs.doy >= vs.doy_start) | (vs.doy <= vs.doy_mid) & (vs.doy > arr0) & (vs.t_grow_root[:, :, vs.tau, :] > 0))
    mask6 = mask_winter & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    # before growing period
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask4, vs.z_evap[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )
    # root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask5, ((vs.z_root_crop_max / 1000) - ((vs.z_root_crop_max - vs.z_evap[:, :, npx.newaxis]) / 1000) * npx.exp(vs.root_growth_rate * vs.t_grow_root[:, :, vs.tau, :])) * 1000, vs.z_root_crop[:, :, vs.tau, :]),
    )
    # harvesting
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask6, vs.z_evap[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )

    mask7 = mask_winter_catch & (vs.t_grow_root[:, :, vs.tau, :] <= 0)
    mask8 = mask_winter_catch & (vs.doy >= vs.doy_start) & (vs.doy <= vs.doy_mid)
    mask9 = mask_winter_catch & (vs.doy > vs.doy_end) & (vs.doy < vs.doy_start)
    # before growing period
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask7, vs.z_evap[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )
    # root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask8, ((vs.z_root_crop_max / 1000) - ((vs.z_root_crop_max - vs.z_evap[:, :, npx.newaxis]) / 1000) * npx.exp(vs.root_growth_rate * vs.t_grow_root[:, :, vs.tau, :])) * 1000, vs.z_root_crop[:, :, vs.tau, :]),
    )
    # harvesting
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask9, vs.z_evap[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )

    mask10 = mask_571 & (vs.doy < vs.doy_start)
    mask11 = mask_571 & (vs.doy >= vs.doy_start) & (vs.doy <= vs.doy_mid)
    # before growing period
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask10, vs.z_evap[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )
    # root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask11, ((vs.z_root_crop_max / 1000) - ((vs.z_root_crop_max - vs.z_evap[:, :, npx.newaxis]) / 1000) * npx.exp(vs.root_growth_rate * vs.t_grow_root[:, :, vs.tau, :])) * 1000, vs.z_root_crop[:, :, vs.tau, :]),
    )

    mask12 = mask_572 & (vs.t_grow_root[:, :, vs.tau, :] <= 0)
    mask13 = mask_572 & (vs.doy >= vs.doy_start) & (vs.doy <= vs.doy_mid)
    # before growing period
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask12, vs.z_evap[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )
    # root growth
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask13, ((vs.z_root_crop_max / 1000) - ((vs.z_root_crop_max - vs.z_evap[:, :, npx.newaxis]) / 1000) * npx.exp(vs.root_growth_rate * vs.t_grow_root[:, :, vs.tau, :])) * 1000, vs.z_root_crop[:, :, vs.tau, :]),
    )

    # root growth stops if 95 % of total soil depth is reached
    mask_stop_growth = (vs.z_root_crop[:, :, vs.tau, :] >= 0.9 * vs.z_soil[:, :, npx.newaxis])
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask_stop_growth, 0.9 * vs.z_soil[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )

    # bare
    vs.z_root_crop = update(
        vs.z_root_crop,
        at[:, :, vs.tau, :], npx.where(mask_bare, vs.z_evap[:, :, npx.newaxis], vs.z_root_crop[:, :, vs.tau, :]),
    )

    return KernelOutput(z_root_crop=vs.z_root_crop)


@roger_kernel
def update_lu_id(state):
    """
    Updates land use while crop rotation
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y"))
    mask1 = (vs.doy >= arr0) & (vs.doy <= vs.doy_end[:, :, 0]) & (vs.doy_start[:, :, 0] != 0) & (vs.doy_end[:, :, 0] != 0)
    mask2 = (vs.doy >= vs.doy_start[:, :, 1]) & (vs.doy <= vs.doy_end[:, :, 1]) & (vs.doy_start[:, :, 1] != 0) & (vs.doy_end[:, :, 1] != 0)
    mask3 = (vs.doy >= vs.doy_end[:, :, 2]) & (vs.doy_start[:, :, 2] != 0) & (vs.doy_end[:, :, 2] != 0)
    mask5 = mask1 | mask2 | mask3
    mask6 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & ~mask3
    mask7 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask8 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & ~mask1 & ~mask2
    mask9 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & ~mask3
    mask10 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask11 = mask6 | mask7 | mask8 | mask9 | mask10

    vs.lu_id = update(
        vs.lu_id,
        at[:, :], 599
    )
    vs.lu_id = update(
        vs.lu_id,
        at[:, :], npx.where(mask1 & ~mask2 & ~mask3, vs.crop_type[:, :, 0], vs.lu_id)
    )
    vs.lu_id = update(
        vs.lu_id,
        at[:, :], npx.where(mask2 & ~mask1 & ~mask3, vs.crop_type[:, :, 1], vs.lu_id)
    )
    vs.lu_id = update(
        vs.lu_id,
        at[:, :], npx.where(mask3 & ~mask1 & ~mask2, vs.crop_type[:, :, 2], vs.lu_id)
    )
    vs.lu_id = update(
        vs.lu_id,
        at[:, :], npx.where(~mask5, 599, vs.lu_id)
    )
    vs.lu_id = update(
        vs.lu_id,
        at[:, :], npx.where(mask11, 599, vs.lu_id)
    )

    return KernelOutput(lu_id=vs.lu_id)


@roger_kernel
def update_ground_cover(state):
    """
    Updates ground cover
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y"))
    mask1 = (vs.doy >= arr0) & (vs.doy <= vs.doy_end[:, :, 0]) & (vs.doy_start[:, :, 0] != 0) & (vs.doy_end[:, :, 0] != 0)
    mask2 = (vs.doy >= vs.doy_start[:, :, 1]) & (vs.doy <= vs.doy_end[:, :, 1]) & (vs.doy_start[:, :, 1] != 0) & (vs.doy_end[:, :, 1] != 0)
    mask3 = (vs.doy >= vs.doy_start[:, :, 2]) & (vs.doy_start[:, :, 2] != 0) & (vs.doy_end[:, :, 2] != 0)
    mask5 = mask1 | mask2 | mask3
    mask6 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & ~mask3
    mask7 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask8 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & ~mask1 & ~mask2
    mask9 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & ~mask3
    mask10 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask11 = mask6 | mask7 | mask8 | mask9 | mask10

    vs.ground_cover = update(
        vs.ground_cover,
        at[:, :, vs.tau], npx.where(mask1 & ~mask2 & ~mask3, vs.ccc[:, :, vs.tau, 0], vs.ground_cover[:, :, vs.tau])
    )
    vs.ground_cover = update(
        vs.ground_cover,
        at[:, :, vs.tau], npx.where(mask2 & ~mask1 & ~mask3, vs.ccc[:, :, vs.tau, 1], vs.ground_cover[:, :, vs.tau])
    )
    vs.ground_cover = update(
        vs.ground_cover,
        at[:, :, vs.tau], npx.where(mask3 & ~mask1 & ~mask2, vs.ccc[:, :, vs.tau, 2], vs.ground_cover[:, :, vs.tau])
    )
    vs.ground_cover = update(
        vs.ground_cover,
        at[:, :, vs.tau], npx.where(~mask5, 0, vs.ground_cover[:, :, vs.tau])
    )
    vs.ground_cover = update(
        vs.ground_cover,
        at[:, :, vs.tau], npx.where(mask11, 0, vs.ground_cover[:, :, vs.tau])
    )

    return KernelOutput(ground_cover=vs.ground_cover)


@roger_kernel
def update_k_stress_transp(state):
    """
    Updates water stress coeffcient of transpiration
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y"))
    mask1 = (vs.doy >= arr0) & (vs.doy <= vs.doy_end[:, :, 0]) & (vs.doy_start[:, :, 0] != 0) & (vs.doy_end[:, :, 0] != 0)
    mask2 = (vs.doy >= vs.doy_start[:, :, 1]) & (vs.doy <= vs.doy_end[:, :, 1]) & (vs.doy_start[:, :, 1] != 0) & (vs.doy_end[:, :, 1] != 0)
    mask3 = (vs.doy >= vs.doy_start[:, :, 2]) & (vs.doy_start[:, :, 2] != 0) & (vs.doy_end[:, :, 2] != 0)
    mask5 = mask1 | mask2 | mask3
    mask6 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & ~mask3
    mask7 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask8 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & ~mask1 & ~mask2
    mask9 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & ~mask3
    mask10 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask11 = mask6 | mask7 | mask8 | mask9 | mask10

    vs.k_stress_transp = update(
        vs.k_stress_transp,
        at[:, :], npx.where(mask1 & ~mask2 & ~mask3, vs.k_stress_transp_crop[:, :, 0], vs.k_stress_transp)
    )
    vs.k_stress_transp = update(
        vs.k_stress_transp,
        at[:, :], npx.where(mask2 & ~mask1 & ~mask3, vs.k_stress_transp_crop[:, :, 1], vs.k_stress_transp)
    )
    vs.k_stress_transp = update(
        vs.k_stress_transp,
        at[:, :], npx.where(mask3 & ~mask1 & ~mask2, vs.k_stress_transp_crop[:, :, 2], vs.k_stress_transp)
    )
    vs.k_stress_transp = update(
        vs.k_stress_transp,
        at[:, :], npx.where(~mask5, 0, vs.k_stress_transp)
    )
    vs.k_stress_transp = update(
        vs.k_stress_transp,
        at[:, :], npx.where(mask11, 0, vs.k_stress_transp)
    )

    return KernelOutput(k_stress_transp=vs.k_stress_transp)


@roger_kernel
def update_basal_transp_coeff(state):
    """
    Updates transpiration coeffcient
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y"))
    mask1 = (vs.doy >= arr0) & (vs.doy <= vs.doy_end[:, :, 0]) & (vs.doy_start[:, :, 0] != 0) & (vs.doy_end[:, :, 0] != 0)
    mask2 = (vs.doy >= vs.doy_start[:, :, 1]) & (vs.doy <= vs.doy_end[:, :, 1]) & (vs.doy_start[:, :, 1] != 0) & (vs.doy_end[:, :, 1] != 0)
    mask3 = (vs.doy >= vs.doy_start[:, :, 2]) & (vs.doy_start[:, :, 2] != 0) & (vs.doy_end[:, :, 2] != 0)
    mask5 = mask1 | mask2 | mask3
    mask6 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & ~mask3
    mask7 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask8 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & ~mask1 & ~mask2
    mask9 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & ~mask3
    mask10 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask11 = mask6 | mask7 | mask8 | mask9 | mask10

    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff,
        at[:, :], npx.where(mask1 & ~mask2 & ~mask3, vs.basal_crop_coeff[:, :, 0], vs.basal_transp_coeff)
    )
    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff,
        at[:, :], npx.where(mask2 & ~mask1 & ~mask3, vs.basal_crop_coeff[:, :, 1], vs.basal_transp_coeff)
    )
    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff,
        at[:, :], npx.where(mask3 & ~mask1 & ~mask2, vs.basal_crop_coeff[:, :, 2], vs.basal_transp_coeff)
    )
    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff,
        at[:, :], npx.where(~mask5, 0, vs.basal_transp_coeff)
    )
    vs.basal_transp_coeff = update(
        vs.basal_transp_coeff,
        at[:, :], npx.where(mask11, 0, vs.basal_transp_coeff)
    )

    return KernelOutput(basal_transp_coeff=vs.basal_transp_coeff)


@roger_kernel
def update_basal_evap_coeff(state):
    """
    Updates evaporation coeffcient
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y"))
    mask1 = (vs.doy >= arr0) & (vs.doy <= vs.doy_end[:, :, 0]) & (vs.doy_start[:, :, 0] != 0) & (vs.doy_end[:, :, 0] != 0)
    mask2 = (vs.doy >= vs.doy_start[:, :, 1]) & (vs.doy <= vs.doy_end[:, :, 1]) & (vs.doy_start[:, :, 1] != 0) & (vs.doy_end[:, :, 1] != 0)
    mask3 = (vs.doy >= vs.doy_start[:, :, 2]) & (vs.doy_start[:, :, 2] != 0) & (vs.doy_end[:, :, 2] != 0)
    mask5 = mask1 | mask2 | mask3
    mask6 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & ~mask3
    mask7 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask8 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & ~mask1 & ~mask2
    mask9 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & ~mask3
    mask10 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask11 = mask6 | mask7 | mask8 | mask9 | mask10

    vs.basal_evap_coeff = update(
        vs.basal_evap_coeff,
        at[:, :], npx.where(mask1 & ~mask2 & ~mask3, vs.basal_evap_coeff_crop[:, :, 0], vs.basal_evap_coeff)
    )
    vs.basal_evap_coeff = update(
        vs.basal_evap_coeff,
        at[:, :], npx.where(mask2 & ~mask1 & ~mask3, vs.basal_evap_coeff_crop[:, :, 1], vs.basal_evap_coeff)
    )
    vs.basal_evap_coeff = update(
        vs.basal_evap_coeff,
        at[:, :], npx.where(mask3 & ~mask1 & ~mask2, vs.basal_evap_coeff_crop[:, :, 2], vs.basal_evap_coeff)
    )
    vs.basal_evap_coeff = update(
        vs.basal_evap_coeff,
        at[:, :], npx.where(~mask5, 1, vs.basal_evap_coeff)
    )
    vs.basal_evap_coeff = update(
        vs.basal_evap_coeff,
        at[:, :], npx.where(mask11, 1, vs.basal_evap_coeff)
    )

    return KernelOutput(basal_evap_coeff=vs.basal_evap_coeff)


@roger_kernel
def update_S_int_ground_tot(state):
    """
    Updates total lower interception storage
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y"))
    mask1 = (vs.doy >= arr0) & (vs.doy <= vs.doy_end[:, :, 0]) & (vs.doy_start[:, :, 0] != 0) & (vs.doy_end[:, :, 0] != 0)
    mask2 = (vs.doy >= vs.doy_start[:, :, 1]) & (vs.doy <= vs.doy_end[:, :, 1]) & (vs.doy_start[:, :, 1] != 0) & (vs.doy_end[:, :, 1] != 0)
    mask3 = (vs.doy >= vs.doy_start[:, :, 2]) & (vs.doy_start[:, :, 2] != 0) & (vs.doy_end[:, :, 2] != 0)
    mask5 = mask1 | mask2 | mask3
    mask6 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & ~mask3
    mask7 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask8 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & ~mask1 & ~mask2
    mask9 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & ~mask3
    mask10 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask11 = mask6 | mask7 | mask8 | mask9 | mask10

    vs.S_int_ground_tot = update(
        vs.S_int_ground_tot,
        at[:, :], npx.where(mask1 & ~mask2 & ~mask3, vs.S_int_tot_crop[:, :, 0], vs.S_int_ground_tot)
    )
    vs.S_int_ground_tot = update(
        vs.S_int_ground_tot,
        at[:, :], npx.where(mask2 & ~mask1 & ~mask3, vs.S_int_tot_crop[:, :, 1], vs.S_int_ground_tot)
    )
    vs.S_int_ground_tot = update(
        vs.S_int_ground_tot,
        at[:, :], npx.where(mask3 & ~mask1 & ~mask2, vs.S_int_tot_crop[:, :, 2], vs.S_int_ground_tot)
    )
    vs.S_int_ground_tot = update(
        vs.S_int_ground_tot,
        at[:, :], npx.where(~mask5, 0, vs.S_int_ground_tot)
    )
    vs.S_int_ground_tot = update(
        vs.S_int_ground_tot,
        at[:, :], npx.where(mask11, 0, vs.S_int_ground_tot)
    )

    return KernelOutput(ground_cover=vs.ground_cover)


@roger_kernel
def update_z_root(state):
    """
    Updates root depth
    """
    vs = state.variables

    arr0 = allocate(state.dimensions, ("x", "y"))
    mask1 = (vs.doy >= arr0) & (vs.doy <= vs.doy_end[:, :, 0]) & (vs.doy_start[:, :, 0] != 0) & (vs.doy_end[:, :, 0] != 0)
    mask2 = (vs.doy >= vs.doy_start[:, :, 1]) & (vs.doy <= vs.doy_end[:, :, 1]) & (vs.doy_start[:, :, 1] != 0) & (vs.doy_end[:, :, 1] != 0)
    mask3 = (vs.doy >= vs.doy_start[:, :, 2]) & (vs.doy_start[:, :, 2] != 0) & (vs.doy_end[:, :, 2] != 0)
    mask5 = mask1 | mask2 | mask3
    mask6 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & ~mask3
    mask7 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & ~mask2 & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask8 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & ~mask1 & ~mask2
    mask9 = (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & ~mask3
    mask10 = (vs.doy_start[:, :, 0] == 0) & (vs.doy_end[:, :, 0] == 0) & (vs.doy_start[:, :, 1] == 0) & (vs.doy_end[:, :, 1] == 0) & (vs.doy_start[:, :, 2] == 0) & (vs.doy_end[:, :, 2] == 0)
    mask11 = mask6 | mask7 | mask8 | mask9 | mask10

    vs.z_root = update(
        vs.z_root,
        at[:, :, vs.tau], npx.where(mask1 & ~mask2 & ~mask3, vs.z_root_crop[:, :, vs.tau, 0], vs.z_root[:, :, vs.tau])
    )
    vs.z_root = update(
        vs.z_root,
        at[:, :, vs.tau], npx.where(mask2 & ~mask1 & ~mask3, vs.z_root_crop[:, :, vs.tau, 1], vs.z_root[:, :, vs.tau])
    )
    vs.z_root = update(
        vs.z_root,
        at[:, :, vs.tau], npx.where(mask3 & ~mask1 & ~mask2, vs.z_root_crop[:, :, vs.tau, 2], vs.z_root[:, :, vs.tau])
    )
    vs.z_root = update(
        vs.z_root,
        at[:, :, vs.tau], npx.where(~mask5, vs.z_evap, vs.z_root[:, :, vs.tau])
    )
    vs.z_root = update(
        vs.z_root,
        at[:, :, vs.tau], npx.where(mask11, vs.z_evap, vs.z_root[:, :, vs.tau])
    )

    return KernelOutput(z_root=vs.z_root)


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

    mask_root_growth = (vs.z_root[:, :, vs.tau] > vs.z_root[:, :, vs.taum1])
    mask_root_loss = (vs.z_root[:, :, vs.tau] < vs.z_root[:, :, vs.taum1])

    uplift_root_growth_lp = update(
        uplift_root_growth_lp,
        at[:, :], ((vs.z_root[:, :, vs.tau] - vs.z_root[:, :, vs.taum1]) / (vs.z_soil - vs.z_root[:, :, vs.taum1])) * vs.S_lp_ss * mask_root_growth
    )
    uplift_root_growth_fp = update(
        uplift_root_growth_fp,
        at[:, :], ((vs.z_root[:, :, vs.tau] - vs.z_root[:, :, vs.taum1]) / (vs.z_soil - vs.z_root[:, :, vs.taum1])) * vs.S_fp_ss * mask_root_growth
    )
    uplift_root_growth_lp = update(
        uplift_root_growth_lp,
        at[:, :], npx.where(uplift_root_growth_lp <= 0, 0, uplift_root_growth_lp)
    )
    uplift_root_growth_fp = update(
        uplift_root_growth_fp,
        at[:, :], npx.where(uplift_root_growth_fp <= 0, 0, uplift_root_growth_fp)
    )

    drain_root_loss_lp = update(
        drain_root_loss_lp,
        at[:, :], ((vs.z_root[:, :, vs.taum1] - vs.z_root[:, :, vs.tau]) / vs.z_root[:, :, vs.taum1]) * vs.S_lp_rz * mask_root_loss
    )
    drain_root_loss_fp = update(
        drain_root_loss_fp,
        at[:, :], ((vs.z_root[:, :, vs.taum1] - vs.z_root[:, :, vs.tau]) / vs.z_root[:, :, vs.taum1]) * vs.S_fp_rz * mask_root_loss
    )
    drain_root_loss_lp = update(
        drain_root_loss_lp,
        at[:, :], npx.where(drain_root_loss_lp <= 0, 0, drain_root_loss_lp)
    )
    drain_root_loss_fp = update(
        drain_root_loss_fp,
        at[:, :], npx.where(drain_root_loss_fp <= 0, 0, drain_root_loss_fp)
    )

    vs.re_rg = update(
        vs.re_rg,
        at[:, :], (uplift_root_growth_fp + uplift_root_growth_lp) * mask_root_growth
    )

    vs.re_rl = update(
        vs.re_rl,
        at[:, :], (drain_root_loss_fp + drain_root_loss_lp) * mask_root_loss
    )

    # uplift from subsoil large pores
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[:, :], - uplift_root_growth_lp * mask_root_growth
    )
    # uplift from subsoil fine pores
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[:, :], - uplift_root_growth_fp * mask_root_growth
    )

    # update root zone storage
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[:, :], vs.re_rg * vs.maskCatch,
    )

    # fine pore excess fills large pores
    mask1 = (vs.S_fp_rz > vs.S_ufc_rz) & (vs.re_rg > 0)
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[:, :], mask1 * (vs.S_fp_rz - vs.S_ufc_rz) * vs.maskCatch,
    )
    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[:, :], npx.where(mask1, vs.S_ufc_rz, vs.S_fp_rz) * vs.maskCatch,
    )

    # drainage from root zone large pores
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[:, :], - drain_root_loss_lp * mask_root_loss
    )
    # drainage from root zone fine pores
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[:, :], - drain_root_loss_fp * mask_root_loss
    )

    # update subsoil storage
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[:, :], vs.re_rl * vs.maskCatch,
    )

    # fine pore excess fills large pores
    mask2 = (vs.S_fp_ss > vs.S_ufc_ss) & (vs.re_rl > 0)
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[:, :], mask2 * (vs.S_fp_ss - vs.S_ufc_ss) * vs.maskCatch,
    )
    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[:, :], npx.where(mask2, vs.S_ufc_ss, vs.S_fp_ss) * vs.maskCatch,
    )

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
        at[:, :], (vs.theta_ac * vs.z_root[:, :, vs.tau]) * vs.maskCatch,
    )

    vs.S_ufc_rz = update(
        vs.S_ufc_rz,
        at[:, :], (vs.theta_ufc * vs.z_root[:, :, vs.tau]) * vs.maskCatch,
    )

    vs.S_pwp_rz = update(
        vs.S_pwp_rz,
        at[:, :], (vs.theta_pwp * vs.z_root[:, :, vs.tau]) * vs.maskCatch,
    )

    vs.S_sat_rz = update(
        vs.S_sat_rz,
        at[:, :], ((vs.theta_ac + vs.theta_ufc + vs.theta_pwp) * vs.z_root[:, :, vs.tau]) * vs.maskCatch,
    )

    vs.S_fc_rz = update(
        vs.S_fc_rz,
        at[:, :], ((vs.theta_ufc + vs.theta_pwp) * vs.z_root[:, :, vs.tau]) * vs.maskCatch,
    )

    vs.S_ac_ss = update(
        vs.S_ac_ss,
        at[:, :], (vs.theta_ac * (vs.z_soil - vs.z_root[:, :, vs.tau])) * vs.maskCatch,
    )

    vs.S_ufc_ss = update(
        vs.S_ufc_ss,
        at[:, :], (vs.theta_ufc * (vs.z_soil - vs.z_root[:, :, vs.tau])) * vs.maskCatch,
    )

    vs.S_pwp_ss = update(
        vs.S_pwp_ss,
        at[:, :], (vs.theta_pwp * (vs.z_soil - vs.z_root[:, :, vs.tau])) * vs.maskCatch,
    )

    vs.S_sat_ss = update(
        vs.S_sat_ss,
        at[:, :], ((vs.theta_ac + vs.theta_ufc + vs.theta_pwp) * (vs.z_soil - vs.z_root[:, :, vs.tau])) * vs.maskCatch,
    )

    vs.S_fc_ss = update(
        vs.S_fc_ss,
        at[:, :], ((vs.theta_ufc + vs.theta_pwp) * (vs.z_soil - vs.z_root[:, :, vs.tau])) * vs.maskCatch,
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

    for i in range(500, 600):
        mask = (vs.crop_type == i)
        row_no = _get_row_no(vs.lut_crops[:, 0], i)
        vs.doy_start = update(
            vs.doy_start,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 1], vs.doy_start),
        )
        vs.doy_mid = update(
            vs.doy_mid,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 2], vs.doy_mid),
        )
        vs.doy_dec = update(
            vs.doy_dec,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 3], vs.doy_dec),
        )
        vs.doy_end = update(
            vs.doy_end,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 4], vs.doy_end),
        )
        vs.ta_base = update(
            vs.ta_base,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 11], vs.ta_base),
        )
        vs.ta_ceil = update(
            vs.ta_ceil,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 12], vs.ta_ceil),
        )
        vs.ccc_min = update(
            vs.ccc_min,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 13], vs.ccc_min),
        )
        vs.ccc_max = update(
            vs.ccc_max,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 14], vs.ccc_max),
        )
        vs.crop_height_max = update(
            vs.crop_height_max,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 16] * 1000, vs.crop_height_max),
        )
        vs.ccc_growth_rate = update(
            vs.ccc_growth_rate,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 18], vs.ccc_growth_rate),
        )
        vs.basal_crop_coeff_mid = update(
            vs.basal_crop_coeff_mid,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 21], vs.basal_crop_coeff_mid),
        )
        vs.z_root_crop_max = update(
            vs.z_root_crop_max,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 15] * 1000, vs.z_root_crop_max),
        )
        vs.root_growth_rate = update(
            vs.root_growth_rate,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 19], vs.root_growth_rate),
        )
        vs.water_stress_coeff_crop = update(
            vs.water_stress_coeff_crop,
            at[:, :, :], npx.where(mask, vs.lut_crops[row_no, 20], vs.water_stress_coeff_crop),
        )

    vs.theta_water_stress_crop = update(
        vs.theta_water_stress_crop,
        at[:, :, :], (1 - vs.water_stress_coeff_crop) * (vs.theta_fc[:, :, npx.newaxis] - vs.theta_pwp[:, :, npx.newaxis]) + vs.theta_pwp[:, :, npx.newaxis],
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
        )


@roger_routine
def calculate_crop_phenology(state):
    """
    Calculates crop phenology
    """
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        if vs.time % (24 * 60 * 60) == 0:
            if settings.enable_crop_water_stress:
                vs.update(calc_k_stress_transp_crop(state))
                vs.update(calc_k_stress_root_growth(state))
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

        if (vs.event_id == 0) & (vs.time % (24 * 60 * 60) == 0):
            vs.update(update_lu_id(state))
            vs.update(update_ground_cover(state))
            vs.update(update_k_stress_transp(state))
            vs.update(update_basal_transp_coeff(state))
            vs.update(update_basal_evap_coeff(state))
            vs.update(update_S_int_ground_tot(state))
            vs.update(update_z_root(state))
            vs.update(recalc_soil_params(state))
            vs.update(redistribution(state))

        if vs.time % (24 * 60 * 60) == 0:
            print("\n", vs.doy, vs.ground_cover[3, 3, 1])
            print(vs.ccc_max[3, 3, 0]/2, vs.ccc[3, 3, 1, 0], vs.t_grow_cc[3, 3, 1, 0])
            print(vs.ccc_max[3, 3, 1]/2, vs.ccc[3, 3, 1, 1], vs.t_grow_cc[3, 3, 1, 1])
            print(vs.ccc_max[3, 3, 2]/2, vs.ccc[3, 3, 1, 2], vs.t_grow_cc[3, 3, 1, 2])
            # if vs.YEAR[vs.itt] == 2018:
            #     print("\n", vs.doy, vs.z_root[3, 3, 1], vs.z_root_crop[3, 3, 1, 0], vs.t_grow_root[3, 3, 1, 0], "\n")
            #     print("\n", vs.doy, vs.ground_cover[3, 3, 1], vs.ccc[3, 3, 1, 0], vs.t_grow_cc[3, 3, 1, 0], "\n")
            # elif vs.YEAR[vs.itt] == 2019:
            #     print("\n", vs.doy, vs.z_root[3, 3, 1], vs.z_root_crop[3, 3, 1, 0], vs.t_grow_root[3, 3, 1, 0], "\n")

        if (vs.YEAR[vs.itt] > vs.YEAR[vs.itt-1]) & (vs.itt > 1):
            if settings.enable_crop_rotation:
                vs.itt_cr = vs.itt_cr + 2
                vs.crop_type = update(vs.crop_type, at[:, :, 0], vs.crop_type[:, :, 2])
                vs.crop_type = update(vs.crop_type, at[:, :, 1], vs.CROP_TYPE[:, :, vs.itt_cr])
                vs.crop_type = update(vs.crop_type, at[:, :, 2], vs.CROP_TYPE[:, :, vs.itt_cr + 1])
                vs.ccc = update(
                    vs.ccc,
                    at[:, :, :2, 0], vs.ccc[:, :, :2, 2],
                )
                vs.z_root_crop = update(
                    vs.z_root_crop,
                    at[:, :, :2, 0], vs.z_root_crop[:, :, :2, 2],
                )
                vs.t_grow_cc = update(
                    vs.t_grow_cc,
                    at[:, :, :, 0], vs.t_grow_cc[:, :, :, 2],
                )
                vs.t_grow_cc = update(
                    vs.t_grow_cc,
                    at[:, :, :, 1:], 0,
                )
                vs.t_grow_root = update(
                    vs.t_grow_root,
                    at[:, :, :, 0], vs.t_grow_root[:, :, :, 2],
                )
                vs.t_grow_root = update(
                    vs.t_grow_root,
                    at[:, :, :, 1], 0,
                )
                vs.t_grow_root = update(
                    vs.t_grow_root,
                    at[:, :, :, 2], 0,
                )
                vs.gdd_sum = update(
                    vs.gdd_sum,
                    at[:, :, :, :], 0,
                )
                vs.ccc_mid = update(
                    vs.ccc_mid,
                    at[:, :, 0], vs.ccc_mid[:, :, 2],
                )
                vs.t_half_mid = update(
                    vs.t_half_mid,
                    at[:, :, 0], vs.t_half_mid[:, :, 2],
                )
                vs.t_half_mid = update(
                    vs.t_half_mid,
                    at[:, :, 2], 0,
                )
                vs.t_half_mid = update(
                    vs.t_half_mid,
                    at[:, :, 1], 0,
                )
                vs.t_decay = update(
                    vs.t_decay,
                    at[:, :, 0], vs.t_decay[:, :, 2],
                )
                vs.t_decay = update(
                    vs.t_decay,
                    at[:, :, 2], 0,
                )
                vs.t_decay = update(
                    vs.t_decay,
                    at[:, :, 1], 0,
                )

            else:
                vs.gdd_sum = update(
                    vs.gdd_sum,
                    at[:, :, :, 0], 0,
                )
                vs.t_half_mid = update(
                    vs.t_half_mid,
                    at[:, :, 0], 0,
                )
                vs.t_decay = update(
                    vs.t_decay,
                    at[:, :, 0], 0,
                )

            vs.update(set_crop_params(state))
