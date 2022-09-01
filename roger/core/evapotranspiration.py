from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at
from roger.core import transport


@roger_kernel
def calc_evap_int_top(state):
    """
    Calculates evaporation from upper interception storage
    """
    vs = state.variables

    mask1 = (vs.S_int_top[:, :, vs.tau] <= vs.S_int_top_tot) & (vs.pet_res <= vs.S_int_top[:, :, vs.tau]) & (vs.S_int_top_tot > 0) & (vs.S_int_top[:, :, vs.tau] > 0)
    mask2 = (vs.S_int_top[:, :, vs.tau] <= vs.S_int_top_tot) & (vs.pet_res > vs.S_int_top[:, :, vs.tau]) & (vs.S_int_top_tot > 0) & (vs.S_int_top[:, :, vs.tau] > 0)

    vs.evap_int_top = update(
        vs.evap_int_top,
        at[2:-2, 2:-2], 0,
    )
    # interception storage will not be fully evaporated
    vs.evap_int_top = update_add(
        vs.evap_int_top,
        at[2:-2, 2:-2], vs.pet_res[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.pet_res = update(
        vs.pet_res,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], 0, vs.pet_res[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # interception storage will be evaporated
    vs.evap_int_top = update_add(
        vs.evap_int_top,
        at[2:-2, 2:-2], vs.S_int_top[2:-2, 2:-2, vs.tau] * mask2[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # residual ET
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.S_int_top[2:-2, 2:-2, vs.tau] * mask2[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update top interception storage
    vs.S_int_top = update_add(
        vs.S_int_top,
        at[2:-2, 2:-2, vs.tau], -vs.evap_int_top[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_int_top=vs.S_int_top, pet_res=vs.pet_res, evap_int_top=vs.evap_int_top)


@roger_kernel
def calc_evap_int_ground(state):
    """
    Calculates evaporation from lower interception storage
    """
    vs = state.variables

    mask1 = (vs.S_int_ground[:, :, vs.tau] <= vs.S_int_ground_tot) & (vs.pet_res <= vs.S_int_ground[:, :, vs.tau]) & (vs.S_int_ground_tot > 0) & (vs.S_int_ground[:, :, vs.tau] > 0)
    mask2 = (vs.S_int_ground[:, :, vs.tau] <= vs.S_int_ground_tot) & (vs.pet_res > vs.S_int_ground[:, :, vs.tau]) & (vs.S_int_ground_tot > 0) & (vs.S_int_ground[:, :, vs.tau] > 0)

    vs.evap_int_ground = update(
        vs.evap_int_ground,
        at[2:-2, 2:-2], 0,
    )
    # interception storage will not be fully evaporated
    vs.evap_int_ground = update_add(
        vs.evap_int_ground,
        at[2:-2, 2:-2], vs.pet_res[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.pet_res = update(
        vs.pet_res,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], 0, vs.pet_res[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # interception storage will be evaporated
    vs.evap_int_ground = update_add(
        vs.evap_int_ground,
        at[2:-2, 2:-2], vs.S_int_ground[2:-2, 2:-2, vs.tau] * mask2[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # residual ET
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.S_int_ground[2:-2, 2:-2, vs.tau] * mask2[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update ground interception storage
    vs.S_int_ground = update_add(
        vs.S_int_ground,
        at[2:-2, 2:-2, vs.tau], -vs.evap_int_ground[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.evap_int = update_add(
        vs.evap_int,
        at[2:-2, 2:-2], vs.evap_int_ground[2:-2, 2:-2] + vs.evap_int_top[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_int_ground=vs.S_int_ground, pet_res=vs.pet_res, evap_int_ground=vs.evap_int_ground, evap_int=vs.evap_int)


@roger_kernel
def calc_evap_dep(state):
    """
    Calculates evaporation from surface depression storage
    """
    vs = state.variables

    # PET exceeds water stored in the depression
    mask1 = (vs.S_dep[:, :, vs.tau] <= vs.pet_res) & (vs.S_dep[:, :, vs.tau] > 0) & (vs.pet_res > 0) & (vs.prec[:, :, vs.tau] <= 0)
    # PET does not exceed water stored in the depression
    mask2 = (vs.S_dep[:, :, vs.tau] > vs.pet_res) & (vs.S_dep[:, :, vs.tau] > 0) & (vs.pet_res > 0) & (vs.prec[:, :, vs.tau] <= 0)

    vs.evap_dep = update(
        vs.evap_dep,
        at[2:-2, 2:-2], 0,
    )
    vs.evap_dep = update_add(
        vs.evap_dep,
        at[2:-2, 2:-2], vs.S_dep[2:-2, 2:-2, vs.tau] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.S_dep[2:-2, 2:-2, vs.tau] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.evap_dep = update_add(
        vs.evap_dep,
        at[2:-2, 2:-2], vs.pet_res[2:-2, 2:-2] * mask2[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.pet_res = update(
        vs.pet_res,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], 0, vs.pet_res[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # update surface depression storage after evaporation
    mask3 = (vs.S_dep[:, :, vs.tau] > 0) & (vs.evap_dep > 0)
    vs.S_dep = update_add(
        vs.S_dep,
        at[2:-2, 2:-2, vs.tau], -vs.evap_dep[2:-2, 2:-2] * mask3[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_dep=vs.S_dep, pet_res=vs.pet_res, evap_dep=vs.evap_dep)


@roger_kernel
def calc_evap_sur(state):
    """
    Calculates surface evaporation
    """
    vs = state.variables

    vs.evap_sur = update(
        vs.evap_sur,
        at[2:-2, 2:-2], vs.evap_int_top[2:-2, 2:-2] + vs.evap_int_ground[2:-2, 2:-2] + vs.evap_dep[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(evap_sur=vs.evap_sur)


@roger_kernel
def calc_evap_soil(state):
    """
    Calculates evaporation from upper soil layer (root zone)
    """
    vs = state.variables

    # calculates water stress of soil evaporation
    mask3 = (vs.de <= vs.rew)
    mask4 = (vs.de > vs.rew) & (vs.de <= vs.tew)
    mask5 = (vs.de > vs.tew)
    vs.k_stress_evap = update(
        vs.k_stress_evap,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], 1, vs.k_stress_evap[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.k_stress_evap = update(
        vs.k_stress_evap,
        at[2:-2, 2:-2], npx.where(mask4[2:-2, 2:-2], (vs.tew[2:-2, 2:-2] - vs.de[2:-2, 2:-2]) / (vs.tew[2:-2, 2:-2] - vs.rew[2:-2, 2:-2]), vs.k_stress_evap[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.k_stress_evap = update(
        vs.k_stress_evap,
        at[2:-2, 2:-2], npx.where(mask5[2:-2, 2:-2], 0, vs.k_stress_evap[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    # calculates coeffcient of soil evaporation
    vs.evap_coeff = update(
        vs.evap_coeff,
        at[2:-2, 2:-2], vs.basal_evap_coeff[2:-2, 2:-2] * vs.k_stress_evap[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # potential soil evaporation
    pevap_soil = allocate(state.dimensions, ("x", "y"))
    # soil evaporation from fine pore storage
    evap_fp = allocate(state.dimensions, ("x", "y"))
    pevap_soil = update(
        pevap_soil,
        at[2:-2, 2:-2], vs.pet_res[2:-2, 2:-2] * vs.evap_coeff[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # conditions for evaporation from fine storage
    mask1 = (vs.S_fp_rz > 0) & (pevap_soil <= vs.S_fp_rz) & (pevap_soil > 0) & (vs.swe[:, :, vs.tau] <= 0) & (vs.prec[:, :, vs.tau] <= 0)
    mask2 = (vs.S_fp_rz > 0) & (pevap_soil > vs.S_fp_rz) & (pevap_soil > 0) & (vs.swe[:, :, vs.tau] <= 0) & (vs.prec[:, :, vs.tau] <= 0)

    # water evaporates from fine pores in root zone
    # some water remains
    evap_fp = update_add(
        evap_fp,
        at[2:-2, 2:-2], pevap_soil[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # residual ET
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -pevap_soil[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.pet_res = update(
        vs.pet_res,
        at[2:-2, 2:-2], npx.where(vs.pet_res[2:-2, 2:-2] < 0, 0, vs.pet_res[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # water evaporates from fine pores in root zone
    # no water remains
    evap_fp = update_add(
        evap_fp,
        at[2:-2, 2:-2], vs.S_fp_rz[2:-2, 2:-2] * mask2[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # residual ET
    vs.pet_res = update_add(
        vs.pet_res,
        at[2:-2, 2:-2], -vs.S_fp_rz[2:-2, 2:-2] * mask2[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.pet_res = update(
        vs.pet_res,
        at[2:-2, 2:-2], npx.where(vs.pet_res[2:-2, 2:-2] < 0, 0, vs.pet_res[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.evap_soil = update(
        vs.evap_soil,
        at[2:-2, 2:-2], evap_fp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update root zone storage after evaporation
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[2:-2, 2:-2], -vs.evap_soil[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_fp_rz=vs.S_fp_rz, pet_res=vs.pet_res, evap_soil=vs.evap_soil, evap_coeff=vs.evap_coeff, k_stress_evap=vs.k_stress_evap)


@roger_kernel
def calc_transp(state):
    """
    Calculates transpiration from upper soil layer (root zone)
    """
    vs = state.variables
    settings = state.settings

    theta_water_stress = allocate(state.dimensions, ("x", "y"))
    theta_water_stress = update(
        theta_water_stress,
        at[2:-2, 2:-2], settings.transp_water_stress * vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    mask_crops = npx.isin(vs.lu_id, npx.arange(500, 600, 1, dtype=int))
    vs.k_stress_transp = update(
        vs.k_stress_transp,
        at[2:-2, 2:-2], npx.where(mask_crops[2:-2, 2:-2], vs.k_stress_transp[2:-2, 2:-2], ((vs.theta_rz[2:-2, 2:-2, vs.tau] - vs.theta_pwp[2:-2, 2:-2]) / (theta_water_stress[2:-2, 2:-2] - vs.theta_pwp[2:-2, 2:-2]))) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.k_stress_transp = update(
        vs.k_stress_transp,
        at[2:-2, 2:-2], npx.where(vs.k_stress_transp[2:-2, 2:-2] > 1, 1, vs.k_stress_transp[2:-2, 2:-2])
    )

    # calculates coeffcient of transpiration
    vs.transp_coeff = update(
        vs.transp_coeff,
        at[2:-2, 2:-2], vs.basal_transp_coeff[2:-2, 2:-2] * vs.k_stress_transp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    transp_fp = allocate(state.dimensions, ("x", "y"))
    transp_lp = allocate(state.dimensions, ("x", "y"))

    # potential transpiration
    vs.ptransp = update(
        vs.ptransp,
        at[2:-2, 2:-2], vs.pet_res[2:-2, 2:-2] * vs.transp_coeff[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # residual transpiration
    vs.ptransp_res = update(
        vs.ptransp_res,
        at[2:-2, 2:-2], vs.ptransp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # water transpires from large pores in root zone
    # some water remains
    mask1 = (vs.S_lp_rz > 0) & (vs.ptransp_res <= vs.S_lp_rz) & (vs.ptransp > 0) & (vs.prec[:, :, vs.tau] <= 0)
    transp_lp = update_add(
        transp_lp,
        at[2:-2, 2:-2], vs.ptransp_res[2:-2, 2:-2] * mask1[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.ptransp_res = update(
        vs.ptransp_res,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], 0, vs.ptransp_res[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # water transpires from large pores in root zone
    # no water remains
    mask2 = (vs.S_lp_rz > 0) & (vs.ptransp_res > vs.S_lp_rz) & (vs.ptransp > 0) & (vs.prec[:, :, vs.tau] <= 0)
    transp_lp = update_add(
        transp_lp,
        at[2:-2, 2:-2], vs.S_lp_rz[2:-2, 2:-2] * mask2[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.ptransp_res = update_add(
        vs.ptransp_res,
        at[2:-2, 2:-2], -vs.S_lp_rz[2:-2, 2:-2] * mask2[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # water transpires from fine pores in root zone
    # some water remains
    mask3 = (vs.S_fp_rz > 0) & (vs.ptransp_res <= vs.S_fp_rz) & (vs.S_lp_rz <= 0) & (vs.ptransp > 0) & (vs.prec[:, :, vs.tau] <= 0)
    transp_fp = update_add(
        transp_fp,
        at[2:-2, 2:-2], vs.ptransp_res[2:-2, 2:-2] * mask3[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.ptransp_res = update(
        vs.ptransp_res,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], 0, vs.ptransp_res[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # water transpires from fine pores in root zone
    # no water remains
    mask4 = (vs.S_fp_rz > 0) & (vs.ptransp_res > vs.S_fp_rz) & (vs.S_lp_rz <= 0) & (vs.ptransp > 0) & (vs.prec[:, :, vs.tau] <= 0)
    transp_fp = update_add(
        transp_fp,
        at[2:-2, 2:-2], vs.S_fp_rz[2:-2, 2:-2] * mask4[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.ptransp_res = update_add(
        vs.ptransp_res,
        at[2:-2, 2:-2], -vs.S_fp_rz[2:-2, 2:-2] * mask4[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.ptransp_res = update(
        vs.ptransp_res,
        at[2:-2, 2:-2], npx.where(vs.ptransp_res[2:-2, 2:-2] < 0, 0, vs.ptransp_res[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # transpiration from root zone (root water uptake)
    vs.transp = update(
        vs.transp,
        at[2:-2, 2:-2], (transp_fp[2:-2, 2:-2] + transp_lp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # update root zone storage after transpiration
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[2:-2, 2:-2], -transp_lp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[2:-2, 2:-2], -transp_fp[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_lp_rz=vs.S_lp_rz, S_fp_rz=vs.S_fp_rz, pet_res=vs.pet_res, transp=vs.transp, ptransp_res=vs.ptransp_res, transp_coeff=vs.transp_coeff, k_stress_transp=vs.k_stress_transp)


@roger_kernel
def calc_acc_evap_soil_deficit(state):
    """
    Calculates accumulation of soil evapotranspiration deficit
    """
    vs = state.variables

    vs.de = update_add(
        vs.de,
        at[2:-2, 2:-2], vs.evap_soil[2:-2, 2:-2] + vs.transp[2:-2, 2:-2] * (vs.z_evap[2:-2, 2:-2] / vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(de=vs.de)


@roger_kernel
def calc_aet_soil(state):
    """
    Calculates evapotranspiration from upper soil layer (root zone)
    """
    vs = state.variables

    vs.aet_soil = update(
        vs.aet_soil,
        at[2:-2, 2:-2], (vs.evap_soil[2:-2, 2:-2] + vs.transp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(aet_soil=vs.aet_soil)


@roger_kernel
def calc_aet(state):
    """
    Calculates actual evapotranspiration
    """
    vs = state.variables

    vs.aet = update(
        vs.aet,
        at[2:-2, 2:-2], (vs.evap_int_top[2:-2, 2:-2] + vs.evap_int_ground[2:-2, 2:-2] + vs.evap_dep[2:-2, 2:-2] + vs.evap_soil[2:-2, 2:-2] + vs.transp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(aet=vs.aet)


@roger_routine
def calculate_evapotranspiration(state):
    """
    Calculates evapotranspiration
    """
    vs = state.variables
    vs.update(calc_evap_int_top(state))
    vs.update(calc_evap_int_ground(state))
    vs.update(calc_evap_dep(state))
    vs.update(calc_evap_sur(state))
    vs.update(calc_evap_soil(state))
    vs.update(calc_transp(state))
    vs.update(calc_acc_evap_soil_deficit(state))
    vs.update(calc_aet_soil(state))
    vs.update(calc_aet(state))


@roger_kernel
def calculate_evaporation_transport_kernel(state):
    """
    Calculates transport of soil evaporation
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.evap_soil, vs.sas_params_evap_soil)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_evap_soil[2:-2, 2:-2, :], axis=-1),
    )

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_evap_soil=vs.tt_evap_soil, TT_evap_soil=vs.TT_evap_soil)


@roger_kernel
def calculate_evaporation_transport_iso_kernel(state):
    """
    Calculates isotope transport of soil evaporation
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.evap_soil, vs.sas_params_evap_soil)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_evap_soil[2:-2, 2:-2, :], axis=-1),
    )

    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_evap_soil = update(
        vs.mtt_evap_soil,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil, vs.msa_rz, alpha)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_evap_soil = update(
        vs.C_evap_soil,
        at[2:-2, 2:-2], transport.calc_conc_iso_flux(state, vs.mtt_evap_soil, vs.tt_evap_soil, vs.evap_soil)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where((vs.sa_rz[2:-2, 2:-2, vs.tau, :] > 0), vs.msa_rz[2:-2, 2:-2, vs.tau, :], npx.nan) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, vs.tau], transport.calc_conc_iso_storage(state, vs.sa_rz, vs.msa_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_evap_soil=vs.tt_evap_soil, TT_evap_soil=vs.TT_evap_soil, msa_rz=vs.msa_rz, C_rz=vs.C_rz, mtt_evap_soil=vs.mtt_evap_soil, C_evap_soil=vs.C_evap_soil)


@roger_kernel
def calculate_transpiration_transport_kernel(state):
    """
    Calculates transport of transpiration
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.transp, vs.sas_params_transp)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_transp[2:-2, 2:-2, :], axis=2),
    )

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_transp, vs.transp)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_transp=vs.tt_transp, TT_transp=vs.TT_transp)


@roger_kernel
def calculate_transpiration_transport_iso_kernel(state):
    """
    Calculates isotope transport of transpiration
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.transp, vs.sas_params_transp)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_transp[2:-2, 2:-2, :], axis=-1),
    )

    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_transp, vs.transp, vs.msa_rz, alpha)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_transp = update(
        vs.C_transp,
        at[2:-2, 2:-2], transport.calc_conc_iso_flux(state, vs.mtt_transp, vs.tt_transp, vs.transp)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_transp, vs.transp)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    # update isotope StorAge
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where((vs.sa_rz[2:-2, 2:-2, vs.tau, :] > 0), vs.msa_rz[2:-2, 2:-2, vs.tau, :], npx.nan) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_transp=vs.tt_transp, TT_transp=vs.TT_transp, msa_rz=vs.msa_rz, mtt_transp=vs.mtt_transp, C_transp=vs.C_transp)


@roger_kernel
def calculate_transpiration_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of transpiration
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_transp = update(
        vs.tt_transp,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.transp, vs.sas_params_transp)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_transp = update(
        vs.TT_transp,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_transp[2:-2, 2:-2, :], axis=2),
    )

    # calculate isotope travel time distribution
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_transp, vs.transp, vs.msa_rz, vs.alpha_transp)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_transp = update(
        vs.C_transp,
        at[2:-2, 2:-2], npx.where(vs.transp > 0, npx.sum(vs.mtt_transp, axis=2) / vs.transp, 0)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_transp, vs.transp)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge of root zone
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], vs.msa_rz[2:-2, 2:-2, vs.tau, :] - vs.mtt_transp[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_transp=vs.tt_transp, TT_transp=vs.TT_transp, msa_rz=vs.msa_rz, mtt_transp=vs.mtt_transp, C_transp=vs.C_transp)


@roger_routine
def calculate_evapotranspiration_transport(state):
    """
    Calculates evapotranspiration transport
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_evaporation_transport_kernel(state))
        vs.update(calculate_transpiration_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_evaporation_transport_iso_kernel(state))
        vs.update(calculate_transpiration_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_evaporation_transport_kernel(state))
        vs.update(calculate_transpiration_transport_anion_kernel(state))
