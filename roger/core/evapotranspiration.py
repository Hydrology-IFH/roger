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

    mask1 = (vs.S_int_top[:, :, vs.tau] <= vs.S_int_top_tot) & (vs.pet_res <= vs.S_int_top[:, :, vs.tau]) & (vs.S_int_top_tot > 0) & (vs.S_int_top[:, :, vs.tau] > 0) & (vs.prec <= 0)
    mask2 = (vs.S_int_top[:, :, vs.tau] <= vs.S_int_top_tot) & (vs.pet_res > vs.S_int_top[:, :, vs.tau]) & (vs.S_int_top_tot > 0) & (vs.S_int_top[:, :, vs.tau] > 0) & (vs.prec <= 0)

    evap_int_top = allocate(state.dimensions, ("x", "y"))
    vs.evap_int_top = update(
        vs.evap_int_top,
        at[:, :], evap_int_top,
    )
    # interception storage will not be fully evaporated
    vs.evap_int_top = update_add(
        vs.evap_int_top,
        at[:, :], vs.pet_res * mask1 * vs.maskCatch,
    )
    vs.pet_res = update(
        vs.pet_res,
        at[:, :], npx.where(mask1, 0, vs.pet_res) * vs.maskCatch,
    )

    # interception storage will be evaporated
    vs.evap_int_top = update_add(
        vs.evap_int_top,
        at[:, :], vs.S_int_top[:, :, vs.tau] * mask2 * vs.maskCatch,
    )
    # residual ET
    vs.pet_res = update_add(
        vs.pet_res,
        at[:, :], -vs.S_int_top[:, :, vs.tau] * mask2 * vs.maskCatch,
    )

    # update top interception storage
    vs.S_int_top = update_add(
        vs.S_int_top,
        at[:, :, vs.tau], -vs.evap_int_top * vs.maskCatch,
    )

    return KernelOutput(S_int_top=vs.S_int_top, pet_res=vs.pet_res, evap_int_top=vs.evap_int_top)


@roger_kernel
def calc_evap_int_ground(state):
    """
    Calculates evaporation from lower interception storage
    """
    vs = state.variables

    mask1 = (vs.S_int_ground[:, :, vs.tau] <= vs.S_int_ground_tot) & (vs.pet_res <= vs.S_int_ground[:, :, vs.tau]) & (vs.S_int_ground_tot > 0) & (vs.S_int_ground[:, :, vs.tau] > 0) & (vs.prec <= 0)
    mask2 = (vs.S_int_ground[:, :, vs.tau] <= vs.S_int_ground_tot) & (vs.pet_res > vs.S_int_ground[:, :, vs.tau]) & (vs.S_int_ground_tot > 0) & (vs.S_int_ground[:, :, vs.tau] > 0) & (vs.prec <= 0)

    evap_int_ground = allocate(state.dimensions, ("x", "y"))
    vs.evap_int_ground = update(
        vs.evap_int_ground,
        at[:, :], evap_int_ground,
    )
    # interception storage will not be fully evaporated
    vs.evap_int_ground = update_add(
        vs.evap_int_ground,
        at[:, :], vs.pet_res * mask1 * vs.maskCatch,
    )
    vs.pet_res = update(
        vs.pet_res,
        at[:, :], npx.where(mask1, 0, vs.pet_res) * vs.maskCatch,
    )

    # interception storage will be evaporated
    vs.evap_int_ground = update_add(
        vs.evap_int_ground,
        at[:, :], vs.S_int_ground[:, :, vs.tau] * mask2 * vs.maskCatch,
    )
    # residual ET
    vs.pet_res = update_add(
        vs.pet_res,
        at[:, :], -vs.S_int_ground[:, :, vs.tau] * mask2 * vs.maskCatch,
    )

    # update ground interception storage
    vs.S_int_ground = update_add(
        vs.S_int_ground,
        at[:, :, vs.tau], -vs.evap_int_ground * vs.maskCatch,
    )

    return KernelOutput(S_int_ground=vs.S_int_ground, pet_res=vs.pet_res, evap_int_ground=vs.evap_int_ground)


@roger_kernel
def calc_evap_dep(state):
    """
    Calculates evaporation from surface depression storage
    """
    vs = state.variables

    # PET exceeds water stored in the depression
    mask1 = (vs.S_dep[:, :, vs.tau] <= vs.pet_res) & (vs.S_dep[:, :, vs.tau] > 0) & (vs.pet_res > 0) & (vs.prec <= 0)
    # PET does not exceed water stored in the depression
    mask2 = (vs.S_dep[:, :, vs.tau] > vs.pet_res) & (vs.S_dep[:, :, vs.tau] > 0) & (vs.pet_res > 0) & (vs.prec <= 0)

    evap_dep = allocate(state.dimensions, ("x", "y"))
    vs.evap_dep = update(
        vs.evap_dep,
        at[:, :], evap_dep,
    )
    vs.evap_dep = update_add(
        vs.evap_dep,
        at[:, :], vs.S_dep[:, :, vs.tau] * mask1 * vs.maskCatch,
    )
    vs.pet_res = update_add(
        vs.pet_res,
        at[:, :], -vs.S_dep[:, :, vs.tau] * mask1 * vs.maskCatch,
    )

    vs.evap_dep = update_add(
        vs.evap_dep,
        at[:, :], vs.pet_res * mask2 * vs.maskCatch,
    )
    vs.pet_res = update(
        vs.pet_res,
        at[:, :], npx.where(mask2, 0, vs.pet_res) * vs.maskCatch,
    )

    # update surface depression storage after evaporation
    mask3 = (vs.S_dep[:, :, vs.tau] > 0) & (vs.evap_dep > 0)
    vs.S_dep = update_add(
        vs.S_dep,
        at[:, :, vs.tau], -vs.evap_dep * mask3 * vs.maskCatch,
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
        at[:, :], vs.evap_int_top + vs.evap_int_ground + vs.evap_dep * vs.maskCatch,
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
        at[:, :], npx.where(mask3, 1, vs.k_stress_evap) * vs.maskCatch,
    )
    vs.k_stress_evap = update(
        vs.k_stress_evap,
        at[:, :], npx.where(mask4, (vs.tew - vs.de) / (vs.tew - vs.rew), vs.k_stress_evap) * vs.maskCatch,
    )
    vs.k_stress_evap = update(
        vs.k_stress_evap,
        at[:, :], npx.where(mask5, 0, vs.k_stress_evap) * vs.maskCatch,
    )
    # calculates coeffcient of soil evaporation
    vs.evap_coeff = update(
        vs.evap_coeff,
        at[:, :], vs.basal_evap_coeff * vs.k_stress_evap * vs.maskCatch,
    )

    # potential soil evaporation
    pevap_soil = allocate(state.dimensions, ("x", "y"))
    # soil evaporation from fine pore storage
    evap_fp = allocate(state.dimensions, ("x", "y"))
    pevap_soil = update(
        pevap_soil,
        at[:, :], vs.pet_res * vs.evap_coeff * vs.maskCatch,
    )

    # conditions for evaporation from fine storage
    mask1 = (vs.S_fp_rz > 0) & (pevap_soil <= vs.S_fp_rz) & (pevap_soil > 0) & (vs.swe[:, :, vs.tau] <= 0) & (vs.prec <= 0)
    mask2 = (vs.S_fp_rz > 0) & (pevap_soil > vs.S_fp_rz) & (pevap_soil > 0) & (vs.swe[:, :, vs.tau] <= 0) & (vs.prec <= 0)

    # water evaporates from fine pores in root zone
    # some water remains
    evap_fp = update_add(
        evap_fp,
        at[:, :], pevap_soil * mask1 * vs.maskCatch,
    )
    # residual ET
    vs.pet_res = update_add(
        vs.pet_res,
        at[:, :], -pevap_soil * mask1 * vs.maskCatch,
    )
    vs.pet_res = update(
        vs.pet_res,
        at[:, :], npx.where(vs.pet_res < 0, 0, vs.pet_res) * vs.maskCatch,
    )

    # water evaporates from fine pores in root zone
    # no water remains
    evap_fp = update_add(
        evap_fp,
        at[:, :], vs.S_fp_rz * mask2 * vs.maskCatch,
    )
    # residual ET
    vs.pet_res = update_add(
        vs.pet_res,
        at[:, :], -vs.S_fp_rz * mask2 * vs.maskCatch,
    )
    vs.pet_res = update(
        vs.pet_res,
        at[:, :], npx.where(vs.pet_res < 0, 0, vs.pet_res) * vs.maskCatch,
    )

    vs.evap_soil = update(
        vs.evap_soil,
        at[:, :], evap_fp * vs.maskCatch,
    )

    # update root zone storage after evaporation
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[:, :], -vs.evap_soil * vs.maskCatch,
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
        at[:, :], ((1 - settings.transp_water_stress) * (vs.theta_fc - vs.theta_pwp) + vs.theta_pwp) * vs.maskCatch,
    )

    mask_crops = npx.isin(vs.lu_id, npx.arange(500, 600, 1, dtype=int))
    vs.k_stress_transp = update(
        vs.k_stress_transp,
        at[:, :], npx.where((vs.theta_rz[:, :, vs.tau] > theta_water_stress) & ~mask_crops, 1, ((vs.theta_rz[:, :, vs.tau] - vs.theta_pwp) / (theta_water_stress - vs.theta_pwp))) * vs.maskCatch,
    )

    # calculates coeffcient of transpiration
    vs.transp_coeff = update(
        vs.transp_coeff,
        at[:, :], vs.basal_transp_coeff * vs.k_stress_transp * vs.maskCatch,
    )

    transp_fp = allocate(state.dimensions, ("x", "y"))
    transp_lp = allocate(state.dimensions, ("x", "y"))

    # potential transpiration
    vs.ptransp = update(
        vs.ptransp,
        at[:, :], vs.pet_res * vs.transp_coeff * vs.maskCatch,
    )

    # residual transpiration
    vs.ptransp_res = update(
        vs.ptransp_res,
        at[:, :], vs.ptransp * vs.maskCatch,
    )

    # water transpires from large pores in root zone
    # some water remains
    mask1 = (vs.S_lp_rz > 0) & (vs.ptransp_res <= vs.S_lp_rz) & (vs.ptransp > 0) & (vs.swe[:, :, vs.tau] <= 0) & (vs.prec <= 0)
    transp_lp = update_add(
        transp_lp,
        at[:, :], vs.ptransp_res * mask1 * vs.maskCatch,
    )
    vs.ptransp_res = update(
        vs.ptransp_res,
        at[:, :], npx.where(mask1, 0, vs.ptransp_res) * vs.maskCatch,
    )

    # water transpires from large pores in root zone
    # no water remains
    mask2 = (vs.S_lp_rz > 0) & (vs.ptransp_res > vs.S_lp_rz) & (vs.ptransp > 0) & (vs.swe[:, :, vs.tau] <= 0) & (vs.prec <= 0)
    transp_lp = update_add(
        transp_lp,
        at[:, :], vs.S_lp_rz * mask2 * vs.maskCatch,
    )
    vs.ptransp_res = update_add(
        vs.ptransp_res,
        at[:, :], -vs.S_lp_rz * mask2 * vs.maskCatch,
    )

    # water transpires from fine pores in root zone
    # some water remains
    mask3 = (vs.S_fp_rz > 0) & (vs.ptransp_res <= vs.S_fp_rz) & (vs.S_lp_rz <= 0) & (vs.ptransp > 0) & (vs.swe[:, :, vs.tau] <= 0) & (vs.prec <= 0)
    transp_fp = update_add(
        transp_fp,
        at[:, :], vs.ptransp_res * mask3 * vs.maskCatch,
    )
    vs.ptransp_res = update(
        vs.ptransp_res,
        at[:, :], npx.where(mask3, 0, vs.ptransp_res) * vs.maskCatch,
    )

    # water transpires from fine pores in root zone
    # no water remains
    mask4 = (vs.S_fp_rz > 0) & (vs.ptransp_res > vs.S_fp_rz) & (vs.S_lp_rz <= 0) & (vs.ptransp > 0) & (vs.swe[:, :, vs.tau] <= 0) & (vs.prec <= 0)
    transp_fp = update_add(
        transp_fp,
        at[:, :], vs.S_fp_rz * mask4 * vs.maskCatch,
    )
    vs.ptransp_res = update_add(
        vs.ptransp_res,
        at[:, :], -vs.S_fp_rz * mask4 * vs.maskCatch,
    )

    vs.ptransp_res = update(
        vs.ptransp_res,
        at[:, :], npx.where(vs.ptransp_res < 0, 0, vs.ptransp_res) * vs.maskCatch,
    )

    # transpiration from root zone (root water uptake)
    vs.transp = update(
        vs.transp,
        at[:, :], (transp_fp + transp_lp) * vs.maskCatch,
    )

    # update root zone storage after transpiration
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[:, :], -transp_lp * vs.maskCatch,
    )
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[:, :], -transp_fp * vs.maskCatch,
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
        at[:, :], vs.evap_soil + vs.transp * (vs.z_evap / vs.z_root[:, :, vs.tau]) * vs.maskCatch,
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
        at[:, :], (vs.evap_soil + vs.transp) * vs.maskCatch,
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
        at[:, :], (vs.evap_int_top + vs.evap_int_ground + vs.evap_dep + vs.evap_soil + vs.transp) * vs.maskCatch,
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
        at[:, :, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[:, :, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.evap_soil, vs.sas_params_evap_soil) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[:, :, 1:], npx.cumsum(vs.tt_evap_soil, axis=2),
    )

    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
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
        at[:, :, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_evap_soil = update(
        vs.tt_evap_soil,
        at[:, :, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.evap_soil, vs.sas_params_evap_soil) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_evap_soil = update(
        vs.TT_evap_soil,
        at[:, :, 1:], npx.cumsum(vs.tt_evap_soil, axis=2),
    )

    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_evap_soil = update(
        vs.mtt_evap_soil,
        at[:, :, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil, vs.msa_rz, alpha) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.C_evap_soil = update(
        vs.C_evap_soil,
        at[:, :], transport.calc_conc_iso_flux(state, vs.mtt_evap_soil, vs.tt_evap_soil) * vs.maskCatch,
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_evap_soil, vs.evap_soil) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge
    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, vs.tau, :], npx.where((vs.sa_rz[:, :, vs.tau, :] > 0), vs.msa_rz[:, :, vs.tau, :], npx.NaN) * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_evap_soil=vs.tt_evap_soil, TT_evap_soil=vs.TT_evap_soil, msa_rz=vs.msa_rz, mtt_evap_soil=vs.mtt_evap_soil, C_evap_soil=vs.C_evap_soil)


@roger_kernel
def calculate_transpiration_transport_kernel(state):
    """
    Calculates travel time of transpiration
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_transp = update(
        vs.tt_transp,
        at[:, :, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.transp, vs.sas_params_transp) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_transp = update(
        vs.TT_transp,
        at[:, :, 1:], npx.cumsum(vs.tt_transp, axis=2),
    )

    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_transp, vs.transp) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
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
        at[:, :, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_transp = update(
        vs.tt_transp,
        at[:, :, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.transp, vs.sas_params_transp) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_transp = update(
        vs.TT_transp,
        at[:, :, 1:], npx.cumsum(vs.tt_transp, axis=2),
    )

    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[:, :, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_transp, vs.transp, vs.msa_rz, alpha) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.C_transp = update(
        vs.C_transp,
        at[:, :], transport.calc_conc_iso_flux(state, vs.mtt_transp, vs.tt_transp) * vs.maskCatch,
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_transp, vs.transp) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge
    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, vs.tau, :], npx.where((vs.sa_rz[:, :, vs.tau, :] > 0), vs.msa_rz[:, :, vs.tau, :], npx.NaN) * vs.maskCatch[:, :, npx.newaxis],
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
        at[:, :, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_transp = update(
        vs.tt_transp,
        at[:, :, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.transp, vs.sas_params_transp) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_transp = update(
        vs.TT_transp,
        at[:, :, 1:], npx.cumsum(vs.tt_transp, axis=2),
    )

    # calculate isotope travel time distribution
    vs.mtt_transp = update(
        vs.mtt_transp,
        at[:, :, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_transp, vs.transp, vs.msa_rz, vs.alpha_transp) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.C_transp = update(
        vs.C_transp,
        at[:, :], npx.where(vs.transp > 0, npx.sum(vs.mtt_transp, axis=2) / vs.transp, 0) * vs.maskCatch,
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_transp, vs.transp) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge of root zone
    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, vs.tau, :], vs.msa_rz[:, :, vs.tau, :] - vs.mtt_transp * vs.maskCatch[:, :, npx.newaxis],
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
