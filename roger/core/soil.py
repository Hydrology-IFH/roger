from roger import roger_kernel, roger_routine, KernelOutput
from roger.core.operators import numpy as npx, update, update_multiply, at, for_loop
from roger.core.utilities import _get_row_no
from roger.variables import allocate
from roger.core import numerics
from roger.core import transport


@roger_kernel
def calc_k(state):
    """
    Calculates hydraulic conductivity of soil
    """
    vs = state.variables

    vs.k = update(
        vs.k,
        at[2:-2, 2:-2, vs.tau],
        (vs.ks[2:-2, 2:-2] / (1 + (vs.theta[2:-2, 2:-2, vs.tau] / vs.theta_sat[2:-2, 2:-2]) ** (-vs.m_bc[2:-2, 2:-2])))
        * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(k=vs.k)


@roger_kernel
def calc_h(state):
    """
    Calculates soil water potential
    """
    vs = state.variables

    vs.h = update(
        vs.h,
        at[2:-2, 2:-2, vs.tau],
        (
            vs.ha[2:-2, 2:-2]
            / ((vs.theta[2:-2, 2:-2, vs.tau] / vs.theta_sat[2:-2, 2:-2]) ** (1 / vs.lambda_bc[2:-2, 2:-2]))
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(h=vs.h)


@roger_kernel
def calc_theta(state):
    """
    Calculates soil water content
    """
    vs = state.variables

    vs.theta = update(
        vs.theta,
        at[2:-2, 2:-2, vs.tau],
        ((vs.S_fp_s[2:-2, 2:-2] + vs.S_lp_s[2:-2, 2:-2]) / vs.z_soil[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(theta=vs.theta)


@roger_kernel
def calc_theta_ff(state):
    """
    Calculates soil water conent including film flow
    """
    vs = state.variables

    vs.theta_ff = update(
        vs.theta_ff,
        at[2:-2, 2:-2, vs.tau],
        npx.sum(vs.S_f[2:-2, 2:-2, :], axis=-1) / vs.z_soil[2:-2, 2:-2] + vs.theta[2:-2, 2:-2, vs.tau],
    )

    return KernelOutput(
        theta_ff=vs.theta_ff,
    )


@roger_kernel
def calc_S(state):
    """
    Calculates soil water content
    """
    vs = state.variables

    vs.S_fp_s = update(
        vs.S_fp_s,
        at[2:-2, 2:-2],
        (vs.S_fp_rz[2:-2, 2:-2] + vs.S_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_lp_s = update(
        vs.S_lp_s,
        at[2:-2, 2:-2],
        (vs.S_lp_rz[2:-2, 2:-2] + vs.S_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_s = update(
        vs.S_s,
        at[2:-2, 2:-2, vs.tau],
        (vs.S_pwp_s[2:-2, 2:-2] + vs.S_fp_s[2:-2, 2:-2] + vs.S_lp_s[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_fp_s=vs.S_fp_s, S_lp_s=vs.S_lp_s, S_s=vs.S_s)


@roger_kernel
def calc_dS(state):
    """
    Calculates storage change
    """
    vs = state.variables

    vs.dS_s = update(
        vs.dS_s,
        at[2:-2, 2:-2],
        (vs.S_s[2:-2, 2:-2, vs.tau] - vs.S_s[2:-2, 2:-2, vs.taum1]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(dS_s=vs.dS_s)


@roger_routine
def calculate_soil(state):
    """
    Calculates soil storage and storage dependent variables
    """
    vs = state.variables
    settings = state.settings

    vs.update(calc_S(state))
    vs.update(calc_dS(state))
    vs.update(calc_theta(state))
    vs.update(calc_k(state))
    vs.update(calc_h(state))

    if settings.enable_film_flow:
        vs.update(calc_theta_ff(state))


@roger_kernel
def calc_parameters_soil_kernel(state):
    vs = state.variables
    settings = state.settings

    vs.S_ac_s = update(
        vs.S_ac_s, at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] * vs.theta_ac[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_ufc_s = update(
        vs.S_ufc_s, at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] * vs.theta_ufc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_pwp_s = update(
        vs.S_pwp_s, at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] * vs.theta_pwp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_fc_s = update(
        vs.S_fc_s,
        at[2:-2, 2:-2],
        (vs.z_soil[2:-2, 2:-2] * (vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2])) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_sat_s = update(
        vs.S_sat_s,
        at[2:-2, 2:-2],
        (vs.z_soil[2:-2, 2:-2] * (vs.theta_ac[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]))
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.theta_sat = update(
        vs.theta_sat,
        at[2:-2, 2:-2],
        (vs.theta_ac[2:-2, 2:-2] + vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.theta_fc = update(
        vs.theta_fc, at[2:-2, 2:-2], (vs.theta_ufc[2:-2, 2:-2] + vs.theta_pwp[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    # pore-size distribution index
    vs.lambda_bc = update(
        vs.lambda_bc,
        at[2:-2, 2:-2],
        (
            (
                npx.log(vs.theta_fc[2:-2, 2:-2] / vs.theta_sat[2:-2, 2:-2])
                - npx.log(vs.theta_pwp[2:-2, 2:-2] / vs.theta_sat[2:-2, 2:-2])
            )
            / (npx.log(15850) - npx.log(63))
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    # air entry value (or bubbling pressure)
    vs.ha = update(
        vs.ha,
        at[2:-2, 2:-2],
        ((vs.theta_pwp[2:-2, 2:-2] / vs.theta_sat[2:-2, 2:-2]) ** (1.0 / vs.lambda_bc[2:-2, 2:-2]) * (-15850))
        * vs.maskCatch[2:-2, 2:-2],
    )
    # pore-connectivity parameter
    vs.m_bc = update(
        vs.m_bc,
        at[2:-2, 2:-2],
        ((settings.a_bc + settings.b_bc * vs.lambda_bc[2:-2, 2:-2]) / vs.lambda_bc[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )
    # salvucci exponent
    vs.n_salv = update(
        vs.n_salv, at[2:-2, 2:-2], (settings.a_bc + settings.b_bc * vs.lambda_bc[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    # wetting front suction (in mm)
    # *(-10) for conversion to mm
    vs.wfs = update(
        vs.wfs,
        at[2:-2, 2:-2],
        (((2 + 3 * vs.lambda_bc[2:-2, 2:-2]) / (1 + 3 * vs.lambda_bc[2:-2, 2:-2]) * vs.ha[2:-2, 2:-2] / 2) * (-10))
        * vs.maskCatch[2:-2, 2:-2],
    )

    # soil water content at different soil water potentials
    vs.theta_27 = update(
        vs.theta_27,
        at[2:-2, 2:-2],
        ((vs.ha[2:-2, 2:-2] / (-(10**2.7))) ** vs.lambda_bc[2:-2, 2:-2] * vs.theta_sat[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_4 = update(
        vs.theta_4,
        at[2:-2, 2:-2],
        ((vs.ha[2:-2, 2:-2] / (-(10**4))) ** vs.lambda_bc[2:-2, 2:-2] * vs.theta_sat[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_6 = update(
        vs.theta_6,
        at[2:-2, 2:-2],
        ((vs.ha[2:-2, 2:-2] / (-(10**6))) ** vs.lambda_bc[2:-2, 2:-2] * vs.theta_sat[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )

    # calculate sand content
    vs.sand = update(vs.sand, at[2:-2, 2:-2], (1 * (vs.theta_ac[2:-2, 2:-2] / 0.24)) * vs.maskCatch[2:-2, 2:-2])
    vs.sand = update(
        vs.sand, at[2:-2, 2:-2], npx.where(vs.sand[2:-2, 2:-2] < 0, 0, vs.sand[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )
    vs.sand = update(
        vs.sand, at[2:-2, 2:-2], npx.where(vs.sand[2:-2, 2:-2] > 1, 1, vs.sand[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    # calculate clay content
    vs.clay = update(
        vs.clay,
        at[2:-2, 2:-2],
        (settings.clay_max * (vs.theta_6[2:-2, 2:-2] - settings.clay_min) / 0.3) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.clay = update(
        vs.clay,
        at[2:-2, 2:-2],
        npx.where(vs.clay[2:-2, 2:-2] < settings.clay_min, settings.clay_min, vs.clay[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )

    # calculate maximum depth of shrinkage cracks (in mm) which depends on
    # soil clay content
    vs.z_sc_max = update(vs.z_sc_max, at[2:-2, 2:-2], (vs.clay[2:-2, 2:-2] * 700) * vs.maskCatch[2:-2, 2:-2])

    # calculate drainage area of vertical macropores
    vs.mp_drain_area = update(
        vs.mp_drain_area,
        at[2:-2, 2:-2],
        1 - npx.exp((-1) * (vs.dmpv[2:-2, 2:-2] / 82) ** 0.887) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        S_ac_s=vs.S_ac_s,
        S_ufc_s=vs.S_ufc_s,
        S_pwp_s=vs.S_pwp_s,
        S_fc_s=vs.S_fc_s,
        S_sat_s=vs.S_sat_s,
        theta_sat=vs.theta_sat,
        theta_fc=vs.theta_fc,
        lambda_bc=vs.lambda_bc,
        ha=vs.ha,
        m_bc=vs.m_bc,
        n_salv=vs.n_salv,
        wfs=vs.wfs,
        theta_27=vs.theta_27,
        theta_4=vs.theta_4,
        theta_6=vs.theta_6,
        sand=vs.sand,
        clay=vs.clay,
        z_sc_max=vs.z_sc_max,
        mp_drain_area=vs.mp_drain_area,
    )


@roger_kernel
def calc_parameters_root_zone_kernel(state):
    vs = state.variables
    settings = state.settings

    # calculate readily evaporable water
    mask1 = vs.theta_pwp < settings.theta_rew_min
    mask2 = (vs.theta_pwp >= settings.theta_rew_min) & (vs.theta_pwp <= settings.theta_rew_max)
    mask3 = vs.theta_pwp > settings.theta_rew_max
    vs.rew = update(
        vs.rew,
        at[2:-2, 2:-2],
        npx.where(mask1[2:-2, 2:-2], settings.rew_min, vs.rew[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.rew = update(
        vs.rew,
        at[2:-2, 2:-2],
        npx.where(mask2[2:-2, 2:-2], vs.theta_pwp[2:-2, 2:-2] / settings.theta_rew_max, vs.rew[2:-2, 2:-2])
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.rew = update(
        vs.rew,
        at[2:-2, 2:-2],
        npx.where(mask3[2:-2, 2:-2], settings.rew_max, vs.rew[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # evaporation depth
    vs.z_evap = update(
        vs.z_evap,
        at[2:-2, 2:-2],
        ((vs.rew[2:-2, 2:-2] / settings.rew_max) * settings.z_evap_max) * vs.maskCatch[2:-2, 2:-2],
    )

    # total evaporable water
    vs.tew = update(
        vs.tew,
        at[2:-2, 2:-2],
        ((vs.theta_fc[2:-2, 2:-2] - 0.5 * vs.theta_pwp[2:-2, 2:-2]) * vs.z_evap[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # root depth
    z_root = allocate(state.dimensions, ("x", "y"))

    z_root = update(z_root, at[2:-2, 2:-2], vs.z_root[2:-2, 2:-2, 0])

    cc_cond = allocate(state.dimensions, ("x", "y"), dtype=bool, fill=False)
    cc_cond = update(
        cc_cond,
        at[:, :],
        npx.isin(vs.lu_id, npx.array([0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 98, 31, 32, 33, 40, 41, 50, 60, 98])),
    )

    # assign land use specific root depth
    def loop_body(i, z_root):
        mask = (vs.lu_id == i) & cc_cond
        row_no = _get_row_no(vs.lut_rdlu[:, 0], i)
        z_root = update(
            z_root,
            at[2:-2, 2:-2],
            npx.where(mask[2:-2, 2:-2], vs.lut_rdlu[row_no, 1], z_root[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
        )

        return z_root

    z_root = for_loop(0, 61, loop_body, z_root)

    z_root = update(
        z_root,
        at[2:-2, 2:-2],
        npx.where(vs.maskRiver[2:-2, 2:-2] | vs.maskLake[2:-2, 2:-2], 0, z_root[2:-2, 2:-2]),
    )

    z_root = update(
        z_root,
        at[2:-2, 2:-2],
        npx.where(
            (
                (vs.lu_id[2:-2, 2:-2] == 10)
                | (vs.lu_id[2:-2, 2:-2] == 11)
                | (vs.lu_id[2:-2, 2:-2] == 12)
                | (vs.lu_id[2:-2, 2:-2] == 15)
                | (vs.lu_id[2:-2, 2:-2] == 16)
                | (vs.lu_id[2:-2, 2:-2] == 17)
            ),
            1500,
            z_root[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    z_root = update(
        z_root,
        at[2:-2, 2:-2],
        npx.where((vs.lu_id[2:-2, 2:-2] == 100), 300, z_root[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    z_root = update(
        z_root,
        at[2:-2, 2:-2],
        npx.where(
            (z_root[2:-2, 2:-2] >= vs.z_soil[2:-2, 2:-2]),
            settings.zroot_to_zsoil_max * vs.z_soil[2:-2, 2:-2],
            z_root[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, 0],
        z_root[2:-2, 2:-2] * vs.c_root[2:-2, 2:-2],
    )

    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, 1],
        z_root[2:-2, 2:-2] * vs.c_root[2:-2, 2:-2],
    )

    # set thickness of upper soil water storage to 20 cm for bare soils
    mask_crops = npx.isin(vs.lu_id, npx.arange(500, 600, 1, dtype=npx.int32))
    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, 0],
        npx.where(mask_crops[2:-2, 2:-2], 200, vs.z_root[2:-2, 2:-2, 0]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, 1],
        npx.where(mask_crops[2:-2, 2:-2], 200, vs.z_root[2:-2, 2:-2, 1]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_root = update(
        vs.z_root,
        at[2:-2, 2:-2, :],
        npx.where(
            vs.z_root[2:-2, 2:-2, :] < vs.z_soil[2:-2, 2:-2, npx.newaxis],
            vs.z_root[2:-2, 2:-2, :],
            vs.z_soil[2:-2, 2:-2, npx.newaxis] * 0.9,
        ),
    )

    vs.S_ac_rz = update(
        vs.S_ac_rz, at[2:-2, 2:-2], (vs.theta_ac[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2]
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

    return KernelOutput(
        z_root=vs.z_root,
        z_evap=vs.z_evap,
        rew=vs.rew,
        tew=vs.tew,
        S_ac_rz=vs.S_ac_rz,
        S_ufc_rz=vs.S_ufc_rz,
        S_pwp_rz=vs.S_pwp_rz,
        S_fc_rz=vs.S_fc_rz,
        S_sat_rz=vs.S_sat_rz,
    )


@roger_kernel
def calc_parameters_subsoil_kernel(state):
    vs = state.variables

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
        S_ac_ss=vs.S_ac_ss,
        S_ufc_ss=vs.S_ufc_ss,
        S_pwp_ss=vs.S_pwp_ss,
        S_fc_ss=vs.S_fc_ss,
        S_sat_ss=vs.S_sat_ss,
    )


@roger_kernel
def calc_parameters_lateral_flow_kernel(state):
    vs = state.variables

    v_mp_layer = allocate(state.dimensions, ("x", "y", 8))

    # assign macropore flow velocity from look up table (in mm/h)
    def loop_body(i, v_mp_layer):
        nrow = _get_row_no(vs.lut_mlms[:, 0], i)
        # convert m/h to mm/h
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 7],
            npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 1] * 1000, v_mp_layer[2:-2, 2:-2, 7])
            * vs.maskCatch[2:-2, 2:-2],
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 6],
            npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 2] * 1000, v_mp_layer[2:-2, 2:-2, 6])
            * vs.maskCatch[2:-2, 2:-2],
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 5],
            npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 3] * 1000, v_mp_layer[2:-2, 2:-2, 5])
            * vs.maskCatch[2:-2, 2:-2],
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 4],
            npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 4] * 1000, v_mp_layer[2:-2, 2:-2, 4])
            * vs.maskCatch[2:-2, 2:-2],
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 3],
            npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 5] * 1000, v_mp_layer[2:-2, 2:-2, 3])
            * vs.maskCatch[2:-2, 2:-2],
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 2],
            npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 6] * 1000, v_mp_layer[2:-2, 2:-2, 2])
            * vs.maskCatch[2:-2, 2:-2],
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 1],
            npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 7] * 1000, v_mp_layer[2:-2, 2:-2, 1])
            * vs.maskCatch[2:-2, 2:-2],
        )
        v_mp_layer = update(
            v_mp_layer,
            at[2:-2, 2:-2, 0],
            npx.where(vs.slope_per[2:-2, 2:-2] == i, vs.lut_mlms[nrow, 8] * 1000, v_mp_layer[2:-2, 2:-2, 0])
            * vs.maskCatch[2:-2, 2:-2],
        )

        return v_mp_layer

    v_mp_layer = for_loop(1, npx.max(vs.slope_per) + 1, loop_body, v_mp_layer)

    vs.v_mp_layer_8 = update(vs.v_mp_layer_8, at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 7] * vs.maskCatch[2:-2, 2:-2])
    vs.v_mp_layer_7 = update(vs.v_mp_layer_7, at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 6] * vs.maskCatch[2:-2, 2:-2])
    vs.v_mp_layer_6 = update(vs.v_mp_layer_6, at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 5] * vs.maskCatch[2:-2, 2:-2])
    vs.v_mp_layer_5 = update(vs.v_mp_layer_5, at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 4] * vs.maskCatch[2:-2, 2:-2])
    vs.v_mp_layer_4 = update(vs.v_mp_layer_4, at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 3] * vs.maskCatch[2:-2, 2:-2])
    vs.v_mp_layer_3 = update(vs.v_mp_layer_3, at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 2] * vs.maskCatch[2:-2, 2:-2])
    vs.v_mp_layer_2 = update(vs.v_mp_layer_2, at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 1] * vs.maskCatch[2:-2, 2:-2])
    vs.v_mp_layer_1 = update(vs.v_mp_layer_1, at[2:-2, 2:-2], v_mp_layer[2:-2, 2:-2, 0] * vs.maskCatch[2:-2, 2:-2])

    return KernelOutput(
        v_mp_layer_8=vs.v_mp_layer_8,
        v_mp_layer_7=vs.v_mp_layer_7,
        v_mp_layer_6=vs.v_mp_layer_6,
        v_mp_layer_5=vs.v_mp_layer_5,
        v_mp_layer_4=vs.v_mp_layer_4,
        v_mp_layer_3=vs.v_mp_layer_3,
        v_mp_layer_2=vs.v_mp_layer_2,
        v_mp_layer_1=vs.v_mp_layer_1,
    )


@roger_kernel
def calc_parameters_crops_kernel(state):
    vs = state.variables

    for i in range(500, 600):
        mask = vs.crop_type == i
        row_no = _get_row_no(vs.lut_crops[:, 0], i)
        vs.doy_start = update(
            vs.doy_start,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 1], vs.doy_start[2:-2, 2:-2, :]),
        )
        vs.doy_mid = update(
            vs.doy_mid,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 2], vs.doy_mid[2:-2, 2:-2, :]),
        )
        vs.doy_dec = update(
            vs.doy_dec,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 3], vs.doy_dec[2:-2, 2:-2, :]),
        )
        vs.doy_end = update(
            vs.doy_end,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 4], vs.doy_end[2:-2, 2:-2, :]),
        )
        vs.ta_base = update(
            vs.ta_base,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 11], vs.ta_base[2:-2, 2:-2, :]),
        )
        vs.ta_ceil = update(
            vs.ta_ceil,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 12], vs.ta_ceil[2:-2, 2:-2, :]),
        )
        vs.ccc_min = update(
            vs.ccc_min,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 13], vs.ccc_min[2:-2, 2:-2, :]),
        )
        vs.ccc_max = update(
            vs.ccc_max,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 14], vs.ccc_max[2:-2, 2:-2, :]),
        )
        vs.crop_height_max = update(
            vs.crop_height_max,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 16] * 1000, vs.crop_height_max[2:-2, 2:-2, :]),
        )
        vs.ccc_growth_rate = update(
            vs.ccc_growth_rate,
            at[2:-2, 2:-2, :],
            npx.where(mask[2:-2, 2:-2], vs.lut_crops[row_no, 18], vs.ccc_growth_rate[2:-2, 2:-2, :]),
        )
        vs.basal_crop_coeff_mid = update(
            vs.basal_crop_coeff_mid,
            at[2:-2, 2:-2, :],
            npx.where(
                mask[2:-2, 2:-2],
                vs.lut_crops[row_no, 21] * vs.lut_crop_scale[2:-2, 2:-2, row_no],
                vs.basal_crop_coeff_mid[2:-2, 2:-2, :],
            ),
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
    )


@roger_routine
def calculate_parameters(state):
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        numerics.validate_parameters_soil(state)
        vs.update(calc_parameters_soil_kernel(state))
        vs.update(calc_parameters_root_zone_kernel(state))
        vs.update(calc_parameters_subsoil_kernel(state))
        if settings.enable_lateral_flow:
            vs.update(calc_parameters_lateral_flow_kernel(state))
        if settings.enable_crop_phenology:
            vs.update(calc_parameters_crops_kernel(state))


@roger_kernel
def calc_initial_conditions_soil_kernel(state):
    vs = state.variables

    vs.S_fp_s = update(
        vs.S_fp_s, at[2:-2, 2:-2], (vs.S_fp_rz[2:-2, 2:-2] + vs.S_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_lp_s = update(
        vs.S_lp_s, at[2:-2, 2:-2], (vs.S_lp_rz[2:-2, 2:-2] + vs.S_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2]
    )

    vs.S_s = update(
        vs.S_s,
        at[2:-2, 2:-2, :2],
        (vs.S_rz[2:-2, 2:-2, :2] + vs.S_ss[2:-2, 2:-2, :2]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.theta = update(
        vs.theta,
        at[2:-2, 2:-2, :2],
        (vs.S_s[2:-2, 2:-2, :2] / vs.z_soil[2:-2, 2:-2, npx.newaxis]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        S_fp_s=vs.S_fp_s,
        S_lp_s=vs.S_lp_s,
        S_s=vs.S_s,
        theta=vs.theta,
    )


@roger_kernel
def calc_initial_conditions_root_zone_kernel(state):
    vs = state.variables

    mask1 = vs.theta_rz[:, :, vs.tau] > vs.theta_pwp
    mask2 = vs.theta_rz[:, :, vs.tau] <= vs.theta_pwp
    vs.theta_fp_rz = update(
        vs.theta_fp_rz,
        at[2:-2, 2:-2],
        npx.where(
            mask1[2:-2, 2:-2], vs.theta_rz[2:-2, 2:-2, vs.tau] - vs.theta_pwp[2:-2, 2:-2], vs.theta_fp_rz[2:-2, 2:-2]
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_fp_rz = update(
        vs.theta_fp_rz,
        at[2:-2, 2:-2],
        npx.where(mask2[2:-2, 2:-2], 0, vs.theta_fp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask3 = vs.theta_fp_rz >= vs.theta_ufc
    vs.theta_fp_rz = update(
        vs.theta_fp_rz,
        at[2:-2, 2:-2],
        npx.where(mask3[2:-2, 2:-2], vs.theta_ufc[2:-2, 2:-2], vs.theta_fp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask4 = vs.theta_rz[:, :, vs.tau] > vs.theta_fc
    mask5 = vs.theta_rz[:, :, vs.tau] <= vs.theta_fc
    vs.theta_lp_rz = update(
        vs.theta_lp_rz,
        at[2:-2, 2:-2],
        npx.where(
            mask4[2:-2, 2:-2], vs.theta_rz[2:-2, 2:-2, vs.tau] - vs.theta_fc[2:-2, 2:-2], vs.theta_lp_rz[2:-2, 2:-2]
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_lp_rz = update(
        vs.theta_lp_rz,
        at[2:-2, 2:-2],
        npx.where(mask5[2:-2, 2:-2], 0, vs.theta_lp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_fp_rz = update(
        vs.S_fp_rz,
        at[2:-2, 2:-2],
        (vs.theta_fp_rz[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_lp_rz = update(
        vs.S_lp_rz,
        at[2:-2, 2:-2],
        (vs.theta_lp_rz[2:-2, 2:-2] * vs.z_root[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_rz = update(
        vs.S_rz,
        at[2:-2, 2:-2, :2],
        (
            vs.S_pwp_rz[2:-2, 2:-2, npx.newaxis]
            + vs.S_fp_rz[2:-2, 2:-2, npx.newaxis]
            + vs.S_lp_rz[2:-2, 2:-2, npx.newaxis]
        )
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        theta_fp_rz=vs.theta_fp_rz,
        theta_lp_rz=vs.theta_lp_rz,
        S_fp_rz=vs.S_fp_rz,
        S_lp_rz=vs.S_lp_rz,
        S_rz=vs.S_rz,
    )


@roger_kernel
def calc_initial_conditions_subsoil_kernel(state):
    vs = state.variables

    mask6 = vs.theta_ss[:, :, vs.tau] > vs.theta_pwp
    mask7 = vs.theta_ss[:, :, vs.tau] <= vs.theta_pwp
    vs.theta_fp_ss = update(
        vs.theta_fp_ss,
        at[2:-2, 2:-2],
        npx.where(
            mask6[2:-2, 2:-2], vs.theta_ss[2:-2, 2:-2, vs.tau] - vs.theta_pwp[2:-2, 2:-2], vs.theta_fp_ss[2:-2, 2:-2]
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_fp_ss = update(
        vs.theta_fp_ss,
        at[2:-2, 2:-2],
        npx.where(mask7[2:-2, 2:-2], 0, vs.theta_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask8 = vs.theta_fp_ss >= vs.theta_ufc
    vs.theta_fp_ss = update(
        vs.theta_fp_ss,
        at[2:-2, 2:-2],
        npx.where(mask8[2:-2, 2:-2], vs.theta_ufc[2:-2, 2:-2], vs.theta_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask9 = vs.theta_ss[:, :, vs.tau] > vs.theta_fc
    mask10 = vs.theta_ss[:, :, vs.tau] <= vs.theta_fc
    vs.theta_lp_ss = update(
        vs.theta_lp_ss,
        at[2:-2, 2:-2],
        npx.where(
            mask9[2:-2, 2:-2], vs.theta_ss[2:-2, 2:-2, vs.tau] - vs.theta_fc[2:-2, 2:-2], vs.theta_lp_ss[2:-2, 2:-2]
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    vs.theta_lp_ss = update(
        vs.theta_lp_ss,
        at[2:-2, 2:-2],
        npx.where(mask10[2:-2, 2:-2], 0, vs.theta_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[2:-2, 2:-2],
        (vs.theta_fp_ss[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]))
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_lp_ss = update(
        vs.S_lp_ss,
        at[2:-2, 2:-2],
        (vs.theta_lp_ss[2:-2, 2:-2] * (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]))
        * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_ss = update(
        vs.S_ss,
        at[2:-2, 2:-2, :2],
        (
            vs.S_pwp_ss[2:-2, 2:-2, npx.newaxis]
            + vs.S_fp_ss[2:-2, 2:-2, npx.newaxis]
            + vs.S_lp_ss[2:-2, 2:-2, npx.newaxis]
        )
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        theta_fp_ss=vs.theta_fp_ss,
        theta_lp_ss=vs.theta_lp_ss,
        S_fp_ss=vs.S_fp_ss,
        S_lp_ss=vs.S_lp_ss,
        S_ss=vs.S_ss,
    )


@roger_kernel
def calc_initial_conditions_kernel(state):
    vs = state.variables

    vs.S = update(
        vs.S,
        at[2:-2, 2:-2, :2],
        vs.S_sur[2:-2, 2:-2, :2] + vs.S_s[2:-2, 2:-2, :2] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        S=vs.S,
    )


@roger_kernel
def calc_initial_conditions_with_gw_kernel(state):
    vs = state.variables

    vs.S = update(
        vs.S,
        at[2:-2, 2:-2, :2],
        vs.S_sur[2:-2, 2:-2, :2]
        + vs.S_s[2:-2, 2:-2, :2]
        + vs.S_vad[2:-2, 2:-2, :2]
        + vs.S_gw[2:-2, 2:-2, :2] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(
        S=vs.S,
    )


@roger_routine
def calculate_initial_conditions(state):
    """
    Calculates initial conditions of soil
    """
    vs = state.variables
    settings = state.settings

    if not settings.enable_offline_transport:
        numerics.validate_initial_conditions_soil(state)
        vs.update(calc_initial_conditions_root_zone_kernel(state))
        vs.update(calc_initial_conditions_subsoil_kernel(state))
        vs.update(calc_initial_conditions_soil_kernel(state))
        if not settings.enable_groundwater:
            vs.update(calc_initial_conditions_kernel(state))
        elif settings.enable_groundwater:
            vs.update(calc_initial_conditions_with_gw_kernel(state))


@roger_kernel
def calculate_soil_transport_kernel(state):
    """
    Calculates StorAge
    """
    vs = state.variables

    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, vs.tau, :],
        vs.sa_rz[2:-2, 2:-2, vs.tau, :] + vs.sa_ss[2:-2, 2:-2, vs.tau, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :, :],
        transport.calc_SA(state, vs.SA_s, vs.sa_s)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(sa_s=vs.sa_s, SA_s=vs.SA_s)


@roger_kernel
def calculate_soil_transport_iso_kernel(state):
    """
    Calculates StorAge and isotope ratio
    """
    vs = state.variables

    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, :, :],
        vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :, :],
        transport.calc_SA(state, vs.SA_s, vs.sa_s)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, :, :],
        npx.where(
            vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] > 0,
            vs.msa_rz[2:-2, 2:-2, :, :]
            * (vs.sa_rz[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :]))
            + vs.msa_ss[2:-2, 2:-2, :, :]
            * (vs.sa_ss[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :])),
            0,
        ),
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, :2, :],
        npx.where(npx.isnan(vs.msa_s[2:-2, 2:-2, :2, :]), 0, vs.msa_s[2:-2, 2:-2, :2, :]),
    )
    vs.csa_s = update(
        vs.csa_s,
        at[2:-2, 2:-2, vs.tau, :],
        transport.conc_to_delta(state, vs.msa_s[2:-2, 2:-2, vs.tau, :]),
    )
    vs.C_s = update(
        vs.C_s,
        at[2:-2, 2:-2, vs.tau],
        transport.calc_conc_iso_storage(state, vs.sa_s, vs.msa_s)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_s = update(
        vs.C_iso_s,
        at[2:-2, 2:-2, vs.tau],
        transport.conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        sa_s=vs.sa_s, SA_s=vs.SA_s, msa_s=vs.msa_s, csa_s=vs.csa_s, C_s=vs.C_s, C_iso_s=vs.C_iso_s, M_s=vs.M_s
    )


@roger_kernel
def calculate_soil_transport_anion_kernel(state):
    """
    Calculates StorAge and solute concentration
    """
    vs = state.variables

    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, :, :],
        vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :, :],
        transport.calc_SA(state, vs.SA_s, vs.sa_s)[2:-2, 2:-2, :, :]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, :, :],
        vs.msa_rz[2:-2, 2:-2, :, :] + vs.msa_ss[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.csa_s = update(
        vs.csa_s,
        at[2:-2, 2:-2, :, :],
        npx.where(vs.sa_s[2:-2, 2:-2, :, :] > 0, vs.msa_s[2:-2, 2:-2, :, :] / vs.sa_s[2:-2, 2:-2, :, :], 0)
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.M_s = update(
        vs.M_s,
        at[2:-2, 2:-2, vs.tau],
        npx.nansum(vs.msa_s[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.C_s = update(
        vs.C_s,
        at[2:-2, 2:-2, vs.tau],
        npx.where(
            npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1) > 0,
            vs.M_s[2:-2, 2:-2, vs.tau] / npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1),
            0,
        ),
    )

    return KernelOutput(sa_s=vs.sa_s, SA_s=vs.SA_s, msa_s=vs.msa_s, csa_s=vs.csa_s, C_s=vs.C_s, M_s=vs.M_s)


@roger_routine
def calculate_soil_transport(state):
    """
    Calculates StorAge (and solute concentration)
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (
        settings.enable_chloride
        | settings.enable_bromide
        | settings.enable_oxygen18
        | settings.enable_deuterium
        | settings.enable_nitrate
    ):
        vs.update(calculate_soil_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_soil_transport_iso_kernel(state))

    if settings.enable_offline_transport and (
        settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate | settings.enable_virtualtracer
    ):
        vs.update(calculate_soil_transport_anion_kernel(state))


@roger_kernel
def rescale_sa_soil_kernel(state):
    """
    Rescale StorAge.
    """
    vs = state.variables

    vs.sa_rz = update_multiply(
        vs.sa_rz,
        at[2:-2, 2:-2, 0, :],
        vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )
    vs.sa_ss = update_multiply(
        vs.sa_ss,
        at[2:-2, 2:-2, 0, :],
        vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )

    vs.sa_rz = update_multiply(
        vs.sa_rz,
        at[2:-2, 2:-2, 1, :],
        vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )
    vs.sa_ss = update_multiply(
        vs.sa_ss,
        at[2:-2, 2:-2, 1, :],
        vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :2, 1:],
        npx.cumsum(vs.sa_rz[2:-2, 2:-2, :2, :], axis=-1),
    )
    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :2, 0],
        0,
    )

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :2, 1:],
        npx.cumsum(vs.sa_ss[2:-2, 2:-2, :2, :], axis=-1),
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :2, 0],
        0,
    )

    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, :2, :],
        vs.sa_rz[2:-2, 2:-2, :2, :] + vs.sa_ss[2:-2, 2:-2, :2, :],
    )

    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :2, 1:],
        npx.cumsum(vs.sa_s[2:-2, 2:-2, :2, :], axis=-1),
    )
    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :2, 0],
        0,
    )

    return KernelOutput(
        sa_rz=vs.sa_rz,
        sa_ss=vs.sa_ss,
        sa_s=vs.sa_s,
        SA_rz=vs.SA_rz,
        SA_ss=vs.SA_ss,
        SA_s=vs.SA_s,
    )


@roger_kernel
def rescale_sa_msa_iso_soil_kernel(state):
    """
    Rescale solute concentration.
    """
    vs = state.variables

    vs.sa_rz = update_multiply(
        vs.sa_rz,
        at[2:-2, 2:-2, 0, :],
        vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )
    vs.sa_ss = update_multiply(
        vs.sa_ss,
        at[2:-2, 2:-2, 0, :],
        vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )
    vs.sa_rz = update_multiply(
        vs.sa_rz,
        at[2:-2, 2:-2, 1, :],
        vs.S_rz_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )
    vs.sa_ss = update_multiply(
        vs.sa_ss,
        at[2:-2, 2:-2, 1, :],
        vs.S_ss_init[2:-2, 2:-2, npx.newaxis] / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
    )
    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :2, 1:],
        npx.cumsum(vs.sa_rz[2:-2, 2:-2, :2, :], axis=-1),
    )
    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :2, 0],
        0,
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :2, 1:],
        npx.cumsum(vs.sa_ss[2:-2, 2:-2, :2, :], axis=-1),
    )
    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :2, 0],
        0,
    )
    vs.sa_s = update(
        vs.sa_s,
        at[2:-2, 2:-2, :, :],
        vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :],
    )
    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :2, 1:],
        npx.cumsum(vs.sa_s[2:-2, 2:-2, :2, :], axis=-1),
    )
    vs.SA_s = update(
        vs.SA_s,
        at[2:-2, 2:-2, :2, 0],
        0,
    )
    vs.C_rz = update(
        vs.C_rz,
        at[2:-2, 2:-2, :2],
        transport.calc_conc_iso_storage(state, vs.sa_rz, vs.msa_rz)[2:-2, 2:-2, npx.newaxis],
    )
    vs.C_iso_rz = update(
        vs.C_iso_rz,
        at[2:-2, 2:-2, vs.taum1],
        transport.conc_to_delta(state, vs.C_rz[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_rz = update(
        vs.C_iso_rz,
        at[2:-2, 2:-2, vs.tau],
        transport.conc_to_delta(state, vs.C_rz[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_ss = update(
        vs.C_ss,
        at[2:-2, 2:-2, :2],
        transport.calc_conc_iso_storage(state, vs.sa_ss, vs.msa_ss)[2:-2, 2:-2, npx.newaxis],
    )
    vs.C_iso_ss = update(
        vs.C_iso_ss,
        at[2:-2, 2:-2, vs.taum1],
        transport.conc_to_delta(state, vs.C_ss[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_ss = update(
        vs.C_iso_ss,
        at[2:-2, 2:-2, vs.tau],
        transport.conc_to_delta(state, vs.C_ss[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, :, :],
        npx.where(
            vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :] > 0,
            vs.msa_rz[2:-2, 2:-2, :, :]
            * (vs.sa_rz[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :]))
            + vs.msa_ss[2:-2, 2:-2, :, :]
            * (vs.sa_ss[2:-2, 2:-2, :, :] / (vs.sa_rz[2:-2, 2:-2, :, :] + vs.sa_ss[2:-2, 2:-2, :, :])),
            0,
        ),
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, :2, :],
        npx.where(npx.isnan(vs.msa_s[2:-2, 2:-2, :2, :]), 0, vs.msa_s[2:-2, 2:-2, :2, :]),
    )
    vs.msa_s = update(
        vs.msa_s,
        at[2:-2, 2:-2, :2, 0],
        0,
    )
    vs.C_s = update(
        vs.C_s,
        at[2:-2, 2:-2, :2],
        transport.calc_conc_iso_storage(state, vs.sa_s, vs.msa_s)[2:-2, 2:-2, npx.newaxis],
    )
    vs.C_iso_s = update(
        vs.C_iso_s,
        at[2:-2, 2:-2, vs.taum1],
        transport.conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_s = update(
        vs.C_iso_s,
        at[2:-2, 2:-2, vs.tau],
        transport.conc_to_delta(state, vs.C_s[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        C_rz=vs.C_rz,
        C_ss=vs.C_ss,
        C_s=vs.C_s,
        msa_rz=vs.msa_rz,
        msa_ss=vs.msa_ss,
        msa_s=vs.msa_s,
        sa_rz=vs.sa_rz,
        sa_ss=vs.sa_ss,
        sa_s=vs.sa_s,
        SA_rz=vs.SA_rz,
        SA_ss=vs.SA_ss,
        SA_s=vs.SA_s,
        C_iso_rz=vs.C_iso_rz,
        C_iso_ss=vs.C_iso_ss,
        C_iso_s=vs.C_iso_s,
    )


@roger_kernel
def rescale_sa_msa_anion_soil_kernel(state):
    """
    Rescale solute concentration.
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_bromide:
        vs.C_rz = update(
            vs.C_rz,
            at[2:-2, 2:-2, :],
            0,
        )
        vs.C_ss = update(
            vs.C_ss,
            at[2:-2, 2:-2, :],
            0,
        )
        vs.C_s = update(
            vs.C_s,
            at[2:-2, 2:-2, :],
            0,
        )
        vs.M_rz = update(
            vs.M_rz,
            at[2:-2, 2:-2, :],
            0,
        )
        vs.M_ss = update(
            vs.M_ss,
            at[2:-2, 2:-2, :],
            0,
        )
        vs.M_s = update(
            vs.M_s,
            at[2:-2, 2:-2, :],
            0,
        )
        vs.msa_rz = update(
            vs.msa_rz,
            at[2:-2, 2:-2, :, :],
            0,
        )
        vs.msa_ss = update(
            vs.msa_ss,
            at[2:-2, 2:-2, :, :],
            0,
        )
        vs.msa_s = update(
            vs.msa_s,
            at[2:-2, 2:-2, :, :],
            0,
        )

        vs.sa_rz = update_multiply(
            vs.sa_rz,
            at[2:-2, 2:-2, 0, :],
            vs.S_rz_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_ss = update_multiply(
            vs.sa_ss,
            at[2:-2, 2:-2, 0, :],
            vs.S_ss_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_rz = update_multiply(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :],
            vs.S_rz_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_ss = update_multiply(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :],
            vs.S_ss_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.SA_rz = update(
            vs.SA_rz,
            at[2:-2, 2:-2, :2, 1:],
            npx.cumsum(vs.sa_rz[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_rz = update(
            vs.SA_rz,
            at[2:-2, 2:-2, :2, 0],
            0,
        )
        vs.SA_ss = update(
            vs.SA_ss,
            at[2:-2, 2:-2, :2, 1:],
            npx.cumsum(vs.sa_ss[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_ss = update(
            vs.SA_ss,
            at[2:-2, 2:-2, :2, 0],
            0,
        )
        vs.sa_s = update(
            vs.sa_s,
            at[2:-2, 2:-2, :2, :],
            vs.sa_rz[2:-2, 2:-2, :2, :] + vs.sa_ss[2:-2, 2:-2, :2, :],
        )
        vs.SA_s = update(
            vs.SA_s,
            at[2:-2, 2:-2, :2, 1:],
            npx.cumsum(vs.sa_s[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_s = update(
            vs.SA_s,
            at[2:-2, 2:-2, :2, 0],
            0,
        )

    elif settings.enable_chloride | settings.enable_nitrate | settings.enable_virtualtracer:
        vs.msa_rz = update_multiply(
            vs.msa_rz,
            at[2:-2, 2:-2, 0, :],
            vs.S_rz_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.msa_ss = update_multiply(
            vs.msa_ss,
            at[2:-2, 2:-2, 0, :],
            vs.S_ss_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.msa_rz = update_multiply(
            vs.msa_rz,
            at[2:-2, 2:-2, 1, :],
            vs.S_rz_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.msa_ss = update_multiply(
            vs.msa_ss,
            at[2:-2, 2:-2, 1, :],
            vs.S_ss_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_rz = update_multiply(
            vs.sa_rz,
            at[2:-2, 2:-2, 0, :],
            vs.S_rz_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_ss = update_multiply(
            vs.sa_ss,
            at[2:-2, 2:-2, 0, :],
            vs.S_ss_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_rz = update_multiply(
            vs.sa_rz,
            at[2:-2, 2:-2, 1, :],
            vs.S_rz_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.sa_ss = update_multiply(
            vs.sa_ss,
            at[2:-2, 2:-2, 1, :],
            vs.S_ss_init[2:-2, 2:-2, npx.newaxis]
            / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.SA_rz = update(
            vs.SA_rz,
            at[2:-2, 2:-2, :2, 1:],
            npx.cumsum(vs.sa_rz[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_rz = update(
            vs.SA_rz,
            at[2:-2, 2:-2, :2, 0],
            0,
        )
        vs.SA_ss = update(
            vs.SA_ss,
            at[2:-2, 2:-2, :2, 1:],
            npx.cumsum(vs.sa_ss[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_ss = update(
            vs.SA_ss,
            at[2:-2, 2:-2, :2, 0],
            0,
        )
        vs.sa_s = update(
            vs.sa_s,
            at[2:-2, 2:-2, :2, :],
            vs.sa_rz[2:-2, 2:-2, :2, :] + vs.sa_ss[2:-2, 2:-2, :2, :],
        )
        vs.SA_s = update(
            vs.SA_s,
            at[2:-2, 2:-2, :2, 1:],
            npx.cumsum(vs.sa_s[2:-2, 2:-2, :2, :], axis=-1),
        )
        vs.SA_s = update(
            vs.SA_s,
            at[2:-2, 2:-2, :2, 0],
            0,
        )
        vs.C_rz = update(
            vs.C_rz,
            at[2:-2, 2:-2, :2],
            npx.sum(vs.msa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis]
            / npx.sum(vs.sa_rz[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.C_ss = update(
            vs.C_ss,
            at[2:-2, 2:-2, :2],
            npx.sum(vs.msa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis]
            / npx.sum(vs.sa_ss[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.msa_s = update(
            vs.msa_s,
            at[2:-2, 2:-2, :, :],
            vs.msa_rz[2:-2, 2:-2, :, :] + vs.msa_ss[2:-2, 2:-2, :, :],
        )
        vs.C_s = update(
            vs.C_s,
            at[2:-2, 2:-2, :2],
            npx.sum(vs.msa_s[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis]
            / npx.sum(vs.sa_s[2:-2, 2:-2, vs.tau, :], axis=-1)[:, :, npx.newaxis],
        )
        vs.csa_rz = update(
            vs.csa_rz,
            at[2:-2, 2:-2, :, :],
            npx.where(vs.sa_rz[2:-2, 2:-2, :, :] > 0, vs.msa_rz[2:-2, 2:-2, :, :] / vs.sa_rz[2:-2, 2:-2, :, :], 0)
            * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
        )
        vs.csa_ss = update(
            vs.csa_ss,
            at[2:-2, 2:-2, :, :],
            npx.where(vs.sa_ss[2:-2, 2:-2, :, :] > 0, vs.msa_ss[2:-2, 2:-2, :, :] / vs.sa_ss[2:-2, 2:-2, :, :], 0)
            * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
        )
        vs.csa_s = update(
            vs.csa_s,
            at[2:-2, 2:-2, :, :],
            npx.where(vs.sa_s[2:-2, 2:-2, :, :] > 0, vs.msa_s[2:-2, 2:-2, :, :] / vs.sa_s[2:-2, 2:-2, :, :], 0)
            * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
        )

    return KernelOutput(
        C_rz=vs.C_rz,
        C_ss=vs.C_ss,
        C_s=vs.C_s,
        msa_rz=vs.msa_rz,
        msa_ss=vs.msa_ss,
        msa_s=vs.msa_s,
        sa_rz=vs.sa_rz,
        sa_ss=vs.sa_ss,
        sa_s=vs.sa_s,
        SA_rz=vs.SA_rz,
        SA_ss=vs.SA_ss,
        SA_s=vs.SA_s,
        csa_rz=vs.csa_rz,
        csa_ss=vs.csa_ss,
        csa_s=vs.csa_s,
    )


@roger_routine
def rescale_SA(state):
    """
    Rescale StorAge after warmup.
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(rescale_sa_msa_iso_soil_kernel(state))
    elif settings.enable_offline_transport and (
        settings.enable_bromide | settings.enable_chloride | settings.enable_nitrate | settings.enable_virtualtracer
    ):
        vs.update(rescale_sa_msa_anion_soil_kernel(state))
    elif settings.enable_offline_transport and not (
        settings.enable_oxygen18
        | settings.enable_deuterium
        | settings.enable_bromide
        | settings.enable_chloride
        | settings.enable_nitrate
        | settings.enable_virtualtracer
    ):
        vs.update(rescale_sa_soil_kernel(state))
