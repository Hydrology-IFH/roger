from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at
from roger.core import transport


@roger_kernel
def calc_S_zsat(state):
    """
    Calculates storage of saturation water level
    """
    vs = state.variables

    vs.S_zsat = update(
        vs.S_zsat,
        at[:, :], vs.z_sat[:, :, vs.tau] * vs.theta_ac * vs.maskCatch,
    )

    return KernelOutput(S_zsat=vs.S_zsat)


@roger_kernel
def calc_z_sat_layer(state):
    """
    Calculates saturation depth
    """
    vs = state.variables

    z_sat_layer_1 = allocate(state.dimensions, ("x", "y"))
    vs.z_sat_layer_1 = update(
        vs.z_sat_layer_1,
        at[:, :, vs.tau], z_sat_layer_1 * vs.maskCatch,
    )
    vs.z_sat_layer_1 = update(
        vs.z_sat_layer_1,
        at[:, :, vs.tau], vs.z_sat[:, :, vs.tau] * vs.maskCatch,
    )
    vs.z_sat_layer_1 = update(
        vs.z_sat_layer_1,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_1[:, :, vs.tau] > 200, 200, vs.z_sat_layer_1[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.z_sat_layer_1 = update(
        vs.z_sat_layer_1,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_1[:, :, vs.tau] <= 0, 0, vs.z_sat_layer_1[:, :, vs.tau]) * vs.maskCatch,
    )

    z_sat_layer_2 = allocate(state.dimensions, ("x", "y"))
    vs.z_sat_layer_2 = update(
        vs.z_sat_layer_2,
        at[:, :, vs.tau], z_sat_layer_2 * vs.maskCatch,
    )
    vs.z_sat_layer_2 = update(
        vs.z_sat_layer_2,
        at[:, :, vs.tau], vs.z_sat[:, :, vs.tau] - 200 * vs.maskCatch,
    )
    vs.z_sat_layer_2 = update(
        vs.z_sat_layer_2,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_2[:, :, vs.tau] > 200, 200, vs.z_sat_layer_2[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.z_sat_layer_2 = update(
        vs.z_sat_layer_2,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_2[:, :, vs.tau] <= 0, 0, vs.z_sat_layer_2[:, :, vs.tau]) * vs.maskCatch,
    )

    z_sat_layer_3 = allocate(state.dimensions, ("x", "y"))
    vs.z_sat_layer_3 = update(
        vs.z_sat_layer_3,
        at[:, :, vs.tau], z_sat_layer_3 * vs.maskCatch,
    )
    vs.z_sat_layer_3 = update(
        vs.z_sat_layer_3,
        at[:, :, vs.tau], vs.z_sat[:, :, vs.tau] - 400 * vs.maskCatch,
    )
    vs.z_sat_layer_3 = update(
        vs.z_sat_layer_3,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_3[:, :, vs.tau] > 200, 200, vs.z_sat_layer_3[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.z_sat_layer_3 = update(
        vs.z_sat_layer_3,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_3[:, :, vs.tau] <= 0, 0, vs.z_sat_layer_3[:, :, vs.tau]) * vs.maskCatch,
    )

    z_sat_layer_4 = allocate(state.dimensions, ("x", "y"))
    vs.z_sat_layer_4 = update(
        vs.z_sat_layer_4,
        at[:, :, vs.tau], z_sat_layer_4 * vs.maskCatch,
    )
    vs.z_sat_layer_4 = update(
        vs.z_sat_layer_4,
        at[:, :, vs.tau], vs.z_sat[:, :, vs.tau] - 600 * vs.maskCatch,
    )
    vs.z_sat_layer_4 = update(
        vs.z_sat_layer_4,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_4[:, :, vs.tau] > 200, 200, vs.z_sat_layer_4[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.z_sat_layer_4 = update(
        vs.z_sat_layer_4,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_4[:, :, vs.tau] <= 0, 0, vs.z_sat_layer_4[:, :, vs.tau]) * vs.maskCatch,
    )

    z_sat_layer_5 = allocate(state.dimensions, ("x", "y"))
    vs.z_sat_layer_5 = update(
        vs.z_sat_layer_5,
        at[:, :, vs.tau], z_sat_layer_5 * vs.maskCatch,
    )
    vs.z_sat_layer_5 = update(
        vs.z_sat_layer_5,
        at[:, :, vs.tau], vs.z_sat[:, :, vs.tau] - 800 * vs.maskCatch,
    )
    vs.z_sat_layer_5 = update(
        vs.z_sat_layer_5,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_5[:, :, vs.tau] > 200, 200, vs.z_sat_layer_5[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.z_sat_layer_5 = update(
        vs.z_sat_layer_5,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_5[:, :, vs.tau] <= 0, 0, vs.z_sat_layer_5[:, :, vs.tau]) * vs.maskCatch,
    )

    z_sat_layer_6 = allocate(state.dimensions, ("x", "y"))
    vs.z_sat_layer_6 = update(
        vs.z_sat_layer_6,
        at[:, :, vs.tau], z_sat_layer_6 * vs.maskCatch,
    )
    vs.z_sat_layer_6 = update(
        vs.z_sat_layer_6,
        at[:, :, vs.tau], vs.z_sat[:, :, vs.tau] - 1000 * vs.maskCatch,
    )
    vs.z_sat_layer_6 = update(
        vs.z_sat_layer_6,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_6[:, :, vs.tau] > 200, 200, vs.z_sat_layer_6[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.z_sat_layer_6 = update(
        vs.z_sat_layer_6,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_6[:, :, vs.tau] <= 0, 0, vs.z_sat_layer_6[:, :, vs.tau]) * vs.maskCatch,
    )

    z_sat_layer_7 = allocate(state.dimensions, ("x", "y"))
    vs.z_sat_layer_7 = update(
        vs.z_sat_layer_7,
        at[:, :, vs.tau], z_sat_layer_7 * vs.maskCatch,
    )
    vs.z_sat_layer_7 = update(
        vs.z_sat_layer_7,
        at[:, :, vs.tau], vs.z_sat[:, :, vs.tau] - 1200 * vs.maskCatch,
    )
    vs.z_sat_layer_7 = update(
        vs.z_sat_layer_7,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_7[:, :, vs.tau] > 200, 200, vs.z_sat_layer_7[:, :, vs.tau]) * vs.maskCatch,
    )
    vs.z_sat_layer_7 = update(
        vs.z_sat_layer_7,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_7[:, :, vs.tau] <= 0, 0, vs.z_sat_layer_7[:, :, vs.tau]) * vs.maskCatch,
    )

    z_sat_layer_8 = allocate(state.dimensions, ("x", "y"))
    vs.z_sat_layer_8 = update(
        vs.z_sat_layer_8,
        at[:, :, vs.tau], z_sat_layer_8 * vs.maskCatch,
    )
    vs.z_sat_layer_8 = update(
        vs.z_sat_layer_8,
        at[:, :, vs.tau], vs.z_sat[:, :, vs.tau] - 1400 * vs.maskCatch,
    )
    vs.z_sat_layer_8 = update(
        vs.z_sat_layer_8,
        at[:, :, vs.tau], npx.where(vs.z_sat_layer_8[:, :, vs.tau] <= 0, 0, vs.z_sat_layer_8[:, :, vs.tau]) * vs.maskCatch,
    )

    return KernelOutput(z_sat_layer_1=vs.z_sat_layer_1,
                        z_sat_layer_2=vs.z_sat_layer_2,
                        z_sat_layer_3=vs.z_sat_layer_3,
                        z_sat_layer_4=vs.z_sat_layer_4,
                        z_sat_layer_5=vs.z_sat_layer_5,
                        z_sat_layer_6=vs.z_sat_layer_6,
                        z_sat_layer_7=vs.z_sat_layer_7,
                        z_sat_layer_8=vs.z_sat_layer_8,
                        )


@roger_kernel
def calc_q_sub_pot(state):
    """
    Calculates potential lateral subsurface runoff
    """
    vs = state.variables
    settings = state.settings

    # potential matrix subsurface runoff (in mm/dt)
    vs.q_sub_mat_pot = update(
        vs.q_sub_mat_pot,
        at[:, :], ((vs.ks * vs.slope * vs.z_sat[:, :, vs.tau] * vs.dt) * 1e-3) * vs.maskCatch,
    )

    # total potential macropore subsurface runoff (in mm/dt)
    # convert mm3 to mm (1e-6)
    vs.q_sub_mp_pot = update(
        vs.q_sub_mp_pot,
        at[:, :], ((vs.z_sat_layer_1[:, :, vs.tau] * vs.v_mp_layer_1 +
                    vs.z_sat_layer_2[:, :, vs.tau] * vs.v_mp_layer_2 +
                    vs.z_sat_layer_3[:, :, vs.tau] * vs.v_mp_layer_3 +
                    vs.z_sat_layer_4[:, :, vs.tau] * vs.v_mp_layer_4 +
                    vs.z_sat_layer_5[:, :, vs.tau] * vs.v_mp_layer_5 +
                    vs.z_sat_layer_6[:, :, vs.tau] * vs.v_mp_layer_6 +
                    vs.z_sat_layer_7[:, :, vs.tau] * vs.v_mp_layer_7 +
                    vs.z_sat_layer_8[:, :, vs.tau] * vs.v_mp_layer_8) * vs.dt * vs.dmph * settings.r_mp**2 * settings.pi * 1e-9) * vs.maskCatch,
    )
    vs.q_sub_mp_pot = update(
        vs.q_sub_mp_pot,
        at[:, :], npx.where(vs.q_sub_mp_pot < 0, 0, vs.q_sub_mp_pot) * vs.maskCatch,
    )

    # potential lateral subsurface runoff
    vs.q_sub_pot = update(
        vs.q_sub_pot,
        at[:, :], (vs.q_sub_mp_pot + vs.q_sub_mat_pot) * vs.maskCatch,
    )

    # contribution of subsurface runoff components
    vs.q_sub_mat_share = update(
        vs.q_sub_mat_share,
        at[:, :], (vs.q_sub_mat_pot/vs.q_sub_pot) * vs.maskCatch,
    )
    vs.q_sub_mat_share = update(
        vs.q_sub_mat_share,
        at[:, :], npx.where(vs.q_sub_pot == 0, 0, vs.q_sub_mat_share) * vs.maskCatch,
    )

    vs.q_sub_mp_share = update(
        vs.q_sub_mp_share,
        at[:, :], (vs.q_sub_mp_pot/vs.q_sub_pot) * vs.maskCatch,
    )
    vs.q_sub_mp_share = update(
        vs.q_sub_mp_share,
        at[:, :], npx.where(vs.q_sub_pot == 0, 0, vs.q_sub_mp_share) * vs.maskCatch,
    )

    # constraining subsurface runoff to water in large pores
    mask1 = (vs.S_zsat < vs.q_sub_pot) & (vs.S_zsat > 0)
    vs.q_sub_pot = update(
        vs.q_sub_pot,
        at[:, :], npx.where(mask1, vs.S_zsat, vs.q_sub_pot) * vs.maskCatch,
    )

    # lateral matrix subsurface runoff
    vs.q_sub_mat_pot = update(
        vs.q_sub_mat_pot,
        at[:, :], vs.q_sub_pot * vs.q_sub_mat_share * vs.maskCatch,
    )
    # lateral macropore subsurface runoff
    vs.q_sub_mp_pot = update(
        vs.q_sub_mp_pot,
        at[:, :], vs.q_sub_pot * vs.q_sub_mp_share * vs.maskCatch,
    )

    return KernelOutput(q_sub_mat_pot=vs.q_sub_mat_pot, q_sub_mp_pot=vs.q_sub_mp_pot, q_sub_pot=vs.q_sub_pot, q_sub_mp_share=vs.q_sub_mp_share, q_sub_mat_share=vs.q_sub_mat_share)


@roger_kernel
def calc_q_sub_rz(state):
    """
    Calculates lateral subsurface runoff in root zone
    """
    vs = state.variables

    # root zone share for subsurface runoff
    rz_share = allocate(state.dimensions, ("x", "y"))
    rz_share = update(
        rz_share,
        at[:, :], ((vs.z_sat[:, :, vs.tau] - (vs.z_soil - vs.z_root[:, :, vs.tau])) / vs.z_sat[:, :, vs.tau]) * vs.maskCatch,
    )
    mask1 = (vs.z_sat[:, :, vs.tau] <= vs.z_soil - vs.z_root[:, :, vs.tau])
    rz_share = update(
        rz_share,
        at[:, :], npx.where(mask1, 0, rz_share) * vs.maskCatch,
    )

    vs.S_zsat_rz = update(
        vs.S_zsat_rz,
        at[:, :], ((vs.z_sat[:, :, vs.tau] * rz_share) / vs.theta_ac) * vs.maskCatch,
    )

    vs.q_sub_rz = update(
        vs.q_sub_rz,
        at[:, :], npx.where(vs.q_sub_pot * rz_share < vs.S_zsat_rz, vs.q_sub_pot * rz_share, vs.S_zsat_rz) * vs.maskCatch,
    )

    # root zone subsurface runoff
    vs.q_sub_mat_rz = update(
        vs.q_sub_mat_rz,
        at[:, :], vs.q_sub_rz * vs.q_sub_mat_share * vs.maskCatch,
    )

    vs.q_sub_mp_rz = update(
        vs.q_sub_mp_rz,
        at[:, :], vs.q_sub_rz * vs.q_sub_mp_share * vs.maskCatch,
    )

    vs.q_sub_mp_pot_rz = update(
        vs.q_sub_mp_pot_rz,
        at[:, :], vs.q_sub_mp_pot * rz_share * vs.maskCatch,
    )

    # update subsoil saturation water level
    vs.z_sat = update_add(
        vs.z_sat,
        at[:, :, vs.tau], -vs.q_sub_rz/vs.theta_ac * vs.maskCatch,
    )

    # update subsoil storage
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[:, :], -vs.q_sub_rz * vs.maskCatch,
    )

    return KernelOutput(q_sub_mp_rz=vs.q_sub_mp_rz, q_sub_mat_rz=vs.q_sub_mat_rz, q_sub_rz=vs.q_sub_rz, z_sat=vs.z_sat, S_lp_rz=vs.S_lp_rz, S_zsat_rz=vs.S_zsat_rz)


@roger_kernel
def calc_q_sub_pot_ss(state):
    """
    Calculates potential lateral subsurface runoff in subsoil
    """
    vs = state.variables

    # subsoil share for subsurface runoff
    ss_share = allocate(state.dimensions, ("x", "y"))
    ss_share = update(
        ss_share,
        at[:, :], ((vs.z_soil - vs.z_root[:, :, vs.tau]) / vs.z_sat[:, :, vs.tau]) * vs.maskCatch,
    )
    mask1 = (vs.z_sat[:, :, vs.tau] <= vs.z_soil - vs.z_root[:, :, vs.tau])
    mask2 = (vs.z_sat[:, :, vs.tau] <= 0)
    ss_share = update(
        ss_share,
        at[:, :], npx.where(mask1, 1, ss_share) * vs.maskCatch,
    )
    ss_share = update(
        ss_share,
        at[:, :], npx.where(mask2, 0, ss_share) * vs.maskCatch,
    )

    # subsoil subsurface runoff
    vs.q_sub_mat_pot_ss = update(
        vs.q_sub_mat_pot_ss,
        at[:, :], vs.q_sub_mat_pot * ss_share * vs.maskCatch,
    )

    vs.q_sub_mp_pot_ss = update(
        vs.q_sub_mp_pot_ss,
        at[:, :], vs.q_sub_mp_pot * ss_share * vs.maskCatch,
    )

    vs.q_sub_pot_ss = update(
        vs.q_sub_pot_ss,
        at[:, :], (vs.q_sub_mat_pot_ss + vs.q_sub_mp_pot_ss) * vs.maskCatch,
    )

    return KernelOutput(q_sub_mp_pot_ss=vs.q_sub_mp_pot_ss, q_sub_mat_pot_ss=vs.q_sub_mat_pot_ss, q_sub_pot_ss=vs.q_sub_pot_ss, S_zsat_rz=vs.S_zsat_rz)


@roger_kernel
def calc_q_sub_ss(state):
    """
    Calculates lateral and vertical subsurface runoff in subsoil
    """
    vs = state.variables

    # fraction of lateral and vertical flow
    fv = allocate(state.dimensions, ("x", "y"))
    fl = allocate(state.dimensions, ("x", "y"))
    fv = update(
        fv,
        at[:, :], npx.where((vs.q_pot_ss + vs.q_sub_pot_ss) > 0, vs.q_pot_ss / (vs.q_pot_ss + vs.q_sub_pot_ss), 0) * vs.maskCatch,
    )
    fl = update(
        fl,
        at[:, :], npx.where((vs.q_pot_ss + vs.q_sub_pot_ss) > 0, vs.q_sub_pot_ss / (vs.q_pot_ss + vs.q_sub_pot_ss), 0) * vs.maskCatch,
    )

    # vertical flow
    vs.q_ss = update(
        vs.q_ss,
        at[:, :], 0,
    )
    vs.q_ss = update(
        vs.q_ss,
        at[:, :], npx.where((vs.q_pot_ss + vs.q_sub_pot_ss) <= vs.S_zsat, (vs.q_pot_ss + vs.q_sub_pot_ss) * fv, vs.S_zsat * fv) * vs.maskCatch,
    )

    # lateral flow
    q_sub_ss = allocate(state.dimensions, ("x", "y"))
    vs.q_sub_ss = update(
        vs.q_sub_ss,
        at[:, :], q_sub_ss,
    )
    vs.q_sub_ss = update(
        vs.q_sub_ss,
        at[:, :], npx.where((vs.q_pot_ss + vs.q_sub_pot_ss) <= vs.S_zsat, (vs.q_pot_ss + vs.q_sub_pot_ss) * fl, vs.S_zsat * fl) * vs.maskCatch,
    )
    vs.q_sub_mat_ss = update(
        vs.q_sub_mat_ss,
        at[:, :], vs.q_sub_ss * vs.q_sub_mat_share * vs.maskCatch,
    )
    vs.q_sub_mp_ss = update(
        vs.q_sub_mp_ss,
        at[:, :], vs.q_sub_ss * vs.q_sub_mp_share * vs.maskCatch,
    )

    # update subsoil saturation water level
    vs.z_sat = update_add(
        vs.z_sat,
        at[:, :, vs.tau], -((vs.q_sub_ss/vs.theta_ac) + (vs.q_ss/vs.theta_ac)) * vs.maskCatch,
    )

    # update subsoil storage
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[:, :], -(vs.q_sub_ss + vs.q_ss) * vs.maskCatch,
    )

    return KernelOutput(q_ss=vs.q_ss, q_sub_ss=vs.q_sub_ss, q_sub_mat_ss=vs.q_sub_mat_ss, q_sub_mp_ss=vs.q_sub_mp_ss, z_sat=vs.z_sat, S_lp_ss=vs.S_lp_ss)


@roger_kernel
def calc_q_sub(state):
    """
    Calculates lateral and vertical subsurface runoff in subsoil
    """
    vs = state.variables

    vs.q_sub = update(
        vs.q_sub,
        at[:, :], (vs.q_sub_rz + vs.q_sub_ss) * vs.maskCatch,
    )

    return KernelOutput(q_sub=vs.q_sub)


@roger_kernel
def calc_dz_sat(state):
    """
    Calculates change of saturation water level in subsoil
    """
    vs = state.variables

    # root zone drainage which fills up large pore storae
    q_lp_in = allocate(state.dimensions, ("x", "y"))
    q_lp_in = update(
        q_lp_in,
        at[:, :], (vs.q_rz + vs.inf_ss - (vs.S_ufc_ss - vs.S_fp_ss)) * vs.maskCatch,
    )
    q_lp_in = update(
        q_lp_in,
        at[:, :], npx.where(q_lp_in < 0, 0, q_lp_in) * vs.maskCatch,
    )

    # vertical length of macropores reaching into subsoil
    lmpv_ss = allocate(state.dimensions, ("x", "y"))
    lmpv_ss = update(
        lmpv_ss,
        at[:, :], vs.lmpv - vs.z_root[:, :, vs.tau] * vs.maskCatch,
    )
    lmpv_ss = update(
        lmpv_ss,
        at[:, :], npx.where(vs.lmpv < vs.z_root[:, :, vs.tau], 0, lmpv_ss) * vs.maskCatch,
    )
    lmpv_ss = update(
        lmpv_ss,
        at[:, :], npx.where(vs.lmpv < vs.z_soil - vs.z_root[:, :, vs.tau], vs.z_soil - vs.z_root[:, :, vs.tau], lmpv_ss) * vs.maskCatch,
    )

    # saturation from top [mm]
    z_sat_top = allocate(state.dimensions, ("x", "y"))
    z_sat_top = update(
        z_sat_top,
        at[:, :], vs.S_lp_ss / vs.theta_ac * vs.maskCatch,
    )
    # vertical distance without macropores [mm]
    z_nomp = allocate(state.dimensions, ("x", "y"))
    z_nomp = update(
        z_nomp,
        at[:, :], (vs.z_soil - vs.z_root[:, :, vs.tau]) - lmpv_ss - vs.z_sat[:, :, vs.tau] * vs.maskCatch,
    )
    # non-saturated large pore storage [mm]
    S_nomp = allocate(state.dimensions, ("x", "y"))
    z_sat_top = update(
        z_sat_top,
        at[:, :], vs.theta_ac * z_nomp * vs.maskCatch,
    )
    # non-saturated distance [mm]
    z_ns = allocate(state.dimensions, ("x", "y"))
    z_ns = update(
        z_ns,
        at[:, :], z_nomp - z_sat_top * vs.maskCatch,
    )
    z_ns = update(
        z_ns,
        at[:, :], npx.where(z_ns < 0, 0, z_ns) * vs.maskCatch,
    )
    # vertical flow [mm/dt]
    qv = allocate(state.dimensions, ("x", "y"))
    qv = update(
        qv,
        at[:, :], ((vs.ks * vs.dt) - z_ns) * vs.maskCatch,
    )
    qv = update(
        qv,
        at[:, :], npx.where(qv < 0, vs.ks * vs.dt, qv) * vs.maskCatch,
    )
    qv = update(
        qv,
        at[:, :], npx.where(qv > vs.S_lp_ss, vs.S_lp_ss, qv) * vs.maskCatch,
    )

    # factor for vertical redistribution [-]
    f_vr = allocate(state.dimensions, ("x", "y"))
    f_vr = update(
        f_vr,
        at[:, :], npx.where(S_nomp != 0, vs.S_lp_ss / S_nomp, 1) * vs.maskCatch,
    )
    f_vr = update(
        f_vr,
        at[:, :], npx.where(f_vr > 1, 1, f_vr) * vs.maskCatch,
    )

    # change in saturation water level
    mask1 = (f_vr > 0) & (f_vr < 1) & (vs.ks * vs.dt < z_ns)
    mask2 = (f_vr > 0) & (f_vr < 1) & (vs.ks * vs.dt >= z_ns)
    mask3 = (f_vr >= 1) & (q_lp_in > 0)
    mask4 = (f_vr >= 1) & (q_lp_in <= 0)
    vs.dz_sat = update(
        vs.dz_sat,
        at[:, :], npx.where(mask1, (qv * f_vr) / vs.theta_ac, vs.dz_sat) * vs.maskCatch,
    )
    vs.dz_sat = update(
        vs.dz_sat,
        at[:, :], npx.where(mask2, qv / vs.theta_ac, vs.dz_sat) * vs.maskCatch,
    )
    vs.dz_sat = update(
        vs.dz_sat,
        at[:, :], npx.where(mask3, q_lp_in / vs.theta_ac, vs.dz_sat) * vs.maskCatch,
    )
    vs.dz_sat = update(
        vs.dz_sat,
        at[:, :], npx.where(mask4, 0, vs.dz_sat) * vs.maskCatch,
    )

    vs.z_sat = update(
        vs.z_sat,
        at[:, :, vs.tau], (vs.z_sat[:, :, vs.taum1] + vs.dz_sat) * vs.maskCatch,
    )

    return KernelOutput(dz_sat=vs.dz_sat, z_sat=vs.z_sat)


@roger_kernel
def calc_perc_pot_rz(state):
    """
    Calculates potential percolation in root zone
    """
    vs = state.variables

    # potential percolation rate
    perc_pot = allocate(state.dimensions, ("x", "y"))
    mask1 = (vs.z_wf[:, :, vs.tau] < vs.z_root[:, :, vs.tau])
    mask2 = (vs.z_wf[:, :, vs.tau] >= vs.z_root[:, :, vs.tau])
    perc_pot = update(
        perc_pot,
        at[:, :], npx.fmin(vs.ks * vs.dt, vs.k_rz[:, :, vs.tau] * vs.dt) * mask1 * vs.maskCatch,
    )
    perc_pot = update(
        perc_pot,
        at[:, :], npx.where(mask2, vs.ks * vs.dt, perc_pot) * vs.maskCatch,
    )

    # drainage of root zone
    mask3 = (perc_pot > 0) & (perc_pot <= vs.S_lp_rz) & (vs.z_root[:, :, vs.taum1] < vs.z_soil - vs.z_sat[:, :, vs.tau])
    mask4 = (perc_pot > 0) & (perc_pot > vs.S_lp_rz) & (vs.z_root[:, :, vs.taum1] < vs.z_soil - vs.z_sat[:, :, vs.tau])

    q_pot_rz = allocate(state.dimensions, ("x", "y"))
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[:, :], q_pot_rz,
    )
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[:, :], npx.where(mask3, perc_pot, vs.q_pot_rz) * vs.maskCatch,
    )
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[:, :], npx.where(mask4, vs.S_lp_rz, vs.q_pot_rz) * vs.maskCatch,
    )

    return KernelOutput(q_pot_rz=vs.q_pot_rz)


@roger_kernel
def calc_perc_rz(state):
    """
    Calculates percolation in root zone
    """
    vs = state.variables

    vs.q_rz = update(
        vs.q_rz,
        at[:, :], vs.q_pot_rz * vs.maskCatch,
    )

    vs.update(calc_dz_sat(state))

    # update root zone storage after root zone drainage
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[:, :], -vs.q_rz * vs.maskCatch,
    )

    # update subsoil storage after percolation
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[:, :], vs.q_rz * vs.maskCatch,
    )

    # subsoil fine pore excess fills subsoil large pores
    mask = (vs.S_fp_ss > vs.S_ufc_ss)
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[:, :], (vs.S_fp_ss - vs.S_ufc_ss) * mask * vs.maskCatch,
    )
    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[:, :], npx.where(mask, vs.S_ufc_ss, vs.S_fp_ss) * vs.maskCatch,
    )

    return KernelOutput(q_rz=vs.q_rz, S_lp_rz=vs.S_lp_rz, S_fp_ss=vs.S_fp_ss, S_lp_ss=vs.S_lp_ss, dz_sat=vs.dz_sat, z_sat=vs.z_sat)


@roger_kernel
def calc_perc_pot_ss(state):
    """
    Calculates potenital percolation in subsoil
    """
    vs = state.variables
    settings = state.settings

    # potential drainage rate
    # calculate potential percolation rate
    perc_pot = allocate(state.dimensions, ("x", "y"))
    perc_pot = update(
        perc_pot,
        at[:, :], npx.fmin(vs.kf * vs.dt, npx.where(vs.z_sat[:, :, vs.tau] > 0, vs.ks * vs.dt, npx.fmin(vs.ks * vs.dt, vs.k_ss[:, :, vs.tau] * vs.dt))) * vs.maskCatch,
    )

    # where drainage occurs
    if settings.enable_groundwater_boundary | settings.enable_groundwater:
        mask1 = (perc_pot > 0) & ((vs.S_zsat > 0) | (vs.z_soil < vs.z_gw[:, :, vs.tau])) & (perc_pot <= vs.S_zsat)
        mask2 = (perc_pot > 0) & ((vs.S_zsat > 0) | (vs.z_soil < vs.z_gw[:, :, vs.tau])) & (perc_pot > vs.S_zsat)
    else:
        mask1 = (perc_pot > 0) & (vs.S_zsat > 0) & (perc_pot <= vs.S_zsat)
        mask2 = (perc_pot > 0) & (vs.S_zsat > 0) & (perc_pot > vs.S_zsat)

    # vertical drainage of subsoil
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[:, :], 0,
    )
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[:, :], npx.where(mask1, perc_pot, vs.q_pot_ss) * vs.maskCatch,
    )
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[:, :], npx.where(mask2, vs.S_zsat, vs.q_pot_ss) * vs.maskCatch,
    )

    return KernelOutput(q_pot_ss=vs.q_pot_ss)


@roger_kernel
def calc_perc_ss(state):
    """
    Calculates percolation in subsoil
    """
    vs = state.variables

    vs.q_ss = update(
        vs.q_ss,
        at[:, :], vs.q_pot_ss * vs.maskCatch,
    )

    # update saturation water level
    vs.z_sat = update_add(
        vs.z_sat,
        at[:, :, vs.tau], -vs.q_ss/vs.theta_ac * vs.maskCatch,
    )

    vs.S_zsat = update(
        vs.S_zsat,
        at[:, :], vs.z_sat[:, :, vs.tau] * vs.theta_ac * vs.maskCatch,
    )

    # update root zone storage after root zone drainage
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[:, :], -vs.q_ss * vs.maskCatch,
    )

    return KernelOutput(q_ss=vs.q_ss, S_lp_ss=vs.S_lp_ss, z_sat=vs.z_sat, S_zsat=vs.S_zsat)


@roger_kernel
def calc_perc_pot_rz_ff(state):
    """
    Calculates potential percolation in root zone
    """
    vs = state.variables

    # potential percolation rate
    perc_pot = allocate(state.dimensions, ("x", "y"))
    perc_pot = update(
        perc_pot,
        at[:, :], npx.fmin(vs.ks * vs.dt, vs.k_rz[:, :, vs.tau] * vs.dt) * vs.maskCatch,
    )
    # drainage of root zone
    mask3 = (perc_pot > 0) & (perc_pot <= vs.S_lp_rz) & (vs.z_root[:, :, vs.taum1] < vs.z_soil - vs.z_sat[:, :, vs.tau])
    mask4 = (perc_pot > 0) & (perc_pot > vs.S_lp_rz) & (vs.z_root[:, :, vs.taum1] < vs.z_soil - vs.z_sat[:, :, vs.tau])

    q_pot_rz = allocate(state.dimensions, ("x", "y"))
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[:, :], q_pot_rz,
    )
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[:, :], npx.where(mask3, perc_pot, vs.q_pot_rz) * vs.maskCatch,
    )
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[:, :], npx.where(mask4, vs.S_lp_rz, vs.q_pot_rz) * vs.maskCatch,
    )

    return KernelOutput(q_pot_rz=vs.q_pot_rz)


@roger_kernel
def calc_perc_rz_ff(state):
    """
    Calculates percolation in root zone
    """
    vs = state.variables

    vs.q_rz = update(
        vs.q_rz,
        at[:, :], vs.q_pot_rz * vs.maskCatch,
    )

    # update root zone storage after root zone drainage
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[:, :], -vs.q_rz * vs.maskCatch,
    )

    # update subsoil storage after percolation
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[:, :], vs.q_rz * vs.maskCatch,
    )

    # subsoil fine pore excess fills subsoil large pores
    mask = (vs.S_fp_ss > vs.S_ufc_ss)
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[:, :], (vs.S_fp_ss - vs.S_ufc_ss) * mask * vs.maskCatch,
    )
    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[:, :], npx.where(mask, vs.S_ufc_ss, vs.S_fp_ss) * vs.maskCatch,
    )

    return KernelOutput(q_rz=vs.q_rz, S_lp_rz=vs.S_lp_rz, S_fp_ss=vs.S_fp_ss, S_lp_ss=vs.S_lp_ss)


@roger_kernel
def calc_perc_pot_ss_ff(state):
    """
    Calculates potenital percolation in subsoil
    """
    vs = state.variables
    settings = state.settings

    # potential drainage rate
    # calculate potential percolation rate
    perc_pot = allocate(state.dimensions, ("x", "y"))
    perc_pot = update(
        perc_pot,
        at[:, :], npx.fmin(vs.kf * vs.dt, npx.where(vs.S_lp_ss > 0, vs.ks * vs.dt, npx.fmin(vs.ks * vs.dt, vs.k_ss[:, :, vs.tau] * vs.dt))) * vs.maskCatch,
    )

    # where drainage occurs
    if settings.enable_groundwater_boundary | settings.enable_groundwater:
        mask1 = (perc_pot > 0) & ((vs.S_lp_ss > 0) | (vs.z_soil < vs.z_gw[:, :, vs.tau])) & (perc_pot <= vs.S_lp_ss)
        mask2 = (perc_pot > 0) & ((vs.S_lp_ss > 0) | (vs.z_soil < vs.z_gw[:, :, vs.tau])) & (perc_pot > vs.S_lp_ss)
    else:
        mask1 = (perc_pot > 0) & (vs.S_lp_ss > 0) & (perc_pot <= vs.S_lp_ss)
        mask2 = (perc_pot > 0) & (vs.S_lp_ss > 0) & (perc_pot > vs.S_lp_ss)

    # vertical drainage of subsoil
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[:, :], 0,
    )
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[:, :], npx.where(mask1, perc_pot, vs.q_pot_ss) * vs.maskCatch,
    )
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[:, :], npx.where(mask2, vs.S_lp_ss, vs.q_pot_ss) * vs.maskCatch,
    )

    return KernelOutput(q_pot_ss=vs.q_pot_ss)


@roger_kernel
def calc_perc_ss_ff(state):
    """
    Calculates percolation in subsoil
    """
    vs = state.variables

    vs.q_ss = update(
        vs.q_ss,
        at[:, :], vs.q_pot_ss * vs.maskCatch,
    )

    # update root zone storage after root zone drainage
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[:, :], -vs.q_ss * vs.maskCatch,
    )

    return KernelOutput(q_ss=vs.q_ss, S_lp_ss=vs.S_lp_ss)


@roger_routine
def calculate_subsurface_runoff(state):
    """
    Calculates subsurface runoff
    """
    vs = state.variables
    settings = state.settings

    vs = state.variables
    settings = state.settings

    if settings.enable_lateral_flow:
        vs.update(calc_perc_pot_rz(state))
        vs.update(calc_perc_rz(state))
        vs.update(calc_S_zsat(state))
        vs.update(calc_z_sat_layer(state))
        vs.update(calc_q_sub_pot(state))
        vs.update(calc_q_sub_rz(state))
        vs.update(calc_q_sub_pot_ss(state))
        vs.update(calc_perc_pot_ss(state))
        vs.update(calc_q_sub_ss(state))
        vs.update(calc_q_sub(state))

    elif not settings.enable_lateral_flow:
        vs.update(calc_perc_pot_rz(state))
        vs.update(calc_perc_rz(state))
        vs.update(calc_S_zsat(state))
        vs.update(calc_perc_pot_ss(state))
        vs.update(calc_perc_ss(state))

    elif not settings.enable_lateral_flow and settings.enable_film_flow:
        vs.update(calc_perc_pot_rz_ff(state))
        vs.update(calc_perc_rz_ff(state))
        vs.update(calc_perc_pot_ss_ff(state))
        vs.update(calc_perc_ss_ff(state))


@roger_kernel
def calculate_percolation_rz_transport_kernel(state):
    """
    Calculates travel time of percolation
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[:, :, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.q_rz, vs.sas_params_q_rz) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[:, :, 1:], npx.cumsum(vs.tt_q_rz, axis=2),
    )

    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[:, :, vs.tau, :], vs.tt_q_rz * vs.q_rz[:, :, npx.newaxis] * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_q_rz=vs.tt_q_rz, TT_q_rz=vs.TT_q_rz, sa_ss=vs.sa_ss)


@roger_kernel
def calculate_percolation_rz_transport_iso_kernel(state):
    """
    Calculates isotope transport of percolation
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[:, :, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.q_rz, vs.sas_params_q_rz) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[:, :, 1:], npx.cumsum(vs.tt_q_rz, axis=2),
    )

    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[:, :, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz, vs.msa_rz, alpha) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.C_q_rz = update(
        vs.C_q_rz,
        at[:, :], transport.calc_conc_iso_flux(state, vs.mtt_q_rz, vs.tt_q_rz) * vs.maskCatch,
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge
    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, vs.tau, :], npx.where((vs.sa_rz[:, :, vs.tau, :] > 0), vs.msa_rz[:, :, vs.tau, :], npx.NaN) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.msa_ss = update(
        vs.msa_ss,
        at[:, :, :, :], transport.calc_msa_iso(state, vs.sa_ss, vs.msa_ss, vs.q_rz, vs.tt_q_rz, vs.mtt_q_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[:, :, vs.tau, :], vs.tt_q_rz * vs.q_rz[:, :, npx.newaxis] * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_q_rz=vs.tt_q_rz, TT_q_rz=vs.TT_q_rz, msa_rz=vs.msa_rz, mtt_q_rz=vs.mtt_q_rz, C_q_rz=vs.C_q_rz, sa_ss=vs.sa_ss, msa_ss=vs.msa_ss)


@roger_kernel
def calculate_percolation_rz_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of percolation
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[:, :, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.q_rz, vs.sas_params_q_rz) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[:, :, 1:], npx.cumsum(vs.tt_q_rz, axis=2),
    )

    # calculate isotope travel time distribution
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[:, :, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz, vs.msa_rz, vs.alpha_q) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.C_q_rz = update(
        vs.C_q_rz,
        at[:, :], npx.where(vs.q_rz > 0, npx.sum(vs.mtt_q_rz, axis=2) / vs.q_rz, 0) * vs.maskCatch,
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[:, :, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge of root zone
    vs.msa_rz = update(
        vs.msa_rz,
        at[:, :, vs.tau, :], vs.msa_rz[:, :, vs.tau, :] - vs.mtt_q_rz * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.msa_ss = update_add(
        vs.msa_ss,
        at[:, :, vs.tau, :], vs.mtt_q_rz * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[:, :, vs.tau, :], vs.tt_q_rz * vs.q_rz[:, :, npx.newaxis] * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_q_rz=vs.tt_q_rz, TT_q_rz=vs.TT_q_rz, msa_rz=vs.msa_rz, mtt_q_rz=vs.mtt_q_rz, C_q_rz=vs.C_q_rz, sa_ss=vs.sa_ss, msa_ss=vs.msa_ss)


@roger_kernel
def calculate_percolation_ss_transport_kernel(state):
    """
    Calculates travel time of percolation
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[:, :, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.q_ss, vs.sas_params_q_ss) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[:, :, 1:], npx.cumsum(vs.tt_q_ss, axis=2),
    )

    vs.sa_ss = update(
        vs.sa_ss,
        at[:, :, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_q_ss=vs.tt_q_ss, TT_q_ss=vs.TT_q_ss)


@roger_kernel
def calculate_percolation_ss_transport_iso_kernel(state):
    """
    Calculates isotope transport of percolation
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[:, :, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.q_ss, vs.sas_params_q_ss) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[:, :, 1:], npx.cumsum(vs.tt_q_ss, axis=2),
    )

    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[:, :, :], transport.calc_mtt(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss, vs.msa_ss, alpha) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.C_q_ss = update(
        vs.C_q_ss,
        at[:, :], transport.calc_conc_iso_flux(state, vs.mtt_q_ss, vs.tt_q_ss) * vs.maskCatch,
    )

    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[:, :, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge
    vs.msa_ss = update(
        vs.msa_ss,
        at[:, :, vs.tau, :], npx.where((vs.sa_ss[:, :, vs.tau, :] > 0), vs.msa_ss[:, :, vs.tau, :], npx.NaN) * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_q_ss=vs.tt_q_ss, TT_q_ss=vs.TT_q_ss, msa_ss=vs.msa_ss, mtt_q_ss=vs.mtt_q_ss, C_q_ss=vs.C_q_ss)


@roger_kernel
def calculate_percolation_ss_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of percolation
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[:, :, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[:, :, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.q_ss, vs.sas_params_q_ss) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[:, :, 1:], npx.cumsum(vs.tt_q_ss, axis=2),
    )

    # calculate isotope travel time distribution
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[:, :, :], transport.calc_mtt(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss, vs.msa_ss, vs.alpha_q) * vs.maskCatch[:, :, npx.newaxis],
    )

    vs.C_q_ss = update(
        vs.C_q_ss,
        at[:, :], npx.where(vs.q_ss > 0, npx.sum(vs.mtt_q_ss, axis=2) / vs.q_ss, 0) * vs.maskCatch,
    )

    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[:, :, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss) * vs.maskCatch[:, :, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge of root zone
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[:, :, vs.tau, :], -vs.mtt_q_ss * vs.maskCatch[:, :, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_q_ss=vs.tt_q_ss, TT_q_ss=vs.TT_q_ss, msa_ss=vs.msa_ss, mtt_q_ss=vs.mtt_q_ss, C_q_ss=vs.C_q_ss)


@roger_routine
def calculate_percolation_rz_transport(state):
    """
    Calculates percolation transport
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_percolation_rz_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_percolation_rz_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_percolation_rz_transport_anion_kernel(state))


@roger_routine
def calculate_percolation_ss_transport(state):
    """
    Calculates percolation transport
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calculate_percolation_ss_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calculate_percolation_ss_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calculate_percolation_ss_transport_anion_kernel(state))
