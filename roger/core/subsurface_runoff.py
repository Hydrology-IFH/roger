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
        at[2:-2, 2:-2], npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] <= vs.z_soil[2:-2, 2:-2], vs.z_sat[2:-2, 2:-2, vs.tau] * vs.theta_ac[2:-2, 2:-2], vs.z_soil[2:-2, 2:-2] * vs.theta_ac[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_zsat_ss = update(
        vs.S_zsat_ss,
        at[2:-2, 2:-2], npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] <= vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau], vs.S_zsat[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]) * vs.theta_ac[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_zsat_rz = update(
        vs.S_zsat_rz,
        at[2:-2, 2:-2], npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] > vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau], (vs.z_sat[2:-2, 2:-2, vs.tau] - (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) * vs.theta_ac[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(S_zsat=vs.S_zsat, S_zsat_rz=vs.S_zsat_rz, S_zsat_ss=vs.S_zsat_ss)


@roger_kernel
def calc_z_sat_layer(state):
    """
    Calculates saturation depth
    """
    vs = state.variables

    vs.z_sat_layer_1 = update(
        vs.z_sat_layer_1,
        at[2:-2, 2:-2, vs.tau], 0,
    )
    vs.z_sat_layer_1 = update(
        vs.z_sat_layer_1,
        at[2:-2, 2:-2, vs.tau], vs.z_sat[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_1 = update(
        vs.z_sat_layer_1,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_1[2:-2, 2:-2, vs.tau] > 200, 200, vs.z_sat_layer_1[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_1 = update(
        vs.z_sat_layer_1,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_1[2:-2, 2:-2, vs.tau] <= 0, 0, vs.z_sat_layer_1[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sat_layer_2 = update(
        vs.z_sat_layer_2,
        at[2:-2, 2:-2, vs.tau], 0,
    )
    vs.z_sat_layer_2 = update(
        vs.z_sat_layer_2,
        at[2:-2, 2:-2, vs.tau], vs.z_sat[2:-2, 2:-2, vs.tau] - 200 * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_2 = update(
        vs.z_sat_layer_2,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_2[2:-2, 2:-2, vs.tau] > 200, 200, vs.z_sat_layer_2[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_2 = update(
        vs.z_sat_layer_2,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_2[2:-2, 2:-2, vs.tau] <= 0, 0, vs.z_sat_layer_2[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sat_layer_3 = update(
        vs.z_sat_layer_3,
        at[2:-2, 2:-2, vs.tau], 0,
    )
    vs.z_sat_layer_3 = update(
        vs.z_sat_layer_3,
        at[2:-2, 2:-2, vs.tau], vs.z_sat[2:-2, 2:-2, vs.tau] - 400 * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_3 = update(
        vs.z_sat_layer_3,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_3[2:-2, 2:-2, vs.tau] > 200, 200, vs.z_sat_layer_3[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_3 = update(
        vs.z_sat_layer_3,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_3[2:-2, 2:-2, vs.tau] <= 0, 0, vs.z_sat_layer_3[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sat_layer_4 = update(
        vs.z_sat_layer_4,
        at[2:-2, 2:-2, vs.tau], 0,
    )
    vs.z_sat_layer_4 = update(
        vs.z_sat_layer_4,
        at[2:-2, 2:-2, vs.tau], vs.z_sat[2:-2, 2:-2, vs.tau] - 600 * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_4 = update(
        vs.z_sat_layer_4,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_4[2:-2, 2:-2, vs.tau] > 200, 200, vs.z_sat_layer_4[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_4 = update(
        vs.z_sat_layer_4,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_4[2:-2, 2:-2, vs.tau] <= 0, 0, vs.z_sat_layer_4[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sat_layer_5 = update(
        vs.z_sat_layer_5,
        at[2:-2, 2:-2, vs.tau], 0,
    )
    vs.z_sat_layer_5 = update(
        vs.z_sat_layer_5,
        at[2:-2, 2:-2, vs.tau], vs.z_sat[2:-2, 2:-2, vs.tau] - 800 * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_5 = update(
        vs.z_sat_layer_5,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_5[2:-2, 2:-2, vs.tau] > 200, 200, vs.z_sat_layer_5[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_5 = update(
        vs.z_sat_layer_5,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_5[2:-2, 2:-2, vs.tau] <= 0, 0, vs.z_sat_layer_5[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sat_layer_6 = update(
        vs.z_sat_layer_6,
        at[2:-2, 2:-2, vs.tau], 0,
    )
    vs.z_sat_layer_6 = update(
        vs.z_sat_layer_6,
        at[2:-2, 2:-2, vs.tau], vs.z_sat[2:-2, 2:-2, vs.tau] - 1000 * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_6 = update(
        vs.z_sat_layer_6,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_6[2:-2, 2:-2, vs.tau] > 200, 200, vs.z_sat_layer_6[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_6 = update(
        vs.z_sat_layer_6,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_6[2:-2, 2:-2, vs.tau] <= 0, 0, vs.z_sat_layer_6[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sat_layer_7 = update(
        vs.z_sat_layer_7,
        at[2:-2, 2:-2, vs.tau], 0,
    )
    vs.z_sat_layer_7 = update(
        vs.z_sat_layer_7,
        at[2:-2, 2:-2, vs.tau], vs.z_sat[2:-2, 2:-2, vs.tau] - 1200 * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_7 = update(
        vs.z_sat_layer_7,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_7[2:-2, 2:-2, vs.tau] > 200, 200, vs.z_sat_layer_7[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_7 = update(
        vs.z_sat_layer_7,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_7[2:-2, 2:-2, vs.tau] <= 0, 0, vs.z_sat_layer_7[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sat_layer_8 = update(
        vs.z_sat_layer_8,
        at[2:-2, 2:-2, vs.tau], 0,
    )
    vs.z_sat_layer_8 = update(
        vs.z_sat_layer_8,
        at[2:-2, 2:-2, vs.tau], vs.z_sat[2:-2, 2:-2, vs.tau] - 1400 * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat_layer_8 = update(
        vs.z_sat_layer_8,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat_layer_8[2:-2, 2:-2, vs.tau] <= 0, 0, vs.z_sat_layer_8[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
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
def calc_potential_lateral_subsurface_runoff(state):
    """
    Calculates potential lateral subsurface runoff
    """
    vs = state.variables
    settings = state.settings

    # calculate potential matrix subsurface runoff with darcy (in mm/dt)
    # convert mm3 to mm (1e-6)
    vs.q_sub_mat_pot = update(
        vs.q_sub_mat_pot,
        at[2:-2, 2:-2], ((vs.ks[2:-2, 2:-2] * vs.slope[2:-2, 2:-2] * vs.z_sat[2:-2, 2:-2, vs.tau] * vs.dt) * 1e-6) * vs.maskCatch[2:-2, 2:-2],
    )

    # total potential macropore subsurface runoff (in mm/dt)
    # convert mm3 to mm (1e-9)
    vs.q_sub_mp_pot = update(
        vs.q_sub_mp_pot,
        at[2:-2, 2:-2], ((vs.z_sat_layer_1[2:-2, 2:-2, vs.tau] * vs.v_mp_layer_1[2:-2, 2:-2] +
                          vs.z_sat_layer_2[2:-2, 2:-2, vs.tau] * vs.v_mp_layer_2[2:-2, 2:-2] +
                          vs.z_sat_layer_3[2:-2, 2:-2, vs.tau] * vs.v_mp_layer_3[2:-2, 2:-2] +
                          vs.z_sat_layer_4[2:-2, 2:-2, vs.tau] * vs.v_mp_layer_4[2:-2, 2:-2] +
                          vs.z_sat_layer_5[2:-2, 2:-2, vs.tau] * vs.v_mp_layer_5[2:-2, 2:-2] +
                          vs.z_sat_layer_6[2:-2, 2:-2, vs.tau] * vs.v_mp_layer_6[2:-2, 2:-2] +
                          vs.z_sat_layer_7[2:-2, 2:-2, vs.tau] * vs.v_mp_layer_7[2:-2, 2:-2] +
                          vs.z_sat_layer_8[2:-2, 2:-2, vs.tau] * vs.v_mp_layer_8[2:-2, 2:-2]) * vs.dt * vs.dmph[2:-2, 2:-2] * settings.r_mp**2 * settings.pi * 1e-9) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_sub_mp_pot = update(
        vs.q_sub_mp_pot,
        at[2:-2, 2:-2], npx.where(vs.q_sub_mp_pot[2:-2, 2:-2] < 0, 0, vs.q_sub_mp_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # potential lateral subsurface runoff
    vs.q_sub_pot = update(
        vs.q_sub_pot,
        at[2:-2, 2:-2], (vs.q_sub_mp_pot[2:-2, 2:-2] + vs.q_sub_mat_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # contribution of subsurface runoff components
    vs.q_sub_mat_share = update(
        vs.q_sub_mat_share,
        at[2:-2, 2:-2], (vs.q_sub_mat_pot[2:-2, 2:-2]/vs.q_sub_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_sub_mat_share = update(
        vs.q_sub_mat_share,
        at[2:-2, 2:-2], npx.where(vs.q_sub_pot[2:-2, 2:-2] == 0, 0, vs.q_sub_mat_share[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sub_mp_share = update(
        vs.q_sub_mp_share,
        at[2:-2, 2:-2], (vs.q_sub_mp_pot[2:-2, 2:-2]/vs.q_sub_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_sub_mp_share = update(
        vs.q_sub_mp_share,
        at[2:-2, 2:-2], npx.where(vs.q_sub_pot[2:-2, 2:-2] == 0, 0, vs.q_sub_mp_share[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # constraining subsurface runoff to water in large pores
    mask1 = (vs.q_sub_pot > vs.S_lp_rz + vs.S_lp_ss)
    vs.q_sub_pot = update(
        vs.q_sub_pot,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], vs.S_lp_rz[2:-2, 2:-2] + vs.S_lp_ss[2:-2, 2:-2], vs.q_sub_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # lateral matrix subsurface runoff
    vs.q_sub_mat_pot = update(
        vs.q_sub_mat_pot,
        at[2:-2, 2:-2], vs.q_sub_pot[2:-2, 2:-2] * vs.q_sub_mat_share[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # lateral macropore subsurface runoff
    vs.q_sub_mp_pot = update(
        vs.q_sub_mp_pot,
        at[2:-2, 2:-2], vs.q_sub_pot[2:-2, 2:-2] * vs.q_sub_mp_share[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_sub_mat_pot=vs.q_sub_mat_pot, q_sub_mp_pot=vs.q_sub_mp_pot, q_sub_pot=vs.q_sub_pot, q_sub_mp_share=vs.q_sub_mp_share, q_sub_mat_share=vs.q_sub_mat_share)


@roger_kernel
def calc_lateral_subsurface_runoff_rz(state):
    """
    Calculates lateral subsurface runoff in root zone
    """
    vs = state.variables

    # root zone share for subsurface runoff
    rz_share = allocate(state.dimensions, ("x", "y"))
    rz_share = update(
        rz_share,
        at[2:-2, 2:-2], ((vs.z_sat[2:-2, 2:-2, vs.tau] - (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])) / vs.z_sat[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask1 = (vs.z_sat[:, :, vs.tau] <= vs.z_soil - vs.z_root[:, :, vs.tau]) | (vs.S_lp_rz <= 0)
    rz_share = update(
        rz_share,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], 0, rz_share[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_zsat_rz = update(
        vs.S_zsat_rz,
        at[2:-2, 2:-2], ((vs.z_sat[2:-2, 2:-2, vs.tau] * rz_share[2:-2, 2:-2]) / vs.theta_ac[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sub_rz = update(
        vs.q_sub_rz,
        at[2:-2, 2:-2], npx.where(vs.q_sub_pot[2:-2, 2:-2] * rz_share[2:-2, 2:-2] < vs.S_zsat_rz[2:-2, 2:-2], vs.q_sub_pot[2:-2, 2:-2] * rz_share[2:-2, 2:-2], vs.S_zsat_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # lateral subsurface runoff in root zone
    vs.q_sub_mat_rz = update(
        vs.q_sub_mat_rz,
        at[2:-2, 2:-2], vs.q_sub_rz[2:-2, 2:-2] * vs.q_sub_mat_share[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sub_mp_rz = update(
        vs.q_sub_mp_rz,
        at[2:-2, 2:-2], vs.q_sub_rz[2:-2, 2:-2] * vs.q_sub_mp_share[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sub_mp_pot_rz = update(
        vs.q_sub_mp_pot_rz,
        at[2:-2, 2:-2], vs.q_sub_mp_pot[2:-2, 2:-2] * rz_share[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update subsoil saturation water level
    vs.z_sat = update_add(
        vs.z_sat,
        at[2:-2, 2:-2, vs.tau], -vs.q_sub_rz[2:-2, 2:-2]/vs.theta_ac[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update subsoil storage
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[2:-2, 2:-2], -vs.q_sub_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_sub_mp_rz=vs.q_sub_mp_rz, q_sub_mat_rz=vs.q_sub_mat_rz, q_sub_rz=vs.q_sub_rz, z_sat=vs.z_sat, S_lp_rz=vs.S_lp_rz, S_zsat_rz=vs.S_zsat_rz)


@roger_kernel
def calc_potential_lateral_subsurface_runoff_ss(state):
    """
    Calculates potential lateral subsurface runoff in subsoil
    """
    vs = state.variables

    # subsoil share for subsurface runoff
    ss_share = allocate(state.dimensions, ("x", "y"))
    ss_share = update(
        ss_share,
        at[2:-2, 2:-2], ((vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]) / vs.z_sat[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask1 = (vs.z_sat[:, :, vs.tau] <= vs.z_soil - vs.z_root[:, :, vs.tau]) | (vs.S_lp_rz <= 0)
    mask2 = (vs.z_sat[:, :, vs.tau] <= 0)
    ss_share = update(
        ss_share,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], 1, ss_share[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    ss_share = update(
        ss_share,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], 0, ss_share[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # lateral subsurface runoff in subsoil
    vs.q_sub_mat_pot_ss = update(
        vs.q_sub_mat_pot_ss,
        at[2:-2, 2:-2], vs.q_sub_mat_pot[2:-2, 2:-2] * ss_share[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sub_mp_pot_ss = update(
        vs.q_sub_mp_pot_ss,
        at[2:-2, 2:-2], vs.q_sub_mp_pot[2:-2, 2:-2] * ss_share[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sub_pot_ss = update(
        vs.q_sub_pot_ss,
        at[2:-2, 2:-2], (vs.q_sub_mat_pot_ss[2:-2, 2:-2] + vs.q_sub_mp_pot_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_sub_mp_pot_ss=vs.q_sub_mp_pot_ss, q_sub_mat_pot_ss=vs.q_sub_mat_pot_ss, q_sub_pot_ss=vs.q_sub_pot_ss, S_zsat_rz=vs.S_zsat_rz)


@roger_kernel
def calc_lateral_subsurface_runoff_ss(state):
    """
    Calculates lateral and vertical subsurface runoff in subsoil
    """
    vs = state.variables

    vs.q_ss = update(
        vs.q_ss,
        at[2:-2, 2:-2], 0,
    )
    vs.q_ss = update(
        vs.q_ss,
        at[2:-2, 2:-2], npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] <= 0, vs.q_pot_ss[2:-2, 2:-2], vs.q_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # update subsoil storage after vertical subsoil drainage
    mask1 = (vs.S_lp_ss < vs.q_pot_ss) & (vs.z_sat[:, :, vs.tau] <= 0)
    mask2 = (vs.S_lp_ss >= vs.q_pot_ss) & (vs.z_sat[:, :, vs.tau] <= 0)
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], -(vs.q_ss[2:-2, 2:-2] - vs.S_lp_ss[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_lp_ss = update(
        vs.S_lp_ss,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], 0, vs.S_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], -vs.q_ss[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # fraction of lateral and vertical flow
    fv = allocate(state.dimensions, ("x", "y"))
    fl = allocate(state.dimensions, ("x", "y"))
    fv = update(
        fv,
        at[2:-2, 2:-2], npx.where((vs.q_pot_ss[2:-2, 2:-2] + vs.q_sub_pot_ss[2:-2, 2:-2]) > 0, vs.q_pot_ss[2:-2, 2:-2] / (vs.q_pot_ss[2:-2, 2:-2] + vs.q_sub_pot_ss[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2],
    )
    fl = update(
        fl,
        at[2:-2, 2:-2], npx.where((vs.q_pot_ss[2:-2, 2:-2] + vs.q_sub_pot_ss[2:-2, 2:-2]) > 0, vs.q_sub_pot_ss[2:-2, 2:-2] / (vs.q_pot_ss[2:-2, 2:-2] + vs.q_sub_pot_ss[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # vertical flow while saturation water level
    q_ss_sat = allocate(state.dimensions, ("x", "y"))
    q_ss_sat = update(
        q_ss_sat,
        at[2:-2, 2:-2], npx.where((vs.q_pot_ss[2:-2, 2:-2] + vs.q_sub_pot_ss[2:-2, 2:-2]) <= vs.S_zsat_ss[2:-2, 2:-2], (vs.q_pot_ss[2:-2, 2:-2] + vs.q_sub_pot_ss[2:-2, 2:-2]) * fv[2:-2, 2:-2], vs.S_zsat_ss[2:-2, 2:-2] * fv[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_ss = update(
        vs.q_ss,
        at[2:-2, 2:-2], npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] > 0, q_ss_sat[2:-2, 2:-2], vs.q_ss[2:-2, 2:-2]),
    )

    # lateral flow
    vs.q_sub_ss = update(
        vs.q_sub_ss,
        at[2:-2, 2:-2], 0,
    )
    vs.q_sub_ss = update(
        vs.q_sub_ss,
        at[2:-2, 2:-2], npx.where((vs.q_pot_ss[2:-2, 2:-2] + vs.q_sub_pot_ss[2:-2, 2:-2]) <= vs.S_zsat_ss[2:-2, 2:-2], (vs.q_pot_ss[2:-2, 2:-2] + vs.q_sub_pot_ss[2:-2, 2:-2]) * fl[2:-2, 2:-2], vs.S_zsat_ss[2:-2, 2:-2] * fl[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_sub_mat_ss = update(
        vs.q_sub_mat_ss,
        at[2:-2, 2:-2], vs.q_sub_ss[2:-2, 2:-2] * vs.q_sub_mat_share[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_sub_mp_ss = update(
        vs.q_sub_mp_ss,
        at[2:-2, 2:-2], vs.q_sub_ss[2:-2, 2:-2] * vs.q_sub_mp_share[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update subsoil storage after lateral subsurface runoff
    mask = (vs.z_sat[:, :, vs.tau] > 0)
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], -vs.q_sub_ss[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )
    # update subsoil storage after vertical subsoil drainage
    mask1 = (vs.S_lp_ss < vs.q_ss) & (vs.z_sat[:, :, vs.tau] > 0)
    mask2 = (vs.S_lp_ss >= vs.q_ss) & (vs.z_sat[:, :, vs.tau] > 0)
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], -(vs.q_ss[2:-2, 2:-2] - vs.S_lp_ss[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_lp_ss = update(
        vs.S_lp_ss,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], 0, vs.S_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], -vs.q_ss[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # update subsoil saturation water level
    vs.z_sat = update_add(
        vs.z_sat,
        at[2:-2, 2:-2, vs.tau], -((vs.q_sub_ss[2:-2, 2:-2] + vs.q_ss[2:-2, 2:-2])/vs.theta_ac[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.z_sat = update(
        vs.z_sat,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] < 0, 0, vs.z_sat[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_zsat = update(
        vs.S_zsat,
        at[2:-2, 2:-2], vs.z_sat[2:-2, 2:-2, vs.tau] * vs.theta_ac[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_ss=vs.q_ss, q_sub_ss=vs.q_sub_ss, q_sub_mat_ss=vs.q_sub_mat_ss, q_sub_mp_ss=vs.q_sub_mp_ss, z_sat=vs.z_sat, S_lp_ss=vs.S_lp_ss, S_fp_ss=vs.S_fp_ss)


@roger_kernel
def calc_lateral_subsurface_runoff(state):
    """
    Calculates lateral and vertical subsurface runoff in subsoil
    """
    vs = state.variables

    vs.q_sub_mat = update(
        vs.q_sub_mat,
        at[2:-2, 2:-2], (vs.q_sub_mat_rz[2:-2, 2:-2] + vs.q_sub_mat_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sub_mp = update(
        vs.q_sub_mp,
        at[2:-2, 2:-2], (vs.q_sub_mp_rz[2:-2, 2:-2] + vs.q_sub_mp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.q_sub = update(
        vs.q_sub,
        at[2:-2, 2:-2], (vs.q_sub_rz[2:-2, 2:-2] + vs.q_sub_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_sub_mat=vs.q_sub_mat, q_sub_mp=vs.q_sub_mp, q_sub=vs.q_sub)


@roger_kernel
def calc_rise_of_saturation_water_table(state):
    """
    Calculates rise of saturation water level
    """
    vs = state.variables

    # vertical length of macropores reaching into subsoil
    lmpv_ss = allocate(state.dimensions, ("x", "y"))
    lmpv_ss = update(
        lmpv_ss,
        at[2:-2, 2:-2], vs.lmpv[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
    )
    lmpv_ss = update(
        lmpv_ss,
        at[2:-2, 2:-2], npx.where(vs.lmpv[2:-2, 2:-2] < vs.z_root[2:-2, 2:-2, vs.tau], 0, lmpv_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # saturation from top [mm]
    z_sat_top = allocate(state.dimensions, ("x", "y"))
    z_sat_top = update(
        z_sat_top,
        at[2:-2, 2:-2], vs.S_lp_ss[2:-2, 2:-2] / vs.theta_ac[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # vertical distance without macropores [mm]
    z_nomp = allocate(state.dimensions, ("x", "y"))
    z_nomp = update(
        z_nomp,
        at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]) - lmpv_ss[2:-2, 2:-2] - vs.z_sat[2:-2, 2:-2, vs.tau] * vs.maskCatch[2:-2, 2:-2],
    )
    z_nomp = update(
        z_nomp,
        at[2:-2, 2:-2], npx.where(z_nomp[2:-2, 2:-2] < 0, 0, z_nomp[2:-2, 2:-2]),
    )

    vs.z_sat = update(
        vs.z_sat,
        at[2:-2, 2:-2, vs.tau], npx.where(z_sat_top[2:-2, 2:-2] > z_nomp[2:-2, 2:-2], npx.where((vs.S_fp_ss[2:-2, 2:-2] >= vs.S_ufc_ss[2:-2, 2:-2]) & ((vs.S_lp_ss[2:-2, 2:-2] / vs.theta_ac[2:-2, 2:-2]) < (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau])), vs.S_lp_ss[2:-2, 2:-2] / vs.theta_ac[2:-2, 2:-2], npx.where((vs.S_fp_rz[2:-2, 2:-2] >= vs.S_ufc_rz[2:-2, 2:-2]) & (vs.S_lp_ss[2:-2, 2:-2] >= vs.S_ac_ss[2:-2, 2:-2]), vs.S_lp_rz[2:-2, 2:-2] / vs.theta_ac[2:-2, 2:-2] + vs.S_lp_ss[2:-2, 2:-2] / vs.theta_ac[2:-2, 2:-2], vs.S_lp_ss[2:-2, 2:-2] / vs.theta_ac[2:-2, 2:-2])), 0) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(z_sat=vs.z_sat)


@roger_kernel
def calc_potential_percolation_rz(state):
    """
    Calculates potential percolation in root zone
    """
    vs = state.variables

    # potential percolation rate
    perc_pot = allocate(state.dimensions, ("x", "y"))
    mask1 = (vs.z_wf[:, :, vs.tau] < vs.z_root[:, :, vs.tau]) & (vs.z_sat[:, :, vs.tau] <= 0)
    mask2 = (vs.z_wf[:, :, vs.tau] >= vs.z_root[:, :, vs.tau]) & (vs.z_sat[:, :, vs.tau] <= 0)
    mask3 = (vs.z_sat[:, :, vs.tau] > 0) & (vs.z_root[:, :, vs.tau] < vs.z_soil - vs.z_sat[:, :, vs.tau])
    perc_pot = update(
        perc_pot,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], vs.k_rz[2:-2, 2:-2, vs.tau] * vs.dt, 0) * vs.maskCatch[2:-2, 2:-2],
    )
    perc_pot = update(
        perc_pot,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.k_rz[2:-2, 2:-2, vs.tau] * vs.dt, perc_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    z = allocate(state.dimensions, ("x", "y"))
    z = update(
        z,
        at[2:-2, 2:-2], (vs.z_soil[2:-2, 2:-2] - vs.z_root[2:-2, 2:-2, vs.tau]) - vs.z_sat[2:-2, 2:-2, vs.tau],
    )
    perc_pot = update(
        perc_pot,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], ((npx.power((z[2:-2, 2:-2])/(-vs.ha[2:-2, 2:-2]*10.2), -vs.n_salv[2:-2, 2:-2]) - npx.power(-vs.h_rz[2:-2, 2:-2, vs.tau]/-vs.ha[2:-2, 2:-2], -vs.n_salv[2:-2, 2:-2]))/(1 + npx.power(-vs.h_rz[2:-2, 2:-2, vs.tau]/-vs.ha[2:-2, 2:-2], -vs.n_salv[2:-2, 2:-2]) + (vs.n_salv[2:-2, 2:-2] - 1) * npx.power((z[2:-2, 2:-2])/(-vs.ha[2:-2, 2:-2]*10.2), -vs.n_salv[2:-2, 2:-2]))) * vs.dt,
                                  perc_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # vertical drainage of root zone
    mask3 = (perc_pot > 0) & (vs.S_lp_rz + vs.S_fp_rz >= perc_pot) & (vs.z_root[:, :, vs.taum1] < vs.z_soil - vs.z_sat[:, :, vs.tau])
    mask4 = (perc_pot > 0) & (vs.S_lp_rz + vs.S_fp_rz < perc_pot) & (vs.z_root[:, :, vs.taum1] < vs.z_soil - vs.z_sat[:, :, vs.tau])

    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[2:-2, 2:-2], 0,
    )
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[2:-2, 2:-2], npx.where(mask3[2:-2, 2:-2], perc_pot[2:-2, 2:-2], vs.q_pot_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[2:-2, 2:-2], npx.where(mask4[2:-2, 2:-2], vs.S_fp_rz[2:-2, 2:-2] + vs.S_lp_rz[2:-2, 2:-2], vs.q_pot_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    mask5 = (vs.q_pot_rz > 0) & ((vs.S_ac_ss + vs.S_ufc_ss) - (vs.S_lp_ss + vs.S_fp_ss) > 1e-6) & (vs.q_pot_rz > (vs.S_ac_ss + vs.S_ufc_ss) - (vs.S_lp_ss + vs.S_fp_ss)) & (vs.z_root[:, :, vs.taum1] < vs.z_soil - vs.z_sat[:, :, vs.tau])
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[2:-2, 2:-2], npx.where(mask5[2:-2, 2:-2], (vs.S_ac_ss[2:-2, 2:-2] + vs.S_ufc_ss[2:-2, 2:-2]) - (vs.S_lp_ss[2:-2, 2:-2] + vs.S_fp_ss[2:-2, 2:-2]), vs.q_pot_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask6 = (vs.S_lp_ss >= vs.S_ac_ss - 1e-6) & (vs.S_fp_ss >= vs.S_ufc_ss - 1e-6)
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[2:-2, 2:-2], npx.where(mask6[2:-2, 2:-2], 0, vs.q_pot_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    mask7 = (vs.z_root[:, :, vs.taum1] >= vs.z_soil - vs.z_sat[:, :, vs.tau])
    vs.q_pot_rz = update(
        vs.q_pot_rz,
        at[2:-2, 2:-2], npx.where(mask7[2:-2, 2:-2], 0, vs.q_pot_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_pot_rz=vs.q_pot_rz)


@roger_kernel
def calc_percolation_rz(state):
    """
    Calculates percolation in root zone
    """
    vs = state.variables

    mask1 = (vs.S_lp_rz < vs.q_pot_rz)
    mask2 = (vs.S_lp_rz >= vs.q_pot_rz)

    vs.q_rz = update(
        vs.q_rz,
        at[2:-2, 2:-2], vs.q_pot_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update root zone storage after vertical root zone drainage
    vs.S_fp_rz = update_add(
        vs.S_fp_rz,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], -(vs.q_rz[2:-2, 2:-2] - vs.S_lp_rz[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_lp_rz = update(
        vs.S_lp_rz,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], 0, vs.S_lp_rz[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_lp_rz = update_add(
        vs.S_lp_rz,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], -vs.q_rz[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # update subsoil storage after percolation
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[2:-2, 2:-2], vs.q_rz[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # subsoil fine pore excess fills subsoil large pores
    mask = (vs.S_fp_ss > vs.S_ufc_ss)
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[2:-2, 2:-2], (vs.S_fp_ss[2:-2, 2:-2] - vs.S_ufc_ss[2:-2, 2:-2]) * mask[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_fp_ss = update(
        vs.S_fp_ss,
        at[2:-2, 2:-2], npx.where(mask[2:-2, 2:-2], vs.S_ufc_ss[2:-2, 2:-2], vs.S_fp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_rz=vs.q_rz, S_fp_rz=vs.S_fp_rz, S_lp_rz=vs.S_lp_rz, S_fp_ss=vs.S_fp_ss, S_lp_ss=vs.S_lp_ss)


@roger_kernel
def calc_potential_percolation_ss(state):
    """
    Calculates potenital percolation in subsoil
    """
    vs = state.variables

    # potential drainage rate
    # calculate potential percolation rate
    perc_pot = allocate(state.dimensions, ("x", "y"))
    z = allocate(state.dimensions, ("x", "y"))
    z = update(
        z,
        at[2:-2, 2:-2], vs.z_gw[2:-2, 2:-2, vs.tau] * 1000 - vs.z_soil[2:-2, 2:-2],
    )
    perc_pot = update(
        perc_pot,
        at[2:-2, 2:-2], npx.where((vs.z_gw[2:-2, 2:-2, vs.tau] > 10) & (vs.z_gw[2:-2, 2:-2, vs.tau] * 1000 > vs.z_soil[2:-2, 2:-2]), npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] > 0, npx.fmin(vs.kf[2:-2, 2:-2] * vs.dt, vs.ks[2:-2, 2:-2] * vs.dt), npx.fmin(vs.kf[2:-2, 2:-2] * vs.dt, vs.k_ss[2:-2, 2:-2, vs.tau] * vs.dt)), npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] > 0, npx.fmin(vs.kf[2:-2, 2:-2] * vs.dt, vs.ks[2:-2, 2:-2] * vs.dt), npx.fmin(vs.kf[2:-2, 2:-2] * vs.dt, (npx.power((z[2:-2, 2:-2])/(-vs.ha[2:-2, 2:-2]*10.2), -vs.n_salv[2:-2, 2:-2])/((1 + (vs.n_salv[2:-2, 2:-2] - 1)) * npx.power((z[2:-2, 2:-2])/(-vs.ha[2:-2, 2:-2]*10.2), -vs.n_salv[2:-2, 2:-2]))) * vs.dt))) * vs.maskCatch[2:-2, 2:-2],
    )

    # where drainage occurs
    mask1 = (perc_pot > 0) & (vs.z_soil < vs.z_gw[:, :, vs.tau] * 1000) & (perc_pot <= vs.S_fp_ss + vs.S_lp_ss)
    mask2 = (perc_pot > 0) & (vs.z_soil < vs.z_gw[:, :, vs.tau] * 1000) & (perc_pot > vs.S_fp_ss + vs.S_lp_ss)

    # vertical drainage of subsoil
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[2:-2, 2:-2], 0,
    )
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], perc_pot[2:-2, 2:-2], vs.q_pot_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], vs.S_fp_ss[2:-2, 2:-2] + vs.S_lp_ss[2:-2, 2:-2], vs.q_pot_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    # percolation stops while shallow groundwater table or groundwater table
    # rises into soil
    z = allocate(state.dimensions, ("x", "y"))
    cpr_pot = allocate(state.dimensions, ("x", "y"))
    z = update(
        z,
        at[2:-2, 2:-2], npx.where(vs.z_gw[2:-2, 2:-2, vs.tau] * 1000 < vs.z_soil[2:-2, 2:-2], 0, vs.z_gw[2:-2, 2:-2, vs.tau] * 1000 - vs.z_soil[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    cpr_pot = update(
        cpr_pot,
        at[2:-2, 2:-2], ((npx.power((z[2:-2, 2:-2])/(-vs.ha[2:-2, 2:-2]*10.2), -vs.n_salv[2:-2, 2:-2]) - npx.power(-vs.h_ss[2:-2, 2:-2, vs.tau]/-vs.ha[2:-2, 2:-2], -vs.n_salv[2:-2, 2:-2]))/(1 + npx.power(-vs.h_ss[2:-2, 2:-2, vs.tau]/-vs.ha[2:-2, 2:-2], -vs.n_salv[2:-2, 2:-2]) + (vs.n_salv[2:-2, 2:-2] - 1) * npx.power((z[2:-2, 2:-2])/(-vs.ha[2:-2, 2:-2]*10.2), -vs.n_salv[2:-2, 2:-2]))) * vs.maskCatch[2:-2, 2:-2],
    )
    cpr_pot = update(
        cpr_pot,
        at[2:-2, 2:-2], npx.where(z[2:-2, 2:-2] > 0, 0, cpr_pot[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.q_pot_ss = update(
        vs.q_pot_ss,
        at[2:-2, 2:-2], npx.where((cpr_pot[2:-2, 2:-2] > 0) & (z[2:-2, 2:-2] > 0), 0, vs.q_pot_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_pot_ss=vs.q_pot_ss)


@roger_kernel
def calc_percolation_ss(state):
    """
    Calculates percolation in subsoil
    """
    vs = state.variables

    vs.q_ss = update(
        vs.q_ss,
        at[2:-2, 2:-2], vs.q_pot_ss[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update saturation water level
    vs.z_sat = update_add(
        vs.z_sat,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] > 0, -vs.q_ss[2:-2, 2:-2]/vs.theta_ac[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.z_sat = update(
        vs.z_sat,
        at[2:-2, 2:-2, vs.tau], npx.where(vs.z_sat[2:-2, 2:-2, vs.tau] < 0, 0, vs.z_sat[2:-2, 2:-2, vs.tau]) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.S_zsat_ss = update(
        vs.S_zsat_ss,
        at[2:-2, 2:-2], vs.z_sat[2:-2, 2:-2, vs.tau] * vs.theta_ac[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update subsoil storage after vertical subsoil drainage
    mask1 = (vs.S_lp_ss < vs.q_pot_ss)
    mask2 = (vs.S_lp_ss >= vs.q_pot_ss)
    vs.S_fp_ss = update_add(
        vs.S_fp_ss,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], -(vs.q_ss[2:-2, 2:-2] - vs.S_lp_ss[2:-2, 2:-2]), 0) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_lp_ss = update(
        vs.S_lp_ss,
        at[2:-2, 2:-2], npx.where(mask1[2:-2, 2:-2], 0, vs.S_lp_ss[2:-2, 2:-2]) * vs.maskCatch[2:-2, 2:-2],
    )
    vs.S_lp_ss = update_add(
        vs.S_lp_ss,
        at[2:-2, 2:-2], npx.where(mask2[2:-2, 2:-2], -vs.q_ss[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(q_ss=vs.q_ss, S_fp_ss=vs.S_fp_ss, S_lp_ss=vs.S_lp_ss, z_sat=vs.z_sat, S_zsat=vs.S_zsat)


@roger_routine
def calculate_subsurface_runoff(state):
    """
    Calculates subsurface runoff
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_lateral_flow:
        vs.update(calc_rise_of_saturation_water_table(state))
        vs.update(calc_S_zsat(state))
        vs.update(calc_z_sat_layer(state))
        vs.update(calc_potential_percolation_rz(state))
        vs.update(calc_percolation_rz(state))
        vs.update(calc_potential_lateral_subsurface_runoff(state))
        vs.update(calc_lateral_subsurface_runoff_rz(state))
        vs.update(calc_potential_lateral_subsurface_runoff_ss(state))
        vs.update(calc_potential_percolation_ss(state))
        vs.update(calc_lateral_subsurface_runoff_ss(state))
        vs.update(calc_lateral_subsurface_runoff(state))

    elif not settings.enable_lateral_flow:
        vs.update(calc_rise_of_saturation_water_table(state))
        vs.update(calc_S_zsat(state))
        vs.update(calc_potential_percolation_rz(state))
        vs.update(calc_percolation_rz(state))
        vs.update(calc_potential_percolation_ss(state))
        vs.update(calc_percolation_ss(state))

    elif not settings.enable_lateral_flow and settings.enable_film_flow:
        vs.update(calc_potential_percolation_rz(state))
        vs.update(calc_percolation_rz(state))
        vs.update(calc_potential_percolation_ss(state))
        vs.update(calc_percolation_ss(state))


@roger_kernel
def calc_subsurface_runoff_routing_1D(state):
    """
    Calculates subsurface runoff routing
    """
    pass


@roger_kernel
def calc_subsurface_runoff_routing_2D(state):
    """
    Calculates subsurface runoff routing
    """
    pass


@roger_kernel
def calc_percolation_rz_transport_kernel(state):
    """
    Calculates travel time of percolation
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.q_rz, vs.sas_params_q_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_rz[2:-2, 2:-2, :], axis=-1),
    )

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, :], vs.tt_q_rz[2:-2, 2:-2, :] * vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_q_rz=vs.tt_q_rz, TT_q_rz=vs.TT_q_rz, sa_ss=vs.sa_ss)


@roger_kernel
def calc_percolation_rz_transport_iso_kernel(state):
    """
    Calculates isotope transport of percolation
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.q_rz, vs.sas_params_q_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_rz[2:-2, 2:-2, :], axis=2),
    )
    # calculate isotope travel time distribution
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.C_q_rz = update(
        vs.C_q_rz,
        at[2:-2, 2:-2], transport.calc_conc_iso_flux(state, vs.mtt_q_rz, vs.tt_q_rz, vs.q_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_q_rz = update(
        vs.C_iso_q_rz,
        at[2:-2, 2:-2], transport.conc_to_delta(state, vs.C_q_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # update isotope StorAge
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.tt_q_rz[2:-2, 2:-2, :] * vs.q_rz[2:-2, 2:-2, npx.newaxis] + vs.sa_ss[2:-2, 2:-2, vs.tau, :] > 0, vs.msa_ss[2:-2, 2:-2, vs.tau, :] * (vs.sa_ss[2:-2, 2:-2, vs.tau, :] / (vs.tt_q_rz[2:-2, 2:-2, :] * vs.q_rz[2:-2, 2:-2, npx.newaxis] + vs.sa_ss[2:-2, 2:-2, vs.tau, :])) + vs.mtt_q_rz[2:-2, 2:-2, :] * ((vs.tt_q_rz[2:-2, 2:-2, :] * vs.q_rz[2:-2, 2:-2, npx.newaxis]) / ((vs.tt_q_rz[2:-2, 2:-2, :] * vs.q_rz[2:-2, 2:-2, npx.newaxis]) + vs.sa_ss[2:-2, 2:-2, vs.tau, :])), vs.msa_ss[2:-2, 2:-2, vs.tau, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, :], vs.tt_q_rz[2:-2, 2:-2, :] * vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    # update isotope StorAge
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.sa_rz[2:-2, 2:-2, vs.tau, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, vs.tau, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_q_rz=vs.tt_q_rz, TT_q_rz=vs.TT_q_rz, msa_rz=vs.msa_rz, mtt_q_rz=vs.mtt_q_rz, C_q_rz=vs.C_q_rz, C_iso_q_rz=vs.C_iso_q_rz, sa_ss=vs.sa_ss, msa_ss=vs.msa_ss)


@roger_kernel
def calc_percolation_rz_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of percolation
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_rz = update(
        vs.tt_q_rz,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.q_rz, vs.sas_params_q_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_rz = update(
        vs.TT_q_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_rz[2:-2, 2:-2, :], axis=2),
    )

    # calculate isotope travel time distribution
    vs.mtt_q_rz = update(
        vs.mtt_q_rz,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_q_rz = update(
        vs.C_q_rz,
        at[2:-2, 2:-2], npx.where(vs.q_rz[2:-2, 2:-2] > 0, npx.sum(vs.mtt_q_rz[2:-2, 2:-2, :], axis=-1) / vs.q_rz[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.M_q_rz = update(
        vs.M_q_rz,
        at[2:-2, 2:-2], npx.sum(vs.mtt_q_rz[2:-2, 2:-2, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_q_rz, vs.q_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge of root zone
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], -vs.mtt_q_rz[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :], vs.mtt_q_rz[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.sa_ss = update_add(
        vs.sa_ss,
        at[2:-2, 2:-2, vs.tau, :], vs.tt_q_rz[2:-2, 2:-2, :] * vs.q_rz[2:-2, 2:-2, npx.newaxis] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_q_rz=vs.tt_q_rz, TT_q_rz=vs.TT_q_rz, msa_rz=vs.msa_rz, mtt_q_rz=vs.mtt_q_rz, C_q_rz=vs.C_q_rz, M_q_rz=vs.M_q_rz, sa_ss=vs.sa_ss, msa_ss=vs.msa_ss)


@roger_kernel
def calc_percolation_ss_transport_kernel(state):
    """
    Calculates travel time of percolation
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.q_ss, vs.sas_params_q_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_ss[2:-2, 2:-2, :], axis=-1),
    )

    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_q_ss=vs.tt_q_ss, TT_q_ss=vs.TT_q_ss)


@roger_kernel
def calc_percolation_ss_transport_iso_kernel(state):
    """
    Calculates isotope transport of percolation
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.q_ss, vs.sas_params_q_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_ss[2:-2, 2:-2, :], axis=2),
    )
    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss, vs.msa_ss, alpha)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.C_q_ss = update(
        vs.C_q_ss,
        at[2:-2, 2:-2], transport.calc_conc_iso_flux(state, vs.mtt_q_ss, vs.tt_q_ss, vs.q_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_q_ss = update(
        vs.C_iso_q_ss,
        at[2:-2, 2:-2], transport.conc_to_delta(state, vs.C_q_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    # update isotope StorAge
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.sa_ss[2:-2, 2:-2, vs.tau, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, vs.tau, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_q_ss=vs.tt_q_ss, TT_q_ss=vs.TT_q_ss, msa_ss=vs.msa_ss, mtt_q_ss=vs.mtt_q_ss, C_q_ss=vs.C_q_ss, C_iso_q_ss=vs.C_iso_q_ss)


@roger_kernel
def calc_percolation_ss_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of percolation
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_ss = update(
        vs.tt_q_ss,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.q_ss, vs.sas_params_q_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_ss = update(
        vs.TT_q_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_ss[2:-2, 2:-2, :], axis=-1),
    )

    # calculate anion travel time distribution
    vs.mtt_q_ss = update(
        vs.mtt_q_ss,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_q_ss = update(
        vs.C_q_ss,
        at[2:-2, 2:-2], npx.where(vs.q_ss[2:-2, 2:-2] > 0, npx.sum(vs.mtt_q_ss[2:-2, 2:-2, :], axis=-1) / vs.q_ss[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    vs.M_q_ss = update(
        vs.M_q_ss,
        at[2:-2, 2:-2], npx.sum(vs.mtt_q_ss[2:-2, 2:-2, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_q_ss, vs.q_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge of subsoil
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :], -vs.mtt_q_ss[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_q_ss=vs.tt_q_ss, TT_q_ss=vs.TT_q_ss, msa_ss=vs.msa_ss, mtt_q_ss=vs.mtt_q_ss, C_q_ss=vs.C_q_ss, M_q_ss=vs.M_q_ss)


@roger_routine
def calculate_percolation_rz_transport(state):
    """
    Calculates transport of percolation in root zone
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calc_percolation_rz_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calc_percolation_rz_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calc_percolation_rz_transport_anion_kernel(state))


@roger_routine
def calculate_percolation_ss_transport(state):
    """
    Calculates transport of percolation in subsoil
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate):
        vs.update(calc_percolation_ss_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calc_percolation_ss_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate):
        vs.update(calc_percolation_ss_transport_anion_kernel(state))


@roger_kernel
def calc_lateral_subsurface_runoff_rz_transport_kernel(state):
    """
    Calculates travel time of lateral subsurface runoff in root zone
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_sub_rz = update(
        vs.tt_q_sub_rz,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.q_sub_rz, vs.sas_params_q_sub_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_sub_rz = update(
        vs.TT_q_sub_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_sub_rz[2:-2, 2:-2, :], axis=-1),
    )

    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_q_sub_rz, vs.q_sub_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_q_sub_rz=vs.tt_q_sub_rz, TT_q_sub_rz=vs.TT_q_sub_rz, sa_ss=vs.sa_ss)


@roger_kernel
def calc_lateral_subsurface_runoff_rz_transport_iso_kernel(state):
    """
    Calculates isotope transport of lateral subsurface runoff in root zone
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_sub_rz = update(
        vs.tt_q_sub_rz,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.q_sub_rz, vs.sas_params_q_sub_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_sub_rz = update(
        vs.TT_q_sub_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_sub_rz[2:-2, 2:-2, :], axis=2),
    )
    # calculate isotope travel time distribution
    vs.mtt_q_sub_rz = update(
        vs.mtt_q_sub_rz,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_q_sub_rz, vs.q_sub_rz, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.C_q_sub_rz = update(
        vs.C_q_sub_rz,
        at[2:-2, 2:-2], transport.calc_conc_iso_flux(state, vs.mtt_q_sub_rz, vs.tt_q_sub_rz, vs.q_sub_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_q_sub_rz = update(
        vs.C_iso_q_sub_rz,
        at[2:-2, 2:-2], transport.conc_to_delta(state, vs.C_q_sub_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    # update isotope StorAge
    vs.msa_rz = update(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.sa_rz[2:-2, 2:-2, vs.tau, :] <= 0, 0, vs.msa_rz[2:-2, 2:-2, vs.tau, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_q_sub_rz=vs.tt_q_sub_rz, TT_q_sub_rz=vs.TT_q_sub_rz, msa_rz=vs.msa_rz, mtt_q_sub_rz=vs.mtt_q_sub_rz, C_q_sub_rz=vs.C_q_sub_rz, C_iso_q_sub_rz=vs.C_iso_q_sub_rz, sa_ss=vs.sa_ss, msa_ss=vs.msa_ss)


@roger_kernel
def calc_lateral_subsurface_runoff_rz_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of lateral subsurface runoff in root zone
    """
    vs = state.variables

    vs.SA_rz = update(
        vs.SA_rz,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_rz, vs.sa_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_sub_rz = update(
        vs.tt_q_sub_rz,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_rz, vs.sa_rz, vs.q_sub_rz, vs.sas_params_q_sub_rz)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_sub_rz = update(
        vs.TT_q_sub_rz,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_sub_rz[2:-2, 2:-2, :], axis=2),
    )

    # calculate isotope travel time distribution
    vs.mtt_q_sub_rz = update(
        vs.mtt_q_sub_rz,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_rz, vs.tt_q_sub_rz, vs.q_sub_rz, vs.msa_rz, vs.alpha_q)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_q_sub_rz = update(
        vs.C_q_sub_rz,
        at[2:-2, 2:-2], npx.where(vs.q_sub_rz[2:-2, 2:-2] > 0, npx.sum(vs.mtt_q_sub_rz[2:-2, 2:-2, :], axis=-1) / vs.q_sub_rz[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # update StorAge with flux
    vs.sa_rz = update(
        vs.sa_rz,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_rz, vs.tt_q_sub_rz, vs.q_sub_rz)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge of root zone
    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :], -vs.mtt_q_sub_rz[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_rz=vs.sa_rz, tt_q_sub_rz=vs.tt_q_sub_rz, TT_q_sub_rz=vs.TT_q_sub_rz, msa_rz=vs.msa_rz, mtt_q_sub_rz=vs.mtt_q_sub_rz, C_q_sub_rz=vs.C_q_sub_rz, sa_ss=vs.sa_ss, msa_ss=vs.msa_ss)


@roger_kernel
def calc_lateral_subsurface_runoff_ss_transport_kernel(state):
    """
    Calculates travel time of lateral subsurface runoff in subsoil
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_sub_ss = update(
        vs.tt_q_sub_ss,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.q_sub_ss, vs.sas_params_q_sub_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_sub_ss = update(
        vs.TT_q_sub_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_sub_ss[2:-2, 2:-2, :], axis=-1),
    )

    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_q_sub_ss, vs.q_sub_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_q_sub_ss=vs.tt_q_sub_ss, TT_q_sub_ss=vs.TT_q_sub_ss)


@roger_kernel
def calc_lateral_subsurface_runoff_ss_transport_iso_kernel(state):
    """
    Calculates isotope transport of lateral subsurface runoff in subsoil
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    vs.tt_q_sub_ss = update(
        vs.tt_q_sub_ss,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.q_sub_ss, vs.sas_params_q_sub_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_sub_ss = update(
        vs.TT_q_sub_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_sub_ss[2:-2, 2:-2, :], axis=2),
    )
    # calculate solute travel time distribution
    alpha = allocate(state.dimensions, ("x", "y"), fill=1)
    vs.mtt_q_sub_ss = update(
        vs.mtt_q_sub_ss,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_ss, vs.tt_q_sub_ss, vs.q_sub_ss, vs.msa_ss, alpha)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    vs.C_q_sub_ss = update(
        vs.C_q_sub_ss,
        at[2:-2, 2:-2], transport.calc_conc_iso_flux(state, vs.mtt_q_sub_ss, vs.tt_q_sub_ss, vs.q_sub_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    vs.C_iso_q_sub_ss = update(
        vs.C_iso_q_sub_ss,
        at[2:-2, 2:-2], transport.conc_to_delta(state, vs.C_q_sub_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )
    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_q_sub_ss, vs.q_sub_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    # update isotope StorAge
    vs.msa_ss = update(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :], npx.where(vs.sa_ss[2:-2, 2:-2, vs.tau, :] <= 0, 0, vs.msa_ss[2:-2, 2:-2, vs.tau, :]) * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_q_sub_ss=vs.tt_q_sub_ss, TT_q_sub_ss=vs.TT_q_sub_ss, msa_ss=vs.msa_ss, mtt_q_sub_ss=vs.mtt_q_sub_ss, C_q_sub_ss=vs.C_q_sub_ss, C_iso_q_sub_ss=vs.C_iso_q_sub_ss)


@roger_kernel
def calc_lateral_subsurface_runoff_ss_transport_anion_kernel(state):
    """
    Calculates chloride/bromide/nitrate transport of lateral subsurface runoff in subsoil
    """
    vs = state.variables

    vs.SA_ss = update(
        vs.SA_ss,
        at[2:-2, 2:-2, :, :], transport.calc_SA(state, vs.SA_ss, vs.sa_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )

    vs.tt_q_sub_ss = update(
        vs.tt_q_sub_ss,
        at[2:-2, 2:-2, :], transport.calc_tt(state, vs.SA_ss, vs.sa_ss, vs.q_sub_ss, vs.sas_params_q_sub_ss)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.TT_q_sub_ss = update(
        vs.TT_q_sub_ss,
        at[2:-2, 2:-2, 1:], npx.cumsum(vs.tt_q_sub_ss[2:-2, 2:-2, :], axis=-1),
    )

    # calculate isotope travel time distribution
    vs.mtt_q_sub_ss = update(
        vs.mtt_q_sub_ss,
        at[2:-2, 2:-2, :], transport.calc_mtt(state, vs.sa_ss, vs.tt_q_sub_ss, vs.q_sub_ss, vs.msa_ss, vs.alpha_q)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.C_q_sub_ss = update(
        vs.C_q_sub_ss,
        at[2:-2, 2:-2], npx.where(vs.q_sub_ss[2:-2, 2:-2] > 0, npx.sum(vs.mtt_q_sub_ss[2:-2, 2:-2, :], axis=-1) / vs.q_sub_ss[2:-2, 2:-2], 0) * vs.maskCatch[2:-2, 2:-2],
    )

    # update StorAge with flux
    vs.sa_ss = update(
        vs.sa_ss,
        at[2:-2, 2:-2, :, :], transport.update_sa(state, vs.sa_ss, vs.tt_q_sub_ss, vs.q_sub_ss)[2:-2, 2:-2, :, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis, npx.newaxis],
    )
    # update solute StorAge of subsoil
    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :], -vs.mtt_q_sub_ss[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return KernelOutput(sa_ss=vs.sa_ss, tt_q_sub_ss=vs.tt_q_sub_ss, TT_q_sub_ss=vs.TT_q_sub_ss, msa_ss=vs.msa_ss, mtt_q_sub_ss=vs.mtt_q_sub_ss, C_q_sub_ss=vs.C_q_sub_ss)


@roger_routine
def calculate_lateral_subsurface_runoff_rz_transport(state):
    """
    Calculates transport of lateral subsurface runoff in root zone
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate | settings.enable_virtualtracer):
        vs.update(calc_lateral_subsurface_runoff_rz_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calc_lateral_subsurface_runoff_rz_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate | settings.enable_virtualtracer):
        vs.update(calc_lateral_subsurface_runoff_rz_transport_anion_kernel(state))


@roger_routine
def calculate_lateral_subsurface_runoffn_ss_transport(state):
    """
    Calculates transport lateral subsurface runoff in subsoil
    """
    vs = state.variables
    settings = state.settings

    if settings.enable_offline_transport and not (settings.enable_chloride | settings.enable_bromide | settings.enable_oxygen18 | settings.enable_deuterium | settings.enable_nitrate | settings.enable_virtualtracer):
        vs.update(calc_lateral_subsurface_runoff_ss_transport_kernel(state))

    if settings.enable_offline_transport and (settings.enable_oxygen18 | settings.enable_deuterium):
        vs.update(calc_lateral_subsurface_runoff_ss_transport_iso_kernel(state))

    if settings.enable_offline_transport and (settings.enable_chloride | settings.enable_bromide | settings.enable_nitrate | settings.enable_virtualtracer):
        vs.update(calc_lateral_subsurface_runoff_ss_transport_anion_kernel(state))
