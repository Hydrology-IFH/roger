from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at


@roger_kernel
def calc_soil_temperature_kernel1(state, ta_year, a_year):
    """Calculates soil temperature."""
    vs = state.variables
    settings = state.settings

    vs.temp_soil = update(
        vs.temp_soil,
        at[2:-2, 2:-2, vs.tau],
        ta_year[2:-2, 2:-2]
        + a_year[2:-2, 2:-2]
        * npx.sin(
            (2 * settings.pi) * (vs.doy[1]/365)
            - (2 * settings.pi) * (vs.phi_soil_temp[2:-2, 2:-2]/365)/2
            - ((0.5 * (vs.z_soil[2:-2, 2:-2]/1000)) / (vs.damp_soil_temp[2:-2, 2:-2] * (vs.S_s[2:-2, 2:-2, vs.tau]/(vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2])))))
        * npx.exp((-0.5 * (vs.z_soil[2:-2, 2:-2]/1000)) / (vs.damp_soil_temp[2:-2, 2:-2] * (vs.S_s[2:-2, 2:-2, vs.tau]/(vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2]))))
        * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        temp_soil=vs.temp_soil,
        )

@roger_kernel
def calc_soil_temperature_kernel2(state):
    """Calculates soil temperature."""
    vs = state.variables
    settings = state.settings

    ta_year = allocate(state.dimensions, ("x", "y"))
    a_year = allocate(state.dimensions, ("x", "y"))

    # calculate annual average air temperature and annual average amplitude of air temperature
    ta_year = update(
        ta_year,
        at[2:-2, 2:-2],
        npx.mean(vs.TA[settings.nitt-364:]),
    )
    a_year = update(
        a_year,
        at[2:-2, 2:-2],
        2 * npx.mean(npx.abs(vs.TA[npx.newaxis, npx.newaxis, settings.nitt-364:] - ta_year[2:-2, 2:-2, npx.newaxis]), axis=-1),
    )

    vs.temp_soil = update(
        vs.temp_soil,
        at[2:-2, 2:-2, vs.tau],
        ta_year[2:-2, 2:-2]
        + a_year[2:-2, 2:-2]
        * npx.sin(
            (2 * settings.pi) * (vs.doy[1]/365)
            - (2 * settings.pi) * (vs.phi_soil_temp[2:-2, 2:-2]/365)/2
            - ((0.5 * (vs.z_soil[2:-2, 2:-2]/1000)) / (vs.damp_soil_temp[2:-2, 2:-2] * (vs.S_s[2:-2, 2:-2, vs.tau]/(vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2])))))
        * npx.exp((-0.5 * (vs.z_soil[2:-2, 2:-2]/1000)) / (vs.damp_soil_temp[2:-2, 2:-2] * (vs.S_s[2:-2, 2:-2, vs.tau]/(vs.S_sat_rz[2:-2, 2:-2] + vs.S_sat_ss[2:-2, 2:-2]))))
        * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        temp_soil=vs.temp_soil,
        )


@roger_kernel
def calc_denit_soil(state, msa, km, Dmax, sa, S_sat):
    """Calculates soil dentirification rate."""
    vs = state.variables
    settings = state.settings

    S = allocate(state.dimensions, ("x", "y"))
    S = update(
        S,
        at[2:-2, 2:-2],
        npx.sum(sa[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # soil temperature coefficient
    soil_temp_coeff = allocate(state.dimensions, ("x", "y"))
    soil_temp_coeff = update(
        soil_temp_coeff,
        at[2:-2, 2:-2],
        npx.where(
            ((vs.temp_soil[2:-2, 2:-2, vs.tau] >= 5) & (vs.temp_soil[2:-2, 2:-2, vs.tau] <= 30)),
            vs.temp_soil[2:-2, 2:-2, vs.tau] / (30 - 5),
            0,
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    soil_temp_coeff = update(
        soil_temp_coeff,
        at[2:-2, 2:-2],
        npx.where(
            (vs.temp_soil[2:-2, 2:-2, vs.tau] > 30),
            1,
            soil_temp_coeff[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    # calculate denitrification rate
    mr = allocate(state.dimensions, ("x", "y", "ages"))
    mr = update(
        mr,
        at[2:-2, 2:-2, :],
        ((Dmax[2:-2, 2:-2, npx.newaxis]/settings.ages) * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100
        * (msa[2:-2, 2:-2, vs.tau, :] / ((km[2:-2, 2:-2, npx.newaxis]/settings.ages) * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100 + msa[2:-2, 2:-2, vs.tau, :])))
        * soil_temp_coeff[2:-2, 2:-2, npx.newaxis]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # no denitrification if storage is lower than 70 % of pore volume
    mr = update(
        mr,
        at[2:-2, 2:-2, :],
        npx.where(
            S[2:-2, 2:-2, npx.newaxis] >= 0.7 * S_sat[2:-2, 2:-2, npx.newaxis],
            mr[2:-2, 2:-2, :],
            0,
        )
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # limit denitrification to available solute mass
    mr = update(
        mr,
        at[2:-2, 2:-2, :],
        npx.where(mr[2:-2, 2:-2, :] > msa[2:-2, 2:-2, vs.tau, :], msa[2:-2, 2:-2, vs.tau, :], mr[2:-2, 2:-2, :])
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return mr


@roger_kernel
def calc_nit_soil(state, Nmin, knit, Dnit, sa, S_sat):
    """Calculates soil nitrification rate."""
    vs = state.variables
    settings = state.settings

    S = allocate(state.dimensions, ("x", "y"))
    S = update(
        S,
        at[2:-2, 2:-2],
        npx.sum(sa[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    # soil temperature coefficient
    soil_temp_coeff = allocate(state.dimensions, ("x", "y"))
    soil_temp_coeff = update(
        soil_temp_coeff,
        at[2:-2, 2:-2],
        npx.where(
            ((vs.temp_soil[2:-2, 2:-2, vs.tau] >= 1) & (vs.temp_soil[2:-2, 2:-2, vs.tau] <= 30)),
            vs.temp_soil[2:-2, 2:-2, vs.tau] / (30 - 1),
            0,
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    soil_temp_coeff = update(
        soil_temp_coeff,
        at[2:-2, 2:-2],
        npx.where(
            (vs.temp_soil[2:-2, 2:-2, vs.tau] > 30),
            1,
            soil_temp_coeff[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    # calculate nitrification rate
    ma = allocate(state.dimensions, ("x", "y", "ages"))
    ma = update(
        ma,
        at[2:-2, 2:-2, :],
        ((Dnit[2:-2, 2:-2, npx.newaxis]/settings.ages) * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100
        * (Nmin[2:-2, 2:-2, vs.tau, :] / ((knit[2:-2, 2:-2, npx.newaxis]/settings.ages) * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100 + Nmin[2:-2, 2:-2, vs.tau, :])))
        * soil_temp_coeff[2:-2, 2:-2, npx.newaxis]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # no nitrification if storage is greater than 70 % of pore volume
    ma = update(
        ma,
        at[2:-2, 2:-2, :],
        npx.where(
            S[2:-2, 2:-2, npx.newaxis] < 0.7 * S_sat[2:-2, 2:-2, npx.newaxis],
            ma[2:-2, 2:-2, :],
            0,
        )
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # limit denitrification to available solute mass
    ma = update(
        ma,
        at[2:-2, 2:-2, :],
        npx.where(ma[2:-2, 2:-2, :] > Nmin[2:-2, 2:-2, vs.tau, :], Nmin[2:-2, 2:-2, vs.tau, :], ma[2:-2, 2:-2, :])
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return ma


@roger_kernel
def calc_min_soil(state, kmin):
    """Calculates soil nitrogen mineralization rate."""
    vs = state.variables
    settings = state.settings

    # soil temperature coefficient
    soil_temp_coeff = allocate(state.dimensions, ("x", "y"))
    soil_temp_coeff = update(
        soil_temp_coeff,
        at[2:-2, 2:-2],
        npx.where(
            ((vs.temp_soil[2:-2, 2:-2, vs.tau] >= 0) & (vs.temp_soil[2:-2, 2:-2, vs.tau] <= 30)),
            vs.temp_soil[2:-2, 2:-2, vs.tau] / (30 - 0),
            0,
        )
        * vs.maskCatch[2:-2, 2:-2],
    )
    soil_temp_coeff = update(
        soil_temp_coeff,
        at[2:-2, 2:-2],
        npx.where(
            (vs.temp_soil[2:-2, 2:-2, vs.tau] > 30),
            1,
            soil_temp_coeff[2:-2, 2:-2],
        )
        * vs.maskCatch[2:-2, 2:-2],
    )

    ma = allocate(state.dimensions, ("x", "y"))
    ma = update(
        ma,
        at[2:-2, 2:-2],
        kmin[2:-2, 2:-2]
        * (vs.dt / (365 * 24))
        * settings.dx
        * settings.dy
        * 100
        * soil_temp_coeff[2:-2, 2:-2]
        * vs.maskCatch[2:-2, 2:-2],
    )

    return ma


@roger_kernel
def calc_n_fixation(state, kfix):
    """Calculates nitrogen fixation."""
    vs = state.variables
    settings = state.settings

    nfix = allocate(state.dimensions, ("x", "y"))
    nfix = update(
        nfix,
        at[2:-2, 2:-2],
        kfix[2:-2, 2:-2] * (vs.dt / (365 * 24)) * settings.dx * settings.dy * 100 * (vs.z_root[2:-2, 2:-2, vs.tau]/(settings.zroot_to_zsoil_max * vs.z_soil[2:-2, 2:-2])) * vs.maskCatch[2:-2, 2:-2],
    )

    # nitrogen fixation of yellow mustard and clover
    lu_id = vs.LU_ID[2:-2, 2:-2, vs.itt]
    mask = npx.isin(lu_id, npx.array([580, 581, 583, 584, 586, 587]))
    nfix = update(nfix, at[2:-2, 2:-2], npx.where(mask, nfix[2:-2, 2:-2], 0))

    return nfix


@roger_kernel
def calc_denit_gw(state, msa, k):
    """Calculates groundwater dentrification rate."""
    vs = state.variables

    # calculate denitrification rate
    age = allocate(state.dimensions, ("x", "y", "ages"))
    mr = allocate(state.dimensions, ("x", "y", "ages"))
    age = update(
        age,
        at[2:-2, 2:-2, :],
        vs.ages[npx.newaxis, npx.newaxis, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    mr = update(
        mr,
        at[2:-2, 2:-2, :],
        msa[2:-2, 2:-2, :]
        * k[2:-2, 2:-2, npx.newaxis]
        * npx.exp(-k[2:-2, 2:-2, npx.newaxis] * age[2:-2, 2:-2, :])
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )
    # limit denitrification to available solute mass
    mr = update(
        mr,
        at[2:-2, 2:-2, :],
        npx.where(mr[2:-2, 2:-2, :] > msa[2:-2, 2:-2, vs.tau, :], msa[2:-2, 2:-2, vs.tau, :], mr[2:-2, 2:-2, :])
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    return mr


@roger_kernel
def calc_nitrogen_cycle_kernel(state):
    vs = state.variables

    vs.Nmin_rz = update_add(
        vs.Nmin_rz,
        at[2:-2, 2:-2, vs.tau, 0],
        calc_n_fixation(state, vs.kfix_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.Nmin_rz = update_add(
        vs.Nmin_rz,
        at[2:-2, 2:-2, vs.tau, 0],
        calc_min_soil(state, vs.kmin_rz)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.Nmin_ss = update_add(
        vs.Nmin_ss,
        at[2:-2, 2:-2, vs.tau, 0],
        calc_min_soil(state, vs.kmin_ss)[2:-2, 2:-2] * vs.maskCatch[2:-2, 2:-2],
    )

    vs.ma_rz = update(
        vs.ma_rz,
        at[2:-2, 2:-2, :],
        calc_nit_soil(state, vs.Nmin_rz, vs.km_nit_rz, vs.dmax_nit_rz, vs.sa_rz, vs.S_sat_rz)[
            2:-2, 2:-2, :
        ]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.Nmin_rz = update_add(
        vs.Nmin_rz,
        at[2:-2, 2:-2, vs.tau, :],
        -vs.ma_rz[2:-2, 2:-2, :],
    )

    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, 0],
        npx.sum(vs.ma_rz[2:-2, 2:-2, :], axis=-1),
    )

    vs.ma_ss = update(
        vs.ma_ss,
        at[2:-2, 2:-2, :],
        calc_nit_soil(state, vs.Nmin_ss, vs.km_nit_ss, vs.dmax_nit_ss, vs.sa_ss, vs.S_sat_ss)[
            2:-2, 2:-2, :
        ]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.Nmin_ss = update_add(
        vs.Nmin_ss,
        at[2:-2, 2:-2, vs.tau, :],
        -vs.ma_ss[2:-2, 2:-2, :],
    )

    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, 0],
        npx.sum(vs.ma_ss[2:-2, 2:-2, :], axis=-1),
    )

    vs.mr_rz = update(
        vs.mr_rz,
        at[2:-2, 2:-2, :],
        calc_denit_soil(state, vs.msa_rz, vs.km_denit_rz, vs.dmax_denit_rz, vs.sa_rz, vs.S_sat_rz)[
            2:-2, 2:-2, :
        ]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.msa_rz = update_add(
        vs.msa_rz,
        at[2:-2, 2:-2, vs.tau, :],
        -vs.mr_rz[2:-2, 2:-2, :],
    )

    vs.mr_ss = update(
        vs.mr_ss,
        at[2:-2, 2:-2, :],
        calc_denit_soil(state, vs.msa_ss, vs.km_denit_ss, vs.dmax_denit_ss, vs.sa_ss, vs.S_sat_ss)[
            2:-2, 2:-2, :
        ]
        * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.msa_ss = update_add(
        vs.msa_ss,
        at[2:-2, 2:-2, vs.tau, :],
        -vs.mr_ss[2:-2, 2:-2, :],
    )

    vs.ma_s = update(
        vs.ma_s,
        at[2:-2, 2:-2, :],
        vs.ma_rz[2:-2, 2:-2, :] + vs.ma_ss[2:-2, 2:-2, :],
    )

    vs.nit_s = update(
        vs.nit_s,
        at[2:-2, 2:-2],
        npx.sum(vs.ma_s[2:-2, 2:-2, :], axis=-1),
    )    

    vs.mr_s = update(
        vs.mr_s,
        at[2:-2, 2:-2, :],
        vs.mr_rz[2:-2, 2:-2, :] + vs.mr_ss[2:-2, 2:-2, :],
    )

    vs.denit_s = update(
        vs.denit_s,
        at[2:-2, 2:-2],
        npx.sum(vs.mr_s[2:-2, 2:-2, :], axis=-1),
    )   

    vs.Nmin_s = update(
        vs.Nmin_s,
        at[2:-2, 2:-2, vs.tau],
        npx.sum(vs.Nmin_rz[2:-2, 2:-2, vs.tau, :], axis=-1) + npx.sum(vs.Nmin_ss[2:-2, 2:-2, vs.tau, :], axis=-1) * vs.maskCatch[2:-2, 2:-2],
    )

    return KernelOutput(
        msa_rz=vs.msa_rz,
        msa_ss=vs.msa_ss,
        ma_rz=vs.ma_rz,
        ma_ss=vs.ma_ss,
        ma_s=vs.ma_s,
        mr_rz=vs.mr_rz,
        mr_ss=vs.mr_ss,
        mr_s=vs.mr_s,
        Nmin_rz=vs.Nmin_rz,
        Nmin_ss=vs.Nmin_ss,
        Nmin_s=vs.Nmin_s,
        nit_s=vs.nit_s,
        denit_s=vs.denit_s,
    )


@roger_kernel
def calc_nitrogen_cycle_gw_kernel(state):
    vs = state.variables

    vs.mr_gw = update(
        vs.mr_gw,
        at[2:-2, 2:-2, :],
        calc_denit_gw(state, vs.msa_gw, vs.k_calc_denit_gw)[2:-2, 2:-2, :] * vs.maskCatch[2:-2, 2:-2, npx.newaxis],
    )

    vs.msa_gw = update_add(
        vs.msa_gw,
        at[2:-2, 2:-2, vs.tau, :],
        -vs.mr_gw[2:-2, 2:-2, :],
    )

    return KernelOutput(msa_rz=vs.msa_rz, msa_ss=vs.msa_ss)


@roger_routine
def calculate_nitrogen_cycle(state):
    """
    Calculates nitrogen cycle
    """
    vs = state.variables
    settings = state.settings

    if vs.itt + 364 < settings.nitt:
        # calculate annual average air temperature and annual average amplitude of air temperature
        ta_year = npx.mean(vs.TA[vs.itt:vs.itt+364])
        a_year = 2 * npx.mean(npx.abs(vs.TA[npx.newaxis, npx.newaxis, vs.itt:vs.itt+364] - ta_year[2:-2, 2:-2, npx.newaxis]), axis=-1)
        vs.update(calc_soil_temperature_kernel1(state, ta_year, a_year))
    else:
        vs.update(calc_soil_temperature_kernel2(state))
    vs.update(calc_nitrogen_cycle_kernel(state))
    if settings.enable_groundwater:
        vs.update(calc_nitrogen_cycle_gw_kernel(state))
