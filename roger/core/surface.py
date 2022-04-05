from roger import roger_kernel, roger_routine, KernelOutput
from roger.core.operators import update, at


@roger_kernel
def calc_S(state):
    """
    Calculates soil water content
    """
    vs = state.variables

    vs.S_sur = update(
        vs.S_sur,
        at[:, :, vs.tau], (vs.S_int_top[:, :, vs.tau] + vs.S_int_ground[:, :, vs.tau] + vs.S_dep[:, :, vs.tau] + vs.S_snow[:, :, vs.tau]) * vs.maskCatch,
    )

    return KernelOutput(S_sur=vs.S_sur)


@roger_kernel
def calc_dS(state):
    """
    Calculates storage change
    """
    vs = state.variables

    vs.dS_sur = update(
        vs.dS_sur,
        at[:, :], (vs.S_sur[:, :, vs.tau] - vs.S_sur[:, :, vs.taum1]) * vs.maskCatch,
    )

    return KernelOutput(dS_sur=vs.dS_sur)


@roger_routine
def calculate_surface(state):
    """
    Calculates soil storage and storage dependent variables
    """
    vs = state.variables
    vs.update(calc_S(state))
    vs.update(calc_dS(state))
