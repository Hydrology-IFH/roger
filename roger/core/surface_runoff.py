from roger import roger_kernel, roger_routine, KernelOutput
from roger.variables import allocate
from roger.core.operators import numpy as npx, update, update_add, at


@roger_kernel
def calc_channel_routing(state):
    """
    Calculates channel routing
    """
    pass


@roger_kernel
def calc_surface_runoff_routing(state):
    """
    Calculates subsurface runoff routing
    """
    pass
