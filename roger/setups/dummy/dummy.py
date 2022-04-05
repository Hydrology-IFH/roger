import roger
roger.runtime_settings.backend = 'numpy'
roger.runtime_settings.force_overwrite = 'True'
from roger import RogerSetup, roger_routine, roger_kernel, KernelOutput
from roger.core.operators import numpy as npx, update, at
from pathlib import Path

BASE_PATH = Path(__file__).parent


class DUMMYSetup(RogerSetup):
    """A DUMMY model.
    """

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "DUMMY"

        settings.nx, settings.ny, settings.nz = 1, 1, 1
        settings.nitt = 10
        settings.nittevent = 1
        settings.nittevent_p1 = settings.nittevent + 1
        settings.runlen = 10 * 24 * 60 * 60

        settings.dx = 1
        settings.dy = 1
        settings.dz = 1

        settings.x_origin = 0.0
        settings.y_origin = 0.0

    @roger_routine
    def set_grid(self, state):
        vs = state.variables
        settings = state.settings

        # temporal grid
        vs.DT_SECS = update(vs.DT_SECS, at[:], npx.arange(0, settings.nitt * 24 * 60 * 60, 24 * 60 * 60, dtype=int))
        vs.DT = update(vs.DT, at[:], vs.DT_SECS / (60 * 60))
        vs.YEAR = update(vs.YEAR, at[:], 2018)
        vs.MONTH = update(vs.MONTH, at[:], 6)
        vs.DOY = update(vs.DOY, at[:], npx.arange(182, 182 + settings.nitt, 1, dtype=int))
        vs.dt_secs = vs.DT_SECS[vs.itt]
        vs.dt = vs.DT[vs.itt]
        vs.year = vs.YEAR[vs.itt]
        vs.month = vs.MONTH[vs.itt]
        vs.doy = vs.DOY[vs.itt]
        vs.t = update(vs.t, at[:], npx.cumsum(vs.DT))
        # spatial grid
        vs.x = update(vs.x, at[2:-2], npx.arange(1, settings.nx + 1) * (settings.dx / 2))
        vs.y = update(vs.y, at[2:-2], npx.arange(1, settings.ny + 1) * (settings.dy / 2))

    @roger_routine
    def set_look_up_tables(self, state):
        pass

    @roger_routine
    def set_topography(self, state):
        pass

    @roger_routine
    def set_parameters(self, state):
        pass

    @roger_routine
    def set_initial_conditions(self, state):
        pass

    @roger_routine
    def set_forcing(self, state):
        pass

    @roger_routine
    def set_diagnostics(self, state):
        pass


model = DUMMYSetup()
model.setup()
model.run()
