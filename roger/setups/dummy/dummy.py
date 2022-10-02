from roger import RogerSetup, roger_routine


class DUMMYSetup(RogerSetup):
    """A DUMMY model.
    """

    @roger_routine
    def set_settings(self, state):
        settings = state.settings
        settings.identifier = "DUMMY"

        settings.nx, settings.ny = 1, 1
        settings.nitt = 24
        settings.runlen = 24 * 60 * 60

        settings.dx = 1
        settings.dy = 1

        settings.x_origin = 0.0
        settings.y_origin = 0.0

    @roger_routine
    def set_grid(self, state):
        pass

    @roger_routine
    def set_look_up_tables(self, state):
        pass

    @roger_routine
    def set_topography(self, state):
        pass

    @roger_routine
    def set_parameters_setup(self, state):
        pass

    @roger_routine
    def set_parameters(self, state):
        pass

    @roger_routine
    def set_initial_conditions_setup(self, state):
        pass

    @roger_routine
    def set_initial_conditions(self, state):
        pass

    @roger_routine
    def set_forcing_setup(self, state):
        pass

    @roger_routine
    def set_forcing(self, state):
        pass

    @roger_routine
    def set_diagnostics(self, state):
        pass

    @roger_routine
    def after_timestep(self, state):
        pass
