from benchmark_base import benchmark_cli


@benchmark_cli
def main(size, timesteps):
    from roger import roger_routine, roger_kernel, KernelOutput
    from roger.models.dummy import DUMMYSetup
    from roger.variables import allocate
    from roger.core.operators import numpy as npx, update, at, random_uniform

    class DUMMY2Benchmark(DUMMYSetup):

        @roger_routine
        def set_settings(self, state):
            settings = state.settings
            settings.identifier = "DUMMY2Benchmark"

            # total grid numbers in x- and y-direction
            settings.nx, settings.ny = size
            settings.runlen = 24 * 60 * 60 * timesteps

            # spatial discretization (in meters)
            settings.dx = 1
            settings.dy = 1

            # temporal discretization (in hours)

            settings.x_origin = 0.0
            settings.y_origin = 0.0

        @roger_routine
        def set_grid(self, state):
            vs = state.variables
            # spatial grid
            dx = allocate(state.dimensions, ("x"))
            dx = update(dx, at[:], 1)
            dy = allocate(state.dimensions, ("y"))
            dy = update(dy, at[:], 1)
            vs.x = update(vs.x, at[3:-2], npx.cumsum(dx[3:-2]))
            vs.y = update(vs.y, at[3:-2], npx.cumsum(dy[3:-2]))

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
        def set_boundary_conditions_setup(self, state):
            pass

        @roger_routine
        def set_boundary_conditions(self, state):
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
            vs = state.variables

            vs.update(after_timestep_kernel(state))

    @roger_kernel
    def after_timestep_kernel(state):
        vs = state.variables

        a = allocate(state.dimensions, ("x", "y"))
        b = allocate(state.dimensions, ("x", "y"))
        c = allocate(state.dimensions, ("x", "y"))
        a = update(
            a,
            at[2:-2, 2:-2], random_uniform(1, 255, tuple((a.shape[0], a.shape[1])))[2:-2, 2:-2],
        )
        b = update(
            b,
            at[2:-2, 2:-2], random_uniform(0.01, 0.99, tuple((b.shape[0], b.shape[1])))[2:-2, 2:-2],
        )
        c = update(
            c,
            at[2:-2, 2:-2], a[2:-2, 2:-2] * b[2:-2, 2:-2],
        )

        d = allocate(state.dimensions, ("x", "y"))
        d = update(
            d,
            at[2:-2, 2:-2], npx.fft.fft2(c[2:-2, 2:-2]),
        )

        vs.ta = update(
            vs.ta,
            at[2:-2, 2:-2, vs.tau], d[2:-2, 2:-2],
        )

        return KernelOutput(
            ta=vs.ta,
        )

    model = DUMMY2Benchmark()
    model.setup()
    model.run()
    return


if __name__ == "__main__":
    main()
