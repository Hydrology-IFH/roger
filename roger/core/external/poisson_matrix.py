from veros.core.operators import update, at, numpy as npx
from veros.variables import allocate


def assemble_poisson_matrix(state):
    return assemble_pressure_matrix(state)


def assemble_pressure_matrix(state):
    main_diag = allocate(state.dimensions, ("x", "y"), fill=1)
    east_diag, west_diag, north_diag, south_diag = (allocate(state.dimensions, ("x", "y")) for _ in range(4))

    vs = state.variables
    settings = state.settings

    maskC = vs.maskCatch[:, :, -1]

    mp_i = maskC[2:-2, 2:-2] * maskC[3:-1, 2:-2]
    mm_i = maskC[2:-2, 2:-2] * maskC[1:-3, 2:-2]

    mp_j = maskC[2:-2, 2:-2] * maskC[2:-2, 3:-1]
    mm_j = maskC[2:-2, 2:-2] * maskC[2:-2, 1:-3]

    main_diag = update(
        main_diag,
        at[2:-2, 2:-2],
        -1
        * mp_i
        * vs.z0[2:-2, 2:-2]
        / settings.dx
        - 1
        * mm_i
        * vs.z0[1:-3, 2:-2]
        / settings.dx
        - 1
        * mp_j
        * vs.z0[2:-2, 2:-2]
        / settings.dy
        - 1
        * mm_j
        * vs.z0[2:-2, 1:-3]
        / settings.dy
        * maskC[2:-2, 2:-2],
    )

    east_diag = update(
        east_diag,
        at[2:-2, 2:-2],
        mp_i
        * vs.z0[2:-2, 2:-2]
        / settings.dx,
    )

    west_diag = update(
        west_diag,
        at[2:-2, 2:-2],
        mm_i
        * vs.z0[1:-3, 2:-2]
        / settings.dx,
    )

    north_diag = update(
        north_diag,
        at[2:-2, 2:-2],
        mp_j
        * vs.z0[2:-2, 2:-2]
        / settings.dy,
    )

    south_diag = update(
        south_diag,
        at[2:-2, 2:-2],
        mm_j
        * vs.z0[2:-2, 1:-3]
        / settings.dy,
    )
    main_diag = main_diag * maskC
    main_diag = npx.where(npx.abs(main_diag) == 0.0, 1, main_diag)

    offsets = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    diags = [main_diag, east_diag, west_diag, north_diag, south_diag]

    return diags, offsets, maskC
