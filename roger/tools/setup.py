from roger.core.operators import numpy as npx
import numpy as onp

import scipy.interpolate
import scipy.spatial


def interpolate(coords, var, interp_coords, missing_value=None, fill=True, kind="linear"):
    """Interpolate globally defined data to a different (regular) grid.

    Arguments:
       coords: Tuple of coordinate arrays for each dimension.
       var (:obj:`ndarray` of dim (nx1, ..., nxd)): Variable data to interpolate.
       interp_coords: Tuple of coordinate arrays to interpolate to.
       missing_value (optional): Value denoting cells of missing data in ``var``.
          Is replaced by `NaN` before interpolating. Defaults to `None`, which means
          no replacement is taking place.
       fill (bool, optional): Whether `NaN` values should be replaced by the nearest
          finite value after interpolating. Defaults to ``True``.
       kind (str, optional): Order of interpolation. Supported are `nearest` and
          `linear` (default).

    Returns:
       :obj:`ndarray` containing the interpolated values on the grid spanned by
       ``interp_coords``.

    """
    if len(coords) != len(interp_coords) or len(coords) != var.ndim:
        raise ValueError("Dimensions of coordinates and values do not match")

    if missing_value is not None:
        invalid_mask = npx.isclose(var, missing_value)
        var = npx.where(invalid_mask, npx.nan, var)

    if var.ndim > 1 and coords[0].ndim == 1:
        interp_grid = npx.rollaxis(npx.array(npx.meshgrid(*interp_coords, indexing="ij")), 0, len(interp_coords) + 1)
    else:
        interp_grid = interp_coords

    coords = [onp.array(c) for c in coords]
    var = scipy.interpolate.interpn(
        coords, onp.array(var), interp_grid, bounds_error=False, fill_value=npx.nan, method=kind
    )
    var = npx.asarray(var)

    if fill:
        var = fill_holes(var)

    return var


def fill_holes(data):
    """A simple inpainting function that replaces NaN values in `data` with the
    nearest finite value.
    """
    data = onp.array(data)
    dim = data.ndim
    flag = ~onp.isnan(data)

    slcs = [slice(None)] * dim

    while onp.any(~flag):
        for i in range(dim):
            slcs1 = slcs[:]
            slcs2 = slcs[:]
            slcs1[i] = slice(0, -1)
            slcs2[i] = slice(1, None)

            slcs1 = tuple(slcs1)
            slcs2 = tuple(slcs2)

            # replace from the right
            repmask = onp.logical_and(~flag[slcs1], flag[slcs2])
            data[slcs1][repmask] = data[slcs2][repmask]
            flag[slcs1][repmask] = True

            # replace from the left
            repmask = onp.logical_and(~flag[slcs2], flag[slcs1])
            data[slcs2][repmask] = data[slcs1][repmask]
            flag[slcs2][repmask] = True

    return npx.asarray(data)


def get_uniform_grid_steps(total_length, stepsize):
    """Get uniform grid step sizes in an interval.

    Arguments:
        total_length (float): total length of the resulting grid
        stepsize (float): grid step size

    Returns:
        :obj:`ndarray` of grid steps

    Example:
        >>> uniform_steps = uniform_grid_setup(6., 0.25)
        >>> uniform_steps
        [ 0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
          0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,  0.25,
          0.25,  0.25,  0.25,  0.25,  0.25,  0.25 ]

    """
    if total_length % stepsize:
        raise ValueError("total length must be an integer multiple of stepsize")
    return stepsize * npx.ones(int(total_length / stepsize))
