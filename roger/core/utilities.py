from roger import runtime_settings
from roger.core.operators import numpy as npx, update, at, where
from roger import roger_kernel


def _get_row_no(arr1d, i, size=1, fill_value=0):
    arr = npx.full((npx.size(arr1d),), dtype=bool, fill_value=False)
    arr = update(
        arr,
        at[:], arr1d == i,
    )
    cond = npx.any(arr)
    if where(arr, size=size, fill_value=fill_value)[0].size > 0:
        val = where(arr, size=size, fill_value=fill_value)[0][0]
    else:
        val = 0
    row_no = npx.int64(npx.where(cond, val, 0))

    arr_row_no = npx.full((1,), dtype=int, fill_value=False)
    arr_row_no = update(
        arr_row_no,
        at[:], row_no,
    )

    return arr_row_no


@roger_kernel(static_args=("local"))
def enforce_boundaries(arr, local=False):
    from roger.distributed import exchange_overlap

    arr = exchange_overlap(arr, ["x", "y"])
    return arr


@roger_kernel
def pad_z_edges(array):
    """
    Pads the z-axis of an array by repeating its edge values
    """
    if array.ndim == 1:
        newarray = npx.pad(array, 1, mode="edge")
    elif array.ndim >= 3:
        newarray = npx.pad(array, ((0, 0), (0, 0), (1, 1)), mode="edge")
    else:
        raise ValueError("Array to pad needs to have 1 or at least 3 dimensions")
    return newarray


@roger_kernel(static_args=("nz"))
def create_catch_masks(ks, nz):
    ks = ks - 1
    land_mask = ks >= 0
    catch_mask = npx.logical_and(
        land_mask[:, :, npx.newaxis], npx.arange(nz)[npx.newaxis, npx.newaxis, :] >= ks[:, :, npx.newaxis]
    )
    edge_mask = npx.logical_and(
        land_mask[:, :, npx.newaxis], npx.arange(nz)[npx.newaxis, npx.newaxis, :] == ks[:, :, npx.newaxis]
    )
    return land_mask, catch_mask, edge_mask


def linear_regression_numpy(x, y, params):
    import numpy as np
    for i in range(params.shape[0]):
        for j in range(params.shape[1]):
            w = np.where(x[i, j, :] > 0, 1/np.max(x[i, j, :]), 0)
            coeff_intcpt = np.polyfit(x[i, j, :], y[i, j, :], 1, w=w)
            params = update(params, at[i, j, 0], coeff_intcpt[0])
            params = update(params, at[i, j, 1], coeff_intcpt[1])

    return params


def linear_regression_jax(x, y, params):
    import jax
    LEARNING_RATE = 0.005

    def init(params):
        """Returns the initial model params."""
        rng = jax.random.PRNGKey(42)
        weights_key, bias_key = jax.random.split(rng)
        params = update(params, at[:, :, 0], jax.random.normal(weights_key, (params.shape[0], params.shape[1])))
        params = update(params, at[:, :, 1], jax.random.normal(bias_key, (params.shape[0], params.shape[1])))
        return params

    def loss(params, xs, ys):
        """Computes the least squares error of the model's predictions on x against y."""
        weights = jax.numpy.where(xs > 0, 1/jax.numpy.max(xs, axis=-1), jax.numpy.NaN)
        pred = params[:, :, 0, jax.numpy.newaxis] * xs * weights + params[:, :, 1, jax.numpy.newaxis]
        return jax.numpy.nanmean((pred - ys) ** 2, axis=-1)

    @roger_kernel
    def update(params, xs, ys):
        """Performs one SGD update step on params using the given data."""
        grad = jax.grad(loss)(params, xs, ys)
        params = update(params, at[:, :, :], jax.tree_multimap(lambda param, g: param - g * LEARNING_RATE, params, grad))

        return params

    params = update(params, at[:, :, :], init(x.shape[0], x.shape[1]))
    for _ in range(1000):
        params = update(params, x, y)

    return params


if runtime_settings.backend == "numpy":
    linear_regression = linear_regression_numpy

elif runtime_settings.backend == "jax":
    linear_regression = linear_regression_jax
