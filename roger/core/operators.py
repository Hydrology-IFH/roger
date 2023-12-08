from contextlib import contextmanager
from time import time_ns
import os

from roger import runtime_settings, runtime_state


class Index:
    __slots__ = ()

    @staticmethod
    def __getitem__(key):
        return key


def noop(*args, **kwargs):
    pass


@contextmanager
def make_writeable(*arrs):
    orig_writeable = [arr.flags.writeable for arr in arrs]
    writeable_arrs = []
    try:
        for arr in arrs:
            arr = arr.copy()
            arr.flags.writeable = True
            writeable_arrs.append(arr)

        if len(writeable_arrs) == 1:
            yield writeable_arrs[0]
        else:
            yield writeable_arrs

    finally:
        for arr, orig_val in zip(writeable_arrs, orig_writeable):
            try:
                arr.flags.writeable = orig_val
            except ValueError:
                pass


def update_numpy(arr, at, to):
    with make_writeable(arr) as warr:
        warr[at] = to
    return warr


def update_add_numpy(arr, at, to):
    with make_writeable(arr) as warr:
        warr[at] += to
    return warr


def update_multiply_numpy(arr, at, to):
    with make_writeable(arr) as warr:
        warr[at] *= to
    return warr


def solve_tridiagonal_numpy(a, b, c, d, water_mask, edge_mask):
    import numpy as np
    from scipy.linalg import lapack

    out = np.zeros(a.shape, dtype=a.dtype)

    if not np.any(water_mask):
        return out

    # remove couplings between slices
    with make_writeable(a, c) as warr:
        a, c = warr
        a[edge_mask] = 0
        c[..., -1] = 0

    sol = lapack.dgtsv(a[water_mask][1:], b[water_mask], c[water_mask][:-1], d[water_mask])[3]
    out[water_mask] = sol
    return out


def fori_numpy(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def where_numpy(*args, **kwargs):
    import numpy as np
    if args.__len__() == 1:
        return np.where(args[0])
    elif args.__len__() == 3:
        return np.where(args[0], args[1], args[2])


def random_uniform_numpy(lower, upper, shape):
    import numpy as np
    seed = int(time_ns() + os.getpid())
    rng = np.random.default_rng(seed)
    return rng.uniform(lower, upper, size=shape[0]*shape[1]).reshape(shape)


def scan_numpy(f, init, xs, length=None):
    import numpy as np

    if xs is None:
        xs = [None] * length

    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)

    return carry, np.stack(ys)


def update_jax(arr, at, to):
    return arr.at[at].set(to)


def update_add_jax(arr, at, to):
    return arr.at[at].add(to)


def update_multiply_jax(arr, at, to):
    return arr.at[at].multiply(to)


def random_uniform_jax(lower, upper, shape):
    import jax
    seed = int(time_ns() + os.getpid())
    key = jax.random.PRNGKey(seed)
    return jax.random.uniform(key, shape=shape, minval=lower, maxval=upper)


def flush_jax():
    import jax

    dummy = jax.device_put(0.0) + 0.0
    try:
        dummy.block_until_ready()
    except AttributeError:
        # if we are jitting, dummy is not a DeviceArray that we can wait for
        pass


numpy = runtime_state.backend_module

if runtime_settings.backend == "numpy":
    import scipy.special
    import scipy.stats

    update = update_numpy
    update_add = update_add_numpy
    update_multiply = update_multiply_numpy
    at = Index()
    for_loop = fori_numpy
    scan = scan_numpy
    flush = noop
    where = where_numpy
    random_uniform = random_uniform_numpy
    scipy_special = scipy.special
    scipy_stats = scipy.stats
    numpy.seterr(all="ignore")

elif runtime_settings.backend == "jax":
    import jax.lax
    import jax.numpy
    import jax.scipy.special
    import jax.scipy.stats

    update = update_jax
    update_add = update_add_jax
    update_multiply = update_multiply_jax
    at = Index()
    for_loop = jax.lax.fori_loop
    scan = jax.lax.scan
    flush = flush_jax
    where = jax.numpy.where
    random_uniform = random_uniform_jax
    scipy_special = jax.scipy.special
    scipy_stats = jax.scipy.stats

else:
    raise ValueError(f"Unrecognized backend {runtime_settings.backend}")
