import warnings
from contextlib import contextmanager

from roger import runtime_settings, runtime_state, roger_kernel


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
    return np.random.uniform(lower, upper, size=shape[0]*shape[1]).reshape(shape)


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


@roger_kernel(static_args=("use_ext",))
def solve_tridiagonal_jax(a, b, c, d, water_mask, edge_mask, use_ext=None):
    import jax.lax
    import jax.numpy as jnp

    from roger.core.special.tdma_ import tdma, HAS_CPU_EXT, HAS_GPU_EXT

    if use_ext is None:
        use_ext = (HAS_CPU_EXT and runtime_settings.device == "cpu") or (
            HAS_GPU_EXT and runtime_settings.device == "gpu"
        )

    if use_ext:
        return tdma(a, b, c, d, water_mask, edge_mask)

    warnings.warn("Could not use custom TDMA implementation, falling back to pure JAX")

    a = water_mask * a * jnp.logical_not(edge_mask)
    b = jnp.where(water_mask, b, 1.0)
    c = water_mask * c
    d = water_mask * d

    def compute_primes(last_primes, x):
        last_cp, last_dp = last_primes
        a, b, c, d = x
        cp = c / (b - a * last_cp)
        dp = (d - a * last_dp) / (b - a * last_cp)
        new_primes = (cp, dp)
        return new_primes, new_primes

    diags_transposed = [jnp.moveaxis(arr, 2, 0) for arr in (a, b, c, d)]
    init = jnp.zeros(a.shape[:-1], dtype=a.dtype)
    _, primes = jax.lax.scan(compute_primes, (init, init), diags_transposed)

    def backsubstitution(last_x, x):
        cp, dp = x
        new_x = dp - cp * last_x
        return new_x, new_x

    _, sol = jax.lax.scan(backsubstitution, init, primes, reverse=True)
    return jnp.moveaxis(sol, 0, 2)


def update_jax(arr, at, to):
    return arr.at[at].set(to)


def update_add_jax(arr, at, to):
    return arr.at[at].add(to)


def update_multiply_jax(arr, at, to):
    return arr.at[at].multiply(to)


def random_uniform_jax(lower, upper, shape):
    import jax
    key = jax.random.PRNGKey(42)
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
    solve_tridiagonal = solve_tridiagonal_numpy
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
    solve_tridiagonal = solve_tridiagonal_jax
    for_loop = jax.lax.fori_loop
    scan = jax.lax.scan
    flush = flush_jax
    where = jax.numpy.where
    random_uniform = random_uniform_jax
    scipy_special = jax.scipy.special
    scipy_stats = jax.scipy.stats

else:
    raise ValueError(f"Unrecognized backend {runtime_settings.backend}")
