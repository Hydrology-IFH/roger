import warnings

BACKENDS = ("numpy", "jax")

BACKEND_MESSAGES = {"jax": "Kernels are compiled during first iteration, be patient"}

_init_done = set()


def init_jax_config():
    if "jax" in _init_done:
        return

    import jax
    from roger import runtime_settings, runtime_state
    from roger.state import (
        RogerState,
        RogerVariables,
        DistSafeVariableWrapper,
        roger_state_pytree_flatten,
        roger_state_pytree_unflatten,
        roger_variables_pytree_flatten,
        roger_variables_pytree_unflatten,
        dist_safe_wrapper_pytree_flatten,
        dist_safe_wrapper_pytree_unflatten,
    )

    if runtime_state.proc_num > 1:
        try:
            import mpi4jax  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Running JAX with MPI requires mpi4jax to be installed") from exc

    if runtime_settings.float_type == "float64":
        jax.config.update("jax_enable_x64", True)
    else:
        # ignore warnings about unavailable x64 types
        warnings.filterwarnings("ignore", message="Explicitly requested dtype.*", module="jax")

    jax.config.update("jax_platform_name", runtime_settings.device)

    jax.tree_util.register_pytree_node(RogerState, roger_state_pytree_flatten, roger_state_pytree_unflatten)
    jax.tree_util.register_pytree_node(RogerVariables, roger_variables_pytree_flatten, roger_variables_pytree_unflatten)
    jax.tree_util.register_pytree_node(
        DistSafeVariableWrapper, dist_safe_wrapper_pytree_flatten, dist_safe_wrapper_pytree_unflatten
    )

    _init_done.add("jax")


def get_backend_module(backend_name):
    if backend_name not in BACKENDS:
        raise ValueError(f"unrecognized backend {backend_name} (must be either of: {list(BACKENDS.keys())!r})")

    backend_module = None

    if backend_name == "jax":
        try:
            import jax  # noqa: F401
        except ImportError:
            pass
        else:
            init_jax_config()
            import jax.numpy as backend_module

    elif backend_name == "numpy":
        import numpy as backend_module

    if backend_module is None:
        raise ValueError(f'backend "{backend_name}" failed to import')

    return backend_module


def get_curent_device_name():
    from roger import runtime_settings

    if runtime_settings.backend != "jax":
        return "cpu"

    return runtime_settings.device
