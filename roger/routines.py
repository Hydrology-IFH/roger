import functools
import inspect
import threading
from contextlib import ExitStack, contextmanager

from roger import logger

from roger.state import RogerState


# stack helpers


class RoutineStack:
    def __init__(self):
        self.keep_full_stack = False
        self._stack = []
        self._current_idx = []

    @property
    def stack_level(self):
        return len(self._current_idx)

    def append(self, val):
        frame = self._stack
        for i in self._current_idx:
            frame = frame[i][1]

        self._current_idx.append(len(frame))
        frame.append([val, []])

    def pop(self):
        frame = self._stack
        for i in self._current_idx[:-1]:
            frame = frame[i][1]

        if self.keep_full_stack:
            last_val = frame[-1][0]
        else:
            last_val = frame.pop()[0]
        self._current_idx.pop()
        return last_val


# global context

CURRENT_CONTEXT = threading.local()
CURRENT_CONTEXT.is_dist_safe = True
CURRENT_CONTEXT.routine_stack = RoutineStack()
CURRENT_CONTEXT.mpi4jax_token = None


@contextmanager
def nullcontext():
    yield


@contextmanager
def enter_routine(name, routine_obj, timer=None, dist_safe=True):
    from roger import runtime_state as rst
    from roger.distributed import abort

    stack = CURRENT_CONTEXT.routine_stack

    logger.trace("{}> {}", "-" * stack.stack_level, name)
    stack.append(routine_obj)

    reset_dist_safe = False
    if CURRENT_CONTEXT.is_dist_safe:
        if not dist_safe and rst.proc_num > 1:
            CURRENT_CONTEXT.is_dist_safe = False
            reset_dist_safe = True

    timer_ctx = nullcontext() if timer is None else timer

    try:
        with timer_ctx:
            yield

    except:  # noqa: E722
        if reset_dist_safe:
            abort()
        raise

    finally:
        if reset_dist_safe:
            CURRENT_CONTEXT.is_dist_safe = True

        r = stack.pop()
        assert r is routine_obj

        exec_time = ""
        if timer is not None:
            exec_time = f"({timer.last_time:.3f}s)"

        logger.trace("<{} {} {}", "-" * stack.stack_level, name, exec_time)


# helper functions


def _get_func_name(function):
    return f"{inspect.getmodule(function).__name__}:{function.__qualname__}"


def _is_method(function):
    if inspect.ismethod(function):
        return True

    # hack for unbound methods: check if first argument is called "self"
    spec = inspect.getfullargspec(function)
    return spec.args and spec.args[0] == "self"


# routine


def roger_routine(function=None, *, dist_safe=True, local_variables=()):
    """
    .. note::

      This decorator should be applied to all functions that access the roger state object
      (even when subclassing :class:`roger.rogerSetup`).

    The first argument to the decorated function must be a RogerState instance.

    roger routines cannot return anything. All changes must be applied to the passed state object.

    Parameters:
        dist_safe (bool): If set to False, all variables specified in local_variables are synced
            to the root process before execution and synced back after. This means that the routine
            will only be executed on rank 0. Has no effect in non-distributed contexts.

        local_variables (Tuple[str]): List of variable names to be synced if dist_safe=False. This
            must include all variables retrieved from the state object throughout the routine (inputs
            *and* outputs).

    Example:
       >>> from roger import rogerSetup, roger_routine
       >>>
       >>> class MyModel(rogerSetup):
       >>>     @roger_routine
       >>>     def set_topography(self, state):
       >>>         vs = state.variables
       >>>         settings = state.settings
       >>>         vs.kbot = npx.random.randint(0, settings.nz, size=vs.kbot.shape)

    """

    def inner_decorator(function):
        narg = 1 if _is_method(function) else 0
        num_params = len(inspect.signature(function).parameters)
        if narg >= num_params:
            raise TypeError("roger routines must take at least one argument")

        routine = rogerRoutine(function, state_argnum=narg, dist_safe=dist_safe, local_variables=local_variables)
        routine = functools.wraps(function)(routine)
        return routine

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


class rogerRoutine:
    """Do not instantiate directly!"""

    def __init__(self, function, dist_safe=True, local_variables=(), state_argnum=0):
        if isinstance(local_variables, str):
            local_variables = (local_variables,)

        self.function = function
        self.dist_safe = dist_safe
        self.local_variables = local_variables
        self.state_argnum = state_argnum
        self.name = _get_func_name(self.function)

    def __call__(self, *args, **kwargs):
        from roger import runtime_state as rst
        from roger.state import RogerState, DistSafeVariableWrapper
        from roger.core.operators import flush

        roger_state = args[self.state_argnum]

        if not isinstance(roger_state, RogerState):
            raise TypeError(f"Argument {self.state_argnum} to this roger routine must be a RogerState object")

        timer = roger_state.profile_timers[self.name]

        with ExitStack() as es:
            vars_initialized = roger_state._variables is not None

            if vars_initialized:
                es.enter_context(roger_state.variables.unlock())

            execute = True
            restore_vars = False

            if not self.dist_safe:
                orig_vars = roger_state._variables
                if not isinstance(orig_vars, DistSafeVariableWrapper):
                    restore_vars = True
                    roger_state._variables = DistSafeVariableWrapper(orig_vars, self.local_variables)
                    roger_state._variables._gather_variables()

                execute = rst.proc_rank == 0

            routine_ctx = enter_routine(name=self.name, routine_obj=self, timer=timer, dist_safe=self.dist_safe)

            out = None
            try:
                with routine_ctx:
                    if execute:
                        out = self.function(*args, **kwargs)

            finally:
                if restore_vars:
                    roger_state._variables._scatter_variables()
                    roger_state._variables = orig_vars

                flush()

        if out is not None:
            logger.warning(
                f"Routine {self.name} returned object of type {type(out)}. Return objects are silently dropped."
            )

    def __get__(self, instance, _):
        return functools.partial(self.__call__, instance)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name} at {hex(id(self))}>"


# kernel


def roger_kernel(function=None, *, static_args=()):
    """Decorator that marks a function as a kernel that can be JIT compiled if supported
    by the backend.

    Kernels cannot modify the roger state object. Instead, all modifications have to be
    returned explicitly.

    Parameters:
        static_args (Tuple[str]): Names of kernel arguments that should be static.

    Example:
        >>> from roger import roger_kernel, KernelOutput
        >>>
        >>> @roger_kernel
        >>> def double_psi(state):
        >>>     vs = state.variables
        >>>     vs.psi = 2 * vs.psi
        >>>     return KernelOutput(psi=vs.psi)

    """

    def inner_decorator(function):
        kernel = rogerKernel(function, static_args=static_args)
        kernel = functools.wraps(function)(kernel)
        return kernel

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


class rogerKernel:
    """Do not instantiate directly!"""

    def __init__(self, function, static_args=()):
        """Do some parameter introspection."""

        # make sure function signature is in the form we need
        self.name = _get_func_name(function)
        self.func_sig = inspect.signature(function)

        func_params = self.func_sig.parameters

        allowed_param_types = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

        if any(p.kind not in allowed_param_types for p in func_params.values()):
            raise ValueError(f"roger kernels do not support *args, **kwargs, or keyword-only parameters ({self.name})")

        # parse static args
        if isinstance(static_args, str):
            static_args = (static_args,)

        func_argnames = list(func_params.keys())

        self.static_argnums = []
        for static_arg in static_args:
            try:
                arg_index = func_argnames.index(static_arg)
            except ValueError:
                raise ValueError(
                    f'roger kernel {self.name} has no argument "{static_arg}", but it is given in static_args'
                ) from None

            self.static_argnums.append(arg_index)

        self.function = function

    def __call__(self, *args, **kwargs):
        from roger import runtime_settings, runtime_state
        from roger.core.operators import flush

        inject_tokens = runtime_settings.backend == "jax" and runtime_state.proc_num > 1

        # apply JIT
        if runtime_settings.backend == "jax":
            import jax

            CompiledFunction = type(jax.jit(lambda: None))

            if not isinstance(self.function, CompiledFunction):
                if inject_tokens:
                    function = self.function

                    @functools.wraps(function)
                    def token_wrapper(*args):
                        inputs = args[:-1]
                        token = args[-1]
                        CURRENT_CONTEXT.mpi4jax_token = token
                        out = function(*inputs)
                        token = CURRENT_CONTEXT.mpi4jax_token
                        return out, token

                    if CURRENT_CONTEXT.mpi4jax_token is None:
                        CURRENT_CONTEXT.mpi4jax_token = jax.lax.create_token()

                    self.function = token_wrapper

                self.function = jax.jit(self.function, static_argnums=self.static_argnums)

        # JAX only accepts positional args when using static_argnums
        # so convert everything to positional for consistency
        bound_args = self.func_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        roger_state = None
        for argval in bound_args.arguments.values():
            if isinstance(argval, RogerState):
                roger_state = argval
                break

        called_with_state = roger_state is not None

        # when profiling, make sure all inputs are ready before starting the timer
        if runtime_settings.profile_mode:
            flush()

        if called_with_state:
            timer = roger_state.profile_timers[self.name]
        else:
            timer = None

        with ExitStack() as es:
            if called_with_state:
                es.enter_context(roger_state.variables.unlock())

            args = list(bound_args.arguments.values())

            if inject_tokens:
                args.append(CURRENT_CONTEXT.mpi4jax_token)

            with enter_routine(self.name, self, timer):
                out = self.function(*args)

                if runtime_settings.profile_mode:
                    flush()

            if inject_tokens:
                out, token = out
                CURRENT_CONTEXT.mpi4jax_token = token

        return out

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name} at {hex(id(self))}>"


def is_roger_routine(func):
    if isinstance(func, functools.partial):
        func = func.func

    if inspect.ismethod(func):
        func = func.__self__

    return isinstance(func, rogerRoutine)


# sync


def roger_sync(function=None):
    """Decorator that marks a function as a sync that is run on a single process
    if backend is set to MPI.

    """

    def inner_decorator(function):
        sync = rogerSync(function)
        sync = functools.wraps(function)(sync)
        return sync

    if function is not None:
        return inner_decorator(function)

    return inner_decorator


class rogerSync:
    """Do not instantiate directly!"""

    def __init__(self, function):
        """Do some parameter introspection."""

        # make sure function signature is in the form we need
        self.name = _get_func_name(function)
        self.func_sig = inspect.signature(function)

        func_params = self.func_sig.parameters

        allowed_param_types = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)

        if any(p.kind not in allowed_param_types for p in func_params.values()):
            raise ValueError(f"roger syncs do not support *args, **kwargs, or keyword-only parameters ({self.name})")

        self.function = function

    def __call__(self, *args, **kwargs):
        from roger import runtime_settings, runtime_state
        from roger.distributed import barrier
        import numpy as onp

        # no MPI support
        if runtime_state.proc_num == 1:
            out = self.function(*args, **kwargs)

        # with MPI support
        elif runtime_state.proc_num > 1:
            # run function on a single process
            barrier()
            if runtime_state.proc_rank != 0:
                out = None
                buffer = onp.ascontiguousarray(onp.empty((10,), dtype=float))
                runtime_settings.mpi_comm.Send(buffer, dest=0, tag=11)
            # let other processes wait
            elif runtime_state.proc_rank == 0:
                for proc in range(1, runtime_state.proc_num):
                    buffer = onp.ascontiguousarray(onp.empty((10,), dtype=float))
                    buffer = buffer.copy()
                    runtime_settings.mpi_comm.Recv(buffer, source=proc, tag=11)
                out = self.function(*args, **kwargs)

        return out

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name} at {hex(id(self))}>"
