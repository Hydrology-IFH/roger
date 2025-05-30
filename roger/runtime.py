import os
from threading import local
from collections import namedtuple

from roger.backend import BACKENDS
from roger.logs import LOGLEVELS


# globals
log_args = local()
log_args.log_all_processes = False
log_args.loglevel = "info"
log_args.log_to_file = True


# MPI helpers


def _default_mpi_comm():
    try:
        from mpi4py import MPI
    except ImportError:
        return None
    else:
        return MPI.COMM_WORLD


# validators


def parse_two_ints(v):
    return (int(v[0]), int(v[1]))


def parse_choice(choices, preserve_case=False):
    def validate(choice):
        if isinstance(choice, str) and not preserve_case:
            choice = choice.lower()

        if choice not in choices:
            raise ValueError(f"must be one of {choices}")

        return choice

    return validate


def parse_bool(obj):
    if not isinstance(obj, str):
        return bool(obj)

    return obj.lower() in {"1", "true", "on"}


def check_mpi_comm(comm):
    if comm is not None:
        from mpi4py import MPI

        if not isinstance(comm, MPI.Comm):
            raise TypeError("mpi_comm must be Comm instance or None")

    return comm


def set_loglevel(val):
    from roger import logs

    log_args.loglevel = parse_choice(LOGLEVELS)(val)
    logs.setup_logging(
        loglevel=log_args.loglevel, log_all_processes=log_args.log_all_processes, log_to_file=log_args.log_to_file
    )
    return log_args.loglevel


def set_log_to_file(val):
    from roger import logs

    log_args.log_to_file = parse_bool(val)
    logs.setup_logging(
        loglevel=log_args.loglevel, log_all_processes=log_args.log_all_processes, log_to_file=log_args.log_to_file
    )
    return log_args.log_to_file


def set_log_all_processes(val):
    from roger import logs

    log_args.log_all_processes = parse_bool(val)
    logs.setup_logging(loglevel=log_args.loglevel, log_all_processes=log_args.log_all_processes)
    return log_args.log_all_processes


DEVICES = ("cpu", "gpu", "tpu")
FLOAT_TYPES = ("float64", "float32")
INT_TYPES = ("int64", "int32")


# settings

RuntimeSetting = namedtuple("RuntimeSetting", ("type", "default", "read_from_env"))
RuntimeSetting.__new__.__defaults__ = (None, None, True)

AVAILABLE_SETTINGS = {
    "backend": RuntimeSetting(parse_choice(BACKENDS), "numpy"),
    "device": RuntimeSetting(parse_choice(DEVICES), "cpu"),
    "float_type": RuntimeSetting(parse_choice(FLOAT_TYPES), "float32"),
    "int_type": RuntimeSetting(parse_choice(INT_TYPES), "int32"),
    "petsc_options": RuntimeSetting(str, ""),
    "monitor_water_balance": RuntimeSetting(parse_bool, False),
    "monitor_tracer_balance": RuntimeSetting(parse_bool, False),
    "num_proc": RuntimeSetting(parse_two_ints, (1, 1), read_from_env=False),
    "profile_mode": RuntimeSetting(parse_bool, False),
    "log_to_file": RuntimeSetting(set_log_to_file, False),
    "loglevel": RuntimeSetting(set_loglevel, "info"),
    "mpi_comm": RuntimeSetting(check_mpi_comm, _default_mpi_comm(), read_from_env=False),
    "log_all_processes": RuntimeSetting(set_log_all_processes, False),
    "use_io_threads": RuntimeSetting(parse_bool, False),
    "io_timeout": RuntimeSetting(float, 20),
    "hdf5_gzip_compression": RuntimeSetting(bool, True),
    "force_overwrite": RuntimeSetting(bool, False),
    "diskless_mode": RuntimeSetting(bool, False),
}


class RuntimeSettings:
    __slots__ = ["__locked__", "__setting_types__", "__settings__", *AVAILABLE_SETTINGS.keys()]

    def __init__(self, **kwargs):
        self.__locked__ = False
        self.__setting_types__ = {}

        for name, setting in AVAILABLE_SETTINGS.items():
            setting_envvar = f"ROGER_{name.upper()}"

            if name in kwargs:
                val = kwargs[name]
            elif setting.read_from_env:
                val = os.environ.get(setting_envvar, setting.default)
            else:
                val = setting.default

            self.__setting_types__[name] = setting.type
            self.__setattr__(name, val)

        self.__settings__ = set(self.__setting_types__.keys())

    def update(self, **kwargs):
        for key, val in kwargs.items():
            if key == "float_type":
                setattr(self, key, val)
                if val == "float64":
                    setattr(self, "int_type", "int64")
                elif val == "float32":
                    setattr(self, "int_type", "int32")
            else:
                setattr(self, key, val)

        return self

    def __setattr__(self, attr, val):
        if getattr(self, "__locked__", False):
            raise RuntimeError("Runtime settings cannot be modified after import of core modules")

        if attr.startswith("_"):
            return super().__setattr__(attr, val)

        # coerce type
        stype = self.__setting_types__.get(attr)
        if stype is not None:
            try:
                val = stype(val)
            except (TypeError, ValueError) as e:
                raise ValueError(f'Got invalid value for runtime setting "{attr}": {e!s}') from None

        return super().__setattr__(attr, val)

    def __repr__(self):
        setval = ", ".join(f"{key}={repr(getattr(self, key))}" for key in self.__settings__)
        return f"{self.__class__.__name__}({setval})"


# state


class RuntimeState:
    """Unifies attributes from various modules in a simple read-only object"""

    __slots__ = ()

    @property
    def proc_rank(self):
        from roger import runtime_settings

        comm = runtime_settings.mpi_comm

        if comm is None:
            return 0

        return comm.Get_rank()

    @property
    def proc_num(self):
        from roger import runtime_settings

        comm = runtime_settings.mpi_comm

        if comm is None:
            return 1

        return comm.Get_size()

    @property
    def proc_idx(self):
        from roger import distributed

        return distributed.proc_rank_to_index(self.proc_rank)

    @property
    def backend_module(self):
        from roger import backend, runtime_settings

        return backend.get_backend_module(runtime_settings.backend)

    @property
    def current_device(self):
        from roger import backend

        return backend.get_curent_device_name()

    def __setattr__(self, attr, val):
        raise TypeError(f"Cannot modify {self.__class__.__name__} objects")
