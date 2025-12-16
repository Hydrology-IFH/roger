"""Roger, Runoff Generation Research"""

import sys
import types

# black magic: ensure lazy imports for public API by overriding module.__class__


def _reraise_exceptions(func):
    import functools

    @functools.wraps(func)
    def reraise_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise ImportError("Critical error during initial import") from e

    return reraise_wrapper


class _PublicAPI(types.ModuleType):
    @property
    @_reraise_exceptions
    def __version__(self):
        from roger._version import get_versions

        return get_versions()["version"]

    @property
    @_reraise_exceptions
    def logger(self):
        if not hasattr(self, "_logger"):
            from roger.logs import setup_logging

            self._logger = setup_logging()
        return self._logger

    @property
    @_reraise_exceptions
    def runtime_settings(self):
        if not hasattr(self, "_runtime_settings"):
            from roger.runtime import RuntimeSettings

            self._runtime_settings = RuntimeSettings()
        return self._runtime_settings

    @property
    @_reraise_exceptions
    def runtime_state(self):
        if not hasattr(self, "_runtime_state"):
            from roger.runtime import RuntimeState

            self._runtime_state = RuntimeState()
        return self._runtime_state

    @property
    @_reraise_exceptions
    def roger_routine(self):
        from roger.routines import roger_routine

        return roger_routine

    @property
    @_reraise_exceptions
    def roger_kernel(self):
        from roger.routines import roger_kernel

        return roger_kernel

    @property
    @_reraise_exceptions
    def roger_sync(self):
        from roger.routines import roger_sync

        return roger_sync

    @property
    @_reraise_exceptions
    def KernelOutput(self):
        from roger.state import KernelOutput

        return KernelOutput

    @property
    @_reraise_exceptions
    def RogerSetup(self):
        from roger.roger import RogerSetup

        return RogerSetup

    @property
    @_reraise_exceptions
    def RogerState(self):
        from roger.state import RogerState

        return RogerState


sys.modules[__name__].__class__ = _PublicAPI

del sys
del types
del _PublicAPI
del _reraise_exceptions
