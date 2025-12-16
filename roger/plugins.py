from collections import namedtuple

from roger.variables import Variable
from roger.settings import Setting

RogerPlugin = namedtuple(
    "RogerPlugin",
    [
        "name",
        "module",
        "setup_entrypoint",
        "run_entrypoint",
        "settings",
        "variables",
        "diagnostics",
    ],
)


def load_plugin(module):
    from roger.diagnostics.base import RogerDiagnostic

    modname = module.__name__

    if not hasattr(module, "__ROGER_INTERFACE__"):
        raise RuntimeError(f"module {modname} is not a valid Roger plugin")

    interface = module.__ROGER_INTERFACE__

    setup_entrypoint = interface.get("setup_entrypoint")

    if not callable(setup_entrypoint):
        raise RuntimeError(f"module {modname} is missing a valid setup entrypoint")

    run_entrypoint = interface.get("run_entrypoint")

    if not callable(run_entrypoint):
        raise RuntimeError(f"module {modname} is missing a valid run entrypoint")

    name = interface.get("name", module.__name__)

    settings = interface.get("settings", [])
    for setting, val in settings.items():
        if not isinstance(val, Setting):
            raise TypeError(f"got unexpected type {type(val)} for setting {setting}")

    variables = interface.get("variables", [])
    for variable, val in variables.items():
        if not isinstance(val, Variable):
            raise TypeError(f"got unexpected type {type(val)} for variable {variable}")

    diagnostics = interface.get("diagnostics", [])
    for diagnostic in diagnostics:
        if not issubclass(diagnostic, RogerDiagnostic):
            raise TypeError(f"got unexpected type {type(diagnostic)} for diagnostic {diagnostic}")

    return RogerPlugin(
        name=name,
        module=module,
        setup_entrypoint=setup_entrypoint,
        run_entrypoint=run_entrypoint,
        settings=settings,
        variables=variables,
        diagnostics=diagnostics,
    )
