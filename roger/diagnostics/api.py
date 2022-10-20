from roger import logger, time


def create_default_diagnostics(state):
    # do not import these at module level to make sure core import is deferred
    from roger.diagnostics.average import Average
    from roger.diagnostics.snapshot import Snapshot
    from roger.diagnostics.collect import Collect
    from roger.diagnostics.constant import Constant
    from roger.diagnostics.rate import Rate
    from roger.diagnostics.minimum import Minimum
    from roger.diagnostics.maximum import Maximum
    from roger.diagnostics.tracer_monitor import TracerMonitor
    from roger.diagnostics.water_monitor import WaterMonitor

    return {Diag.name: Diag(state) for Diag in (Average, Snapshot, Collect, Constant, Rate, Minimum, Maximum, TracerMonitor, WaterMonitor)}


def initialize(state):
    for name, diagnostic in state.diagnostics.items():
        diagnostic.initialize(state)

        if diagnostic.output_frequency:
            t, unit = time.format_time(diagnostic.output_frequency)
            logger.info(f' Writing output for diagnostic "{name}" every {t:.1f} {unit}')


def diagnose(state):
    for diagnostic in state.diagnostics.values():
        if diagnostic.sampling_frequency:
            diagnostic.diagnose(state)


def output(state):
    vs = state.variables

    for diagnostic in state.diagnostics.values():
        # daily
        if diagnostic.output_frequency == 24 * 60 * 60 and vs.time % (24 * 60 * 60) == 0:
            diagnostic.output(state)
        # hourly
        elif diagnostic.output_frequency == 60 * 60 and ((vs.time % (60 * 60) == 0) or (vs.time % (24 * 60 * 60) == 0)):
            diagnostic.output(state)
        # 10 minutes
        elif diagnostic.output_frequency == 10 * 60 and ((vs.time % (10 * 60) == 0) or (vs.time % (60 * 60) == 0) or (vs.time % (24 * 60 * 60) == 0)):
            diagnostic.output(state)
        # sampling of constant values
        elif diagnostic.output_frequency == 0 and ((vs.time == 10 * 60) or (vs.time == 60 * 60) or (vs.time == 24 * 60 * 60)):
            diagnostic.output(state)
