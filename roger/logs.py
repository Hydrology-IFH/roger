import sys
import warnings


LOGLEVELS = ("trace", "debug", "info", "warning", "error")


def _inject_proc_rank(record):
    from roger import runtime_state

    return record["extra"].update(proc_rank=runtime_state.proc_rank)


def setup_logging(loglevel="info", stream_sink=sys.stdout, log_all_processes=False, log_to_file=False):
    from loguru import logger

    handler_conf = dict(
        sink=stream_sink,
        level=loglevel.upper(),
        colorize=sys.stdout.isatty(),
    )

    if not hasattr(logger, "diagnostic"):
        logger.level("DIAGNOSTIC", no=45)

    logger.level("TRACE", color="<dim>")
    logger.level("DEBUG", color="<dim><cyan>")
    logger.level("INFO", color="")
    logger.level("SUCCESS", color="<dim><green>")
    logger.level("WARNING", color="<magenta>")
    logger.level("ERROR", color="<bold><red>")
    logger.level("DIAGNOSTIC", color="<bold><magenta>")
    logger.level("CRITICAL", color="<bold><red><WHITE>")

    logger = logger.patch(_inject_proc_rank)
    if log_all_processes:
        handler_conf.update(format="{extra[proc_rank]} | <level>{message}</level>")
    else:
        handler_conf.update(format="<level>{message}</level>", filter=lambda record: record["extra"]["proc_rank"] == 0)

    def diagnostic(_, message, *args, **kwargs):
        logger.opt(depth=1).log("DIAGNOSTIC", message, *args, **kwargs)

    logger.__class__.diagnostic = diagnostic

    def showwarning(message, cls, source, lineno, *args):
        logger.warning(
            "{warning}: {message} ({source}:{lineno})",
            message=message,
            warning=cls.__name__,
            source=source,
            lineno=lineno,
        )

    warnings.showwarning = showwarning

    logger.configure(handlers=[handler_conf])
    logger.enable("roger")
    if log_to_file:
        logger.add("out.log", backtrace=True, diagnose=True, level=loglevel)

    return logger
