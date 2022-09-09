import os
import click


def roger_base_cli(func):
    @click.option("-b", "--backend", type=click.Choice(["numpy", "jax"]), default="numpy")
    @click.option("-d", "--device", type=click.Choice(["cpu", "gpu"]), default="cpu")
    @click.option("-n", "--nproc", type=int, nargs=2, default=(1, 1))
    @click.option("--float-type", type=click.Choice(["float64", "float32"]), default="float64")
    @click.option("-v", "--loglevel", type=click.Choice(["debug", "trace", "error"]), default="error")
    @click.option("--log-to-file", is_flag=True)
    @click.option("--log-all-processes", is_flag=True)
    @click.option("--profile-mode", is_flag=True)
    @click.command("roger-run-base", short_help="Run a roger setup")
    def inner(backend, device, nproc, float_type, loglevel, log_to_file, log_all_processes, profile_mode, **kwargs):
        from roger import runtime_settings, runtime_state

        runtime_settings.update(
            backend=backend,
            device=device,
            float_type=float_type,
            num_proc=nproc,
            loglevel=loglevel,
            log_to_file=log_to_file,
            log_all_processes=log_all_processes,
            profile_mode=profile_mode,
        )

        if device == "gpu" and runtime_state.proc_num > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime_state.proc_rank)

        return func(**kwargs)

    return inner
