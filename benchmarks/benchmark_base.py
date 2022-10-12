import os
import click


def benchmark_cli(func):
    @click.option("--size", type=int, nargs=2, required=True)
    @click.option("--timesteps", type=int, required=True)
    @click.option("-b", "--backend", type=click.Choice(["numpy", "jax"]), default="numpy")
    @click.option("-d", "--device", type=click.Choice(["cpu", "gpu"]), default="cpu")
    @click.option("-n", "--nproc", type=int, nargs=2, default=(1, 1))
    @click.option("--float-type", type=click.Choice(["float64", "float32"]), default="float64")
    @click.option("-v", "--loglevel", type=click.Choice(["debug", "trace"]), default="debug")
    @click.command()
    def inner(backend, device, nproc, float_type, loglevel, **kwargs):
        from roger import runtime_settings, runtime_state

        runtime_settings.update(
            backend=backend,
            device=device,
            float_type=float_type,
            num_proc=nproc,
            loglevel=loglevel,
            profile_mode=True,
        )

        if device == "gpu" and runtime_state.proc_num > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(runtime_state.proc_rank)

        return func(**kwargs)

    return inner
