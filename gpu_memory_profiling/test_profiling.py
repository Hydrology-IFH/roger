from pathlib import Path
import jax
import jax.numpy as jnp
import jax.profiler


def main():
    base_path = Path(__file__).parent

    def func1(x):
      return jnp.tile(x, 10) * 0.5

    def func2(x):
      y = func1(x)
      return y, jnp.tile(x, 10) + 1


    x = jax.random.normal(jax.random.PRNGKey(42), (1000, 1000))
    y, z = func2(x)

    z.block_until_ready()

    file = base_path / "test_memory.prof"
    jax.profiler.save_device_memory_profile(file)

    return


if __name__ == "__main__":
    main()
