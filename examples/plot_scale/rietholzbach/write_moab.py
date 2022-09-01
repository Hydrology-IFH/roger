from pathlib import Path
import os
import subprocess
import click


@click.option("--job-type", type=click.Choice(['serial', 'single-node', 'multi-node', 'single-node-gpu', 'single-node-multi-gpu', 'multi-node-multi-gpu']), default='serial')
@click.command("main")
def main(job_type):
    # write shell-scripts to run simulations from BinAC computing cluster
    base_path = Path(__file__).parent
    list_dir = os.listdir(base_path)

    for d in list_dir:
        file = base_path / d / 'write_moab.py'
        if os.path.exists(file):
            path_dir = base_path / d
            os.chdir(path_dir)
            subprocess.Popen(f"python write_moab.py --job-type {job_type}", shell=True)
    return


if __name__ == "__main__":
    main()
