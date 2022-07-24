from pathlib import Path
import os
import subprocess

# write shell-scripts to run simulations from BwUniCluster 2.0
base_path = Path(__file__).parent
list_dir = os.listdir(base_path)

for d in list_dir:
    file = base_path / d / 'write_slurm_gpu.py'
    if os.path.exists(file):
        path_dir = base_path / d
        os.chdir(path_dir)
        subprocess.Popen("python write_slurm_gpu.py", shell=True)
