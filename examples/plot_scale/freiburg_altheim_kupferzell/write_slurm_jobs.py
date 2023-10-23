from pathlib import Path
import subprocess
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    base_path_bwuc = "/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell"
    base_path_ws = Path("/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell/output")

    locations = ["freiburg", "altheim", "kupferzell"]
    land_cover_scenarios = ["corn", "corn_catch_crop", "crop_rotation"]
    climate_scenarios = ["CCCma-CanESM2_CCLM4-8-17", "MPI-M-MPI-ESM-LR_RCA4"]
    periods = ["1985-2014", "2030-2059", "2070-2099"]

    # --- jobs to calculate fluxes and states --------------------------------------------------------
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            for climate_scenario in climate_scenarios:
                for period in periods:
                    script_name = f"svat_{location}_{land_cover_scenario}_{climate_scenario}_{period}"
                    output_path_ws = base_path_ws / "svat"
                    lines = []
                    lines.append("#!/bin/bash\n")
                    lines.append("#SBATCH --time=9:00:00\n")
                    lines.append("#SBATCH --nodes=1\n")
                    lines.append("#SBATCH --ntasks=1\n")
                    lines.append("#SBATCH --cpus-per-task=1\n")
                    lines.append("#SBATCH --mem=12000\n")
                    lines.append("#SBATCH --mail-type=FAIL\n")
                    lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                    lines.append(f"#SBATCH --job-name={script_name}\n")
                    lines.append(f"#SBATCH --output={script_name}.out\n")
                    lines.append(f"#SBATCH --error={script_name}_err.out\n")
                    lines.append("#SBATCH --export=ALL\n")
                    lines.append(" \n")
                    lines.append('eval "$(conda shell.bash hook)"\n')
                    lines.append("conda activate roger\n")
                    lines.append(f"cd {base_path_bwuc}/svat\n")
                    lines.append(" \n")
                    lines.append(
                        'python svat_crop.py -b numpy -d cpu --location %s --land-cover-scenario %s --climate-scenario %s --period %s -td "${TMPDIR}"\n'
                        % (location, land_cover_scenario, climate_scenario, period)
                    )
                    lines.append("# Move output from local SSD to global workspace\n")
                    lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                    lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                    lines.append('mv "${TMPDIR}"/SVAT_*.nc %s\n' % (output_path_ws.as_posix()))
                    file_path = base_path / "svat" / f"{script_name}_slurm.sh"
                    file = open(file_path, "w")
                    file.writelines(lines)
                    file.close()
                    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            script_name = f"svat_{location}_{land_cover_scenario}_observed_2016-2021"
            output_path_ws = base_path_ws / "svat"
            lines = []
            lines.append("#!/bin/bash\n")
            lines.append("#SBATCH --time=2:00:00\n")
            lines.append("#SBATCH --nodes=1\n")
            lines.append("#SBATCH --ntasks=1\n")
            lines.append("#SBATCH --cpus-per-task=1\n")
            lines.append("#SBATCH --mem=12000\n")
            lines.append("#SBATCH --mail-type=FAIL\n")
            lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
            lines.append(f"#SBATCH --job-name={script_name}\n")
            lines.append(f"#SBATCH --output={script_name}.out\n")
            lines.append(f"#SBATCH --error={script_name}_err.out\n")
            lines.append("#SBATCH --export=ALL\n")
            lines.append(" \n")
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append("conda activate roger\n")
            lines.append(f"cd {base_path_bwuc}/svat\n")
            lines.append(" \n")
            lines.append(
                'python svat_crop.py -b numpy -d cpu --location %s --land-cover-scenario %s --climate-scenario observed --period 2016-2021 -td "${TMPDIR}"\n'
                % (location, land_cover_scenario)
            )
            lines.append("# Move output from local SSD to global workspace\n")
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVAT_*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / "svat" / f"{script_name}_slurm.sh"
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        for climate_scenario in climate_scenarios:
            for period in periods:
                script_name = f"svat_{location}_grass_{climate_scenario}_{period}"
                output_path_ws = base_path_ws / "svat"
                lines = []
                lines.append("#!/bin/bash\n")
                lines.append("#SBATCH --time=9:00:00\n")
                lines.append("#SBATCH --nodes=1\n")
                lines.append("#SBATCH --ntasks=1\n")
                lines.append("#SBATCH --cpus-per-task=1\n")
                lines.append("#SBATCH --mem=12000\n")
                lines.append("#SBATCH --mail-type=FAIL\n")
                lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                lines.append(f"#SBATCH --job-name={script_name}\n")
                lines.append(f"#SBATCH --output={script_name}.out\n")
                lines.append(f"#SBATCH --error={script_name}_err.out\n")
                lines.append("#SBATCH --export=ALL\n")
                lines.append(" \n")
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append("conda activate roger\n")
                lines.append(f"cd {base_path_bwuc}/svat\n")
                lines.append(" \n")
                lines.append(
                    'python svat.py -b numpy -d cpu --location %s --climate-scenario %s --period %s -td "${TMPDIR}"\n'
                    % (location, climate_scenario, period)
                )
                lines.append("# Move output from local SSD to global workspace\n")
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVAT_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / "svat" / f"{script_name}_slurm.sh"
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        script_name = f"svat_{location}_grass_observed_2016-2021"
        output_path_ws = base_path_ws / "svat"
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=2:00:00\n")
        lines.append("#SBATCH --nodes=1\n")
        lines.append("#SBATCH --ntasks=1\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=12000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append(" \n")
        lines.append('eval "$(conda shell.bash hook)"\n')
        lines.append("conda activate roger\n")
        lines.append(f"cd {base_path_bwuc}/svat\n")
        lines.append(" \n")
        lines.append(
            'python svat.py -b numpy -d cpu --location %s --climate-scenario observed --period 2016-2021 -td "${TMPDIR}"\n'
            % (location)
        )
        lines.append("# Move output from local SSD to global workspace\n")
        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
        lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
        lines.append('mv "${TMPDIR}"/SVAT_*.nc %s\n' % (output_path_ws.as_posix()))
        file_path = base_path / "svat" / f"{script_name}_slurm.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)

    # --- jobs to calculate concentrations and water ages --------------------------------------------
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            for climate_scenario in climate_scenarios:
                for period in periods:
                    script_name = f"svat_transport_{location}_{land_cover_scenario}_{climate_scenario}_{period}"
                    output_path_ws = base_path_ws / "svat_transport"
                    lines = []
                    lines.append("#!/bin/bash\n")
                    lines.append("#SBATCH --time=27:00:00\n")
                    lines.append("#SBATCH --nodes=1\n")
                    lines.append("#SBATCH --ntasks=4\n")
                    lines.append("#SBATCH --cpus-per-task=1\n")
                    lines.append("#SBATCH --mem=32000\n")
                    lines.append("#SBATCH --mail-type=FAIL\n")
                    lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                    lines.append(f"#SBATCH --job-name={script_name}\n")
                    lines.append(f"#SBATCH --output={script_name}.out\n")
                    lines.append(f"#SBATCH --error={script_name}_err.out\n")
                    lines.append("#SBATCH --export=ALL\n")
                    lines.append(" \n")
                    lines.append("# load module dependencies\n")
                    lines.append("module load devel/cuda/10.2\n")
                    lines.append("module load devel/cudnn/10.2\n")
                    lines.append("module load lib/hdf5/1.12.2-gnu-12.1-openmpi-4.1\n")
                    lines.append("# prevent memory issues for Open MPI 4.1.x\n")
                    lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                    lines.append("export OMP_NUM_THREADS=1\n")
                    lines.append('eval "$(conda shell.bash hook)"\n')
                    lines.append("conda activate roger-mpi\n")
                    lines.append(f"cd {base_path_bwuc}/svat_transport\n")
                    lines.append(" \n")
                    lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                    lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                    lines.append("# Compares hashes\n")
                    file_nc = "SVAT_%s_%s_%s_%s.nc" % (location, land_cover_scenario, climate_scenario, period)
                    lines.append(
                        f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n'
                    )
                    lines.append("checksum_ssd=0a\n")
                    lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                    lines.append("# Wait for termination of moving files\n")
                    lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                    lines.append("    sleep 10\n")
                    lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                    lines.append("done\n")
                    lines.append('echo "Copying was successful"\n')
                    lines.append(" \n")
                    lines.append(
                        'mpirun --bind-to core --map-by core -report-bindings python svat_crop_transport.py -b jax -d cpu -n 4 1 --location %s --land-cover-scenario %s --climate-scenario %s --period %s -td "${TMPDIR}"\n'
                        % (location, land_cover_scenario, climate_scenario, period)
                    )
                    lines.append("# Move output from local SSD to global workspace\n")
                    lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                    lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                    lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                    file_path = base_path / "svat_transport" / f"{script_name}_cpumpi_slurm.sh"
                    file = open(file_path, "w")
                    file.writelines(lines)
                    file.close()
                    subprocess.Popen(f"chmod +x {file_path}", shell=True)

                    lines = []
                    lines.append("#!/bin/bash\n")
                    lines.append("#SBATCH --time=21:00:00\n")
                    lines.append("#SBATCH --gres=gpu:1\n")
                    lines.append("#SBATCH --ntasks=1\n")
                    lines.append("#SBATCH --cpus-per-task=1\n")
                    lines.append("#SBATCH --mem=12000\n")
                    lines.append("#SBATCH --mail-type=FAIL\n")
                    lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                    lines.append(f"#SBATCH --job-name={script_name}\n")
                    lines.append(f"#SBATCH --output={script_name}.out\n")
                    lines.append(f"#SBATCH --error={script_name}_err.out\n")
                    lines.append("#SBATCH --export=ALL\n")
                    lines.append(" \n")
                    lines.append("# load module dependencies\n")
                    lines.append("module load devel/cudnn/10.2\n")
                    lines.append("module load devel/cuda/12.0\n")
                    lines.append("module load lib/hdf5/1.12.2-gnu-12.1-openmpi-4.1\n")
                    lines.append("# prevent memory issues for Open MPI 4.1.x\n")
                    lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                    lines.append("export OMP_NUM_THREADS=1\n")
                    lines.append('eval "$(conda shell.bash hook)"\n')
                    lines.append("conda activate roger-gpu\n")
                    lines.append(f"cd {base_path_bwuc}/svat_transport\n")
                    lines.append(" \n")
                    lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                    lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                    lines.append("# Compares hashes\n")
                    file_nc = "SVAT_%s_%s_%s_%s.nc" % (location, land_cover_scenario, climate_scenario, period)
                    lines.append(
                        f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n'
                    )
                    lines.append("checksum_ssd=0a\n")
                    lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                    lines.append("# Wait for termination of moving files\n")
                    lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                    lines.append("    sleep 10\n")
                    lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                    lines.append("done\n")
                    lines.append('echo "Copying was successful"\n')
                    lines.append(" \n")
                    lines.append(
                        'python svat_crop_transport.py -b jax -d gpu --location %s --land-cover-scenario %s --climate-scenario %s --period %s -td "${TMPDIR}"\n'
                        % (location, land_cover_scenario, climate_scenario, period)
                    )
                    lines.append("# Move output from local SSD to global workspace\n")
                    lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                    lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                    lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                    file_path = base_path / "svat_transport" / f"{script_name}_gpu_slurm.sh"
                    file = open(file_path, "w")
                    file.writelines(lines)
                    file.close()
                    subprocess.Popen(f"chmod +x {file_path}", shell=True)

                    lines = []
                    lines.append("#!/bin/bash\n")
                    lines.append("#SBATCH --time=48:00:00\n")
                    lines.append("#SBATCH --nodes=1\n")
                    lines.append("#SBATCH --ntasks=1\n")
                    lines.append("#SBATCH --cpus-per-task=1\n")
                    lines.append("#SBATCH --mem=16000\n")
                    lines.append("#SBATCH --mail-type=FAIL\n")
                    lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                    lines.append(f"#SBATCH --job-name={script_name}\n")
                    lines.append(f"#SBATCH --output={script_name}.out\n")
                    lines.append(f"#SBATCH --error={script_name}_err.out\n")
                    lines.append("#SBATCH --export=ALL\n")
                    lines.append(" \n")
                    lines.append('eval "$(conda shell.bash hook)"\n')
                    lines.append("conda activate roger\n")
                    lines.append(f"cd {base_path_bwuc}/svat_transport\n")
                    lines.append(" \n")
                    lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                    lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                    lines.append("# Compares hashes\n")
                    file_nc = "SVAT_%s_%s_%s_%s.nc" % (location, land_cover_scenario, climate_scenario, period)
                    lines.append(
                        f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n'
                    )
                    lines.append("checksum_ssd=0a\n")
                    lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                    lines.append("# Wait for termination of moving files\n")
                    lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                    lines.append("    sleep 10\n")
                    lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                    lines.append("done\n")
                    lines.append('echo "Copying was successful"\n')
                    lines.append(" \n")
                    lines.append(
                        'python svat_crop_transport.py -b jax -d cpu --location %s --land-cover-scenario %s --climate-scenario %s --period %s -td "${TMPDIR}"\n'
                        % (location, land_cover_scenario, climate_scenario, period)
                    )
                    lines.append("# Move output from local SSD to global workspace\n")
                    lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                    lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                    lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                    file_path = base_path / "svat_transport" / f"{script_name}_cpu_slurm.sh"
                    file = open(file_path, "w")
                    file.writelines(lines)
                    file.close()
                    subprocess.Popen(f"chmod +x {file_path}", shell=True)

                    script_name = f"svat_transport_{location}_{land_cover_scenario}_observed_2016-2021"
                    output_path_ws = base_path_ws / "svat_transport"
                    lines = []
                    lines.append("#!/bin/bash\n")
                    lines.append("#SBATCH --time=24:00:00\n")
                    lines.append("#SBATCH --nodes=1\n")
                    lines.append("#SBATCH --ntasks=1\n")
                    lines.append("#SBATCH --cpus-per-task=1\n")
                    lines.append("#SBATCH --mem=12000\n")
                    lines.append("#SBATCH --mail-type=FAIL\n")
                    lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                    lines.append(f"#SBATCH --job-name={script_name}\n")
                    lines.append(f"#SBATCH --output={script_name}.out\n")
                    lines.append(f"#SBATCH --error={script_name}_err.out\n")
                    lines.append("#SBATCH --export=ALL\n")
                    lines.append(" \n")
                    lines.append('eval "$(conda shell.bash hook)"\n')
                    lines.append("conda activate roger\n")
                    lines.append(f"cd {base_path_bwuc}/svat_transport\n")
                    lines.append(" \n")
                    lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                    lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                    lines.append("# Compares hashes\n")
                    file_nc = "SVAT_%s_%s_%s_%s.nc" % (location, land_cover_scenario, climate_scenario, period)
                    lines.append(
                        f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n'
                    )
                    lines.append("checksum_ssd=0a\n")
                    lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                    lines.append("# Wait for termination of moving files\n")
                    lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                    lines.append("    sleep 10\n")
                    lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                    lines.append("done\n")
                    lines.append('echo "Copying was successful"\n')
                    lines.append(" \n")
                    lines.append(
                        'python svat_crop_transport.py -b jax -d cpu -n 4 1 --location %s --land-cover-scenario %s --climate-scenario %s --period %s -td "${TMPDIR}"\n'
                        % (location, land_cover_scenario, climate_scenario, period)
                    )
                    lines.append("# Move output from local SSD to global workspace\n")
                    lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                    lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                    lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                    file_path = base_path / "svat_transport" / f"{script_name}_cpu_slurm.sh"
                    file = open(file_path, "w")
                    file.writelines(lines)
                    file.close()
                    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        for climate_scenario in climate_scenarios:
            for period in periods:
                script_name = f"svat_transport_{location}_grass_{climate_scenario}_{period}"
                output_path_ws = base_path_ws / "svat_transport"
                lines = []
                lines.append("#!/bin/bash\n")
                lines.append("#SBATCH --time=18:00:00\n")
                lines.append("#SBATCH --nodes=1\n")
                lines.append("#SBATCH --ntasks=4\n")
                lines.append("#SBATCH --cpus-per-task=1\n")
                lines.append("#SBATCH --mem=20000\n")
                lines.append("#SBATCH --mail-type=FAIL\n")
                lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                lines.append(f"#SBATCH --job-name={script_name}\n")
                lines.append(f"#SBATCH --output={script_name}.out\n")
                lines.append(f"#SBATCH --error={script_name}_err.out\n")
                lines.append("#SBATCH --export=ALL\n")
                lines.append(" \n")
                lines.append("# load module dependencies\n")
                lines.append("module load devel/cuda/10.2\n")
                lines.append("module load devel/cudnn/10.2\n")
                lines.append("module load lib/hdf5/1.12.2-gnu-12.1-openmpi-4.1\n")
                lines.append("# prevent memory issues for Open MPI 4.1.x\n")
                lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                lines.append("export OMP_NUM_THREADS=1\n")
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append("conda activate roger-mpi\n")
                lines.append(f"cd {base_path_bwuc}/svat_transport\n")
                lines.append(" \n")
                lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append("# Compares hashes\n")
                file_nc = "SVAT_%s_grass_%s_%s.nc" % (location, climate_scenario, period)
                lines.append(
                    f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n'
                )
                lines.append("checksum_ssd=0a\n")
                lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                lines.append("# Wait for termination of moving files\n")
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append("    sleep 10\n")
                lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append("done\n")
                lines.append('echo "Copying was successful"\n')
                lines.append(" \n")
                lines.append(
                    'mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b jax -d cpu -n 4 1 --location %s --land-cover-scenario grass --climate-scenario %s --period %s -td "${TMPDIR}"\n'
                    % (location, climate_scenario, period)
                )
                lines.append("# Move output from local SSD to global workspace\n")
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / "svat_transport" / f"{script_name}_cpumpi_slurm.sh"
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        script_name = f"svat_transport_{location}_grass_observed_2016-2021"
        output_path_ws = base_path_ws / "svat_transport"
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=24:00:00\n")
        lines.append("#SBATCH --nodes=1\n")
        lines.append("#SBATCH --ntasks=4\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=12000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append(" \n")
        lines.append("# load module dependencies\n")
        lines.append("module load devel/cuda/10.2\n")
        lines.append("module load devel/cudnn/10.2\n")
        lines.append("module load lib/hdf5/1.12.2-gnu-12.1-openmpi-4.1\n")
        lines.append("# prevent memory issues for Open MPI 4.1.x\n")
        lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
        lines.append("export OMP_NUM_THREADS=1\n")
        lines.append('eval "$(conda shell.bash hook)"\n')
        lines.append("conda activate roger-gpu\n")
        lines.append(f"cd {base_path_bwuc}/svat_transport\n")
        lines.append(" \n")
        lines.append("# Copy fluxes and states from global workspace to local SSD\n")
        lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
        lines.append("# Compares hashes\n")
        file_nc = "SVAT_%s_grass_%s_%s.nc" % (location, climate_scenario, period)
        lines.append(
            f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n'
        )
        lines.append("checksum_ssd=0a\n")
        lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
        lines.append("# Wait for termination of moving files\n")
        lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
        lines.append("    sleep 10\n")
        lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
        lines.append("done\n")
        lines.append('echo "Copying was successful"\n')
        lines.append(" \n")
        lines.append(
            'mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b jax -d cpu -n 4 1 --location %s --land-cover-scenario grass --climate-scenario observed --period 2016-2021 -td "${TMPDIR}"\n'
            % (location)
        )
        lines.append("# Move output from local SSD to global workspace\n")
        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
        lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
        lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
        file_path = base_path / "svat_transport" / f"{script_name}_cpumpi_slurm.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        for climate_scenario in climate_scenarios:
            for period in periods:
                script_name = f"svat_transport_{location}_grass_{climate_scenario}_{period}"
                output_path_ws = base_path_ws / "svat_transport"
                lines = []
                lines.append("#!/bin/bash\n")
                lines.append("#SBATCH --time=48:00:00\n")
                lines.append("#SBATCH --nodes=1\n")
                lines.append("#SBATCH --ntasks=1\n")
                lines.append("#SBATCH --cpus-per-task=1\n")
                lines.append("#SBATCH --mem=18000\n")
                lines.append("#SBATCH --mail-type=FAIL\n")
                lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                lines.append(f"#SBATCH --job-name={script_name}\n")
                lines.append(f"#SBATCH --output={script_name}.out\n")
                lines.append(f"#SBATCH --error={script_name}_err.out\n")
                lines.append("#SBATCH --export=ALL\n")
                lines.append(" \n")
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append("conda activate roger\n")
                lines.append(f"cd {base_path_bwuc}/svat_transport\n")
                lines.append(" \n")
                lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append("# Compares hashes\n")
                file_nc = "SVAT_%s_grass_%s_%s.nc" % (location, climate_scenario, period)
                lines.append(
                    f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n'
                )
                lines.append("checksum_ssd=0a\n")
                lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                lines.append("# Wait for termination of moving files\n")
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append("    sleep 10\n")
                lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append("done\n")
                lines.append('echo "Copying was successful"\n')
                lines.append(" \n")
                lines.append(
                    'python svat_transport.py -b jax -d cpu --location %s --land-cover-scenario grass --climate-scenario %s --period %s -td "${TMPDIR}"\n'
                    % (location, climate_scenario, period)
                )
                lines.append("# Move output from local SSD to global workspace\n")
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / "svat_transport" / f"{script_name}_cpu_slurm.sh"
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
