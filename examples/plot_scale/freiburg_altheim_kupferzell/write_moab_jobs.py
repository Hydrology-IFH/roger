from pathlib import Path
import subprocess
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/freiburg_altheim_kupferzell'
    base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')

    transport_models_abrev = {'time-variant advection-dispersion-power': 'adpt'}
    
    locations = ['freiburg', 'altheim', 'kupferzell']
    land_cover_scenarios = ['corn', 'corn_catch_crop', 'crop_rotation']
    climate_scenarios = ['CCCma-CanESM2_CCLM4-8-17', 'MPI-M-MPI-ESM-LR_RCA4']
    periods = ['2016-2021', '1985-2005', '2040-2060', '2080-2100']
    
    #--- jobs to calculate fluxes and states -------------------------------------------------------- 
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            for climate_scenario in climate_scenarios:
                for period in periods:
                    script_name = f'svat_{location}_{land_cover_scenario}_{climate_scenario}_{period}'
                    output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat'
                    lines = []
                    lines.append('#!/bin/bash\n')
                    lines.append('#PBS -l nodes=1:ppn=1\n')
                    lines.append('#PBS -l walltime=6:00:00\n')
                    lines.append('#PBS -l pmem=80000mb\n')
                    lines.append(f'#PBS -N {script_name}\n')
                    lines.append('#PBS -m a\n')
                    lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                    lines.append(' \n')
                    lines.append('eval "$(conda shell.bash hook)"\n')
                    lines.append('conda activate roger\n')
                    lines.append(f'cd {base_path_binac}/svat\n')
                    lines.append(' \n')
                    lines.append('python svat_crop.py -b numpy -d cpu --location %s --land-cover-scenario %s --climate-scenario %s --period %s -td "${TMPDIR}"\n' % (location, land_cover_scenario, climate_scenario, period))
                    lines.append('# Move output from local SSD to global workspace\n')
                    lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                    lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                    lines.append('mv "${TMPDIR}"/SVAT_*.nc %s\n' % (output_path_ws.as_posix()))
                    file_path = base_path / 'svat' / f'{script_name}.sh'
                    file = open(file_path, "w")
                    file.writelines(lines)
                    file.close()
                    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            script_name = f'svat_{location}_{land_cover_scenario}_observed_2016-2021'
            output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat'
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=1\n')
            lines.append('#PBS -l walltime=4:00:00\n')
            lines.append('#PBS -l pmem=80000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m a\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger\n')
            lines.append(f'cd {base_path_binac}/svat\n')
            lines.append(' \n')
            lines.append('python svat_crop.py -b numpy -d cpu --location %s --land-cover-scenario %s --climate-scenario observed --period 2016-2021 -td "${TMPDIR}"\n' % (location, land_cover_scenario))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVAT_*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / 'svat' / f'{script_name}.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        for climate_scenario in climate_scenarios:
            for period in periods:
                script_name = f'svat_{location}_grass_{climate_scenario}_{period}'
                output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat'
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=1:ppn=1\n')
                lines.append('#PBS -l walltime=4:00:00\n')
                lines.append('#PBS -l pmem=80000mb\n')
                lines.append(f'#PBS -N {script_name}\n')
                lines.append('#PBS -m a\n')
                lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                lines.append(' \n')
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append('conda activate roger\n')
                lines.append(f'cd {base_path_binac}/svat\n')
                lines.append(' \n')
                lines.append('python svat.py -b numpy -d cpu --location %s --land-cover-scenario grass --climate-scenario %s --period %s -td "${TMPDIR}"\n' % (location, climate_scenario, period))
                lines.append('# Move output from local SSD to global workspace\n')
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVAT_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / 'svat' / f'{script_name}.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        script_name = f'svat_{location}_grass_observed_2016-2021'
        output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat'
        lines = []
        lines.append('#!/bin/bash\n')
        lines.append('#PBS -l nodes=1:ppn=1\n')
        lines.append('#PBS -l walltime=2:00:00\n')
        lines.append('#PBS -l pmem=80000mb\n')
        lines.append(f'#PBS -N {script_name}\n')
        lines.append('#PBS -m a\n')
        lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
        lines.append(' \n')
        lines.append('eval "$(conda shell.bash hook)"\n')
        lines.append('conda activate roger\n')
        lines.append(f'cd {base_path_binac}/svat\n')
        lines.append(' \n')
        lines.append('python svat.py -b numpy -d cpu --location %s --land-cover-scenario grass --climate-scenario observed --period 2016-2021 -td "${TMPDIR}"\n' % (location))
        lines.append('# Move output from local SSD to global workspace\n')
        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
        lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
        lines.append('mv "${TMPDIR}"/SVAT_*.nc %s\n' % (output_path_ws.as_posix()))
        file_path = base_path / 'svat' / f'{script_name}.sh'
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)

    #--- jobs to calculate concentrations and water ages --------------------------------------------
    tm = 'time-variant advection-dispersion-power'
    tm1 = transport_models_abrev[tm]
    tms = tm.replace(" ", "_")
    for location in locations:
        for land_cover_scenario in land_cover_scenarios:
            for climate_scenario in climate_scenarios:
                for period in periods:
                    script_name = f'svat_transport_{location}_{land_cover_scenario}_{climate_scenario}_{period}'
                    output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat_transport'
                    tms = tm.replace(" ", "_")
                    lines = []
                    lines.append('#!/bin/bash\n')
                    lines.append('#PBS -l nodes=1:ppn=4\n')
                    lines.append('#PBS -l walltime=46:00:00\n')
                    lines.append('#PBS -l pmem=4000mb\n')
                    lines.append(f'#PBS -N {script_name}\n')
                    lines.append('#PBS -m a\n')
                    lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                    lines.append(' \n')
                    lines.append('# load module dependencies\n')
                    lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
                    lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                    lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                    lines.append('eval "$(conda shell.bash hook)"\n')
                    lines.append('conda activate roger-mpi\n')
                    lines.append(f'cd {base_path_binac}/svat_transport\n')
                    lines.append(' \n')
                    lines.append('# Copy fluxes and states from global workspace to local SSD\n')
                    lines.append(f'echo "Copy fluxes and states from global workspace to local SSD"\n')
                    lines.append('# Compares hashes\n')
                    file_nc = 'SVAT_%s_%s_%s_%s.nc' % (location, land_cover_scenario, climate_scenario, period)
                    lines.append(f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n')
                    lines.append('checksum_ssd=0a\n')
                    lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                    lines.append('# Wait for termination of moving files\n')
                    lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                    lines.append('    sleep 10\n')
                    lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                    lines.append('done\n')
                    lines.append(f'echo "Copying was successful"\n')
                    lines.append(' \n')
                    lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_crop_transport.py -b jax -d cpu -n 4 1 --location %s --land-cover-scenario %s --climate-scenario %s --period %s -td "${TMPDIR}"\n' % (location, land_cover_scenario, climate_scenario, period))
                    lines.append('# Move output from local SSD to global workspace\n')
                    lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                    lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                    lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                    file_path = base_path / 'svat_transport' / f'{script_name}_cpumpi.sh'
                    file = open(file_path, "w")
                    file.writelines(lines)
                    file.close()
                    subprocess.Popen(f"chmod +x {file_path}", shell=True)

                    script_name = f'svat_transport_{location}_{land_cover_scenario}_observed_2016-2021'
                    output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat_transport'
                    tms = tm.replace(" ", "_")
                    lines = []
                    lines.append('#!/bin/bash\n')
                    lines.append('#PBS -l nodes=1:ppn=4\n')
                    lines.append('#PBS -l walltime=36:00:00\n')
                    lines.append('#PBS -l pmem=4000mb\n')
                    lines.append(f'#PBS -N {script_name}\n')
                    lines.append('#PBS -m a\n')
                    lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                    lines.append(' \n')
                    lines.append('# load module dependencies\n')
                    lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
                    lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                    lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                    lines.append('eval "$(conda shell.bash hook)"\n')
                    lines.append('conda activate roger-mpi\n')
                    lines.append(f'cd {base_path_binac}/svat_transport\n')
                    lines.append(' \n')
                    lines.append('# Copy fluxes and states from global workspace to local SSD\n')
                    lines.append(f'echo "Copy fluxes and states from global workspace to local SSD"\n')
                    lines.append('# Compares hashes\n')
                    file_nc = 'SVAT_%s_%s_%s_%s.nc' % (location, land_cover_scenario, climate_scenario, period)
                    lines.append(f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n')
                    lines.append('checksum_ssd=0a\n')
                    lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                    lines.append('# Wait for termination of moving files\n')
                    lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                    lines.append('    sleep 10\n')
                    lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                    lines.append('done\n')
                    lines.append(f'echo "Copying was successful"\n')
                    lines.append(' \n')
                    lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_crop_transport.py -b jax -d cpu -n 4 1 --location %s --land-cover-scenario %s --climate-scenario %s --period %s -td "${TMPDIR}"\n' % (location, land_cover_scenario, climate_scenario, period))
                    lines.append('# Move output from local SSD to global workspace\n')
                    lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                    lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                    lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                    file_path = base_path / 'svat_transport' / f'{script_name}_cpumpi.sh'
                    file = open(file_path, "w")
                    file.writelines(lines)
                    file.close()
                    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        for climate_scenario in climate_scenarios:
            for period in periods:
                script_name = f'svat_transport_{location}_grass_{climate_scenario}_{period}'
                output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat_transport'
                tms = tm.replace(" ", "_")
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=1:ppn=1:gpus=1:default\n')
                lines.append('#PBS -l walltime=14:00:00\n')
                lines.append('#PBS -l pmem=8000mb\n')
                lines.append(f'#PBS -N {script_name}\n')
                lines.append('#PBS -m a\n')
                lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                lines.append(' \n')
                lines.append('# load module dependencies\n')
                lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
                lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                lines.append('module load lib/cudnn/8.2-cuda-11.4\n')
                lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append('conda activate roger-gpu\n')
                lines.append(f'cd {base_path_binac}/svat_transport\n')
                lines.append(' \n')
                lines.append('# Copy fluxes and states from global workspace to local SSD\n')
                lines.append(f'echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append('# Compares hashes\n')
                file_nc = 'SVAT_%s_grass_%s_%s.nc' % (location, climate_scenario, period)
                lines.append(f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n')
                lines.append('checksum_ssd=0a\n')
                lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                lines.append('# Wait for termination of moving files\n')
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append('    sleep 10\n')
                lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append('done\n')
                lines.append(f'echo "Copying was successful"\n')
                lines.append(' \n')
                lines.append('python svat_transport.py -b jax -d gpu --location %s --land-cover-scenario grass --climate-scenario %s --period %s -td "${TMPDIR}"\n' % (location, climate_scenario, period))
                lines.append('# Move output from local SSD to global workspace\n')
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / 'svat_transport' / f'{script_name}_gpu.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        script_name = f'svat_transport_{location}_grass_observed_2016-2021'
        output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat_transport'
        tms = tm.replace(" ", "_")
        lines = []
        lines.append('#!/bin/bash\n')
        lines.append('#PBS -l nodes=1:ppn=1:gpus=1:default\n')
        lines.append('#PBS -l walltime=6:00:00\n')
        lines.append('#PBS -l pmem=8000mb\n')
        lines.append(f'#PBS -N {script_name}\n')
        lines.append('#PBS -m a\n')
        lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
        lines.append(' \n')
        lines.append('# load module dependencies\n')
        lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
        lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
        lines.append('module load lib/cudnn/8.2-cuda-11.4\n')
        lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
        lines.append('eval "$(conda shell.bash hook)"\n')
        lines.append('conda activate roger-gpu\n')
        lines.append(f'cd {base_path_binac}/svat_transport\n')
        lines.append(' \n')
        lines.append('# Copy fluxes and states from global workspace to local SSD\n')
        lines.append(f'echo "Copy fluxes and states from global workspace to local SSD"\n')
        lines.append('# Compares hashes\n')
        file_nc = 'SVAT_%s_grass_%s_%s.nc' % (location, climate_scenario, period)
        lines.append(f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n')
        lines.append('checksum_ssd=0a\n')
        lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
        lines.append('# Wait for termination of moving files\n')
        lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
        lines.append('    sleep 10\n')
        lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
        lines.append('done\n')
        lines.append(f'echo "Copying was successful"\n')
        lines.append(' \n')
        lines.append('python svat_transport.py -b jax -d gpu --location %s --land-cover-scenario grass --climate-scenario observed --period 2016-2021 -td "${TMPDIR}"\n' % (location))
        lines.append('# Move output from local SSD to global workspace\n')
        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
        lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
        lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
        file_path = base_path / 'svat_transport' / f'{script_name}_gpu.sh'
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        for climate_scenario in climate_scenarios:
            for period in periods:
                script_name = f'svat_transport_{location}_grass_{climate_scenario}_{period}'
                output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat_transport'
                tms = tm.replace(" ", "_")
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=1:ppn=1\n')
                lines.append('#PBS -l walltime=48:00:00\n')
                lines.append('#PBS -l pmem=8000mb\n')
                lines.append(f'#PBS -N {script_name}\n')
                lines.append('#PBS -m a\n')
                lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                lines.append(' \n')
                lines.append('# load module dependencies\n')
                lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
                lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append('conda activate roger\n')
                lines.append(f'cd {base_path_binac}/svat_transport\n')
                lines.append(' \n')
                lines.append('# Copy fluxes and states from global workspace to local SSD\n')
                lines.append(f'echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append('# Compares hashes\n')
                file_nc = 'SVAT_%s_grass_%s_%s.nc' % (location, climate_scenario, period)
                lines.append(f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n')
                lines.append('checksum_ssd=0a\n')
                lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                lines.append('# Wait for termination of moving files\n')
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append('    sleep 10\n')
                lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append('done\n')
                lines.append(f'echo "Copying was successful"\n')
                lines.append(' \n')
                lines.append('python svat_transport.py -b jax -d cpu --location %s --land-cover-scenario grass --climate-scenario %s --period %s -td "${TMPDIR}"\n' % (location, climate_scenario, period))
                lines.append('# Move output from local SSD to global workspace\n')
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / 'svat_transport' / f'{script_name}_cpu.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for location in locations:
        for climate_scenario in climate_scenarios:
            for period in periods:
                script_name = f'svat_transport_{location}_grass_{climate_scenario}_{period}'
                output_path_ws = base_path_ws / 'freiburg_altheim_kupferzell' / 'svat_transport'
                tms = tm.replace(" ", "_")
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=1:ppn=4\n')
                lines.append('#PBS -l walltime=36:00:00\n')
                lines.append('#PBS -l pmem=4000mb\n')
                lines.append(f'#PBS -N {script_name}\n')
                lines.append('#PBS -m a\n')
                lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                lines.append(' \n')
                lines.append('# load module dependencies\n')
                lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
                lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append('conda activate roger-mpi\n')
                lines.append(f'cd {base_path_binac}/svat_transport\n')
                lines.append(' \n')
                lines.append('# Copy fluxes and states from global workspace to local SSD\n')
                lines.append(f'echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append('# Compares hashes\n')
                file_nc = 'SVAT_%s_grass_%s_%s.nc' % (location, climate_scenario, period)
                lines.append(f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat/{file_nc} | cut -f 1 -d " ")\n')
                lines.append('checksum_ssd=0a\n')
                lines.append('cp %s/svat/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                lines.append('# Wait for termination of moving files\n')
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append('    sleep 10\n')
                lines.append('    checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append('done\n')
                lines.append(f'echo "Copying was successful"\n')
                lines.append(' \n')
                lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b jax -d cpu -n 4 1 --location %s --land-cover-scenario grass --climate-scenario %s --period %s -td "${TMPDIR}"\n' % (location, climate_scenario, period))
                lines.append('# Move output from local SSD to global workspace\n')
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVATTRANSPORT_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / 'svat_transport' / f'{script_name}_cpumpi.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
