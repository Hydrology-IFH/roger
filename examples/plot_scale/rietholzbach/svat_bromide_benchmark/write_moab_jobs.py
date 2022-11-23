from pathlib import Path
import subprocess
import click
import numpy as onp


@click.option("--job-type", type=click.Choice(['serial']), default='serial')
@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.command("main")
def main(job_type, sas_solver):
    base_path = Path(__file__).parent
    base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_bromide_benchmark'
    base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')
    transport_models_abrev = {'piston': 'pi',
                              'complete-mixing': 'cm',
                              'preferential': 'pf',
                              'advection-dispersion': 'ad',
                              'time-variant preferential': 'pft',
                              'time-variant advection-dispersion': 'adt',
                              'time-variant': 'tv',
                              'power': 'pow',
                              'time-variant power': 'powt'}

    tracer = 'bromide'
    transport_models = ['piston', 'complete-mixing', 'advection-dispersion', 'time-variant advection-dispersion']
    if job_type == 'serial':
        years = onp.arange(1997, 2007).tolist()
        for tm in transport_models:
            for year in years:
                tm1 = transport_models_abrev[tm]
                tms = tm.replace(" ", "_")
                script_name = f'{tracer}_{sas_solver}_svat_{tm1}_{year}'
                output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport'
                tms = tm.replace(" ", "_")
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=1:ppn=1\n')
                lines.append('#PBS -l walltime=8:00:00\n')
                lines.append('#PBS -l pmem=4000mb\n')
                lines.append(f'#PBS -N {script_name}\n')
                lines.append('#PBS -m bea\n')
                lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                lines.append(' \n')
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append('conda activate roger\n')
                lines.append(f'cd {base_path_binac}\n')
                lines.append(' \n')
                lines.append('python svat_transport.py -b jax -d cpu -tms %s -td "${TMPDIR}" -ss %s -y %s\n' % (tms, sas_solver, year))
                lines.append('# Move output from local SSD to global workspace\n')
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / f'{script_name}.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

    return


if __name__ == "__main__":
    main()
