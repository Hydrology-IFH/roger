from pathlib import Path
import subprocess
import click


@click.option("--job-type", type=click.Choice(['serial', 'single-node', 'multi-node']), default='serial')
@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.command("main")
def main(job_type, sas_solver):
    if job_type == 'serial':
        nsamples = 250
        subprocess.Popen("python bootstrap.py --resample-size 250", shell=True)
    elif job_type == 'single-node':
        nsamples = 2000
        subprocess.Popen("python bootstrap.py --resample-size 2000", shell=True)
    elif job_type == 'multi-node':
        nsamples = 10000
        subprocess.Popen("python bootstrap.py --resample-size 10000", shell=True)
    base_path = Path(__file__).parent
    base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo'
    base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')

    transport_models_abrev = {'complete-mixing': 'cm',
                              'piston': 'pi',
                              'preferential': 'pf',
                              'preferential1': 'pf1',
                              'preferential2': 'pf2',
                              'advection-dispersion': 'ad',
                              'advection-dispersion1': 'ad1',
                              'advection-dispersion2': 'ad2',
                              'time-variant preferential': 'pft',
                              'time-variant preferential1': 'pft1',
                              'time-variant preferential2': 'pft2',
                              'time-variant advection-dispersion': 'adt',
                              'time-variant advection-dispersion1': 'adt1',
                              'time-variant advection-dispersion2': 'adt2',
                              'time-variant': 'tv',
                              'time-variant1': 'tv1',
                              'time-variant2': 'tv2',
                              'preferential + advection-dispersion': 'pfad',
                              'time-variant preferential + advection-dispersion': 'pfadt',
                              'power': 'pow',
                              'time-variant power': 'powt',
                              'time-variant power reverse': 'powtr'}

    tracer = 'oxygen18'
    transport_models = ['preferential', 'preferential1', 'preferential2',
                        'advection-dispersion', 'advection-dispersion1', 'advection-dispersion2',
                        'time-variant preferential', 'time-variant preferential1', 'time-variant preferential2',
                        'time-variant advection-dispersion', 'time-variant advection-dispersion1', 'time-variant advection-dispersion2',
                        'time-variant', 'time-variant1', 'time-variant2',
                        'preferential + advection-dispersion', 'time-variant preferential + advection-dispersion',
                        'power', 'time-variant power', 'time-variant power reverse']
    for tm in transport_models:
        if job_type == 'serial':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{sas_solver}_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=1\n')
            lines.append('#PBS -l walltime=48:00:00\n')
            lines.append('#PBS -l pmem=4000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('python svat_transport.py -b numpy -d cpu -ns %s -tms %s -td "${TMPDIR}" -ss %s\n' % (nsamples, tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f'{script_name}_moab.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_moab.sh", shell=True)

        elif job_type == 'single-node':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{sas_solver}_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=20\n')
            lines.append('#PBS -l walltime=120:00:00\n')
            lines.append('#PBS -l pmem=6000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('# load module dependencies\n')
            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
            lines.append('export OMP_NUM_THREADS=1\n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger-mpi\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('# adapt command to your available scheduler / MPI implementation\n')
            lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes -b numpy -d cpu -n 20 1 -ns %s -tms %s -td "${TMPDIR}" -ss %s\n' % (nsamples, tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f'{script_name}_moab.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_moab.sh", shell=True)

        elif job_type == 'multi-node':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{sas_solver}_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=5:ppn=20\n')
            lines.append('#PBS -l walltime=120:00:00\n')
            lines.append('#PBS -l pmem=4000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('# load module dependencies\n')
            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
            lines.append('export OMP_NUM_THREADS=1\n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger-mpi\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('# adapt command to your available scheduler / MPI implementation\n')
            lines.append(f'mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes -b numpy -d cpu -n 100 1 -ns {nsamples} -tms {tms} -td {output_path_ws.as_posix()} -ss {sas_solver}\n')
            file_path = base_path / f'{script_name}_moab.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_moab.sh", shell=True)

    return


if __name__ == "__main__":
    main()
