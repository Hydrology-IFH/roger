from pathlib import Path
import subprocess
import click


@click.option("--job-type", type=click.Choice(['single-node', 'multi-node']), default='single-node')
@click.command("main")
def main(job_type):
    base_path = Path(__file__).parent
    base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_crop_nitrate_monte_carlo'
    base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')
    transport_models_abrev = {'complete-mixing': 'cm',
                              'advection-dispersion-power': 'ad',
                              'time-variant_advection-dispersion-power': 'adt'}

    tracer = 'nitrate'
    lysimeters = ['lys2', 'lys3', 'lys8']
    transport_models = ['complete-mixing', 'advection-dispersion-power',
                        'time-variant_advection-dispersion-power']
    for lys in lysimeters:
        for tm in transport_models:
            if job_type == 'single-node':
                tm1 = transport_models_abrev[tm]
                tms = tm.replace(" ", "_")
                script_name = f'svat_crop_{tracer}_{tm1}_mc_{lys}'
                output_path_ws = base_path_ws / 'reckenholz' / 'svat_crop_nitrate_monte_carlo'
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=1:ppn=25\n')
                lines.append('#PBS -l walltime=18:00:00\n')
                lines.append('#PBS -l pmem=4000mb\n')
                lines.append(f'#PBS -N {script_name}\n')
                lines.append('#PBS -m bea\n')
                lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                lines.append(' \n')
                lines.append('# load module dependencies\n')
                lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                lines.append('export OMP_NUM_THREADS=1\n')
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append('conda activate roger-mpi\n')
                lines.append(f'cd {base_path_binac}\n')
                lines.append(' \n')
                lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append("# Compares hashes\n")
                file_nc = "SVATCROP_%s_bootstrap.nc" % (lys)
                lines.append(
                    f'checksum_gws=$(shasum -a 256 {output_path_ws.as_posix()}/{file_nc} | cut -f 1 -d " ")\n'
                )
                lines.append("checksum_ssd=0a\n")
                lines.append('cp %s/%s "${TMPDIR}"\n' % (output_path_ws.as_posix(), file_nc))
                lines.append("# Wait for termination of moving files\n")
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append("sleep 60\n")
                lines.append('checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append("done\n")
                lines.append('echo "Copying was successful"\n')
                lines.append(" \n")
                lines.append('# adapt command to your available scheduler / MPI implementation\n')
                lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_crop_nitrate.py -b jax -d cpu -n 25 1 -lys %s -tms %s -td "${TMPDIR}"\n' % (lys, tms))
                lines.append('# Write output to temporary SSD of computing node\n')
                lines.append('echo "Write output to $TMPDIR"\n')
                lines.append('# Move output from temporary SSD to workspace\n')
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / f'{script_name}_moab.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)
            elif job_type == 'multi-node':
                tm1 = transport_models_abrev[tm]
                tms = tm.replace(" ", "_")
                script_name = f'{tracer}_{lys}_svat_crop_{tm1}_mc'
                output_path_ws = base_path_ws / 'reckenholz' / 'svat_crop_nitrate_monte_carlo'
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=2:ppn=20\n')
                lines.append('#PBS -l walltime=48:00:00\n')
                lines.append('#PBS -l pmem=4000mb\n')
                lines.append(f'#PBS -N {script_name}\n')
                lines.append('#PBS -m bea\n')
                lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                lines.append(' \n')
                lines.append('# load module dependencies\n')
                lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
                lines.append('export OMP_NUM_THREADS=1\n')
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append('conda activate roger-mpi\n')
                lines.append(f'cd {base_path_binac}\n')
                lines.append(' \n')
                lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append("# Compares hashes\n")
                file_nc = "SVATCROP_%s_bootstrap.nc" % (lys)
                lines.append(
                    f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/{file_nc} | cut -f 1 -d " ")\n'
                )
                lines.append("checksum_ssd=0a\n")
                lines.append('cp %s/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                lines.append("# Wait for termination of moving files\n")
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append("sleep 60\n")
                lines.append('checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append("done\n")
                lines.append('echo "Copying was successful"\n')
                lines.append(" \n")
                lines.append('# adapt command to your available scheduler / MPI implementation\n')
                lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_crop_nitrate.py -b jax -d cpu -n 40 1 -lys %s -tms %s -td "${TMPDIR}"\n' % (lys, tms))
                file_path = base_path / f'{script_name}_moab.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)
    return


if __name__ == "__main__":
    main()
