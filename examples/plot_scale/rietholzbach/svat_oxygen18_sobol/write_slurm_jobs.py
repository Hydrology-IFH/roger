from pathlib import Path
import subprocess
import click


@click.option("--job-type", type=click.Choice(["parallel", "gpu"]), default="parallel")
@click.command("main")
def main(job_type):
    base_path = Path(__file__).parent
    base_path_bwuc = "/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach"

    transport_models_abrev = {
        "complete-mixing": "cm",
        "piston": "pi",
        "advection-dispersion-power": "adp",
        "time-variant advection-dispersion-power": "adpt",
    }

    transport_models_nsamples = {
        "complete-mixing": 10240,
        "piston": 10240,
        "advection-dispersion-power": 13312,
        "time-variant advection-dispersion-power": 16384,
    }

    transport_models_ntasks = {
        "complete-mixing": 20,
        "piston": 20,
        "advection-dispersion-power": 32,
        "time-variant advection-dispersion-power": 32,
    }

    transport_models_mem = {
        "complete-mixing": 80000,
        "piston": 80000,
        "advection-dispersion-power": 100000,
        "time-variant advection-dispersion-power": 120000,
    }

    transport_models_time = {
        "complete-mixing": "12:00:00",
        "piston": "12:00:00",
        "advection-dispersion-power": "12:00:00",
        "time-variant advection-dispersion-power": "12:00:00",
    }

    transport_models = [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
    ]
    for tm in transport_models:
        if job_type == "parallel":
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f"svat18O_{tm1}_sa"
            output_path_ws = Path(base_path_bwuc) / "svat_oxygen18_sensitivity" / "output"
            lines = []
            lines.append("#!/bin/bash\n")
            lines.append(f"#SBATCH --time={transport_models_time[tm]}\n")
            lines.append("#SBATCH --nodes=1\n")
            lines.append(f"#SBATCH --ntasks={transport_models_ntasks[tm]}\n")
            lines.append("#SBATCH --cpus-per-task=1\n")
            lines.append(f"#SBATCH --mem={transport_models_mem[tm]}\n")
            lines.append("#SBATCH --mail-type=FAIL\n")
            lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
            lines.append(f"#SBATCH --job-name={script_name}\n")
            lines.append(f"#SBATCH --output={script_name}.out\n")
            lines.append(f"#SBATCH --error={script_name}_err.out\n")
            lines.append("#SBATCH --export=ALL\n")
            lines.append(" \n")
            lines.append("# load module dependencies\n")
            lines.append("module load lib/hdf5/1.14.4-gnu-13.3-openmpi-5.0\n")
            lines.append("# prevent memory issues for Open MPI 4.1.x\n")
            lines.append('export OMPI_MCA_btl="self,smcuda,vader,tcp"\n')
            lines.append("export OMP_NUM_THREADS=1\n")
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append("conda activate roger-mpi\n")
            lines.append(f"cd {base_path_bwuc}/svat_oxygen18_sensitivity\n")
            lines.append(" \n")
            lines.append(
                'mpirun --bind-to core --map-by core -report-bindings python svat_oxygen18.py -b jax -d cpu -n %s 1 --float-type float64 -ns %s -tms %s -td "${TMPDIR}"\n'
                % (transport_models_ntasks[tm], transport_models_nsamples[tm], tms)
            )
            lines.append("# Move output from local SSD to global workspace\n")
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVATOXYGEN18_*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f"{script_name}_parallel_slurm.sh"
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {file_path}", shell=True)

        elif job_type == "gpu":
            pass

    return


if __name__ == "__main__":
    main()
