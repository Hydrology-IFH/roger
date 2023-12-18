from pathlib import Path
import pandas as pd
import subprocess
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    base_path_bwuc = "/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh"
    base_path_ws = Path("/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/boadkh/output")

    locations = [
        "singen",
        "azenweiler",
        "unterraderach",
        "muellheim",
        "freiburg",
        "ihringen",
        "altheim",
        "kirchen",
        "maehringen",
        "heidelsheim",
        "elsenz",
        "zaberfeld",
        "kupferzell",
        "stachenhausen",
        "oehringen",
    ]
    crop_rotation_scenarios = ["summer-wheat_clover_winter-wheat", "summer-wheat_winter-wheat", 
                               "summer-wheat_winter-wheat_corn", "summer-wheat_winter-wheat_winter-rape", 
                               "winter-wheat_clover", "winter-wheat_clover_corn", "winter-wheat_corn", 
                               "winter-wheat_sugar-beet_corn", "winter-wheat_winter-rape",
                               "winter-wheat_winter-grain-pea_winter-rape", "summer-wheat_winter-wheat_yellow-mustard", 
                               "summer-wheat_winter-wheat_corn_yellow-mustard", "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                               "winter-wheat_corn_yellow-mustard", "winter-wheat_sugar-beet_corn_yellow-mustard",
                               "winter-wheat_winter-rape_yellow-mustard"]
    
    csv_file = base_path.parent / "clust-id_shp-id_clust-flag.csv"
    df = pd.read_csv(csv_file, sep=";", skiprows=1)
    df.loc[:, "clust-id_shp-id_clust-flag"] = df.loc[:, "CLUST_ID"].astype(str) + "_" + df.loc[:, "SHP_ID"].astype(str) + "_" + df.loc[:, "CLUST_FLAG"].astype(str)
    ids = df.loc[:, "clust-id_shp-id_clust-flag"].values.astype(str).tolist()

    # --- jobs to calculate fluxes and states --------------------------------------------------------
    for location in locations:
        for crop_rotation_scenario in crop_rotation_scenarios:
            script_name = f"svat_crop_{location}_{crop_rotation_scenario}"
            output_path_ws = base_path_ws / "svat_crop"
            lines = []
            lines.append("#!/bin/bash\n")
            lines.append("#SBATCH --time=2:00:00\n")
            lines.append("#SBATCH --nodes=1\n")
            lines.append("#SBATCH --ntasks=1\n")
            lines.append("#SBATCH --cpus-per-task=1\n")
            lines.append("#SBATCH --mem=4000\n")
            lines.append("#SBATCH --mail-type=FAIL\n")
            lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
            lines.append(f"#SBATCH --job-name={script_name}\n")
            lines.append(f"#SBATCH --output={script_name}.out\n")
            lines.append(f"#SBATCH --error={script_name}_err.out\n")
            lines.append("#SBATCH --export=ALL\n")
            lines.append(" \n")
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append("conda activate roger\n")
            lines.append(f"cd {base_path_bwuc}/svat_crop\n")
            lines.append(" \n")
            lines.append(
                'python svat_crop.py -b numpy -d cpu --location %s --crop-rotation-scenario %s -td "${TMPDIR}"\n'
                % (location, crop_rotation_scenario)
            )
            lines.append("# Move output from local SSD to global workspace\n")
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVATCROP_*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / "svat_crop" / f"{script_name}_slurm.sh"
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {file_path}", shell=True)

    # --- jobs to calculate sensitivities of fluxes and states ---------------------------------------
    for location in locations:
        for crop_rotation_scenario in crop_rotation_scenarios:
            script_name = f"svat_crop_{location}_{crop_rotation_scenario}_sa"
            output_path_ws = base_path_ws / "svat_crop_sobol"
            lines = []
            lines.append("#!/bin/bash\n")
            lines.append("#SBATCH --time=8:00:00\n")
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
            lines.append(f"cd {base_path_bwuc}/svat_crop_sobol\n")
            lines.append(" \n")
            lines.append(
                'python svat_crop.py -b numpy -d cpu --location %s --crop-rotation-scenario %s -td "${TMPDIR}"\n'
                % (location, crop_rotation_scenario)
            )
            lines.append("# Move output from local SSD to global workspace\n")
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVATCROP_*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / "svat_crop_sobol" / f"{script_name}_slurm.sh"
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {file_path}", shell=True)

    # --- jobs to calculate nitrate concentrations and water ages --------------------------------------------
    for location in locations:
        for crop_rotation_scenario in crop_rotation_scenarios:
            script_name = f"svat_crop_nitrate_{location}_{crop_rotation_scenario}"
            output_path_ws = base_path_ws / "svat_crop_nitrate"
            lines = []
            lines.append("#!/bin/bash\n")
            lines.append("#SBATCH --time=6:00:00\n")
            lines.append("#SBATCH --nodes=1\n")
            lines.append("#SBATCH --ntasks=1\n")
            lines.append("#SBATCH --cpus-per-task=1\n")
            lines.append("#SBATCH --mem=8000\n")
            lines.append("#SBATCH --mail-type=FAIL\n")
            lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
            lines.append(f"#SBATCH --job-name={script_name}\n")
            lines.append(f"#SBATCH --output={script_name}.out\n")
            lines.append(f"#SBATCH --error={script_name}_err.out\n")
            lines.append("#SBATCH --export=ALL\n")
            lines.append(" \n")
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append("conda activate roger\n")
            lines.append(f"cd {base_path_bwuc}/svat_crop_nitrate\n")
            lines.append(" \n")
            lines.append("# Copy fluxes and states from global workspace to local SSD\n")
            lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
            lines.append("# Compares hashes\n")
            file_nc = "SVATCROP_%s_%s.nc" % (location, crop_rotation_scenario)
            lines.append(
                f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat_crop/{file_nc} | cut -f 1 -d " ")\n'
            )
            lines.append("checksum_ssd=0a\n")
            lines.append('cp %s/svat_crop/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
            lines.append("# Wait for termination of moving files\n")
            lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
            lines.append("sleep 10\n")
            lines.append('checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
            lines.append("done\n")
            lines.append('echo "Copying was successful"\n')
            lines.append(" \n")
            lines.append(
                'python svat_crop_nitrate.py -b jax -d cpu --float-type float64 --location %s --crop-rotation-scenario %s -td "${TMPDIR}"\n'
                % (location, crop_rotation_scenario)
            )
            lines.append("# Move output from local SSD to global workspace\n")
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVATCROPNITRATE_*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / "svat_crop_nitrate" / f"{script_name}_slurm.sh"
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {file_path}", shell=True)

    # --- jobs to calculate sensitivities of nitrate concentrations and water ages --------------------------------------------
    for location in locations:
        for crop_rotation_scenario in crop_rotation_scenarios:
            for x, id in enumerate(ids):
                script_name = f"svat_crop_nitrate_{id}_{location}_{crop_rotation_scenario}"
                output_path_ws = base_path_ws / "svat_crop_nitrate"
                lines = []
                lines.append("#!/bin/bash\n")
                lines.append("#SBATCH --time=12:00:00\n")
                lines.append("#SBATCH --nodes=1\n")
                lines.append("#SBATCH --ntasks=32\n")
                lines.append("#SBATCH --cpus-per-task=1\n")
                lines.append("#SBATCH --mem=16000\n")
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
                lines.append(f"cd {base_path_bwuc}/svat_crop_nitrate_sobol\n")
                lines.append(" \n")
                lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append("# Compares hashes\n")
                file_nc = "SVATCROP_%s_%s.nc" % (location, crop_rotation_scenario)
                lines.append(
                    f'checksum_gws=$(shasum -a 256 {output_path_ws.parent.as_posix()}/svat_crop/{file_nc} | cut -f 1 -d " ")\n'
                )
                lines.append("checksum_ssd=0a\n")
                lines.append('cp %s/svat_crop/%s "${TMPDIR}"\n' % (output_path_ws.parent.as_posix(), file_nc))
                lines.append("# Wait for termination of moving files\n")
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append("sleep 10\n")
                lines.append('checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append("done\n")
                lines.append('echo "Copying was successful"\n')
                lines.append(" \n")
                lines.append(
                    'python svat_crop_nitrate.py -b jax -d cpu --float-type float64 --row %s --id %s --location %s --crop-rotation-scenario %s -td "${TMPDIR}"\n'
                    % (x, id, location, crop_rotation_scenario)
                )
                lines.append("# Move output from local SSD to global workspace\n")
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVATCROPNITRATE_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / "svat_crop_nitrate_sobol" / f"{script_name}_slurm.sh"
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
