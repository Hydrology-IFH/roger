from pathlib import Path
import yaml
import subprocess
import os
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    dir_name = os.path.basename(str(Path(__file__).parent))
    base_path_bwhpc = f"/pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/{dir_name}"
    base_path_ws = Path(f"/pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/{dir_name}")

    # identifiers of the simulations
    stress_tests_meteo = ["base", "base_2000-2024", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]
    stress_test_meteo_magnitudes = [0, 2]
    stress_test_meteo_durations = [0, 3]
    scenario_flags = []
    script_names = []
    for stress_test_meteo in stress_tests_meteo:
        if stress_test_meteo == "base":
            scenario_flags.append('--stress-test-meteo %s --soil-compaction' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction --irrigation' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction --yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --irrigation --yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction --irrigation --yellow-mustard' % (stress_test_meteo))

            script_names.append('oneD_crop_%s_soil-compaction' % (stress_test_meteo))
            script_names.append('oneD_crop_%s' % (stress_test_meteo))
            script_names.append('oneD_crop_%s_soil-compaction_irrigation' % (stress_test_meteo))
            script_names.append('oneD_crop_%s_soil-compaction_yellow-mustard' % (stress_test_meteo))
            script_names.append('oneD_crop_%s_irrigation_yellow-mustard' % (stress_test_meteo))
            script_names.append('oneD_crop_%s_soil-compaction_irrigation_yellow-mustard' % (stress_test_meteo))

        elif stress_test_meteo == "base_2000-2024":
            scenario_flags.append('--stress-test-meteo %s --soil-compaction' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction --irrigation' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction --yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --irrigation --yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction --irrigation --yellow-mustard' % (stress_test_meteo))

            script_names.append('oneD_crop_%s_soil-compaction' % (stress_test_meteo))
            script_names.append('oneD_crop_%s' % (stress_test_meteo))
            script_names.append('oneD_crop_%s_soil-compaction_irrigation' % (stress_test_meteo))
            script_names.append('oneD_crop_%s_soil-compaction_yellow-mustard' % (stress_test_meteo))
            script_names.append('oneD_crop_%s_irrigation_yellow-mustard' % (stress_test_meteo))
            script_names.append('oneD_crop_%s_soil-compaction_irrigation_yellow-mustard' % (stress_test_meteo))

        else:
            for magnitude in stress_test_meteo_magnitudes:
                for duration in stress_test_meteo_durations:
                    if magnitude == 0 and duration == 0:
                        pass
                    else:
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction --irrigation' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction --yellow-mustard' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --irrigation --yellow-mustard' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction --irrigation --yellow-mustard' % (stress_test_meteo, magnitude, duration))

                        script_names.append('oneD_crop_%s_magnitude%s_duration%s_soil-compaction' % (stress_test_meteo, magnitude, duration))
                        script_names.append('oneD_crop_%s_magnitude%s_duration%s' % (stress_test_meteo, magnitude, duration))
                        script_names.append('oneD_crop_%s_magnitude%s_duration%s_soil-compaction_irrigation' % (stress_test_meteo, magnitude, duration))
                        script_names.append('oneD_crop_%s_magnitude%s_duration%s_soil-compaction_yellow-mustard' % (stress_test_meteo, magnitude, duration))
                        script_names.append('oneD_crop_%s_magnitude%s_duration%s_irrigation_yellow-mustard' % (stress_test_meteo, magnitude, duration))
                        script_names.append('oneD_crop_%s_magnitude%s_duration%s_soil-compaction_irrigation_yellow-mustard' % (stress_test_meteo, magnitude, duration))

    jobs = []
    for scenario_flag, script_name in zip(scenario_flags[:2], script_names[:2]):
        output_path_ws = base_path_ws / "output"
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=12:00:00\n")
        lines.append("#SBATCH --gres=gpu:a100:1\n")
        lines.append("#SBATCH --ntasks=1\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=24000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append("module load lib/hdf5/1.12-gnu-14.2-openmpi-4.1\n")
        lines.append("module load devel/cuda/12.6\n")
        lines.append('module load devel/miniforge\n')
        lines.append("conda activate roger-gpu\n")
        lines.append(f"cd {base_path_bwhpc}\n")
        lines.append("\n")
        lines.append("mkdir ${TMPDIR}/roger\n")
        lines.append("mkdir ${TMPDIR}/roger/examples\n")
        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale\n")
        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen\n")
        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n")
        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output\n")
        lines.append("cp -r %s/oneD_crop.py ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc)) 
        lines.append("cp -r %s/parameters_roger.nc ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
        lines.append("cp -r %s/config.yml ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
        lines.append("cp -r %s/input ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
        lines.append('sleep 120\n')
        lines.append("cd ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n")
        lines.append('python oneD_crop.py -b jax -d gpu %s -td "${TMPDIR}"' % (scenario_flag))
        lines.append("# Move output from local SSD to global workspace\n")
        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
        lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
        lines.append('mv "${TMPDIR}"/roger/examples/catchment_scale/dreisam_moehlin_neumagen/output/ONEDCROP_*.nc %s' % (output_path_ws.as_posix()))
        file_path = base_path / f"{script_name}.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)
        jobs.append(f"{script_name}.sh")

    file_path = base_path / "submit_gpu_jobs.sh"
    with open(file_path, "w") as job_file:
        job_file.write("#!/bin/bash\n")
        job_file.write("\n")
        for job in jobs:
            job_file.write(f"sbatch -p gpu {job}\n")
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
