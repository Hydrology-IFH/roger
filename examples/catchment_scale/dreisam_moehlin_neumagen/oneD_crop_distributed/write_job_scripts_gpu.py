from pathlib import Path
import yaml
import subprocess
import os
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    dir_name = os.path.basename(str(Path(__file__).parent))

    # identifiers of the simulations
    stress_tests_meteo = ["base", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]
    stress_test_meteo_magnitudes = [0, 2]
    stress_test_meteo_durations = [0, 3]

    jobs = []
    jobs.append("#!/bin/bash\n")
    jobs.append("\n")
    for stress_test_meteo in stress_tests_meteo:
        if stress_test_meteo == "base":
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --soil-compaction\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --soil-compaction --irrigation\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --soil-compaction --yellow-mustard\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --irrigation --yellow-mustard\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --soil-compaction --irrigation --yellow-mustard\n' % (stress_test_meteo))
        elif stress_test_meteo == "base_2000-2024":
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --soil-compaction\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --soil-compaction --irrigation\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --soil-compaction --yellow-mustard\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --irrigation --yellow-mustard\n' % (stress_test_meteo))
            jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --soil-compaction --irrigation --yellow-mustard\n' % (stress_test_meteo))

        else:
            for magnitude in stress_test_meteo_magnitudes:
                for duration in stress_test_meteo_durations:
                    if magnitude == 0 and duration == 0:
                        pass
                    else:
                        jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction\n' % (stress_test_meteo, magnitude, duration))
                        jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s\n' % (stress_test_meteo, magnitude, duration))
                        jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction --irrigation\n' % (stress_test_meteo, magnitude, duration))
                        jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction --yellow-mustard\n' % (stress_test_meteo, magnitude, duration))
                        jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --irrigation --yellow-mustard\n' % (stress_test_meteo, magnitude, duration))
                        jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction --irrigation --yellow-mustard\n' % (stress_test_meteo, magnitude, duration))

    file_path = base_path / "gpu_jobs.sh"
    file = open(file_path, "w")
    file.writelines(jobs)
    file.close()
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()

