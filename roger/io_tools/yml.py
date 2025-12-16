
import yaml
from pathlib import Path


def read_config(path_to_file: Path):
    """Reads the configuration-file

    Data is imported from .yml files and stored in a dictionary.

    Args
    ----------
    path_to_file : Path
        path to configuration file

    Returns
    ----------
    config : Dict
        Model configuration
    """

    with open(path_to_file) as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

        return config
