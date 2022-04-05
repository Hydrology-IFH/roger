# -*- coding: utf-8 -*-

"""
roger.utils.config
~~~~~~~~~~~
This module is used to read the configuration file.
:2019 by Robin Schwemmle.
:license: GNU GPLv3, see LICENSE for more details.
"""

from collections import OrderedDict
from pathlib import Path
from typing import List, Union, Any

import pandas as pd
from ruamel.yaml import YAML
import os


class Config:
    """Configuration"""

    def __init__(self, cfg_path: Path):
        """Read run configuration from the specified path and parse it into a
        dict.

        During parsing, config keys that contain 'dir', 'file', or 'path' will
        be converted to pathlib.Path instances. Configuration keys ending with
        '_date' will be parsed to pd.Timestamps. The expected format is
        DD/MM/YYYY.

        Args
        ----------
        cfg_path : Path
            Path to the config file.
        """
        self._cfg = self._read_config(cfg_path=cfg_path)
        ll = ['crop_phenology', 'crop_rotation', 'result_dir', 'result_dir_st']
        for key in ll:
            if key not in list(self._cfg.keys()):
                self._cfg[key] = None

    def as_dict(self) -> dict:
        """Return run configuration as dictionary.

        Returns
        -------
        dict
            The run configuration, as defined in the .yml file.
        """
        return self._cfg

    def dump_config(self, folder: Path, filename: str = 'config.yml'):
        """Save the run configuration as a .yml file to disk.

        Args
        ----------
        folder : Path
            Folder in which the configuration will be stored.
        filename : str, optional
            Name of the file that will be stored. Default: 'config.yml'.

        Raises
        ------
        FileExistsError
            If the specified folder already contains a file named `filename`.
        """
        cfg_path = folder / filename
        if cfg_path.exists():
            os.remove(cfg_path)

        if not cfg_path.exists():
            with cfg_path.open('w') as fp:
                temp_cfg = {}
                for key, val in self._cfg.items():
                    if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                        if isinstance(val, list):
                            temp_list = []
                            for elem in val:
                                temp_list.append(str(elem))
                            temp_cfg[key] = temp_list
                        else:
                            temp_cfg[key] = str(val)
                    elif key.endswith('_date'):
                        if isinstance(val, list):
                            temp_list = []
                            for elem in val:
                                temp_list.append(elem.strftime(format="%d.%m.%y %H:%M:%S"))
                            temp_cfg[key] = temp_list
                        else:
                            if isinstance(val, pd.Timestamp):
                                temp_cfg[key] = val.strftime(format="%d.%m.%y %H:%M:%S")
                    else:
                        temp_cfg[key] = val

                yaml = YAML()
                yaml.dump(dict(OrderedDict(sorted(temp_cfg.items()))), fp)
        else:
            FileExistsError(cfg_path)

    @staticmethod
    def _as_default_list(value: Any) -> list:
        if value is None:
            return []
        elif isinstance(value, list):
            return value
        else:
            return [value]

    @staticmethod
    def _as_default_dict(value: Any) -> dict:
        if value is None:
            return {}
        elif isinstance(value, dict):
            return value
        else:
            raise RuntimeError(f"Incompatible type {type(value)}. Expected `dict` or `None`.")

    def _get_value_verbose(self, key: str) -> Union[float, int, str, bool, list, dict, Path, pd.Timestamp]:
        """Use this function internally to return attributes of the config that
        are mandatory"""
        if key not in self._cfg.keys():
            raise ValueError(f"{key} is not specified in the config (.yml).")
        elif self._cfg[key] is None:
            raise ValueError(f"{key} is mandatory but 'None' in the config.")
        else:
            return self._cfg[key]

    def _get_value(self, key: str) -> Union[float, int, str, bool, list, dict, Path, pd.Timestamp, None]:
        """Use this function internally to return attributes of the config that
        are mandatory"""
        if key not in self._cfg.keys():
            raise ValueError(f"{key} is not specified in the config (.yml).")
        else:
            return self._cfg[key]

    def _get_sas_dict(self, key: str) -> dict:
        """Use this function internally to return attributes of the config that
        are mandatory"""
        if key not in self._cfg.keys():
            raise ValueError(f"{key} is not specified in the config (.yml).")
        elif self._cfg[key] is None:
            raise ValueError(f"{key} is mandatory but 'None' in the config.")
        else:
            return {k: v for d in self._cfg[key] for k, v in d.items()}

    @staticmethod
    def _parse_config(cfg: dict) -> dict:
        for key, val in cfg.items():
            # convert all path strings to PosixPath objects
            if any([key.endswith(x) for x in ['_dir', '_path', '_file', '_files']]):
                if (val is not None) and (val != "None"):
                    if isinstance(val, list):
                        temp_list = []
                        for element in val:
                            temp_list.append(Path(element))
                        cfg[key] = temp_list
                    else:
                        cfg[key] = Path(val)
                else:
                    cfg[key] = None

            # convert Dates to pandas Datetime index
            elif key.endswith('_date'):
                if isinstance(val, list):
                    temp_list = []
                    for elem in val:
                        temp_list.append(pd.to_datetime(elem, format='%d.%m.%Y %H:%M:%S'))
                    cfg[key] = temp_list
                else:
                    cfg[key] = pd.to_datetime(val, format='%d.%m.%Y %H:%M:%S')

            else:
                pass

        # Add more config parsing if necessary
        return cfg

    def _read_config(self, cfg_path: Path):
        if cfg_path.exists():
            with cfg_path.open('r') as fp:
                yaml = YAML(typ="safe")
                cfg = yaml.load(fp)
        else:
            raise FileNotFoundError(cfg_path)

        cfg = self._parse_config(cfg)

        return cfg

    @property
    def experiment_name(self) -> str:
        if self._cfg.get("experiment_name", None) is None:
            return "run"
        else:
            return self._cfg["experiment_name"]

    @experiment_name.setter
    def experiment_name(self, name: str):
        self._cfg["experiment_name"] = name

    @property
    def tm_structure(self) -> str:
        return self._get_value_verbose("tm_structure")

    @tm_structure.setter
    def tm_structure(self, tm_structure: str):
        self._cfg["tm_structure"] = tm_structure

    @property
    def prec_corr(self) -> str:
        if self._cfg.get("prec_corr", None) is None:
            return False
        else:
            return self._cfg["prec_corr"]

    @property
    def enable_distributed_forcing(self) -> bool:
        return self._get_value("enable_distributed_forcing")

    @enable_distributed_forcing.setter
    def enable_distributed_forcing(self, val: bool):
        self._cfg["enable_distributed_forcing"] = val

    @property
    def enable_crop_phenology(self):
        return self._get_value("enable_crop_phenology")

    @enable_crop_phenology.setter
    def crop_phenology(self, val: str):
        self._cfg["crop_phenology"] = val

    @property
    def enable_crop_rotation(self) -> bool:
        return self._get_value("enable_crop_rotation")

    @enable_crop_rotation.setter
    def enable_crop_rotation(self, val: bool):
        self._cfg["enable_crop_rotation"] = val

    @property
    def enable_film_flow(self) -> bool:
        return self._get_value("enable_film_flow")

    @enable_film_flow.setter
    def enable_film_flow(self, val: bool):
        self._cfg["enable_film_flow"] = val

    @property
    def enable_lateral_flow(self) -> bool:
        return self._get_value("enable_lateral_flow")

    @enable_lateral_flow.setter
    def enable_lateral_flow(self, val: bool):
        self._cfg["enable_lateral_flow"] = val

    @property
    def enable_groundwater(self) -> bool:
        return self._get_value("enable_groundwater")

    @enable_groundwater.setter
    def enable_groundwater(self, val: bool):
        self._cfg["enable_groundwater"] = val

    @property
    def enable_groundwater_boundary(self) -> bool:
        return self._get_value("enable_groundwater_boundary")

    @enable_groundwater_boundary.setter
    def enable_groundwater_boundary(self, val: bool):
        self._cfg["enable_groundwater_boundary"] = val

    @property
    def enable_routing(self) -> bool:
        return self._get_value("enable_routing")

    @enable_routing.setter
    def enable_routing(self, val: bool):
        self._cfg["enable_routing"] = val

    @property
    def enable_urban(self) -> bool:
        return self._get_value("enable_urban")

    @enable_urban.setter
    def enable_urban(self, val: bool):
        self._cfg["enable_urban"] = val

    @property
    def enable_runon_infiltration(self) -> bool:
        return self._get_value("enable_runon_infiltration")

    @enable_runon_infiltration.setter
    def enable_runon_infiltration(self, val: bool):
        self._cfg["enable_runon_infiltration"] = val

    @property
    def enable_offline_transport(self) -> bool:
        return self._get_value("enable_offline_transport")

    @enable_offline_transport.setter
    def enable_offline_transport(self, val: bool):
        self._cfg["enable_offline_transport"] = val

    @property
    def enable_chloride(self) -> bool:
        return self._get_value("enable_chloride")

    @enable_chloride.setter
    def enable_chloride(self, val: bool):
        self._cfg["enable_chloride"] = val

    @property
    def enable_bromide(self) -> bool:
        return self._get_value("enable_bromide")

    @enable_bromide.setter
    def enable_bromide(self, val: bool):
        self._cfg["enable_bromide"] = val

    @property
    def enable_deuterium(self) -> bool:
        return self._get_value("enable_deuterium")

    @enable_deuterium.setter
    def enable_deuterium(self, val: bool):
        self._cfg["enable_deuterium"] = val

    @property
    def enable_oxygen18(self) -> bool:
        return self._get_value("enable_oxygen18")

    @enable_oxygen18.setter
    def enable_oxygen18(self, val: bool):
        self._cfg["enable_oxygen18"] = val

    @property
    def enable_nitrate(self) -> bool:
        return self._get_value("enable_nitrate")

    @enable_nitrate.setter
    def enable_nitrate(self, val: bool):
        self._cfg["enable_nitrate"] = val

    @property
    def nrows(self) -> int:
        return self._get_value_verbose("nrows")

    @nrows.setter
    def nrows(self, val: int):
        self._cfg["nrows"] = val

    @property
    def ncols(self) -> int:
        return self._get_value_verbose("ncols")

    @ncols.setter
    def ncols(self, val: int):
        self._cfg["ncols"] = val

    @property
    def ntime(self) -> int:
        return self._cfg.get("ntime", None)

    @ntime.setter
    def ntime(self, val: int):
        self._cfg["ntime"] = val

    @property
    def nage(self) -> int:
        return self._cfg.get("nage", None)

    @nage.setter
    def nage(self, val: int):
        self._cfg["nage"] = val

    @property
    def cell_width(self) -> int:
        return self._get_value_verbose("cell_width")

    @property
    def dt_gw(self) -> float:
        return self._get_value_verbose("dt")

    @dt_gw.setter
    def dt_gw(self, val: float):
        self._cfg["dt_gw"] = val

    @property
    def output_variables(self) -> List[str]:
        if "output_variables" in self._cfg.keys():
            return self._cfg["output_variables"]
        else:
            raise ValueError("No output variables ('output_variables') defined in the config.")

    @output_variables.setter
    def output_variables(self, ll: List):
        self._cfg["output_variables"] = ll

    @property
    def verbose(self) -> int:
        """Defines level of verbosity.

        0: Only log info messages, don't show progress bars
        1: Log info messages and show progress bars

        Returns
        -------
        int
            Level of verbosity.
        """
        return self._cfg.get("verbose", 1)


def write_config(run_dir, experiment_name, output_variables,
                 cell_width=1, nrows=1, ncols=1,
                 enable_groundwater_boundary=False,
                 enable_film_flow=False, enable_crop_phenology=False,
                 enable_crop_rotation=False, enable_lateral_flow=False,
                 enable_groundwater=False, enable_routing=False,
                 enable_offline_transport=False,
                 enable_bromide=False,
                 enable_chloride=False,
                 enable_deuterium=False,
                 enable_oxygen18=False,
                 enable_nitrate=False,
                 tm_structure=None):
    """Write configuration template

    Args
    ----------
    run_dir : Path
        working directory

    experiment_name : str
        name of experiment

    tms : List
        names of transport models

    solute : str
        names of solute

    output_variables : List
        output variables

    cell_width : float, optional
        spatial resolution (in meter)

    crop_phenology : str, optional
        names of crop phenologic model

    eval_file : str, optional
        path to file containing observed data
    """
    dict_config = {}
    dict_config['experiment_name'] = experiment_name

    dict_config['enable_film_flow'] = enable_film_flow
    dict_config['enable_lateral_flow'] = enable_lateral_flow
    dict_config['enable_groundwater'] = enable_groundwater
    dict_config['enable_groundwater_boundary'] = enable_groundwater_boundary
    dict_config['enable_offline_transport'] = enable_offline_transport
    dict_config['enable_chloride'] = enable_chloride
    dict_config['enable_bromide'] = enable_bromide
    dict_config['enable_deuterium'] = enable_deuterium
    dict_config['enable_oxygen18'] = enable_oxygen18
    dict_config['enable_nitrate'] = enable_nitrate
    dict_config['enable_crop_phenology'] = enable_crop_phenology
    dict_config['enable_crop_rotation'] = enable_crop_rotation

    dict_config['nrows'] = nrows
    dict_config['ncols'] = ncols
    dict_config['cell_width'] = cell_width

    dict_config['output_variables'] = output_variables

    dict_config['tm_structure'] = tm_structure

    cfg_path = run_dir / 'config.yml'
    with cfg_path.open('w') as fp:
        yaml = YAML()
        yaml.dump(dict(OrderedDict(sorted(dict_config.items()))), fp)
