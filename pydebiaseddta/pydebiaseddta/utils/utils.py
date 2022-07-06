from typing import Dict
import json
from .. import package_path


def load_sample_dta_data(mini=False):
    """Here is the summary

    :param [mini]: [some desc], defaults to [False]
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    sample_data_path = f"{package_path}/data/dta_sample_data/dta_sample_data.json"
    if mini:
        sample_data_path = f"{package_path}/data/dta/dta_sample_data.mini.json"
    with open(sample_data_path) as f:
        return json.load(f)


def load_sample_smiles():
    """Here is the summary

    :param [ParamName]: [ParamDescription], defaults to [DefaultParamVal]
    ...
    :raises [ErrorType]: [ErrorDescription]
    ...
    :return: [ReturnDescription]
    :rtype: [ReturnType]
    """
    sample_data_path = f"{package_path}/data/sequence/chembl27.mini.smiles"
    with open(sample_data_path) as f:
        return [line.strip() for line in f.readlines()]


def save_json(obj: Dict, path: str):
    """A utility function to save dictionaries in json.
    The indent is set to 4 for readability.
    Parameters
    ----------
    obj : Dict
        Dictionary to save
    path : str
        Path to store the .json file
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
