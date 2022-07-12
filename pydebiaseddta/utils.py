from typing import Dict
import json
from . import package_path


def load_sample_dta_data(mini=False):
    sample_data_path = f"{package_path}/data/dta_sample_data/dta_sample_data.json"
    if mini:
        sample_data_path = f"{package_path}/data/dta/dta_sample_data.mini.json"
    with open(sample_data_path) as f:
        return json.load(f)


def load_sample_smiles():
    """
    Return a list of random ingredients as strings.

    :param kind: Optional "kind" of ingredients.
    :type kind: list[str] or None
    :raise lumache.InvalidKindError: If the kind is invalid.
    :return: The ingredients list.
    :rtype: list[str]
    """
    sample_data_path = f"{package_path}/data/sequence/chembl27.mini.smiles"
    with open(sample_data_path) as f:
        return [line.strip() for line in f.readlines()]


def save_json(obj: Dict, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
