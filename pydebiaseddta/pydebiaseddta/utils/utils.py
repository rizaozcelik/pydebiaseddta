import json
from .. import package_path


def load_sample_dta_data(mini=False):
    sample_data_path = (
        f"{package_path}/data/dta_sample_data/dta_sample_data.json"
    )
    if mini:
        sample_data_path = f"{package_path}/data/dta/dta_sample_data.mini.json"
    with open(sample_data_path) as f:
        return json.load(f)


def load_sample_smiles():
    sample_data_path = f"{package_path}/data/sequence/chembl27.mini.smiles"
    with open(sample_data_path) as f:
        return [line.strip() for line in f.readlines()]


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
