from typing import Dict, List
import json
from . import package_path


def load_sample_dta_data(mini: bool = False) -> Dict[str, List]:
    """Loads a portion of the `BDB dataset <https://arxiv.org/pdf/2107.05556.pdf>`_ for fast experimenting.

    Parameters
    ----------
    mini : bool, optional
        Whether to load all drug-target pairs embedded in the library, or a mini version.
        Set to ``True`` for fast prototyping and ``False`` to quickly train a model.
        Defaults to ``False``.

    Returns
    -------
    Dict[str, List]
        The dictionary has three keys: "train", "val", and "test", each corresponding to different folds of the dataset.
        Each key maps to a list with three elements: *list of chemicals*, *list of proteins*, and *list of affinity scores*. 
        The elements in the same index of the lists correspond to a drug-target affinity measurement.
    """
    sample_data_path = f"{package_path}/data/dta_sample_data/dta_sample_data.json"
    if mini:
        sample_data_path = f"{package_path}/data/dta/dta_sample_data.mini.json"
    with open(sample_data_path) as f:
        return json.load(f)


def load_sample_smiles() -> List[str]:
    """Returns examples SMILES strings from ChEMBL for testing.

    Returns
    -------
    List[str]
        SMILES examples from ChEMBL.
    """
    sample_data_path = f"{package_path}/data/sequence/chembl27.mini.smiles"
    with open(sample_data_path) as f:
        return [line.strip() for line in f.readlines()]


def save_json(obj: Dict, path: str) -> None:
    """Saves a dictionary in json format. The indent is set to 4 for readability.
    
    Parameters
    ----------
    obj : Dict
        Dictionary to store.
    path : str
        Path to store the .json file.

    Returns
    -------
    None
    """
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(path: str) -> Dict:
    """Loads a json file into a dictionary.
    
    Parameters
    ----------
    path : str
        Path to the .json file to load.

    Returns
    -------
    Dict
        Content of the .json file as a dictionary.
    """
    with open(path, "r") as f:
        return json.load(f)
