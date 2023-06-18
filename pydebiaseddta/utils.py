from typing import Dict, List, Tuple, Union
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import json
import os
import numpy as np
import pandas as pd
from . import package_path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def load_sample_dta_data(
        mini: bool = False, split: str = None
        ) -> Union[Dict[str, pd.DataFrame], Tuple[List[str], List[str], List[float]]]:
    """Loads a sample from [the KIBA dataset](https://doi.org/10.1021/ci400709d).

    Parameters
    ----------
    mini : bool, optional
        Whether to load all drug-target pairs embedded in the library, or a mini version.
        Set to `True` for fast prototyping and `False` to quickly train a model.
        Defaults to `False`.
    split : str, optional
        If split is provided, a tuple of ligands, proteins, and affinity scores list is returned. Else,
        pd.DataFrame objects for all splits are returned.

    Returns
    -------
    Union[Dict[str, pd.DataFrame], Tuple[List[str], List[str], List[float]]
        If split is provided, a tuple of ligands, proteins, and affinity scores list is returned. Else, a dictionary of
        pd.DataFrame objects for all splits are returned The dictionary has three keys: `"train"`, `"val"`, and `"test"`,
        each corresponding to different folds of the dataset.
    """
    sample_data_folder = f"{package_path}/data/dta/dta_sample_data{'_mini' if mini else ''}/"
    if split:
        df = pd.read_csv(sample_data_folder + split + ".csv")
        return df["smiles"].tolist(), df["aa_sequence"].tolist(), df["affinity_score"].tolist()
    else:
        return {split: pd.read_csv(sample_data_folder + split + ".csv") for split in ["train", "val", "test"]}
    

def load_sample_prot_sim_matrix() -> pd.DataFrame:
    """Loads the Smith-Waterman protein similarity matrix for sample data provided.

    Returns
    -------
    pd.DataFrame
        Similarity matrix with protein ids in the index and columns.
    """
    return pd.read_csv(f"{package_path}/data/dta/dta_sample_sw_sim_matrix.csv", index_col=0)


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


def get_ranks(vec: np.ndarray) -> np.ndarray:
    """Obtains percentile ranks for a vector of observations.

    Parameters
    ----------
    vec : np.ndarray
        An array of real valued observations.

    Returns
    -------
    np.array
        Percentile ranks of the elements of the vector.
    """
    return pd.Series(vec).rank(pct=True).values


def create_ligands_proteins_df(df: pd.DataFrame, radius: int = 2, n_bits: int = 2048) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a DataFrame given input ligands, adds Morgan fingerprints.

    Parameters
    ----------
    df : pd.DataFrame
        An input DataFrame that includes ligand ids under "ligand_id", protein ids under
        "protein_id", SMILES strings under "smiles" and amino-acid sequence strings under
        "aa_sequence".
    radius : int, optional
        radius hyperparameter for Morgan fingerprint computation.
    n_bits : int, optional
        nBits hyperparameter for Morgan fingerprint computation.
    
    Returns
    -------
    df_ligands : pd.DataFrame
        A DataFrame object that includes rdkit.Chem.rdchem.Mol representation of
        ligands under column `mol` as well as their Morgan fingerprint under `morgan_fp`.
        Also includes SMILES representation under "smiles" and includes ligand id as index.
    df_proteins : pd.DataFrame
        A DataFrame object that includes protein id as index and amino-acid sequences under
        column "aa_sequence".
    """
    ligands = {str(row.ligand_id): row.smiles for i, row in df.iterrows()}
    proteins = {row.prot_id: row.aa_sequence for i, row in df.iterrows()}

    df_ligands = pd.DataFrame.from_dict(ligands, orient='index')
    df_ligands.columns = ['smiles']
    df_ligands.index.name = 'ligand_id'
    df_ligands['mol'] = df_ligands['smiles'].apply(Chem.MolFromSmiles)
    df_ligands['morgan_fp'] = df_ligands['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, radius=radius, nBits=n_bits))

    df_proteins = pd.DataFrame.from_dict(proteins, orient='index')
    df_proteins.columns = ['aa_sequence']
    df_proteins.index.name = 'prot_id'
    return df_ligands, df_proteins


def compute_ligand_distances(df_ligands: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise Tanimoto distances between ligands.
    
    Parameters
    ----------
    df_ligands : pd.DataFrame 
        A DataFrame object that includes rdkit.Chem.rdchem.Mol representation of
        ligands under column `mol` as well as their Morgan fingerprint under `morgan_fp`.
        Also includes SMILES representation under "smiles" and includes ligand id as index.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame that includes pairwise Tanimoto distances between ligands, where rows
        and columns are indexed by ligand id.
    """
    tanimoto_scores = [[DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity) for fp2 in df_ligands['morgan_fp']] for fp1 in df_ligands['morgan_fp']]
    tanimoto_distances = 1 - pd.DataFrame(tanimoto_scores, index=df_ligands.index, columns=df_ligands.index)
    return tanimoto_distances


def compute_protein_distances(df_proteins: pd.DataFrame, prot_sim_matrix: pd.DataFrame) -> pd.DataFrame:
    """Computes pairwise Tanimoto distances between ligands.
    
    Parameters
    ----------
    df_proteins : pd.DataFrame
        A DataFrame object that includes protein id as index and amino-acid sequences under
        column "aa_sequence".
    prot_sim_matrix : pd.DataFrame
        Similarity matrix with protein ids in the index and columns.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame that includes pairwise distances between proteins, where rows
        and columns are indexed by protein id.
    """
    return 1 - prot_sim_matrix


def compute_average_distances(
        df_1: pd.DataFrame,
        prot_sim_matrix: pd.DataFrame,
        df_2: pd.DataFrame = None,
        radius: int = 2,
        n_bits: int = 2048,
        ) -> Tuple[pd.Series, pd.Series]:
    """Computes average distances of each ligand and protein in df_1 to those in df_2.

    If no df_2 is provided, then ligands and proteins in df_1 are compared against
    those in df_1.

    Parameters
    ----------
    df_1 : pd.DataFrame
        An input DataFrame that includes ligand ids under "ligand_id", protein ids under
        "protein_id", SMILES strings under "smiles" and amino-acid sequence strings under
        "aa_sequence". This DataFrame includes ligands and proteins for which the
        average distances will be computed.
    prot_sim_matrix : pd.DataFrame
        A DataFrame that includes the pairwise similarity between aminoacid sequences, where rows
        and columns must include the protein ids in df_proteins. Smith-Waterman similarity
        is a desirable option and can be precomputed with various open source libraries.
    df_2 : pd.DataFrame
        An input DataFrame that includes ligand ids under "ligand_id", protein ids under
        "protein_id", SMILES strings under "smiles" and amino-acid sequence strings under
        "aa_sequence". This DataFrame includes ligands and proteins against which the
        average distances will be computed. If none, df_1 will be used as df_2.
    radius : int, optional
        radius hyperparameter for Morgan fingerprint computation.
    n_bits : int, optional
        nBits hyperparameter for Morgan fingerprint computation.
    
    Returns
    -------
    ligand_avg_distance : pd.Series
        A column the same length as the original dataset `df`, each row corresponding to the
        average distance of the ligand in the original row from the rest of the dataset.
    prot_avg_distance : pd.Series
        A column the same length as the original dataset `df`, each row corresponding to the
        average distance of the protein in the original row from the rest of the dataset.
    """
    try:
        df_1["ligand_id"] = df_1.ligand_id.astype(int).astype(str)
        df_2["ligand_id"] = df_2.ligand_id.astype(int).astype(str)
    except:
        pass
    if df_2 is None:
        df = df_1
        df_2 = df_1
    else:
        df = pd.concat([df_1, df_2])
    df_ligands, df_proteins = create_ligands_proteins_df(df, radius, n_bits)

    dist_ligands = compute_ligand_distances(df_ligands)
    df_ligand_distances = (dist_ligands.loc[:, df_2.ligand_id.value_counts().index] * (df_2.ligand_id.value_counts()/len(df_2))[df_2.ligand_id.value_counts().index]).sum(1)
    df_ligand_distances = df_ligand_distances.reset_index().rename(columns={0: "avg_ligand_distance"})
    df_1 = df_1.merge(df_ligand_distances, how="left", on="ligand_id")
    dist_prots = compute_protein_distances(df_proteins, prot_sim_matrix).loc[
        df.prot_id.value_counts().index, df.prot_id.value_counts().index
        ]
    df_prot_distances = (dist_prots.loc[:, df_2.prot_id.value_counts().index] * (df_2.prot_id.value_counts()/len(df_2))[df_2.prot_id.value_counts().index]).sum(1)
    df_prot_distances = df_prot_distances.reset_index().rename(columns={'index': 'prot_id', 0: "avg_prot_distance"})
    df_1 = df_1.merge(df_prot_distances, how="left", on="prot_id")

    return df_1["avg_ligand_distance"], df_1["avg_prot_distance"]


def compute_inv_frequency(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Creates a DataFrame given input ligands, adds Morgan fingerprints.

    Parameters
    ----------
    df : pd.DataFrame
        An input DataFrame that includes ligand ids under "ligand_id", protein ids under
        "protein_id", SMILES strings under "smiles" and amino-acid sequence strings under
        "aa_sequence".
    
    Returns
    -------
    ligand_inv_frequency : pd.Series
        A column the same length as the original dataset `df`, each row corresponding to the
        inverse frequency of the row's ligand in the dataset.
    prot_inv_frequency : pd.Series
        A column the same length as the original dataset `df`, each row corresponding to the
        inverse frequency of the row's protein in the dataset.
    """
    df = df.merge((1/df.ligand_id.value_counts()).reset_index().rename(columns={"index": "ligand_id", "ligand_id": "ligand_inv_frequency"}), how="left", on="ligand_id")
    df = df.merge((1/df.prot_id.value_counts()).reset_index().rename(columns={"index": "prot_id", "prot_id": "prot_inv_frequency"}), how="left", on="prot_id")
    return df["ligand_inv_frequency"], df["prot_inv_frequency"]


def robust_standardize(x: np.ndarray) -> np.ndarray:
    """Standardize a given vector of values using median and mean absolute deviation.
    
    Paramaters
    ----------
    x : np.ndarray
        Input vector
    
    Returns
    -------
    np.ndarray
        Standardized input vector.
    """
    return (x - np.median(x)) / np.mean(np.abs(x - np.median(x)))


def time_diff(now: float, then: float):
    """Computes the difference between two time points provided in seconds.
    
    Parameters
    ----------
    now : float
        Final time in seconds from a fixed starting point (e.g. "epoch").
    then : float
        Initial time in seconds from a fixed starting point (e.g. "epoch").
    """
    min, sec = divmod(round(now - then), 60)
    hour, min = divmod(min, 60)
    return f"{hour:02d}:{min:02d}:{sec:02d}"