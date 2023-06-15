from typing import Dict, List, Tuple
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import json
import numpy as np
import pandas as pd
from . import package_path


def load_sample_dta_data(mini: bool = False) -> Dict[str, List]:
    """Loads a portion of [the BDB dataset](https://arxiv.org/pdf/2107.05556.pdf) for fast experimenting.

    Parameters
    ----------
    mini : bool, optional
        Whether to load all drug-target pairs embedded in the library, or a mini version.
        Set to `True` for fast prototyping and `False` to quickly train a model.
        Defaults to `False`.

    Returns
    -------
    Dict[str, List]
        The dictionary has three keys: `"train"`, `"val"`, and `"test"`, each corresponding to different folds of the dataset.
        Each key maps to a list with three elements: *list of ligands*, *list of proteins*, and *list of affinity scores*. 
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


def create_ligands_proteins_df(df: pd.Dataframe, radius: int = 2, n_bits: int = 2048) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Creates a DataFrame given input ligands, adds Morgan fingerprints.

    Parameters
    ----------
    df : pd.DataFrame
        An input DataFrame that includes ligand ids under `ligand_id`, protein ids under
        `protein_id`, SMILES strings under `smiles` and amino-acid sequence strings under
        `aa_sequence`.
    radius : int, optional
        radius hyperparameter for Morgan fingerprint computation.
    n_bits : int, optional
        nBits hyperparameter for Morgan fingerprint computation.
    
    Returns
    -------
    df_ligands : pd.DataFrame
        A DataFrame object that includes rdkit.Chem.rdchem.Mol representation of
        ligands under column `mol` as well as their Morgan fingerprint under `morgan_fp`.
        Also includes SMILES representation under `smiles` and includes ligand id as index.
    df_proteins : pd.DataFrame
        A DataFrame object that includes protein id as index and amino-acid sequences under
        column `aa_sequence`.
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
        Also includes SMILES representation under `smiles` and includes ligand id as index.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame that includes pairwise Tanimoto distances between ligands, where rows
        and columns are indexed by ligand id.
    """
    tanimoto_scores = [[DataStructs.FingerprintSimilarity(fp1, fp2, metric=DataStructs.TanimotoSimilarity) for fp2 in df_ligands['morgan_fp']] for fp1 in df_ligands['morgan_fp']]
    tanimoto_distances = 1 - pd.DataFrame(tanimoto_scores, index=df_ligands.index, columns=df_ligands.index)
    return tanimoto_distances


def compute_protein_distances(df_proteins: pd.DataFrame, prot_sim_matrix_path: str) -> pd.DataFrame:
    """Computes pairwise Tanimoto distances between ligands.
    
    Parameters
    ----------
    df_proteins : pd.DataFrame
        A DataFrame object that includes protein id as index and amino-acid sequences under
        column `aa_sequence`.
    prot_sim_matrix_path : str
        A path that includes the pairwise similarity between aminoacid sequences, where rows
        and columns must include the protein ids in df_proteins.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame that includes pairwise distances between proteins, where rows
        and columns are indexed by protein id.
    """
    sw_matrix = pd.read_csv(prot_sim_matrix_path, index_col=0).loc[df_proteins.index, df_proteins.index]
    sw_distances = 1 - sw_matrix
    return sw_distances


def compute_average_distance(
        df: pd.DataFrame,
        prot_sim_matrix_path: pd.DataFrame,
        radius: int = 2,
        n_bits: int = 2048,
        ) -> Tuple[pd.Series, pd.Series]:
    """Computes average distances of each ligand and protein from the rest of the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        An input DataFrame that includes ligand ids under `ligand_id`, protein ids under
        `protein_id`, SMILES strings under `smiles` and amino-acid sequence strings under
        `aa_sequence`.
    prot_sim_matrix_path : str
        A path that includes the pairwise similarity between aminoacid sequences, where rows
        and columns must include the protein ids in df_proteins. Smith-Waterman similarity
        is a desirable option and can be precomputed with various open source libraries.
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
    df_ligands, df_proteins = create_ligands_proteins_df(df, radius, n_bits)
    dist_ligands = compute_ligand_distances(df_ligands)
    dist_prots = compute_protein_distances(df_proteins, prot_sim_matrix_path)

    df_ligand_distances = (dist_ligands * (df.ligand_id.value_counts()/len(df))[dist_ligands.index]).sum(1)
    df_ligand_distances = df_ligand_distances.reset_index().rename(columns={0: "avg_ligand_distance"})
    df = df.merge(df_ligand_distances, how="left", on="ligand_id")

    dist_prots = dist_prots.loc[df.prot_id.value_counts().index, df.prot_id.value_counts().index]
    df_prot_distances = (dist_prots * (df.prot_id.value_counts()/len(df))[dist_prots.index]).sum(1)
    df_prot_distances = df_prot_distances.reset_index().rename(columns={'index': 'prot_id', 0: "avg_prot_distance"})
    df = df.merge(df_prot_distances, how="left", on="prot_id")
    
    return df["avg_ligand_distance"], df["avg_prot_distance"]


def compute_inv_frequency(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Creates a DataFrame given input ligands, adds Morgan fingerprints.

    Parameters
    ----------
    df : pd.DataFrame
        An input DataFrame that includes ligand ids under `ligand_id`, protein ids under
        `protein_id`, SMILES strings under `smiles` and amino-acid sequence strings under
        `aa_sequence`.
    
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
    return (x - np.median(x)) / np.median(np.abs(x - np.median(x)))