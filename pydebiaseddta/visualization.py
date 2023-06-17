import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import pandas as pd
from .predictors import Predictor
from itertools import product
import numpy as np


def ligand_protein_affinity_histograms(df: pd.DataFrame, bins: int = 50, figsize: Tuple[int,int] = (6, 6)):
    """Plots histograms for ligands, proteins, and affinity scores in the data.

    Parameters
    ----------
    df : pd.DataFrame
        An input DataFrame that includes ligand ids under "ligand_id", protein ids under
        "protein_id", SMILES strings under "smiles" and amino-acid sequence strings under
        "aa_sequence".
    bins : int
        Number of bins.
    figsize : Tuple[int, int]
        Figure size.
    
    Returns
    -------
    matplotlib.figure.Figure
        Figure object for the plot.
    """
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    axes[0].hist(df.ligand_id.value_counts(), bins=bins);
    axes[0].set_title("Ligand Counts Histogram")
    axes[0].set_xlabel("Ligand counts")
    axes[0].set_ylabel("Frequency")
    axes[1].hist(df.prot_id.value_counts(), bins=bins);
    axes[1].set_title("Protein Counts Histogram")
    axes[1].set_xlabel("Protein counts")
    axes[1].set_ylabel("Frequency")
    axes[2].hist(df.query("affinity_score != 0").affinity_score, bins=bins);
    axes[2].set_title("Affinity Score Histogram")
    axes[1].set_xlabel("Affinity score")
    axes[1].set_ylabel("Frequency")
    fig.tight_layout()
    return fig


def get_missing_interactions_df(
        df: pd.DataFrame,
        desired_ligands: List[str],
        desired_proteins: List[str],
        ) -> pd.DataFrame:
    """For a given dataset and a set of ligands and proteins, returns a dataframe with all missing affinities.

    Parameters
    ----------
    df : pd.DataFrame
        An input DataFrame that includes ligand ids under "ligand_id", protein ids under
        "protein_id", SMILES strings under "smiles" and amino-acid sequence strings under
        "aa_sequence", as well as affinity scores under "affinity_score".
    desired_ligands : List[str]
        A list of ligands for which the function gets the interactions with the provided proteins which
        are missing in df.
    desired_proteins : List[str]
        A list of proteins for which the function gets the interactions with the provided ligands which
        are missing in df.

    Returns
    -------
    pd.DataFrame
        A dataframe which includes all missing affinities between the desired sets of 
        ligands and proteins. Has same structure with df, only with the difference that 
        "affinity_score" column has null values.
    """
    df_desired = pd.DataFrame(np.array(list(product(desired_ligands, desired_proteins))), columns=["ligand_id", "prot_id"])
    df_desired["smiles"] = df.groupby("ligand_id")["smiles"].first()[df_desired["ligand_id"]].values
    df_desired["aa_sequence"] = df.groupby("prot_id")["aa_sequence"].first()[df_desired["prot_id"]].values
    df_augmented = df.merge(df_desired, on=["ligand_id", "smiles", "prot_id", "aa_sequence"], how="outer")
    df_missing_interactions = df_augmented.loc[df_augmented["affinity_score"].isnull()]
    return df_missing_interactions


def plot_prevalent_affinities(
        df: pd.DataFrame,
        k_prevalence: int,
        predictor: Predictor = None,
        ) -> Tuple[List[str], List[str], matplotlib.figure.Figure]:
    """Plots affinities for k prevalent ligands and proteins, and predicted affinities if model provided.

    Parameters
    ----------
    df : pd.DataFrame
        An input DataFrame that includes ligand ids under "ligand_id", protein ids under
        "protein_id", SMILES strings under "smiles" and amino-acid sequence strings under
        "aa_sequence", as well as affinity scores under "affinity_score".
    k_prevalence : int
        k for getting the k most prevalent proteins and ligands in the dataset and their interactions.
    predictor : Predictor
        When not provided, existing affinities within the data between k most prevalent proteins and
        ligands are plotted. If a predictor is provided, then the model's predictions for missing
        affinities are computed, and plotted both separately and combined with existing affinities.

    Returns
    -------
    prevalent_ligands : List[str]
        List of the most prevalent ligands plotted.
    prevalent_proteins : List[str]
        List of the most prevalent proteins plotted.
    fig : matplotlib.figure.Figure
        Figure object for the plot.
    """
    ligandinvmap = df.ligand_id.unique()
    ligandmap = {ligand: idd for idd, ligand in enumerate(ligandinvmap)}
    protinvmap = df.prot_id.unique()
    protmap = {prot: idd for idd, prot in enumerate(protinvmap)}

    prevalent_ligands = df.ligand_id.value_counts().iloc[:k_prevalence].index.tolist()
    prevalent_proteins = df.prot_id.value_counts().iloc[:k_prevalence].index.tolist()
    full_affinity_matrix = np.nan*np.ones((df.ligand_id.nunique(), df.prot_id.nunique()))
    for _, row in df.iterrows():
        full_affinity_matrix[ligandmap[row.ligand_id], protmap[row.prot_id]] = row.affinity_score
    affinity_matrix = full_affinity_matrix[[ligandmap[lig] for lig in prevalent_ligands], :]
    affinity_matrix = affinity_matrix[:, [protmap[prot] for prot in prevalent_proteins]]
    if not predictor:
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(affinity_matrix)
        ax.set_xlabel("Proteins"); ax.set_ylabel("Ligands")
        ax.set_title(f"Affinities for {k_prevalence} prevalent ligands and proteins")
        fig.colorbar(im, ax=ax, shrink=0.7, label="Affinity Score")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(17,6), sharey=True)
        axes[0].imshow(affinity_matrix)
        axes[0].set_xlabel("Proteins"); axes[0].set_ylabel("Ligands")
        axes[0].set_title(f"Affinities for {k_prevalence} prevalent ligands and proteins")
        df_missing_interactions = get_missing_interactions_df(df, prevalent_ligands, prevalent_proteins)
        df_missing_interactions["affinity_score"] = np.array(
            predictor.predict(ligands=df_missing_interactions.smiles.tolist(),
            proteins=df_missing_interactions.aa_sequence.tolist())
            ).flatten()
        new_full_affinity_matrix = np.nan*np.ones((df.ligand_id.nunique(), df.prot_id.nunique()))
        for i_, row in df_missing_interactions.iterrows():
            full_affinity_matrix[ligandmap[row.ligand_id], protmap[row.prot_id]] = row.affinity_score
            new_full_affinity_matrix[ligandmap[row.ligand_id], protmap[row.prot_id]] = row.affinity_score
        affinity_matrix = full_affinity_matrix[[ligandmap[lig] for lig in prevalent_ligands], :]
        affinity_matrix = affinity_matrix[:, [protmap[prot] for prot in prevalent_proteins]]
        new_affinity_matrix = new_full_affinity_matrix[[ligandmap[lig] for lig in prevalent_ligands], :]
        new_affinity_matrix = new_affinity_matrix[:, [protmap[prot] for prot in prevalent_proteins]]
        axes[1].imshow(new_affinity_matrix)
        axes[1].set_xlabel("Proteins");
        axes[1].set_title(f"Predicted missing affinities")
        im = axes[2].imshow(affinity_matrix)
        axes[2].set_xlabel("Proteins");
        axes[2].set_title(f"Existing and predicted affinities combined")
        fig.colorbar(im, ax=axes, label="Affinity Score", location="right", shrink=0.6)
    return prevalent_ligands, prevalent_proteins, fig


def plot_distance_error_split(
        dfs: Dict[str, pd.DataFrame],
        error_col: str,
        sample_per_split: int = 50,
        is_log_errors : bool = True,
        seed: int = 0,
        ) -> matplotlib.figure.Figure:
    """Plots prediction errors in relation to average distances and splits.

    dfs : Dict[str, pd.DataFrame]
        A dictionary that contains split names as keys and a corresponding pd.DataFrame. 
        Each DataFrame needs to include an "avg_ligand_distance" that holds the average
        ligand distances for each interaction's ligand to a set of ligands, an
        "avg_protein_distance" that holds the average protein distances for each interaction's
        proteins to a set of proteins, and a column that holds the prediction errors 
        under `error_col`.
    error_col : str
        The column name that holds error column in the DataFrame's provided.
    sample_per_split : int, optional
        The number of samples to be used from each split for a less cluttered visualization.
    is_log_errors : bool, optional
        Whether to take the log of the error values for visualization purposes.
    seed : int, optional
        The seed for the samples from splits.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object for the plot.
    """
    opt_log = lambda x: np.log(x) if is_log_errors else x 
    assert (len(dfs) in [1, 2, 3, 4])
    markers = ["o", "x", "s", "^"]

    np.random.seed(seed)
    dfs_sample = {key: dfs[key].sample(sample_per_split) for key in dfs.keys()}
    fig, ax = plt.subplots()
    errors = pd.concat([value for key, value in dfs_sample.items()])[error_col]
    vmin, vmax = opt_log(errors).min(), opt_log(errors).max()
    for i, (key, df_sample) in enumerate(dfs_sample.items()):
        im = ax.scatter(df_sample.avg_ligand_distance, df_sample.avg_protein_distance, vmin=vmin, vmax=vmax, alpha=0.8, c=opt_log(df_sample[error_col]), marker=markers[i], label=key)
    ax.legend(loc="lower right")
    ax.set_ylabel("Protein Average Distance to Training Proteins")
    ax.set_xlabel("Ligand Average Distance to Training Ligands")
    fig.colorbar(im, ax=ax, label="Log Error", location="right", shrink=0.7)
    return fig