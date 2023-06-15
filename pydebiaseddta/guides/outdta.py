from typing import List, Dict
import numpy as np
import pandas as pd
import math
from .abstract_guide import Guide
from ..utils import compute_average_distance, compute_inv_frequency, robust_standardize

class OutDTA(Guide):
    def __init__(
            self,
            ligands: pd.Series,
            proteins: pd.Series,
            ligand_id: pd.Series,
            protein_id: pd.Series,
            rarity_indicator: str,
            combination_method="prod",
            prot_sim_matrix_path=None,
            **kwargs
            ):
        """Constructor to create a OutDTA model.
        
        OutDTA computes the "rarity" of each ligand and protein in the dataset, and creates
        an importance weight according to the selected rarity indicator. The available rarity
        indicators are "avg_distance" that computes the ligand similarity based on Tanimoto distance
        with Morgan fingerprint, and 1 - protein similarity based on the protein similarity matrix
        inputted to the `compute_rarity` function (the default recommendation is Smith-Waterman algorithm).

        The function admits four pandas Series that constitute a single dataset of length N. Slightly unintuitive
        method for providing the data is due to compatibility reasons.

        Parameters
        ----------
        ligands : pd.Series
            A series of length N that includes SMILES representation of ligands.
        proteins : pd.Series
            A series of length N that includes amino-acid sequences for proteins.
        ligand_id : pd.Series
            A series of length N that includes ligand ids.
        protein_id : pd.Series
            A series of length N that includes protein ids.
        rarity_indicator : str
            What rarity indicator to use while computing importance weights. Available options are
            "avg_distance" and "inv_frequency".
        combination_method : str, optional
            How to combine the standardized rarities of ligands and proteins to create a single rarity
            indicator. Available options are "prod", "min", "max", "sum".
        prot_sim_matrix_path : str, optional
            Protein pairwise similarity matrix path. Only used if rarity_indicator == "avg_distance".
        
        Raises
        ------
        KeyError
            If any combinationmethod other than the options described above is provided.
        """
        self.df = pd.DataFrame({
            "ligand_id": ligand_id, "protein_id": protein_id, "smiles": ligands, "aa_sequence": proteins
            })
        self.rarity_indicator = rarity_indicator
        self.combination_function = {
            "sum": sum, "prod": lambda a, b: math.prod((a, b)), "min": np.minimum, "max": np.maximum
            }[combination_method]
        self.prot_sim_matrix_path = prot_sim_matrix_path

    def compute_importance_weight(self, df=None):
        """Compute rarity of each ligand and protein in the dataset.

        Parameters
        ----------
        df : pd.DataFrame, optional
            An input DataFrame that includes ligand ids under `ligand_id`, protein ids under
            `protein_id`, SMILES strings under `smiles` and amino-acid sequence strings under
            `aa_sequence`.

        Returns
        -------
        importance_weights : pd.Series
            A pandas Series that includes the importance weights for each row of the input df.
        
        Raises
        ------
        ValueError
            If any rarity_indicator outside the options described above is provided.

        """
        if not df:
            df = self.df
        if self.rarity_indicator == "avg_distance":
            average_ligand_distance, average_prot_distance =  compute_average_distance(self.df, prot_sim_matrix_path=self.prot_sim_matrix_path)
        elif self.rarity_indicator == "inv_frequency":
            average_ligand_distance, average_prot_distance =  compute_inv_frequency(self.df)
        else:
            raise ValueError("The rarity indicator you have selected does not exist.")
        del self.df
        ald, apd = robust_standardize(average_ligand_distance.values), robust_standardize(average_prot_distance.values)
        return self.combination_function(ald, apd)