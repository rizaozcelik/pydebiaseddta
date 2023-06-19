from typing import List, Dict
import numpy as np
import pandas as pd
import math
from .abstract_guide import Guide
from ..utils import compute_average_distances, compute_inv_frequency, robust_standardize

class OutDTA(Guide):
    def __init__(
            self,
            df: pd.DataFrame,
            rarity_indicator: str,
            combination_method: str = "prod",
            prot_sim_matrix: pd.DataFrame = None,
            **kwargs
            ):
        """Constructor to create a OutDTA model.
        
        OutDTA computes the "rarity" of each ligand and protein in the dataset, and creates
        an importance weight according to the selected rarity indicator. The available rarity
        indicators are "avg_distance" that computes the ligand similarity based on Tanimoto distance
        with Morgan fingerprint, and 1 - protein similarity based on the protein similarity matrix
        inputted to the `compute_rarity` function (the default recommendation is Smith-Waterman algorithm).

        The function admits a DataFrame that include id information of ligands and proteins in addition
        to SMILES and amino-acid representations. Slightly unintuitive method for providing the data
        is for compatibility reasons.

        Parameters
        ----------
        df : pd.DataFrame
            An input DataFrame that includes ligand ids under "ligand_id", protein ids under
            "protein_id", SMILES strings under "smiles" and amino-acid sequence strings under
            "aa_sequence".
        rarity_indicator : str
            What rarity indicator to use while computing importance weights. Available options are
            "avg_distance" and "inv_frequency".
        combination_method : str, optional
            How to combine the standardized rarities of ligands and proteins to create a single rarity
            indicator. Available options are "prod", "min", "max", "sum".
        prot_sim_matrix : pd.DataFrame
            Similarity matrix with protein ids in the index and columns. Only used if
            rarity_indicator == "avg_distance".
        
        Raises
        ------
        KeyError
            If any combinationmethod other than the options described above is provided.
        """
        self.df = df
        self.rarity_indicator = rarity_indicator
        self.combination_function = {
            "sum": sum, "prod": lambda a, b: math.prod((a, b)), "min": np.minimum, "max": np.maximum
            }[combination_method]
        self.prot_sim_matrix = prot_sim_matrix

    def train(self, df=None):
        """Compute rarity of each ligand and protein in the dataset.

        Parameters
        ----------
        df : pd.DataFrame, optional
            An input DataFrame that includes ligand ids under "ligand_id", protein ids under
            "protein_id", SMILES strings under "smiles" and amino-acid sequence strings under
            "aa_sequence".

        Raises
        ------
        ValueError
            If any rarity_indicator outside the options described above is provided.

        """
        if not df:
            df = self.df
        if self.rarity_indicator == "avg_distance":
            average_ligand_distance, average_prot_distance =  compute_average_distances(self.df, prot_sim_matrix=self.prot_sim_matrix)
        elif self.rarity_indicator == "inv_frequency":
            average_ligand_distance, average_prot_distance =  compute_inv_frequency(self.df)
        else:
            raise ValueError("The rarity indicator you have selected does not exist.")
        del self.df
        ald, apd = robust_standardize(average_ligand_distance.values), robust_standardize(average_prot_distance.values)
        self.importance_weights = self.combination_function(ald, apd)

    def get_importance_weights(self):
        """Retrieve the computed importance weights.

        Returns
        -------
        importance_weights : pd.Series
            A pandas Series that includes the importance weights for each row of the input df.
        """
        return self.importance_weights

    def predict(self):
        """Predict function for the class.

        Raises
        ------
        Exception
            This guide does not use predict function to provide importance weights.
        """
        raise Exception("This guide cannot be used to obtain predictions.")