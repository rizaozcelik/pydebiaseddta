from typing import List
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from .abstract_guide import Guide


def _list_to_numpy(lst):
    return np.array(lst).reshape(-1, 1)


class IDDTA(Guide):
    def __init__(self,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 criterion="squared_error",
                 **kwargs):
        """Constructor to create an IDDTA model.
        IDDTA represents the proteins and ligands with one-hot vectors of their identities
        and uses a decision tree for prediction. 

        Parameters
        ----------
        max_depth : int, optional
            Determines the maximum depth of the decision tree regressor.
        min_samples_split : int, optional
            Determines the minimum samples to split a leaf for the decision
            tree regressor.
        min_samples_leaf : int, optional
            Determines the minimum samples a leaf can have for the decision
            tree regressor.
        criterion : str, optional
            Criterion according to which the decision tree regressor will be trained.
        """
        self.prediction_model =  DecisionTreeRegressor(
            max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion
            )
        self.ligand_encoder = OneHotEncoder(handle_unknown="ignore")
        self.protein_encoder = OneHotEncoder(handle_unknown="ignore")

    def vectorize_ligands(self, ligands: List[str]) -> np.array:
        """Creates one-hot vectors of the ligands.

        Parameters
        ----------
        ligands : List[str]
            SMILES strings of the input ligands (other representations are also possible, but SMILES is used in this study).

        Returns
        -------
        np.array
            One-hot encoded vectors of the ligands.
        """
        ligands = np.array(ligands).reshape(-1, 1)
        return self.ligand_encoder.transform(ligands).todense()

    def vectorize_proteins(self, proteins: List[str]) -> np.array:
        """Creates one-hot vectors of the proteins.

        Parameters
        ----------
        proteins : List[str]
            Amino-acid sequences of the input proteins.

        Returns
        -------
        np.array
            One-hot encoded vectors of the proteins.
        """
        proteins = np.array(proteins).reshape(-1, 1)
        return self.protein_encoder.transform(proteins).todense()

    def train(
        self,
        train_ligands: List[str],
        train_proteins: List[str],
        train_labels: List[float],
    ):
        """Trains the IDDTA model. IDDTA represents the biomolecules with 
        one-hot-encoding of their identities and applies decision tree for affinity prediction.

        Parameters
        ----------
        train_ligands : List[str]
            SMILES strings of the training ligands.
        train_proteins : List[str]
            Amino-acid sequences of the training proteins.
        train_labels : List[float]
            Affinity scores of the interactions.
        """
        ligand_vecs = self.ligand_encoder.fit_transform(
            _list_to_numpy(train_ligands)
        ).todense()
        protein_vecs = self.protein_encoder.fit_transform(
            _list_to_numpy(train_proteins)
        ).todense()

        X_train = np.hstack([ligand_vecs, protein_vecs])
        self.prediction_model.fit(X_train, train_labels)

    def predict(self, ligands: List[str], proteins: List[str]) -> List[float]:
        """Predicts the affinities of a list of protein-ligand pairs.

        Parameters
        ----------
        ligands : List[str]
            SMILES strings of the ligands.
        proteins : List[str]
            Amino-acid sequences of the proteins.

        Returns
        -------
        List[float]
            Predicted affinities.
        """
        ligand_vecs = self.vectorize_ligands(ligands)
        protein_vecs = self.vectorize_proteins(proteins)
        X_test = np.hstack([ligand_vecs, protein_vecs])
        return self.prediction_model.predict(X_test)
